from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from semantic_atdom.semantics.orchestrator import (
    AcquisitionOverlap,
    AsyncProgressiveSemanticHook,
    FrozenGaussianSnapshot,
    FrozenSemanticSnapshot,
    MapPublicationRequest,
    OnlineSemanticConfig,
    OnlineSemanticOrchestrator,
    ProgressiveSemanticHook,
    ProjectedObservation,
    SemanticSyncResult,
    label_boundary_weights,
)
from semantic_atdom.semantics.publication import (
    MapGridSpec,
    MapPublicationPolicy,
)
from semantic_atdom.semantics.tdom import SemanticTdomConfig, finalize_semantic_tdom
from semantic_atdom.semantics.schema import (
    ObservationKey,
    SemanticClassSpec,
    SemanticDataset,
    SemanticObservation,
    SemanticSchemaError,
    sha256_file,
)
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.topology import TopologyDelta


CLASS_SPEC_VALUE = {
    "schema_version": 1,
    "ignore_id": 255,
    "classes": [
        {"id": 0, "name": "ground", "color": [0, 0, 0]},
        {"id": 1, "name": "building", "color": [255, 0, 0]},
    ],
}
CLASS_SPEC = SemanticClassSpec.from_dict(CLASS_SPEC_VALUE)


def observation(image_id, label_hash=None):
    if label_hash is None:
        label_hash = image_id[-1] * 64
    return SemanticObservation(
        key=ObservationKey(image_id, "sam3-v1"),
        labels=np.zeros((1, 1), dtype=np.uint8),
        scores=None,
        metadata={
            "schema_version": 1,
            "image_id": image_id,
            "label_version": "sam3-v1",
            "model": "sam3",
            "checkpoint_sha256": "a" * 64,
            "prompt_config": {"classes": ["ground", "building"]},
            "thresholds": {"min_score": 0.2},
            "class_spec_sha256": CLASS_SPEC.fingerprint,
            "label_sha256": label_hash,
            "score_sha256": None,
            "width": 1,
            "height": 1,
        },
        label_path=Path(image_id + ".png"),
        score_path=None,
    )


def projection(distribution, support):
    support_tensor = torch.tensor([support], dtype=torch.float32)
    return ProjectedObservation(
        torch.tensor([distribution], dtype=torch.float32),
        support_tensor,
        support_tensor.clone(),
    )


def overlap(image_ids, values=None):
    matrix = np.eye(len(image_ids), dtype=np.float64) * 100.0
    for first, second, value in values or ():
        i = image_ids.index(first)
        j = image_ids.index(second)
        matrix[i, j] = matrix[j, i] = value
    return AcquisitionOverlap(tuple(image_ids), matrix)


class AcquisitionOverlapTests(unittest.TestCase):
    def test_file_rows_are_reordered_by_true_acquisition_number(self):
        with TemporaryDirectory() as directory:
            sparse = Path(directory) / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "imagesNames.txt").write_text(
                "000003.jpg,000001.jpg,000002.jpg\n", encoding="utf-8"
            )
            (sparse / "imageMatchMatrix.txt").write_text(
                "100,20,80\n20,100,50\n80,50,100\n", encoding="utf-8"
            )

            result = AcquisitionOverlap.load(Path(directory))

        self.assertEqual(
            result.acquisition_order, ("000001", "000002", "000003")
        )
        self.assertEqual(result.normalized("000001", "000002"), 0.5)
        self.assertEqual(result.normalized("000002", "000003"), 0.8)

    def test_fixed_layout_parses_inline_markers_and_decimal_timestamps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker = "00007FFE9CAB9DE0"
            (root / "match_matrix_image_name.txt").write_text(
                "1658137108.010.jpg,1658137107.900.jpg" + marker,
                encoding="utf-8",
            )
            (root / "match_matrix_value.txt").write_text(
                "100,25" + marker + "25,50" + marker,
                encoding="utf-8",
            )

            result = AcquisitionOverlap.load(root)

        self.assertEqual(
            result.acquisition_order,
            ("1658137107.900", "1658137108.010"),
        )
        self.assertEqual(
            result.normalized("1658137108.010", "1658137107.900"), 0.5
        )

    def test_anchor_overlap_prevents_transitive_group_growth(self):
        matrix = overlap(
            ["000001", "000002", "000003"],
            [
                ("000001", "000002", 60),
                ("000002", "000003", 80),
                ("000001", "000003", 20),
            ],
        )
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig()
        )
        runner.ingest(observation("000001"), projection([1, 0], 1), matrix)
        runner.ingest(observation("000002"), projection([1, 0], 1), matrix)
        runner.ingest(observation("000003"), projection([0, 1], 1), matrix)

        self.assertEqual(runner.open_member_ids, ("000003",))
        self.assertEqual(state.group_count.item(), 2)
        torch.testing.assert_close(
            state.alpha_raw, torch.tensor([[1.5, 1.5]])
        )


class OnlineSemanticOrchestratorTests(unittest.TestCase):
    def test_open_group_update_replaces_previous_provisional_evidence(self):
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig()
        )
        matrix = overlap(
            ["000001", "000002"], [("000001", "000002", 80)]
        )

        runner.ingest(observation("000001"), projection([1, 0], 0.5), matrix)
        runner.ingest(observation("000002"), projection([0, 1], 0.5), matrix)

        torch.testing.assert_close(
            state.alpha_raw, torch.tensor([[0.875, 0.875]])
        )
        self.assertEqual(state.group_count.item(), 1)
        self.assertEqual(state.semantic_support.item(), 1.0)
        self.assertEqual(runner.open_member_ids, ("000001", "000002"))

    def test_three_view_group_retains_only_anchor_and_latest(self):
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig(max_group_size=4)
        )
        matrix = overlap(
            ["000001", "000002", "000003"],
            [
                ("000001", "000002", 80),
                ("000001", "000003", 80),
            ],
        )
        runner.ingest(
            observation("000001"), projection([1.0, 0.0], 1.0), matrix
        )
        runner.ingest(
            observation("000002"), projection([0.0, 1.0], 1.0), matrix
        )
        runner.ingest(
            observation("000003"), projection([1.0, 0.0], 1.0), matrix
        )

        self.assertEqual(
            runner.open_group["member_image_ids"],
            ["000001", "000002", "000003"],
        )
        self.assertEqual(
            runner.open_group["representative_image_ids"],
            ["000001", "000003"],
        )
        self.assertEqual(state.open_distributions.shape[0], 2)
        torch.testing.assert_close(
            state.open_distributions[:, 0],
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        )

    def test_max_group_size_closes_group_and_third_group_is_soft_weighted(self):
        config = OnlineSemanticConfig(
            max_group_size=1, reliability_floor=0.1, temperature=0.1
        )
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(state, CLASS_SPEC, config)
        matrix = overlap(["000001", "000002", "000003"])

        runner.ingest(observation("000001"), projection([1, 0], 1), matrix)
        runner.ingest(observation("000002"), projection([1, 0], 1), matrix)
        runner.ingest(observation("000003"), projection([0, 1], 1), matrix)

        self.assertEqual(runner.open_member_ids, ())
        self.assertEqual(state.group_count.item(), 3)
        torch.testing.assert_close(
            state.alpha_raw, torch.tensor([[2.5, 1.5]])
        )
        self.assertEqual(state.alpha_robust[0, 0].item(), 2.5)
        self.assertGreaterEqual(state.alpha_robust[0, 1].item(), 0.6)
        self.assertLess(state.alpha_robust[0, 1].item(), 1.5)

    def test_online_loo_consensus_excludes_prior_and_requires_support(self):
        config = OnlineSemanticConfig(
            max_group_size=1,
            reliability_floor=0.1,
            temperature=0.1,
            min_consensus_support=1.0,
        )
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(state, CLASS_SPEC, config)
        matrix = overlap(["000001", "000002", "000003"])

        runner.ingest(observation("000001"), projection([1, 0], 1), matrix)
        runner.ingest(observation("000002"), projection([1, 0], 1), matrix)
        runner.ingest(observation("000003"), projection([0, 1], 1), matrix)

        expected_reliability = 0.1 + 0.9 * torch.exp(
            torch.tensor(-torch.log(torch.tensor(2.0)).item() / 0.1)
        )
        self.assertAlmostEqual(
            float(state.alpha_robust[0, 1] - 0.5),
            float(expected_reliability),
            places=5,
        )

        low_support_state = SemanticGaussianState.initialize(1, 2)
        low_support_runner = OnlineSemanticOrchestrator(
            low_support_state, CLASS_SPEC, config
        )
        low_support_runner.ingest(
            observation("000001"), projection([1, 0], 0.2), matrix
        )
        low_support_runner.ingest(
            observation("000002"), projection([1, 0], 0.2), matrix
        )
        low_support_runner.ingest(
            observation("000003"), projection([0, 1], 1.0), matrix
        )
        torch.testing.assert_close(
            low_support_state.alpha_robust, low_support_state.alpha_raw
        )

    def test_ledger_is_idempotent_and_rejects_changed_hash_for_same_key(self):
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig(max_group_size=1)
        )
        matrix = overlap(["000001"])
        original = observation("000001", "1" * 64)
        runner.ingest(original, projection([1, 0], 1), matrix)
        before = state.alpha_raw.clone()

        self.assertFalse(
            runner.ingest(original, projection([0, 1], 1), matrix)
        )
        torch.testing.assert_close(state.alpha_raw, before)
        with self.assertRaisesRegex(SemanticSchemaError, "conflicting"):
            runner.ingest(
                observation("000001", "2" * 64),
                projection([0, 1], 1),
                matrix,
            )
        torch.testing.assert_close(state.alpha_raw, before)

    def test_open_group_round_trip_can_continue_replacement(self):
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig()
        )
        matrix = overlap(
            ["000001", "000002"], [("000001", "000002", 80)]
        )
        runner.ingest(observation("000001"), projection([1, 0], 0.5), matrix)
        context = runner.recovery_context()
        recovery_hash = state.validated_recovery_context_sha256(
            context, CLASS_SPEC.ontology_fingerprint, 1.0, 2
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_state.pt"
            state.save(
                path,
                "a" * 64,
                "b" * 64,
                CLASS_SPEC.ontology_fingerprint,
                CLASS_SPEC.palette_fingerprint,
                context,
            )
            restored = SemanticGaussianState.load(
                path,
                expected_projection_geometry_hash="a" * 64,
                expected_ontology_hash=CLASS_SPEC.ontology_fingerprint,
                expected_gaussian_count=1,
                expected_recovery_context_sha256=recovery_hash,
            )

        resumed = OnlineSemanticOrchestrator.from_state(
            restored, CLASS_SPEC, OnlineSemanticConfig()
        )
        resumed.ingest(
            observation("000002"), projection([0, 1], 0.5), matrix
        )
        torch.testing.assert_close(
            restored.alpha_raw, torch.tensor([[0.875, 0.875]])
        )
        self.assertEqual(resumed.open_member_ids, ("000001", "000002"))

    def test_topology_change_transforms_open_group_cache_and_evidence(self):
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig()
        )
        runner.ingest(
            observation("000001"),
            projection([0.75, 0.25], 0.8),
            overlap(["000001"]),
        )

        state.apply_topology(TopologyDelta.clone(1, torch.tensor([0])))

        self.assertEqual(tuple(state.open_distributions.shape), (1, 2, 2))
        torch.testing.assert_close(
            state.open_supports, torch.tensor([[0.4, 0.4]])
        )
        torch.testing.assert_close(
            state.open_raw_evidence.sum(dim=0), torch.tensor([1.2, 0.4])
        )

        runner.refresh_open_group(
            {
                "000001": ProjectedObservation(
                    torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                    torch.tensor([0.5, 0.25]),
                    torch.tensor([3.0, 2.0]),
                )
            }
        )
        torch.testing.assert_close(
            state.open_raw_evidence,
            torch.tensor([[0.0, 0.5], [0.25, 0.0]]),
        )
        torch.testing.assert_close(
            state.semantic_support, torch.tensor([3.0, 2.0])
        )
        self.assertFalse(bool(state.active_projection_dirty.any()))
        self.assertTrue(bool(state.dirty.any()))
        self.assertEqual(len(runner.ledger.entries), 1)

    def test_invalid_projection_does_not_partially_mutate_open_group(self):
        state = SemanticGaussianState.initialize(1, 2)
        runner = OnlineSemanticOrchestrator(
            state, CLASS_SPEC, OnlineSemanticConfig()
        )
        matrix = overlap(["000001"])
        before = state.alpha_raw.clone()

        with self.assertRaisesRegex(Exception, "semantic support"):
            runner.ingest(
                observation("000001"),
                ProjectedObservation(
                    torch.tensor([[1.0, 0.0]]),
                    torch.tensor([1.0]),
                    torch.tensor([1.0, 2.0]),
                ),
                matrix,
            )

        torch.testing.assert_close(state.alpha_raw, before)
        self.assertIsNone(runner.open_group)
        self.assertEqual(len(runner.ledger.entries), 0)

    def test_boundary_ramp_preserves_small_regions_and_zeros_unknown(self):
        labels = np.array(
            [[0, 0, 255], [0, 1, 255], [0, 0, 0]], dtype=np.uint8
        )
        weights = label_boundary_weights(labels, bandwidth=2, floor=0.25)

        self.assertGreater(weights[1, 1], 0)
        self.assertLess(weights[1, 1], 1)
        self.assertEqual(weights[0, 2], 0)


class ProgressiveSemanticHookTests(unittest.TestCase):
    def test_online_config_declares_evidence_method_and_schema(self):
        payload = OnlineSemanticConfig().to_dict()

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["semantic_method_version"], "g1-evidence-v2")

    def test_semantic_snapshot_omits_appearance_but_pair_snapshot_can_render_rgb(self):
        class Gaussians:
            active_sh_degree = 2
            max_sh_degree = 3
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.ones((1, 1, 3))
            _features_rest = torch.ones((1, 15, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        semantic = FrozenSemanticSnapshot(Gaussians())
        paired = FrozenGaussianSnapshot(Gaussians())

        self.assertEqual(
            set(semantic._values), {"xyz", "opacity", "scaling", "rotation"}
        )
        self.assertFalse(hasattr(semantic, "get_features_dc"))
        self.assertEqual(paired.active_sh_degree, 2)
        self.assertEqual(paired.max_sh_degree, 3)
        torch.testing.assert_close(paired.get_features_dc, torch.ones((1, 1, 3)))
        torch.testing.assert_close(paired.get_features, torch.ones((1, 16, 3)))

    def test_core_publishes_rgb_and_semantics_from_the_same_frozen_snapshot(self):
        class Camera:
            image_name = "000001"
            image_height = 1
            image_width = 1
            FoVx = 1.0
            FoVy = 1.0
            world_view_transform = torch.eye(4)
            full_proj_transform = torch.eye(4)
            camera_center = torch.zeros(3)
            Tri_Mask = None

        class Gaussians:
            active_sh_degree = 0
            max_sh_degree = 0
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.ones((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

            @property
            def get_xyz(self):
                return self._xyz

        class NeverProject:
            def project(self, *args):
                raise AssertionError("missing labels must not be projected")

        class Pipeline:
            compute_cov3D_python = False
            GetNormal = False

        grid = MapGridSpec.create(
            plane_normal_world=(0.0, 0.0, 1.0),
            pixel_x_axis_world=(1.0, 0.0, 0.0),
            pixel_y_axis_world=(0.0, 1.0, 0.0),
            plane_offset_world=0.0,
            first_bounds=(0.0, 1.0, 0.0, 1.0),
            requested_gsd=(1.0, 1.0),
            tile_shape=(1, 1),
        )
        window = grid.window_for_bounds((0.0, 1.0, 0.0, 1.0))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic_root = root / "semantic"
            semantic_root.mkdir()
            (semantic_root / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC_VALUE), encoding="utf-8"
            )
            sparse = root / "1" / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "imagesNames.txt").write_text(
                "000001.jpg\n", encoding="utf-8"
            )
            (sparse / "imageMatchMatrix.txt").write_text(
                "100\n", encoding="utf-8"
            )
            dataset = SemanticDataset(root)
            state = SemanticGaussianState.initialize(1, 2)
            runner = OnlineSemanticOrchestrator(
                state, dataset.class_spec, OnlineSemanticConfig()
            )
            hook = ProgressiveSemanticHook(
                dataset,
                runner,
                NeverProject(),
                root / "state",
                publication_policy=MapPublicationPolicy(
                    checkpoint_stage_interval=1
                ),
            )
            frozen = FrozenGaussianSnapshot(Gaussians())
            rendered_models = []

            def rgb_renderer(camera, model, pipeline, background, **kwargs):
                rendered_models.append(model)
                return {"render": torch.tensor([[[0.0]], [[0.5]], [[1.0]]])}

            publication = MapPublicationRequest(
                product_key="stage_000001",
                output_root=root / "products" / "stage_000001",
                camera=Camera(),
                alignment={"grid": grid.to_dict(), "window": window.to_dict()},
                rgb_renderer=rgb_renderer,
                rgb_pipeline=Pipeline(),
                rgb_background=torch.zeros(3),
                semantic_config=SemanticTdomConfig(),
            )
            semantic_result = finalize_semantic_tdom(
                torch.zeros((3, 1, 1)),
                torch.zeros((1, 1)),
                CLASS_SPEC,
                SemanticTdomConfig(),
            )
            with patch(
                "semantic_atdom.semantics.orchestrator.rasterize_semantic_tdom",
                return_value=semantic_result,
            ), patch(
                "semantic_atdom.semantics.orchestrator.rasterize_rgb_tdom_striped",
                return_value=torch.tensor([[[0.0]], [[0.5]], [[1.0]]]),
            ) as striped_rgb:
                result = hook.synchronize(
                    1,
                    [Camera()],
                    frozen,
                    root / "1",
                    publication_request=publication,
                )
                first_slot = json.loads(
                    (root / "state" / "latest.json").read_text(encoding="utf-8")
                )["slot"]
                second_sparse = root / "2" / "sparse" / "0"
                second_sparse.mkdir(parents=True)
                (second_sparse / "imagesNames.txt").write_text(
                    "000001.jpg\n", encoding="utf-8"
                )
                (second_sparse / "imageMatchMatrix.txt").write_text(
                    "100\n", encoding="utf-8"
                )
                second_publication = MapPublicationRequest(
                    product_key="stage_000002",
                    output_root=root / "products" / "stage_000002",
                    camera=Camera(),
                    alignment={
                        "grid": grid.to_dict(),
                        "window": window.to_dict(),
                    },
                    rgb_renderer=rgb_renderer,
                    rgb_pipeline=Pipeline(),
                    rgb_background=torch.zeros(3),
                    semantic_config=SemanticTdomConfig(render_tile_rows=16),
                )
                second_result = hook.synchronize(
                    2,
                    [Camera()],
                    FrozenGaussianSnapshot(Gaussians()),
                    root / "2",
                    publication_request=second_publication,
                )

            pair = json.loads(
                (
                    root
                    / "products"
                    / "stage_000001"
                    / "pair_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(result.publication_status, "complete")
            self.assertEqual(second_result.publication_status, "complete")
            self.assertEqual(rendered_models[0], frozen)
            self.assertEqual(len(rendered_models), 1)
            striped_rgb.assert_called_once()
            self.assertEqual(
                pair["projection_geometry_hash"],
                frozen.semantic_projection_sha256(),
            )
            self.assertEqual(pair["rgb_model_hash"], frozen.rgb_model_sha256())
            second_pair = json.loads(
                (
                    root
                    / "products"
                    / "stage_000002"
                    / "pair_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsNotNone(second_pair["semantic_checkpoint_hash"])
            self.assertNotEqual(
                json.loads(
                    (root / "state" / "latest.json").read_text(
                        encoding="utf-8"
                    )
                )["slot"],
                first_slot,
            )

    def test_final_hook_rebuilds_every_ready_label_exactly_twice(self):
        class Camera:
            def __init__(self, image_name):
                self.image_name = image_name

        class Gaussians:
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.zeros((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

            @property
            def get_xyz(self):
                return self._xyz

            def geometry_sha256(self):
                return FrozenGaussianSnapshot(self).geometry_sha256()

        class Projector:
            def __init__(self):
                self.calls = 0

            def project(self, gaussians, camera, item):
                self.calls += 1
                value = [1.0, 0.0] if item.key.image_id != "000003" else [0.0, 1.0]
                return projection(value, 1.0)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            (semantic / "labels").mkdir(parents=True)
            (semantic / "metadata").mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC_VALUE), encoding="utf-8"
            )
            for image_id in ("000001", "000002", "000003"):
                label_path = semantic / "labels" / (image_id + ".png")
                Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(label_path)
                metadata = observation(image_id).metadata.copy()
                metadata["class_spec_sha256"] = CLASS_SPEC.fingerprint
                metadata["label_sha256"] = sha256_file(label_path)
                (semantic / "metadata" / (image_id + ".json")).write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
            sparse = root / "3" / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "imagesNames.txt").write_text(
                "000001,000002,000003\n", encoding="utf-8"
            )
            (sparse / "imageMatchMatrix.txt").write_text(
                "100,0,0\n0,100,0\n0,0,100\n", encoding="utf-8"
            )
            dataset = SemanticDataset(root)
            projector = Projector()
            runner = OnlineSemanticOrchestrator(
                SemanticGaussianState.initialize(1, 2),
                dataset.class_spec,
                OnlineSemanticConfig(max_group_size=1),
            )
            hook = ProgressiveSemanticHook(
                dataset, runner, projector, root / "output"
            )

            preview = hook.synchronize(
                3,
                [Camera("000001"), Camera("000002"), Camera("000003")],
                Gaussians(),
                root / "3",
            )
            self.assertEqual(preview.status, "approx_online")
            result = hook.synchronize(
                3,
                [Camera("000001"), Camera("000002"), Camera("000003")],
                Gaussians(),
                root / "3",
                final=True,
            )

            self.assertEqual(result.errors, ())
            self.assertEqual(result.status, "exact_final")
            self.assertEqual(projector.calls, 9)
            self.assertTrue(
                (root / "output" / "final_rebuild_metrics.json").is_file()
            )
            self.assertEqual(hook.orchestrator.state.group_count.item(), 3)
            self.assertEqual(hook.orchestrator.open_member_ids, ())
            stale_result = hook.synchronize(
                4,
                [Camera("000001"), Camera("000002"), Camera("000003")],
                Gaussians(),
                root / "3",
            )
            self.assertEqual(stale_result.status, "stale_online")
            self.assertTrue(stale_result.historical_evidence_stale)

            missing_projector = Projector()
            missing_runner = OnlineSemanticOrchestrator(
                SemanticGaussianState.initialize(1, 2),
                dataset.class_spec,
                OnlineSemanticConfig(max_group_size=1),
            )
            missing_hook = ProgressiveSemanticHook(
                dataset,
                missing_runner,
                missing_projector,
                root / "missing-output",
            )
            missing_result = missing_hook.synchronize(
                3,
                [Camera("000001"), Camera("000002")],
                Gaussians(),
                root / "3",
                final=True,
            )
            self.assertIn("missing cameras", missing_result.errors[0])
            self.assertEqual(missing_result.status, "exact_partial")
            self.assertEqual(missing_projector.calls, 6)
            self.assertTrue(
                (root / "missing-output" / "final_rebuild_metrics.json").exists()
            )

            unbound_hook = ProgressiveSemanticHook(
                dataset,
                OnlineSemanticOrchestrator(
                    SemanticGaussianState.initialize(1, 2),
                    dataset.class_spec,
                    OnlineSemanticConfig(max_group_size=1),
                ),
                Projector(),
                root / "unbound-output",
                final_authoritative_complete=False,
            )
            unbound_result = unbound_hook.synchronize(
                3,
                [Camera("000001"), Camera("000002"), Camera("000003")],
                Gaussians(),
                root / "3",
                final=True,
            )
            self.assertEqual(unbound_result.status, "exact_partial")

            failing_projector = Projector()
            failing_runner = OnlineSemanticOrchestrator(
                SemanticGaussianState.initialize(1, 2),
                dataset.class_spec,
                OnlineSemanticConfig(max_group_size=1),
            )
            online_state = failing_runner.state
            failing_hook = ProgressiveSemanticHook(
                dataset,
                failing_runner,
                failing_projector,
                root / "failing-output",
            )
            original_save = failing_hook._save_json_atomic
            failed_once = {"value": False}

            def fail_metrics_once(path, payload):
                if path.name == "final_rebuild_metrics.json" and not failed_once["value"]:
                    failed_once["value"] = True
                    raise OSError("injected metrics failure")
                return original_save(path, payload)

            failing_hook._save_json_atomic = fail_metrics_once
            failing_result = failing_hook.synchronize(
                3,
                [Camera("000001"), Camera("000002"), Camera("000003")],
                Gaussians(),
                root / "3",
                final=True,
            )
            self.assertIn("injected metrics failure", failing_result.errors[0])
            self.assertEqual(failing_result.status, "online_fallback")
            self.assertIs(failing_hook.orchestrator.state, online_state)

            recovery_projector = Projector()
            recovery_state = SemanticGaussianState.initialize(1, 2)
            recovery_state.invalidate("injected online failure")
            recovery_hook = ProgressiveSemanticHook(
                dataset,
                OnlineSemanticOrchestrator(
                    recovery_state,
                    dataset.class_spec,
                    OnlineSemanticConfig(max_group_size=1),
                ),
                recovery_projector,
                root / "recovery-output",
            )
            recovery_result = recovery_hook.synchronize(
                3,
                [Camera("000001"), Camera("000002"), Camera("000003")],
                Gaussians(),
                root / "3",
                final=True,
            )
            self.assertEqual(recovery_result.status, "exact_final")
            self.assertEqual(recovery_projector.calls, 6)
            self.assertTrue(recovery_hook.orchestrator.state.is_valid)

    def test_missing_offline_label_is_deferred_without_projection(self):
        class NeverProject:
            def project(self, *args):
                raise AssertionError("missing labels must not be projected")

        class Camera:
            image_name = "000001"

        class Gaussians:
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.zeros((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

            @property
            def get_xyz(self):
                return self._xyz

            def geometry_sha256(self):
                return FrozenGaussianSnapshot(self).geometry_sha256()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            semantic.mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC_VALUE), encoding="utf-8"
            )
            sparse = root / "1" / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "imagesNames.txt").write_text(
                "000001\n", encoding="utf-8"
            )
            (sparse / "imageMatchMatrix.txt").write_text(
                "100\n", encoding="utf-8"
            )
            dataset = SemanticDataset(root)
            state = SemanticGaussianState.initialize(1, 2)
            runner = OnlineSemanticOrchestrator(
                state, dataset.class_spec, OnlineSemanticConfig()
            )
            hook = ProgressiveSemanticHook(
                dataset,
                runner,
                NeverProject(),
                root / "output",
                publication_policy=MapPublicationPolicy(
                    checkpoint_stage_interval=1
                ),
            )

            result = hook.synchronize(
                1, [Camera()], Gaussians(), root / "1"
            )

            self.assertEqual(result.deferred_image_ids, ("000001",))
            self.assertEqual(result.lag_batches, 1)
            self.assertTrue(result.within_lag_budget)
            self.assertEqual(result.status, "stale_online")
            self.assertTrue((root / "output" / "latest.json").is_file())
            self.assertTrue(
                any((root / "output").glob("semantic_state.*.pt"))
            )
            latest = json.loads(
                (root / "output" / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(latest["schema_version"], 5)
            self.assertEqual(
                latest["semantic_method_version"], "g1-evidence-v2"
            )
            self.assertEqual(
                latest["ontology_hash"], dataset.class_spec.ontology_fingerprint
            )
            self.assertEqual(
                latest["palette_hash"], dataset.class_spec.palette_fingerprint
            )
            self.assertIn("projection_geometry_hash", latest)
            self.assertIn("rgb_model_hash", latest)
            self.assertIsNone(latest["rgb_model_hash"])
            self.assertEqual(
                latest["semantic_state_sha256"],
                sha256_file(root / "output" / latest["semantic_state"]),
            )
            self.assertTrue(
                (root / "output" / latest["geometry_checkpoint"]).is_file()
            )
            first_slot = latest["slot"]
            appearance_changed = Gaussians()
            appearance_changed._features_dc = torch.ones((1, 1, 3))
            restored = ProgressiveSemanticHook.create(
                root,
                root / "output",
                appearance_changed,
                projector=NeverProject(),
                publication_policy=MapPublicationPolicy(
                    checkpoint_stage_interval=1
                ),
            )
            self.assertEqual(restored._stage_boundaries, ["000001"])
            restored.synchronize(2, [Camera()], Gaussians(), root / "1")
            latest = json.loads(
                (root / "output" / "latest.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(first_slot, latest["slot"])
            (root / "output" / "semantic_state.{}.pt".format(first_slot)).write_text(
                "interrupted inactive slot", encoding="utf-8"
            )
            recovered = ProgressiveSemanticHook.create(
                root,
                root / "output",
                Gaussians(),
                projector=NeverProject(),
            )
            self.assertEqual(recovered.orchestrator.state.num_gaussians, 1)
            recovered.orchestrator.state.invalidate("injected unrecoverable state")
            failed = recovered.synchronize(
                3, [Camera()], Gaussians(), root / "1", final=True
            )
            self.assertEqual(failed.status, "failed")
            persisted_status = json.loads(
                (root / "output" / "sync_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(persisted_status["status"], "failed")
            fresh = ProgressiveSemanticHook.create(
                root,
                root / "output",
                Gaussians(),
                projector=NeverProject(),
                resume=False,
            )
            self.assertTrue(fresh.orchestrator.state.is_valid)
            self.assertEqual(fresh._stage_boundaries, [])

    def test_async_hook_returns_before_semantic_projection_finishes(self):
        started = threading.Event()
        release = threading.Event()
        snapshot_types = []

        class Core:
            def __init__(self):
                class Runner:
                    state = SemanticGaussianState.initialize(1, 2)

                class ClassSpec:
                    num_classes = 2

                class Dataset:
                    class_spec = ClassSpec()

                self.orchestrator = Runner()
                self.dataset = Dataset()
                self.publication_policy = MapPublicationPolicy()

            def checkpoint_required(self, *args):
                return True

            def synchronize(
                self, stage, cameras, gaussians, root, final, boundaries, *tdom
            ):
                snapshot_types.append(type(gaussians))
                started.set()
                release.wait(timeout=2)
                return type_result(stage)

            def begin_final(self):
                return None

        class Camera:
            image_name = "000001"
            image_height = 1
            image_width = 1
            FoVx = 1.0
            FoVy = 1.0
            world_view_transform = torch.eye(4)
            full_proj_transform = torch.eye(4)
            camera_center = torch.zeros(3)
            Tri_Mask = None

        class Gaussians:
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.zeros((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        def type_result(stage):
            from semantic_atdom.semantics.orchestrator import SemanticSyncResult

            return SemanticSyncResult(stage, "000001", (), (), 0, True, ())

        hook = AsyncProgressiveSemanticHook(Core())
        before = time.monotonic()
        result = hook.synchronize(1, [Camera()], Gaussians(), Path("unused"))
        elapsed = time.monotonic() - before
        self.assertTrue(started.wait(timeout=1))
        self.assertLess(elapsed, 0.5)
        self.assertEqual(result.lag_batches, 1)
        release.set()
        final = hook.synchronize(
            1, [Camera()], Gaussians(), Path("unused"), final=True
        )
        self.assertEqual(final.semantic_through_image, "000001")
        self.assertEqual(snapshot_types, [FrozenSemanticSnapshot] * 2)
        hook.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            hook.synchronize(1, [Camera()], Gaussians(), Path("unused"))

    def test_latest_wins_skips_old_maps_but_newest_stage_keeps_all_cameras(self):
        started = threading.Event()
        release = threading.Event()
        observed = []

        class Core:
            def __init__(self):
                class Runner:
                    state = SemanticGaussianState.initialize(1, 2)

                class ClassSpec:
                    num_classes = 2

                class Dataset:
                    class_spec = ClassSpec()

                self.orchestrator = Runner()
                self.dataset = Dataset()
                self.publication_policy = MapPublicationPolicy()

            def checkpoint_required(self, *args):
                return False

            def synchronize(
                self, stage, cameras, gaussians, root, final, boundaries, publication
            ):
                observed.append(
                    (
                        stage,
                        tuple(camera.image_name for camera in cameras),
                        None if publication is None else publication.product_key,
                    )
                )
                if stage == 1:
                    started.set()
                    release.wait(timeout=2)
                return SemanticSyncResult(
                    stage,
                    cameras[-1].image_name,
                    tuple(camera.image_name for camera in cameras),
                    (),
                    0,
                    True,
                    (),
                )

            def begin_final(self):
                return None

        class Camera:
            image_height = 1
            image_width = 1
            FoVx = 1.0
            FoVy = 1.0
            world_view_transform = torch.eye(4)
            full_proj_transform = torch.eye(4)
            camera_center = torch.zeros(3)
            Tri_Mask = None

            def __init__(self, image_name):
                self.image_name = image_name

        class Gaussians:
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.zeros((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

            @property
            def get_xyz(self):
                return self._xyz

        class Pipeline:
            compute_cov3D_python = False
            GetNormal = False

        hook = AsyncProgressiveSemanticHook(Core())
        hook.synchronize(1, [Camera("000001")], Gaussians(), Path("unused"))
        self.assertTrue(started.wait(timeout=1))
        hook.synchronize(
            2,
            [Camera("000001"), Camera("000002")],
            Gaussians(),
            Path("unused"),
            publication_request=MapPublicationRequest(
                product_key="stage_000002",
                output_root=Path("unused") / "stage_000002",
                camera=Camera("semantic_tdom"),
                alignment={
                    "geometry_stage": 2,
                    "grid": {"grid_id": "fixed"},
                    "window": {"grid_id": "fixed"},
                },
                rgb_renderer=lambda *args, **kwargs: None,
                rgb_pipeline=Pipeline(),
                rgb_background=torch.zeros(3),
                semantic_config=SemanticTdomConfig(),
            ),
        )
        hook.synchronize(
            3,
            [Camera("000001"), Camera("000002"), Camera("000003")],
            Gaussians(),
            Path("unused"),
            publication_request=MapPublicationRequest(
                product_key="stage_000003",
                output_root=Path("unused") / "stage_000003",
                camera=Camera("semantic_tdom"),
                alignment={
                    "geometry_stage": 3,
                    "grid": {"grid_id": "fixed"},
                    "window": {"grid_id": "fixed"},
                },
                rgb_renderer=lambda *args, **kwargs: None,
                rgb_pipeline=Pipeline(),
                rgb_background=torch.zeros(3),
                semantic_config=SemanticTdomConfig(),
                publication_due=False,
            ),
        )
        hook.synchronize(
            4,
            [
                Camera("000001"),
                Camera("000002"),
                Camera("000003"),
                Camera("000004"),
            ],
            Gaussians(),
            Path("unused"),
        )
        release.set()
        hook.synchronize(
            5,
            [
                Camera("000001"),
                Camera("000002"),
                Camera("000003"),
                Camera("000004"),
                Camera("000005"),
            ],
            Gaussians(),
            Path("unused"),
            final=True,
        )

        self.assertEqual([stage for stage, _, _ in observed], [1, 3, 4, 5])
        self.assertEqual(
            observed[1][1], ("000001", "000002", "000003")
        )
        self.assertEqual(observed[1][2], "stage_000003")
        self.assertEqual(
            observed[2][1],
            ("000001", "000002", "000003", "000004"),
        )
        self.assertIsNone(observed[2][2])

    def test_output_preflight_skips_full_rgb_snapshot(self):
        observed = []

        class Core:
            def __init__(self):
                class Runner:
                    state = SemanticGaussianState.initialize(1, 2)

                class ClassSpec:
                    num_classes = 2

                class Dataset:
                    class_spec = ClassSpec()

                self.orchestrator = Runner()
                self.dataset = Dataset()
                self.publication_policy = MapPublicationPolicy(
                    max_pair_output_bytes=1
                )

            def synchronize(
                self, stage, cameras, gaussians, root, final, boundaries, publication
            ):
                observed.append((type(gaussians), publication))
                return SemanticSyncResult(stage, "000001", (), (), 0, True, ())

        class Camera:
            image_name = "000001"
            image_height = 1
            image_width = 1
            FoVx = 1.0
            FoVy = 1.0
            world_view_transform = torch.eye(4)
            full_proj_transform = torch.eye(4)
            camera_center = torch.zeros(3)
            Tri_Mask = None

        class Gaussians:
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.zeros((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

            @property
            def get_xyz(self):
                return self._xyz

        class Pipeline:
            compute_cov3D_python = False
            GetNormal = False

        hook = AsyncProgressiveSemanticHook(Core())
        result = hook.synchronize(
            1,
            [Camera()],
            Gaussians(),
            Path("unused"),
            publication_request=MapPublicationRequest(
                product_key="stage_000001",
                output_root=Path("unused") / "stage_000001",
                camera=Camera(),
                alignment={
                    "grid": {"grid_id": "fixed"},
                    "window": {"grid_id": "fixed"},
                },
                rgb_renderer=lambda *args, **kwargs: None,
                rgb_pipeline=Pipeline(),
                rgb_background=torch.zeros(3),
                semantic_config=SemanticTdomConfig(),
            ),
        )
        self.assertEqual(result.publication_status, "skipped_output_budget")
        self.assertTrue(hook._idle_event.wait(timeout=2))
        self.assertEqual(observed, [(FrozenSemanticSnapshot, None)])
        hook.close()

    def test_async_final_rebuild_runs_after_online_worker_failure(self):
        class Core:
            def __init__(self):
                class Runner:
                    state = SemanticGaussianState.initialize(1, 2)

                self.orchestrator = Runner()

            def synchronize(
                self, stage, cameras, gaussians, root, final, boundaries, *tdom
            ):
                if not final:
                    raise RuntimeError("injected online failure")
                self.orchestrator.state = SemanticGaussianState.initialize(1, 2)
                return SemanticSyncResult(
                    stage,
                    "000001",
                    (),
                    (),
                    0,
                    True,
                    (),
                    "exact_final",
                    False,
                )

            def begin_final(self):
                return None

        class Camera:
            image_name = "000001"
            image_height = 1
            image_width = 1
            FoVx = 1.0
            FoVy = 1.0
            world_view_transform = torch.eye(4)
            full_proj_transform = torch.eye(4)
            camera_center = torch.zeros(3)
            Tri_Mask = None

        class Gaussians:
            _xyz = torch.zeros((1, 3))
            _features_dc = torch.zeros((1, 1, 3))
            _features_rest = torch.zeros((1, 0, 3))
            _opacity = torch.zeros((1, 1))
            _scaling = torch.zeros((1, 3))
            _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        hook = AsyncProgressiveSemanticHook(Core())
        hook.synchronize(1, [Camera()], Gaussians(), Path("unused"))
        self.assertTrue(hook._idle_event.wait(timeout=2))
        self.assertFalse(hook.is_valid)

        result = hook.synchronize(
            1, [Camera()], Gaussians(), Path("unused"), final=True
        )

        self.assertEqual(result.status, "exact_final")

    def test_async_hook_applies_topology_only_through_each_frozen_snapshot(self):
        started = threading.Event()
        release = threading.Event()
        observed = []

        class Core:
            def __init__(self):
                class Runner:
                    state = SemanticGaussianState.initialize(1, 2)

                self.orchestrator = Runner()

            def synchronize(
                self, stage, cameras, gaussians, root, final, boundaries, *tdom
            ):
                if stage == 1:
                    started.set()
                    release.wait(timeout=2)
                state_count = self.orchestrator.state.num_gaussians
                snapshot_count = gaussians.get_xyz.shape[0]
                observed.append((stage, state_count, snapshot_count))
                if state_count != snapshot_count:
                    raise RuntimeError(
                        "semantic state/snapshot topology mismatch: {} != {}".format(
                            state_count, snapshot_count
                        )
                    )
                return type_result(stage)

            def begin_final(self):
                return None

        class Camera:
            image_name = "000001"
            image_height = 1
            image_width = 1
            FoVx = 1.0
            FoVy = 1.0
            world_view_transform = torch.eye(4)
            full_proj_transform = torch.eye(4)
            camera_center = torch.zeros(3)
            Tri_Mask = None

        class Gaussians:
            def __init__(self, count):
                self._xyz = torch.zeros((count, 3))
                self._features_dc = torch.zeros((count, 1, 3))
                self._features_rest = torch.zeros((count, 0, 3))
                self._opacity = torch.zeros((count, 1))
                self._scaling = torch.zeros((count, 3))
                self._rotation = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]]
                ).repeat(count, 1)

        def type_result(stage):
            from semantic_atdom.semantics.orchestrator import SemanticSyncResult

            return SemanticSyncResult(stage, "000001", (), (), 0, True, ())

        hook = AsyncProgressiveSemanticHook(Core())
        hook.synchronize(1, [Camera()], Gaussians(1), Path("unused"))
        self.assertTrue(started.wait(timeout=1))
        hook.apply_topology(TopologyDelta.append_prior(1, 1))
        hook.synchronize(2, [Camera()], Gaussians(2), Path("unused"))
        hook.apply_topology(TopologyDelta.append_prior(2, 1))
        release.set()

        final = hook.synchronize(
            3, [Camera()], Gaussians(3), Path("unused"), final=True
        )

        self.assertEqual(final.semantic_through_image, "000001")
        self.assertEqual(observed, [(1, 1, 1), (2, 2, 2), (3, 3, 3)])

    def test_async_topology_targets_state_replaced_by_online_rebuild(self):
        started = threading.Event()
        release = threading.Event()

        class Runner:
            def __init__(self, count):
                self.state = SemanticGaussianState.initialize(count, 2)

        class Core:
            def __init__(self):
                self.orchestrator = Runner(1)

            def synchronize(self, *args):
                started.set()
                release.wait(timeout=2)
                # Online out-of-order replay atomically installs a freshly
                # rebuilt orchestrator while later topology work is queued.
                self.orchestrator = Runner(1)
                return type_result(1)

        def type_result(stage):
            from semantic_atdom.semantics.orchestrator import SemanticSyncResult

            return SemanticSyncResult(stage, "000001", (), (), 0, True, ())

        hook = AsyncProgressiveSemanticHook(Core())
        try:
            snapshot = type("Snapshot", (), {"wait_ready": lambda self: None})()
            request = type(
                "Request",
                (),
                {
                    "args": (1, (), snapshot, Path("unused"), False, (), None),
                    "publication_status_override": None,
                },
            )()
            hook._stage_pending = True
            hook._pending_stage_count = 1
            hook._idle_event.clear()
            future = hook._executor.submit(hook._run_stage, request)
            future.add_done_callback(hook._stage_done)
            self.assertTrue(started.wait(timeout=1))

            hook.apply_topology(TopologyDelta.append_prior(1, 1))
            release.set()
            hook.wait_for_idle()

            self.assertEqual(hook.num_gaussians, 2)
            self.assertEqual(hook.orchestrator.state.num_gaussians, 2)
        finally:
            release.set()
            hook.close(wait=True)


if __name__ == "__main__":
    unittest.main()
