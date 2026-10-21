from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

import semantic_atdom.semantics.publication as publication_module
from semantic_atdom.semantics.publication import (
    MapGridSpec,
    MapPublicationPolicy,
    publish_tdom_pair,
)
from semantic_atdom.semantics.schema import SemanticClassSpec, sha256_file
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    finalize_semantic_tdom,
)


CLASS_SPEC = SemanticClassSpec.from_dict(
    {
        "schema_version": 1,
        "ignore_id": 255,
        "classes": [
            {"id": 0, "name": "ground", "color": [10, 20, 30]},
            {"id": 1, "name": "building", "color": [40, 50, 60]},
        ],
    }
)


class MapGridSpecTests(unittest.TestCase):
    def test_grid_identity_is_fixed_while_tile_window_expands(self):
        grid = MapGridSpec.create(
            plane_normal_world=(0.0, 0.0, 1.0),
            pixel_x_axis_world=(1.0, 0.0, 0.0),
            pixel_y_axis_world=(0.0, 1.0, 0.0),
            plane_offset_world=7.0,
            first_bounds=(-1.2, 4.1, 2.2, 5.6),
            requested_gsd=(0.1, 0.2),
            tile_shape=(16, 8),
        )

        first = grid.window_for_bounds((-1.2, 4.1, 2.2, 5.6))
        expanded = grid.window_for_bounds((-3.0, 20.0, -2.0, 9.0))

        self.assertEqual(grid.origin_plane, (-1.2, 2.2))
        self.assertEqual(grid.gsd, (0.1, 0.2))
        self.assertEqual(first.grid_id, grid.grid_id)
        self.assertEqual(expanded.grid_id, grid.grid_id)
        self.assertLessEqual(expanded.tile_min_x, first.tile_min_x)
        self.assertGreaterEqual(expanded.tile_max_x, first.tile_max_x)
        self.assertEqual(expanded.width_pixels % 16, 0)
        self.assertEqual(expanded.height_pixels % 8, 0)
        self.assertEqual(
            expanded.pixel_axis_convention,
            "pixel centers; columns increase along pixel_x_axis_world; "
            "rows increase along pixel_y_axis_world",
        )

    def test_grid_round_trip_validates_its_content_addressed_id(self):
        grid = MapGridSpec.create(
            plane_normal_world=(0.0, 0.0, 1.0),
            pixel_x_axis_world=(1.0, 0.0, 0.0),
            pixel_y_axis_world=(0.0, 1.0, 0.0),
            plane_offset_world=0.0,
            first_bounds=(0.0, 10.0, 0.0, 5.0),
            requested_gsd=(0.0, 0.0),
            tile_shape=(512, 512),
            default_width_pixels=4000,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "map_grid.json"
            grid.save(path)
            restored = MapGridSpec.load(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["gsd"][0] *= 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "grid_id"):
                MapGridSpec.load(path)

        self.assertEqual(restored, grid)
        self.assertAlmostEqual(grid.gsd[0], 10.0 / 4000.0)
        self.assertEqual(grid.gsd[0], grid.gsd[1])

    def test_grid_rejects_origin_world_inconsistent_with_plane(self):
        grid = MapGridSpec.create(
            plane_normal_world=(0.0, 0.0, 1.0),
            pixel_x_axis_world=(1.0, 0.0, 0.0),
            pixel_y_axis_world=(0.0, 1.0, 0.0),
            plane_offset_world=2.0,
            first_bounds=(0.0, 1.0, 0.0, 1.0),
            requested_gsd=(1.0, 1.0),
            tile_shape=(1, 1),
        )
        payload = dict(grid.to_dict())
        payload["origin_world"] = [1.0, 0.0, 2.0]

        with self.assertRaisesRegex(ValueError, "origin_world"):
            MapGridSpec.from_dict(payload)


class MapPublicationPolicyTests(unittest.TestCase):
    def test_publication_and_checkpoint_cadences_are_independent(self):
        policy = MapPublicationPolicy(
            publication_stage_interval=3,
            checkpoint_stage_interval=5,
            min_free_cuda_bytes=123,
            max_pair_output_bytes=456,
            online_time_budget_seconds=7.5,
        )

        self.assertFalse(policy.should_publish(31, final=False))
        self.assertTrue(policy.should_publish(33, final=False))
        self.assertTrue(policy.should_publish(31, final=True))
        self.assertFalse(policy.should_checkpoint(31, final=False, exists=False))
        self.assertFalse(policy.should_checkpoint(31, final=False, exists=True))
        self.assertTrue(policy.should_checkpoint(35, final=False, exists=True))
        self.assertFalse(policy.should_checkpoint(31, final=False, exists=True))
        self.assertEqual(policy.overload_policy, "degrade-diagnostics-then-skip")
        self.assertEqual(policy.queue_policy, "latest-wins")

    def test_zero_checkpoint_interval_keeps_only_final_archive(self):
        policy = MapPublicationPolicy(checkpoint_stage_interval=0)

        self.assertFalse(policy.should_checkpoint(13, final=False, exists=False))
        self.assertFalse(policy.should_checkpoint(487, final=False, exists=True))
        self.assertTrue(policy.should_checkpoint(487, final=True, exists=False))


class PairPublisherTests(unittest.TestCase):
    def test_release_directory_updates_parent_map_pointer(self):
        semantic = finalize_semantic_tdom(
            torch.tensor([[[0.7]], [[0.2]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            SemanticTdomConfig(),
        )
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
            maps = Path(directory) / "maps"
            output = maps / "releases" / "stage_000001"
            checkpoint = maps / "state.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"state")
            publish_tdom_pair(
                output_root=output,
                rgb_image=np.zeros((1, 1, 3), dtype=np.uint8),
                semantic_result=semantic,
                class_spec=CLASS_SPEC,
                semantic_config=SemanticTdomConfig(),
                shared_manifest={
                    "product_key": "stage_000001",
                    "geometry_stage": 1,
                    "projection_geometry_hash": "a" * 64,
                    "rgb_model_hash": "b" * 64,
                    "semantic_state_hash": sha256_file(checkpoint),
                    "semantic_status": "approx_online",
                    "alignment": {
                        "grid": grid.to_dict(),
                        "window": window.to_dict(),
                        "image_width": 1,
                        "image_height": 1,
                    },
                },
            )

            latest = json.loads((maps / "latest.json").read_text())
            self.assertEqual(
                latest["pair_manifest"],
                "releases/stage_000001/pair_manifest.json",
            )
            self.assertFalse((maps / "releases" / "latest.json").exists())

    def test_pair_manifest_commits_two_products_and_all_hashes_last(self):
        semantic = finalize_semantic_tdom(
            torch.tensor([[[0.7]], [[0.2]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            SemanticTdomConfig(),
        )
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
            checkpoint = root / "semantic_state.0.pt"
            checkpoint.write_bytes(b"semantic state")
            output = root / "products" / "stage_000031"
            manifest_path = publish_tdom_pair(
                output_root=output,
                rgb_image=np.asarray([[[1, 2, 3]]], dtype=np.uint8),
                semantic_result=semantic,
                class_spec=CLASS_SPEC,
                semantic_config=SemanticTdomConfig(),
                shared_manifest={
                    "product_key": "stage_000031",
                    "geometry_stage": 31,
                    "projection_geometry_hash": "a" * 64,
                    "rgb_model_hash": "b" * 64,
                    "semantic_state_hash": sha256_file(checkpoint),
                    "semantic_status": "approx_online",
                    "alignment": {
                        "grid": grid.to_dict(),
                        "window": window.to_dict(),
                        "image_width": 1,
                        "image_height": 1,
                    },
                },
            )

            pair = json.loads(manifest_path.read_text(encoding="utf-8"))
            rgb_manifest = output / pair["rgb_manifest"]["path"]
            semantic_manifest = output / pair["semantic_manifest"]["path"]
            latest = json.loads(
                (output.parent / "latest.json").read_text(encoding="utf-8")
            )
            rgb_payload = json.loads(rgb_manifest.read_text(encoding="utf-8"))
            semantic_payload = json.loads(
                semantic_manifest.read_text(encoding="utf-8")
            )

            self.assertEqual(pair["publication_status"], "complete")
            self.assertEqual(pair["product_key"], "stage_000031")
            self.assertEqual(pair["grid_id"], grid.grid_id)
            self.assertEqual(pair["projection_geometry_hash"], "a" * 64)
            self.assertEqual(pair["rgb_model_hash"], "b" * 64)
            self.assertEqual(len(pair["alignment_sha256"]), 64)
            self.assertEqual(
                rgb_payload["alignment_sha256"], pair["alignment_sha256"]
            )
            self.assertEqual(
                semantic_payload["alignment_sha256"], pair["alignment_sha256"]
            )
            self.assertEqual(
                pair["semantic_state_hash"], sha256_file(checkpoint)
            )
            self.assertEqual(
                pair["rgb_manifest"]["sha256"], sha256_file(rgb_manifest)
            )
            self.assertEqual(
                pair["semantic_manifest"]["sha256"],
                sha256_file(semantic_manifest),
            )
            self.assertEqual(
                pair["product_files"]["rgb/tdom.png"],
                sha256_file(output / "rgb" / "tdom.png"),
            )
            self.assertEqual(
                {name for name in pair["product_files"] if name.endswith((".png", ".tif", ".npy"))},
                {"rgb/tdom.png", "semantic/semantic_color.png"},
            )
            self.assertEqual(latest["pair_manifest_sha256"], sha256_file(manifest_path))
            self.assertFalse(Path(str(output) + ".publishing").exists())

    def test_failed_republication_preserves_the_previous_complete_pair(self):
        semantic = finalize_semantic_tdom(
            torch.tensor([[[0.7]], [[0.2]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            SemanticTdomConfig(),
        )
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
        shared = {
            "product_key": "stage_000031",
            "geometry_stage": 31,
            "projection_geometry_hash": "a" * 64,
            "rgb_model_hash": "b" * 64,
            "semantic_state_hash": "c" * 64,
            "semantic_status": "approx_online",
            "alignment": {
                "grid": grid.to_dict(),
                "window": window.to_dict(),
                "image_width": 1,
                "image_height": 1,
            },
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "products" / "stage_000031"
            first = publish_tdom_pair(
                output_root=output,
                rgb_image=np.asarray([[[1, 2, 3]]], dtype=np.uint8),
                semantic_result=semantic,
                class_spec=CLASS_SPEC,
                semantic_config=SemanticTdomConfig(),
                shared_manifest=shared,
            )
            first_hash = sha256_file(first)
            with self.assertRaisesRegex(ValueError, "budget"):
                publish_tdom_pair(
                    output_root=output,
                    rgb_image=np.asarray([[[3, 2, 1]]], dtype=np.uint8),
                    semantic_result=semantic,
                    class_spec=CLASS_SPEC,
                    semantic_config=SemanticTdomConfig(),
                    shared_manifest=dict(
                        shared,
                        publication_policy={"max_pair_output_bytes": 1},
                    ),
                )

            self.assertEqual(sha256_file(first), first_hash)
            self.assertFalse(Path(str(output) + ".publishing").exists())

            save_json_atomic = publication_module._save_json_atomic

            def fail_latest(path, payload):
                if Path(path).name == "latest.json":
                    raise OSError("injected latest pointer failure")
                return save_json_atomic(path, payload)

            with patch.object(
                publication_module,
                "_save_json_atomic",
                side_effect=fail_latest,
            ):
                with self.assertRaisesRegex(OSError, "latest pointer"):
                    publish_tdom_pair(
                        output_root=output,
                        rgb_image=np.asarray([[[3, 2, 1]]], dtype=np.uint8),
                        semantic_result=semantic,
                        class_spec=CLASS_SPEC,
                        semantic_config=SemanticTdomConfig(),
                        shared_manifest=shared,
                    )

            self.assertEqual(sha256_file(first), first_hash)
            self.assertFalse(Path(str(output) + ".previous").exists())

    def test_pair_rejects_alignment_dimensions_that_disagree_with_window(self):
        semantic = finalize_semantic_tdom(
            torch.tensor([[[0.7]], [[0.2]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            SemanticTdomConfig(),
        )
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
            with self.assertRaisesRegex(ValueError, "image_width"):
                publish_tdom_pair(
                    output_root=Path(directory) / "stage",
                    rgb_image=np.asarray([[[1, 2, 3]]], dtype=np.uint8),
                    semantic_result=semantic,
                    class_spec=CLASS_SPEC,
                    semantic_config=SemanticTdomConfig(),
                    shared_manifest={
                        "product_key": "stage",
                        "geometry_stage": 1,
                        "projection_geometry_hash": "a" * 64,
                        "rgb_model_hash": "b" * 64,
                        "semantic_state_hash": "c" * 64,
                        "semantic_status": "approx_online",
                        "alignment": {
                            "grid": grid.to_dict(),
                            "window": window.to_dict(),
                            "image_width": 2,
                            "image_height": 1,
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
