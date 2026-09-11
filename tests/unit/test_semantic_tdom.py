import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image

from semantic_atdom.semantics.schema import SemanticClassSpec, sha256_file
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    UnknownReason,
    _semantic_render_features,
    _trusted_geometry_mask,
    finalize_semantic_tdom,
    load_semantic_tdom_palette,
    write_semantic_tdom,
)
from semantic_atdom.semantics.state import SemanticGaussianState


CLASS_SPEC = SemanticClassSpec.from_dict(
    {
        "schema_version": 1,
        "ignore_id": 255,
        "classes": [
            {"id": 0, "name": "ground", "color": [10, 20, 30]},
            {"id": 1, "name": "building", "color": [200, 100, 50]},
        ],
    }
)


class SemanticTdomTests(unittest.TestCase):
    def test_output_budget_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_output_bytes"):
            SemanticTdomConfig(max_output_bytes=0)
        with self.assertRaisesRegex(ValueError, "min_free_cuda_bytes"):
            SemanticTdomConfig(min_free_cuda_bytes=-1)
        with self.assertRaisesRegex(ValueError, "max_group_disagreement"):
            SemanticTdomConfig(max_group_disagreement=2.01)
        with self.assertRaisesRegex(ValueError, "max_gaussian_scale_ratio"):
            SemanticTdomConfig(max_gaussian_scale_ratio=1.0)

    def test_trusted_geometry_mask_rejects_only_scale_outliers(self):
        mask, metrics = _trusted_geometry_mask(
            torch.tensor(
                [
                    [0.2, 0.1, 0.1],
                    [0.2, 0.2, 0.1],
                    [0.2, 0.1, 0.2],
                    [0.2, 0.2, 0.2],
                    [5.0, 0.2, 0.2],
                ]
            ),
            max_scale_ratio=20.0,
        )

        self.assertEqual(mask.tolist(), [True, True, True, True, False])
        self.assertEqual(metrics["input_gaussian_count"], 5)
        self.assertEqual(metrics["trusted_gaussian_count"], 4)
        self.assertEqual(metrics["excluded_gaussian_count"], 1)
        self.assertAlmostEqual(metrics["median_max_world_scale"], 0.2)
        self.assertAlmostEqual(
            metrics["max_trusted_world_scale"], 4.0, places=6
        )

    def test_render_palette_overrides_colors_without_changing_class_spec(self):
        original_fingerprint = CLASS_SPEC.fingerprint
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tdom_palette.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "overrides": {"building": [255, 0, 0]},
                    }
                ),
                encoding="utf-8",
            )
            palette = load_semantic_tdom_palette(path, CLASS_SPEC)

        self.assertEqual(palette, ((10, 20, 30), (255, 0, 0)))
        self.assertEqual(CLASS_SPEC.fingerprint, original_fingerprint)
        result = finalize_semantic_tdom(
            torch.tensor([[[0.1]], [[0.8]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            SemanticTdomConfig(render_palette=palette),
        )
        self.assertEqual(result.colors[:, 0, 0].tolist(), [255, 0, 0])

    def test_render_palette_rejects_invalid_schema_classes_and_colors(self):
        invalid_payloads = (
            {"schema_version": True, "overrides": {}},
            {"schema_version": 1, "overrides": {"tree": [1, 2, 3]}},
            {"schema_version": 1, "overrides": {"building": [256, 0, 0]}},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tdom_palette.json"
            for payload in invalid_payloads:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_semantic_tdom_palette(path, CLASS_SPEC)

    def test_soft_probabilities_are_mixed_before_unknown_gate(self):
        rendered = torch.tensor(
            [
                [[0.6, 0.2]],
                [[0.2, 0.2]],
                [[0.8, 0.4]],
            ]
        )
        coverage = torch.tensor([[0.9, 0.9]])
        config = SemanticTdomConfig(
            min_coverage=0.1,
            min_absolute_support=0.1,
            min_support_ratio=0.1,
            min_margin=0.1,
        )

        result = finalize_semantic_tdom(
            rendered, coverage, CLASS_SPEC, config
        )

        self.assertEqual(result.class_ids.tolist(), [[0, 255]])
        torch.testing.assert_close(
            result.probabilities[:, 0, 0], torch.tensor([0.75, 0.25])
        )
        self.assertEqual(result.colors[:, 0, 0].tolist(), [10, 20, 30])
        self.assertEqual(result.colors[:, 0, 1].tolist(), [0, 0, 0])
        self.assertFalse(torch.isnan(result.entropy[0, 1]))
        self.assertEqual(
            int(result.unknown_reason[0, 1]), int(UnknownReason.LOW_MARGIN)
        )

    def test_entropy_and_unknown_reasons_remain_explainable(self):
        rendered = torch.tensor(
            [
                [[0.45, 0.0]],
                [[0.35, 0.0]],
                [[0.80, 0.0]],
            ]
        )
        coverage = torch.tensor([[0.90, 0.0]])
        result = finalize_semantic_tdom(
            rendered,
            coverage,
            CLASS_SPEC,
            SemanticTdomConfig(min_margin=0.2),
        )

        self.assertTrue(result.unknown_mask[0, 0])
        self.assertFalse(torch.isnan(result.entropy[0, 0]))
        self.assertTrue(
            int(result.unknown_reason[0, 0]) & int(UnknownReason.LOW_MARGIN)
        )
        self.assertTrue(torch.isnan(result.entropy[0, 1]))
        self.assertTrue(
            int(result.unknown_reason[0, 1])
            & int(UnknownReason.NO_SEMANTIC_EVIDENCE)
        )

    def test_zero_coverage_is_geometry_invalid_when_gate_is_disabled(self):
        config = SemanticTdomConfig(
            min_coverage=0.0,
            min_absolute_support=0.0,
            min_support_ratio=0.0,
            min_margin=0.0,
        )
        result = finalize_semantic_tdom(
            torch.zeros((CLASS_SPEC.num_classes + 1, 1, 1)),
            torch.zeros((1, 1)),
            CLASS_SPEC,
            config,
        )
        self.assertTrue(
            int(result.unknown_reason[0, 0]) & int(UnknownReason.LOW_COVERAGE)
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_semantic_tdom(result, root, CLASS_SPEC, {}, config)
            with Image.open(root / "geometry_valid_mask.png") as image:
                self.assertEqual(image.getpixel((0, 0)), 0)

    def test_enhanced_features_produce_separate_uncertainty_maps(self):
        # K class numerators, support, E[||mu||^2], group variance and
        # Dirichlet trace variance.
        rendered = torch.tensor(
            [
                [[0.60]],
                [[0.20]],
                [[0.80]],
                [[0.68]],
                [[0.16]],
                [[0.08]],
            ]
        )
        result = finalize_semantic_tdom(
            rendered, torch.tensor([[0.90]]), CLASS_SPEC
        )

        torch.testing.assert_close(
            result.mixture_disagreement, torch.tensor([[0.225]])
        )
        torch.testing.assert_close(
            result.group_disagreement, torch.tensor([[0.20]])
        )
        torch.testing.assert_close(
            result.posterior_variance, torch.tensor([[0.125]])
        )

    def test_group_disagreement_keeps_full_trace_range(self):
        rendered = torch.tensor(
            [
                [[0.50]],
                [[0.50]],
                [[1.00]],
                [[0.50]],
                [[1.60]],
                [[0.10]],
            ]
        )
        result = finalize_semantic_tdom(
            rendered, torch.tensor([[1.0]]), CLASS_SPEC
        )
        torch.testing.assert_close(
            result.group_disagreement, torch.tensor([[1.60]])
        )

    def test_render_features_keep_group_trace_above_one(self):
        state = SemanticGaussianState.initialize(1, 2)
        state.add_evidence(torch.tensor([[1.0, 1.0]]))
        state.group_variance[0] = torch.tensor([0.8, 0.8])

        features = _semantic_render_features(
            state, torch.device("cpu"), use_robust=True, eps=1e-8
        )

        evidence_ratio = 2.0 / 3.0
        self.assertAlmostEqual(
            float(features[0, CLASS_SPEC.num_classes + 2]),
            evidence_ratio * 1.6,
            places=6,
        )

    def test_uncertainty_is_undefined_without_semantic_evidence(self):
        result = finalize_semantic_tdom(
            torch.zeros((CLASS_SPEC.num_classes + 4, 1, 1)),
            torch.ones((1, 1)),
            CLASS_SPEC,
        )
        for name in (
            "entropy",
            "normalized_entropy",
            "mixture_disagreement",
            "group_disagreement",
            "posterior_variance",
        ):
            self.assertTrue(torch.isnan(getattr(result, name)).all(), name)

    def test_writer_publishes_full_products_and_manifest(self):
        config = SemanticTdomConfig(
            save_probabilities=True,
            save_diagnostics=True,
            render_palette=((10, 20, 30), (255, 0, 0)),
            render_palette_source="/dataset/semantic/tdom_palette.json",
        )
        result = finalize_semantic_tdom(
            torch.tensor([[[0.8]], [[0.1]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            config,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_semantic_tdom(
                result,
                root,
                CLASS_SPEC,
                {
                    "geometry_stage": 31,
                    "semantic_through_image": "000031",
                    "label_version": "sam3-v1",
                    "geometry_hash": "a" * 64,
                },
                config,
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            expected = {
                "semantic_id.png",
                "semantic_color.png",
                "unknown_mask.png",
                "unknown_reason.png",
                "geometry_valid_mask.png",
                "semantic_support.tif",
                "semantic_entropy.tif",
                "coverage.tif",
                "support_ratio.tif",
                "top1_probability.tif",
                "margin.tif",
                "mixture_disagreement.tif",
                "group_disagreement.tif",
                "posterior_variance.tif",
                "semantic_probabilities.npy",
                "class_spec.json",
                "render_palette.json",
            }
            self.assertEqual(set(payload["files"]), expected)
            for name in expected:
                self.assertEqual(payload["files"][name], sha256_file(root / name))
            self.assertEqual(payload["geometry_stage"], 31)
            self.assertEqual(payload["class_spec_sha256"], CLASS_SPEC.fingerprint)
            self.assertEqual(payload["class_spec"], CLASS_SPEC.to_dict())
            self.assertEqual(
                payload["render_palette_sha256"],
                sha256_file(root / "render_palette.json"),
            )
            self.assertEqual(
                payload["render_palette"]["source"],
                "/dataset/semantic/tdom_palette.json",
            )
            self.assertEqual(
                payload["render_palette"]["classes"][1]["color"],
                [255, 0, 0],
            )
            self.assertNotIn("render_palette", payload["unknown_gate"])
            self.assertNotIn("render_palette_source", payload["unknown_gate"])
            with Image.open(root / "geometry_valid_mask.png") as image:
                self.assertEqual(image.getpixel((0, 0)), 255)
            self.assertEqual(
                payload["raster_encodings"]["semantic_support.tif"]["dtype"],
                "uint16",
            )
            self.assertEqual(
                payload["raster_encodings"]["group_disagreement.tif"]["scale"],
                2.0 / 65534.0,
            )

            second_manifest = write_semantic_tdom(
                result,
                root,
                CLASS_SPEC,
                {
                    "geometry_stage": 32,
                    "semantic_through_image": "000032",
                    "label_version": "sam3-v1",
                    "geometry_hash": "b" * 64,
                },
                config,
            )
            second = json.loads(second_manifest.read_text(encoding="utf-8"))
            self.assertEqual(second["geometry_stage"], 32)
            self.assertFalse(root.with_name(root.name + ".publishing").exists())
            self.assertFalse(root.with_name(root.name + ".previous").exists())

    def test_writer_defaults_to_core_product_without_full_diagnostics(self):
        config = SemanticTdomConfig()
        result = finalize_semantic_tdom(
            torch.tensor([[[0.8]], [[0.1]], [[0.9]]]),
            torch.tensor([[0.95]]),
            CLASS_SPEC,
            config,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_semantic_tdom(
                result, root, CLASS_SPEC, {}, config
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload["raster_encodings"]),
                {"semantic_support.tif", "semantic_entropy.tif"},
            )

    def test_channel_count_must_match_class_spec(self):
        with self.assertRaisesRegex(ValueError, "num_classes"):
            finalize_semantic_tdom(
                torch.zeros((2, 1, 1)), torch.zeros((1, 1)), CLASS_SPEC
            )


if __name__ == "__main__":
    unittest.main()
