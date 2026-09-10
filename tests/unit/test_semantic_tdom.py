import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from semantic_atdom.semantics.schema import SemanticClassSpec, sha256_file
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    UnknownReason,
    finalize_semantic_tdom,
    write_semantic_tdom,
)


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

    def test_writer_publishes_default_products_and_manifest(self):
        config = SemanticTdomConfig(save_probabilities=True)
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
            }
            self.assertEqual(set(payload["files"]), expected)
            for name in expected:
                self.assertEqual(payload["files"][name], sha256_file(root / name))
            self.assertEqual(payload["geometry_stage"], 31)
            self.assertEqual(payload["class_spec_sha256"], CLASS_SPEC.fingerprint)
            self.assertEqual(payload["class_spec"], CLASS_SPEC.to_dict())
            self.assertEqual(
                payload["raster_encodings"]["semantic_support.tif"]["dtype"],
                "uint16",
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

    def test_channel_count_must_match_class_spec(self):
        with self.assertRaisesRegex(ValueError, "num_classes"):
            finalize_semantic_tdom(
                torch.zeros((2, 1, 1)), torch.zeros((1, 1)), CLASS_SPEC
            )


if __name__ == "__main__":
    unittest.main()
