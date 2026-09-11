import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from semantic_atdom.semantics.schema import (
    LedgerEntry,
    SemanticClassSpec,
    SemanticDataset,
    SemanticLedger,
    SemanticSchemaError,
    sha256_file,
)


CLASS_SPEC = {
    "schema_version": 1,
    "ignore_id": 255,
    "classes": [
        {
            "id": 0,
            "name": "background",
            "color": [0, 0, 0],
            "is_background": True,
        },
        {
            "id": 1,
            "name": "building",
            "color": [255, 0, 0],
            "is_background": False,
        },
    ],
}


class SemanticSchemaTests(unittest.TestCase):
    def test_class_spec_is_configurable_and_has_stable_fingerprint(self):
        first = SemanticClassSpec.from_dict(CLASS_SPEC)
        second = SemanticClassSpec.from_dict(
            {
                "classes": list(CLASS_SPEC["classes"]),
                "ignore_id": 255,
                "schema_version": 1,
            }
        )

        self.assertEqual(first.num_classes, 2)
        self.assertEqual(first.background_id, 0)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_ontology_and_display_palette_have_independent_fingerprints(self):
        first = SemanticClassSpec.from_dict(CLASS_SPEC)
        recolored = dict(CLASS_SPEC)
        recolored["classes"] = [dict(item) for item in CLASS_SPEC["classes"]]
        recolored["classes"][1]["color"] = [1, 2, 3]
        second = SemanticClassSpec.from_dict(recolored)

        self.assertEqual(first.ontology_fingerprint, second.ontology_fingerprint)
        self.assertNotEqual(first.palette_fingerprint, second.palette_fingerprint)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_class_ids_must_be_contiguous(self):
        invalid = dict(CLASS_SPEC)
        invalid["classes"] = [
            {"id": 0, "name": "background", "color": [0, 0, 0]},
            {"id": 2, "name": "building", "color": [255, 0, 0]},
        ]

        with self.assertRaisesRegex(SemanticSchemaError, "contiguous"):
            SemanticClassSpec.from_dict(invalid)

    def test_malformed_class_is_reported_as_schema_error(self):
        invalid = dict(CLASS_SPEC)
        invalid["classes"] = ["not-an-object"]
        with self.assertRaisesRegex(SemanticSchemaError, "JSON object"):
            SemanticClassSpec.from_dict(invalid)

    def test_dataset_validates_label_scores_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            (semantic / "labels").mkdir(parents=True)
            (semantic / "scores").mkdir()
            (semantic / "metadata").mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC), encoding="utf-8"
            )

            spec = SemanticClassSpec.load(semantic / "class_spec.json")
            label = np.array([[0, 1], [255, 1]], dtype=np.uint8)
            Image.fromarray(label).save(semantic / "labels" / "000001.png")
            scores = np.zeros((2, 2, 2), dtype=np.float32)
            np.save(str(semantic / "scores" / "000001.npy"), scores)
            metadata = {
                "schema_version": 1,
                "image_id": "000001",
                "label_version": "sam3-test-v1",
                "model": "sam3",
                "checkpoint_sha256": "a" * 64,
                "prompt_config": {"classes": ["background", "building"]},
                "thresholds": {"min_score": 0.2},
                "class_spec_sha256": spec.fingerprint,
                "label_sha256": sha256_file(
                    semantic / "labels" / "000001.png"
                ),
                "score_sha256": sha256_file(
                    semantic / "scores" / "000001.npy"
                ),
                "width": 2,
                "height": 2,
            }
            (semantic / "metadata" / "000001.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            observation = SemanticDataset(root).load_observation("000001")

            np.testing.assert_array_equal(observation.labels, label)
            self.assertEqual(observation.scores.shape, (2, 2, 2))
            self.assertEqual(observation.key.image_id, "000001")
            self.assertEqual(observation.key.label_version, "sam3-test-v1")

            Image.fromarray(np.array([[1, 1], [1, 1]], dtype=np.uint8)).save(
                semantic / "labels" / "000001.png"
            )
            with self.assertRaisesRegex(SemanticSchemaError, "label_sha256"):
                SemanticDataset(root).load_observation("000001")

    def test_dataset_accepts_png_only_when_scores_are_optional(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            (semantic / "labels").mkdir(parents=True)
            (semantic / "metadata").mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC), encoding="utf-8"
            )
            spec = SemanticClassSpec.load(semantic / "class_spec.json")
            Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(
                semantic / "labels" / "frame.png"
            )
            (semantic / "metadata" / "frame.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "image_id": "frame",
                        "label_version": "labels-only-v1",
                        "model": "sam3",
                        "checkpoint_sha256": "a" * 64,
                        "prompt_config": {"classes": ["background", "building"]},
                        "thresholds": {"min_score": 0.2},
                        "class_spec_sha256": spec.fingerprint,
                        "label_sha256": sha256_file(
                            semantic / "labels" / "frame.png"
                        ),
                        "score_sha256": None,
                        "width": 2,
                        "height": 1,
                    }
                ),
                encoding="utf-8",
            )

            dataset = SemanticDataset(root)
            self.assertIsNone(dataset.load_observation("frame").scores)
            with self.assertRaisesRegex(SemanticSchemaError, "scores"):
                dataset.load_observation("frame", require_scores=True)

    def test_invalid_label_id_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            (semantic / "labels").mkdir(parents=True)
            (semantic / "metadata").mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC), encoding="utf-8"
            )
            spec = SemanticClassSpec.load(semantic / "class_spec.json")
            Image.fromarray(np.array([[0, 2]], dtype=np.uint8)).save(
                semantic / "labels" / "frame.png"
            )
            (semantic / "metadata" / "frame.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "image_id": "frame",
                        "label_version": "bad-label-v1",
                        "model": "sam3",
                        "checkpoint_sha256": "a" * 64,
                        "prompt_config": {"classes": ["background", "building"]},
                        "thresholds": {"min_score": 0.2},
                        "class_spec_sha256": spec.fingerprint,
                        "label_sha256": sha256_file(
                            semantic / "labels" / "frame.png"
                        ),
                        "score_sha256": None,
                        "width": 2,
                        "height": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SemanticSchemaError, "invalid class IDs"):
                SemanticDataset(root).load_observation("frame")

    def test_ledger_is_idempotent_and_detects_conflicts(self):
        entry = LedgerEntry("000001", "sam3-v1", "a" * 64, "b" * 64)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = SemanticLedger()

            self.assertTrue(ledger.record(entry))
            self.assertFalse(ledger.record(entry))
            with self.assertRaisesRegex(SemanticSchemaError, "conflicting"):
                ledger.record(
                    LedgerEntry("000001", "sam3-v1", "c" * 64, "b" * 64)
                )

            ledger.save(path)
            restored = SemanticLedger.load(path)
            self.assertTrue(restored.contains("000001", "sam3-v1"))
            self.assertEqual(restored.get("000001", "sam3-v1"), entry)

    def test_ledger_rejects_invalid_content_hashes(self):
        with self.assertRaisesRegex(SemanticSchemaError, "label_sha256"):
            LedgerEntry("000001", "sam3-v1", "not-a-hash")

    def test_nonfinite_scores_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            semantic.mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(CLASS_SPEC), encoding="utf-8"
            )
            dataset = SemanticDataset(root)
            with self.assertRaisesRegex(SemanticSchemaError, "finite"):
                dataset._validate_scores(
                    np.array([[[np.nan]], [[0.0]]], dtype=np.float32),
                    (1, 1),
                )


if __name__ == "__main__":
    unittest.main()
