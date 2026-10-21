from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import numpy as np
from PIL import Image

from semantic_atdom.semantics.input_binding import validate_final_preflight_binding
from semantic_atdom.semantics.schema import (
    SemanticClassSpec,
    SemanticDataset,
    SemanticSchemaError,
    sha256_file,
    generation_fingerprint,
)


CLASS_SPEC_VALUE = {
    "schema_version": 1,
    "ignore_id": 255,
    "classes": [
        {"id": 0, "name": "ground", "color": [0, 0, 0]},
        {"id": 1, "name": "building", "color": [255, 0, 0]},
    ],
}


class FinalPreflightBindingTests(unittest.TestCase):
    def _fixture(self, root):
        dataset_root = root / "dataset"
        stage_root = dataset_root / "1"
        sparse = stage_root / "sparse" / "0"
        images = stage_root / "images"
        sparse.mkdir(parents=True)
        images.mkdir()
        for name, content in (
            ("images.bin", b"images"),
            ("cameras.bin", b"cameras"),
            ("points3D.bin", b"points"),
            ("imagesNames.txt", b"000001\n"),
            ("imageMatchMatrix.txt", b"100\n"),
        ):
            (sparse / name).write_bytes(content)

        rgb_path = images / "000001.png"
        Image.fromarray(np.full((2, 2, 3), 10, dtype=np.uint8)).save(rgb_path)

        semantic_root = root / "semantic-data" / "semantic"
        (semantic_root / "labels").mkdir(parents=True)
        (semantic_root / "metadata").mkdir()
        (semantic_root / "class_spec.json").write_text(
            json.dumps(CLASS_SPEC_VALUE), encoding="utf-8"
        )
        class_spec = SemanticClassSpec.from_dict(CLASS_SPEC_VALUE)
        label_path = semantic_root / "labels" / "000001.png"
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(label_path)
        metadata = {
            "schema_version": 1,
            "image_id": "000001",
            "label_version": "labels-v1",
            "model": "fixture",
            "checkpoint_sha256": "a" * 64,
            "prompt_config": {},
            "thresholds": {},
            "class_spec_sha256": class_spec.fingerprint,
            "label_sha256": sha256_file(label_path),
            "score_sha256": None,
            "source_image_sha256": sha256_file(rgb_path),
            "width": 2,
            "height": 2,
        }
        (semantic_root / "metadata" / "000001.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        dataset = SemanticDataset(root / "semantic-data")
        report = {
            "schema_version": 1,
            "dataset_root": str(dataset_root.resolve()),
            "semantic_data_root": str((root / "semantic-data").resolve()),
            "selected_stages": [1],
            "stage_records": [
                {
                    "stage": 1,
                    "images_bin_sha256": sha256_file(sparse / "images.bin"),
                    "cameras_bin_sha256": sha256_file(sparse / "cameras.bin"),
                    "images_names_sha256": sha256_file(sparse / "imagesNames.txt"),
                    "overlap_matrix_sha256": sha256_file(
                        sparse / "imageMatchMatrix.txt"
                    ),
                }
            ],
            "final_image_ids": ["000001"],
            "labels_complete": True,
            "source_binding_complete": True,
            "authoritative_exact_complete": True,
            "class_spec_sha256": class_spec.fingerprint,
            "ontology_sha256": class_spec.ontology_fingerprint,
            "observations": [
                {
                    "image_id": "000001",
                    "label_version": "labels-v1",
                    "label_sha256": metadata["label_sha256"],
                    "generation_sha256": generation_fingerprint(metadata),
                    "score_sha256": metadata.get("score_sha256"),
                    "rgb_sha256": metadata["source_image_sha256"],
                    "source_binding": "verified",
                }
            ],
        }
        report_path = root / "preflight.json"
        report_path.write_text(
            json.dumps(report, sort_keys=True), encoding="utf-8"
        )
        cameras = {"000001": SimpleNamespace(image_path=str(rgb_path))}
        return dataset, stage_root, report_path, cameras, rgb_path, label_path

    def test_binding_accepts_unchanged_inputs_and_rejects_rgb_mutation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, stage, report, cameras, rgb_path, _ = self._fixture(root)
            report_hash = sha256_file(report)

            binding = validate_final_preflight_binding(
                report_path=report,
                expected_report_sha256=report_hash,
                semantic_dataset=dataset,
                image_ids=("000001",),
                camera_by_id=cameras,
                snapshot_root=stage,
                geometry_stage=1,
            )
            self.assertEqual(binding.report_sha256, report_hash)

            Image.fromarray(
                np.full((2, 2, 3), 20, dtype=np.uint8)
            ).save(rgb_path)
            with self.assertRaisesRegex(
                SemanticSchemaError, "RGB content changed since preflight"
            ):
                validate_final_preflight_binding(
                    report_path=report,
                    expected_report_sha256=report_hash,
                    semantic_dataset=dataset,
                    image_ids=("000001",),
                    camera_by_id=cameras,
                    snapshot_root=stage,
                    geometry_stage=1,
                )

    def test_binding_rejects_consistently_replaced_label(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, stage, report, cameras, _, label_path = self._fixture(root)
            report_hash = sha256_file(report)
            Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(label_path)
            metadata_path = label_path.parents[1] / "metadata" / "000001.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["label_sha256"] = sha256_file(label_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(
                SemanticSchemaError, "label content changed since preflight"
            ):
                validate_final_preflight_binding(
                    report_path=report,
                    expected_report_sha256=report_hash,
                    semantic_dataset=dataset,
                    image_ids=("000001",),
                    camera_by_id=cameras,
                    snapshot_root=stage,
                    geometry_stage=1,
                )

    def test_binding_rejects_generation_and_optional_score_drift(self):
        changes = {
            "model": "different-model",
            "checkpoint_sha256": "b" * 64,
            "prompt_config": {"prompt": "changed"},
            "thresholds": {"confidence": 0.9},
            "score_sha256": "add-score-file",
        }
        for field, value in changes.items():
            with self.subTest(field=field), TemporaryDirectory() as directory:
                dataset, stage, report, cameras, _, _ = self._fixture(Path(directory))
                path = dataset.semantic_root / "metadata" / "000001.json"
                metadata = json.loads(path.read_text())
                if field == "score_sha256":
                    scores = dataset.semantic_root / "scores"
                    scores.mkdir()
                    np.save(scores / "000001.npy", np.full((2, 2, 2), .5, dtype=np.float32))
                    value = sha256_file(scores / "000001.npy")
                metadata[field] = value
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(SemanticSchemaError, "generation|score"):
                    validate_final_preflight_binding(report_path=report,
                        expected_report_sha256=sha256_file(report), semantic_dataset=dataset,
                        image_ids=("000001",), camera_by_id=cameras, snapshot_root=stage, geometry_stage=1)

    def test_binding_accepts_formatting_and_presentation_only_changes(self):
        with TemporaryDirectory() as directory:
            dataset, stage, report, cameras, _, _ = self._fixture(Path(directory))
            metadata_path = dataset.semantic_root / "metadata" / "000001.json"
            metadata = json.loads(metadata_path.read_text())
            metadata_path.write_text(json.dumps(metadata, indent=4, sort_keys=True))
            spec_path = dataset.semantic_root / "class_spec.json"
            spec = json.loads(spec_path.read_text())
            spec["classes"][0]["color"] = [17, 18, 19]
            spec_path.write_text(json.dumps(spec, indent=2))
            binding = validate_final_preflight_binding(
                report_path=report, expected_report_sha256=sha256_file(report),
                semantic_dataset=dataset, image_ids=("000001",), camera_by_id=cameras,
                snapshot_root=stage, geometry_stage=1,
            )
            self.assertEqual(binding.ontology_sha256, dataset.class_spec.ontology_fingerprint)

    def test_binding_reloads_disk_ontology_instead_of_trusting_cached_spec(self):
        with TemporaryDirectory() as directory:
            dataset, stage, report, cameras, _, _ = self._fixture(Path(directory))
            path = dataset.semantic_root / "class_spec.json"
            spec = json.loads(path.read_text())
            spec["classes"][0]["name"] = "different-class"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(SemanticSchemaError, "ontology|specification"):
                validate_final_preflight_binding(report_path=report,
                    expected_report_sha256=sha256_file(report), semantic_dataset=dataset,
                    image_ids=("000001",), camera_by_id=cameras, snapshot_root=stage, geometry_stage=1)


if __name__ == "__main__":
    unittest.main()
