from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import struct
import unittest

import numpy as np
from PIL import Image

from semantic_atdom.semantics.preflight import run_dataset_preflight
from semantic_atdom.semantics.schema import SemanticClassSpec, SemanticSchemaError, sha256_file


def _write_images_binary(path, names):
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(names)))
        for index, name in enumerate(names, start=1):
            stream.write(
                struct.pack(
                    "<idddddddi",
                    index,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1,
                )
            )
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))


class DatasetPreflightTests(unittest.TestCase):
    def _fixture(self, root):
        dataset = root / "dataset"
        semantic_data = root / "semantic-data"
        semantic = semantic_data / "semantic"
        (semantic / "labels").mkdir(parents=True)
        (semantic / "metadata").mkdir()
        spec_payload = {
            "schema_version": 1,
            "ignore_id": 255,
            "classes": [
                {"id": 0, "name": "ground", "color": [1, 2, 3]},
                {"id": 1, "name": "building", "color": [4, 5, 6]},
            ],
        }
        (semantic / "class_spec.json").write_text(
            json.dumps(spec_payload), encoding="utf-8"
        )
        spec = SemanticClassSpec.from_dict(spec_payload)
        stage_names = {
            1: ("000001.jpg",),
            2: ("000001.jpg", "000003.jpg"),
        }
        for stage, names in stage_names.items():
            sparse = dataset / str(stage) / "sparse" / "0"
            images = dataset / str(stage) / "images"
            sparse.mkdir(parents=True)
            images.mkdir()
            _write_images_binary(sparse / "images.bin", names)
            (sparse / "cameras.bin").write_bytes(b"camera")
            (sparse / "points3D.bin").write_bytes(b"points")
            (sparse / "imagesNames.txt").write_text(
                ",".join(names), encoding="utf-8"
            )
            matrix = np.eye(len(names), dtype=np.int64)
            np.savetxt(sparse / "imageMatchMatrix.txt", matrix, fmt="%d", delimiter=",")
            for name in names:
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(images / name)
        for image_id in ("000001", "000003"):
            rgb_path = dataset / "2" / "images" / (image_id + ".jpg")
            label_path = semantic / "labels" / (image_id + ".png")
            Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(label_path)
            metadata = {
                "schema_version": 1,
                "image_id": image_id,
                "label_version": "fixture-v1",
                "model": "fixture",
                "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
                "prompt_config": {},
                "thresholds": {},
                "width": 2,
                "height": 2,
                "class_spec_sha256": spec.fingerprint,
                "label_sha256": sha256_file(label_path),
                "source_image_sha256": sha256_file(rgb_path),
            }
            (semantic / "metadata" / (image_id + ".json")).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        return dataset, semantic_data

    def test_preflight_validates_cumulative_stages_and_final_observations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, semantic_data = self._fixture(root)
            result = run_dataset_preflight(
                dataset_root=dataset,
                semantic_data_root=semantic_data,
                output_root=root / "output",
                start_stage=1,
                end_stage=2,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

            self.assertEqual(result.selected_stages, (1, 2))
            self.assertEqual(result.final_image_ids, ("000001", "000003"))
            self.assertTrue(result.authoritative_exact_complete)
            self.assertEqual(report["class_pixel_counts"], [0, 8])
            self.assertEqual(sha256_file(result.report_path), result.report_sha256)

    def test_preflight_requires_explicit_semantic_root_and_external_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, semantic_data = self._fixture(root)
            with self.assertRaisesRegex(SemanticSchemaError, "required"):
                run_dataset_preflight(
                    dataset_root=dataset,
                    semantic_data_root=None,
                    output_root=root / "output",
                    start_stage=1,
                    end_stage=2,
                )
            with self.assertRaisesRegex(SemanticSchemaError, "outside"):
                run_dataset_preflight(
                    dataset_root=dataset,
                    semantic_data_root=semantic_data,
                    output_root=dataset / "output",
                    start_stage=1,
                    end_stage=2,
                )

    def test_fixed_layout_uses_cumulative_overlap_and_shared_rgb_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, semantic_data = self._fixture(root)
            fixed = root / "scene" / "GaussianSplatting"
            shared_images = root / "scene" / "interval5_CAM"
            shared_images.mkdir(parents=True)
            ids = ("1658137107.900", "1658137108.010")
            semantic = semantic_data / "semantic"
            spec = SemanticClassSpec.load(semantic / "class_spec.json")
            for index, image_id in enumerate(ids, start=1):
                rgb_path = shared_images / (image_id + ".jpg")
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(rgb_path)
                label_path = semantic / "labels" / (image_id + ".png")
                Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(label_path)
                metadata = {
                    "schema_version": 1,
                    "image_id": image_id,
                    "label_version": "fixture-v1",
                    "model": "fixture",
                    "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
                    "prompt_config": {},
                    "thresholds": {},
                    "width": 2,
                    "height": 2,
                    "class_spec_sha256": spec.fingerprint,
                    "label_sha256": sha256_file(label_path),
                    "source_image_sha256": sha256_file(rgb_path),
                }
                (semantic / "metadata" / (image_id + ".json")).write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
            marker = "00007FFE9CAB9DE0"
            for stage in (1, 2):
                stage_root = fixed / str(stage)
                binary = stage_root / "bin"
                binary.mkdir(parents=True)
                local_name = ids[stage - 1] + ".jpg"
                _write_images_binary(binary / "images.bin", (local_name,))
                (binary / "cameras.bin").write_bytes(b"camera")
                (binary / "points3D.bin").write_bytes(b"points")
                names = ids[:stage]
                (stage_root / "match_matrix_image_name.txt").write_text(
                    ",".join(item + ".jpg" for item in names) + marker,
                    encoding="utf-8",
                )
                rows = []
                for row in np.eye(stage, dtype=np.int64):
                    rows.append(",".join(str(value) for value in row))
                (stage_root / "match_matrix_value.txt").write_text(
                    marker.join(rows) + marker, encoding="utf-8"
                )

            result = run_dataset_preflight(
                dataset_root=fixed,
                semantic_data_root=semantic_data,
                output_root=root / "fixed-output",
                start_stage=1,
                end_stage=2,
                image_folder_path=shared_images,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

            self.assertEqual(result.final_image_ids, ids)
            self.assertEqual(report["stage_records"][1]["camera_count"], 2)
            self.assertEqual(report["stage_records"][1]["snapshot_camera_count"], 1)


if __name__ == "__main__":
    unittest.main()
