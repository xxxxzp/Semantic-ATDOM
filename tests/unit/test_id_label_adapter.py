from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
from PIL import Image

from semantic_atdom.semantics.id_label_adapter import prepare_id_label_dataset
from semantic_atdom.semantics.schema import SemanticDataset, SemanticSchemaError


class IdLabelAdapterTests(unittest.TestCase):
    def test_explicit_mapping_preserves_decimal_ids_and_binds_rgb(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "raw-labels"
            images = root / "images"
            labels.mkdir()
            images.mkdir()
            image_id = "1658137107.639350179"
            Image.fromarray(np.array([[5, 9]], dtype=np.uint8)).save(
                labels / (image_id + ".png")
            )
            Image.fromarray(np.zeros((1, 2, 3), dtype=np.uint8)).save(
                images / (image_id + ".jpg")
            )
            spec = {
                "schema_version": 1,
                "label_version": "fixture-v1",
                "producer": "fixture",
                "source_id_to_class_id": {"5": 1, "9": 0},
                "class_spec": {
                    "schema_version": 1,
                    "ignore_id": 255,
                    "classes": [
                        {"id": 0, "name": "zero", "color": [0, 0, 0]},
                        {"id": 1, "name": "one", "color": [1, 1, 1]},
                    ],
                },
            }
            spec_path = root / "adapter.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            result = prepare_id_label_dataset(
                label_root=labels,
                image_root=images,
                adapter_spec_path=spec_path,
                output_root=root / "prepared",
            )
            observation = SemanticDataset(result.dataset_root).load_observation(image_id)

            np.testing.assert_array_equal(
                observation.labels, np.array([[1, 0]], dtype=np.uint8)
            )
            self.assertEqual(observation.key.image_id, image_id)
            self.assertEqual(result.image_count, 1)

    def test_unmapped_source_id_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "raw-labels"
            images = root / "images"
            labels.mkdir()
            images.mkdir()
            Image.fromarray(np.array([[2]], dtype=np.uint8)).save(labels / "1.png")
            Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(images / "1.jpg")
            spec = {
                "schema_version": 1,
                "label_version": "fixture-v1",
                "producer": "fixture",
                "source_id_to_class_id": {"1": 0},
                "class_spec": {
                    "schema_version": 1,
                    "ignore_id": 255,
                    "classes": [{"id": 0, "name": "zero", "color": [0, 0, 0]}],
                },
            }
            spec_path = root / "adapter.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(SemanticSchemaError, "unmapped"):
                prepare_id_label_dataset(
                    label_root=labels,
                    image_root=images,
                    adapter_spec_path=spec_path,
                    output_root=root / "prepared",
                )


if __name__ == "__main__":
    unittest.main()
