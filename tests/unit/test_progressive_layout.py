from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from semantic_atdom.scene.progressive_layout import (
    resolve_progressive_dataset_layout,
    select_progressive_stages,
)
from semantic_atdom.scene.dataset_readers import _image_identity


class ProgressiveDatasetLayoutTests(unittest.TestCase):
    def test_camera_identity_preserves_decimal_timestamp(self):
        self.assertEqual(
            _image_identity("1658137107.639350179.jpg"),
            "1658137107.639350179",
        )
        self.assertEqual(
            _image_identity("1658137107.639350179"),
            "1658137107.639350179",
        )

    def test_stage_range_treats_equal_end_as_one_stage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in (13, 14, 15):
                (root / str(stage)).mkdir()
            (root / "cache").mkdir()

            self.assertEqual(select_progressive_stages(root, 13, 13), (13,))
            self.assertEqual(select_progressive_stages(root, 14, -1), (14, 15))

    def test_scene_root_discovers_fixed_progressive_and_shared_image_roots(self):
        with TemporaryDirectory() as directory:
            scene = Path(directory) / "interval5_AMtown01"
            progressive = scene / "GaussianSplatting"
            images = scene / "interval5_CAM"
            progressive.mkdir(parents=True)
            images.mkdir()
            (images / "1658137107.1.jpg").write_bytes(b"rgb")

            layout = resolve_progressive_dataset_layout(scene)

            self.assertEqual(layout.scene_root, scene.resolve())
            self.assertEqual(layout.progressive_root, progressive.resolve())
            self.assertEqual(layout.image_root, images.resolve())

    def test_explicit_image_root_wins_and_ambiguous_discovery_fails(self):
        with TemporaryDirectory() as directory:
            scene = Path(directory) / "scene"
            (scene / "GaussianSplatting").mkdir(parents=True)
            first = scene / "first_CAM"
            second = scene / "second_CAM"
            first.mkdir()
            second.mkdir()
            (first / "1.jpg").write_bytes(b"rgb")
            (second / "1.jpg").write_bytes(b"rgb")

            with self.assertRaisesRegex(ValueError, "multiple shared"):
                resolve_progressive_dataset_layout(scene)

            layout = resolve_progressive_dataset_layout(scene, second)
            self.assertEqual(layout.image_root, second.resolve())


if __name__ == "__main__":
    unittest.main()
