import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tools.train_progressive import GaussianTrainer


class _BoundedRenderImage:
    live = 0
    peak = 0

    def __init__(self):
        type(self).live += 1
        type(self).peak = max(type(self).peak, type(self).live)
        if type(self).live > 2:
            raise MemoryError("rendered CUDA images accumulated across frames")

    def __del__(self):
        type(self).live -= 1

    def min(self):
        return 0.0

    def max(self):
        return 1.0

    def __sub__(self, _other):
        return self

    def __truediv__(self, _other):
        return self

    def __mul__(self, _other):
        return self

    def permute(self, *_axes):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((2, 2, 3), dtype=np.float32)


class _SavedImage:
    def save(self, path):
        Path(path).touch()


class FinalRenderMemoryTests(unittest.TestCase):
    def setUp(self):
        _BoundedRenderImage.live = 0
        _BoundedRenderImage.peak = 0

    def test_full_render_does_not_retain_previous_cuda_frames(self):
        camera_mask = object()
        cameras = [
            SimpleNamespace(
                image_name=f"image_{index}",
                original_image=SimpleNamespace(cuda=lambda: object()),
                Tri_Mask=camera_mask,
            )
            for index in range(8)
        ]
        trainer = object.__new__(GaussianTrainer)
        trainer.scene = SimpleNamespace(
            getTrainCameras=lambda: cameras.copy()
        )
        trainer.gaussians = object()
        trainer.bg = object()
        trainer.AlreadyTrainingIterations = 100800

        render_masks = []
        metric_masks = []

        def fake_render(*_args, **kwargs):
            render_masks.append(kwargs["Render_Mask"])
            return {
                "render": _BoundedRenderImage(),
                "visibility_filter": torch.tensor([True, False, True]),
            }

        def fake_psnr(*args):
            metric_masks.append(args[2] if len(args) > 2 else None)
            return torch.tensor([3.0])

        class _LegacyAllOnesMask:
            def cuda(self):
                return self

        with tempfile.TemporaryDirectory() as root, mock.patch(
            "tools.train_progressive.render", new=fake_render
        ), mock.patch(
            "tools.train_progressive.psnr",
            new=fake_psnr,
        ), mock.patch(
            "tools.train_progressive.torch.ones",
            return_value=_LegacyAllOnesMask(),
        ), mock.patch(
            "tools.train_progressive.tqdm",
            new=lambda *_args, **_kwargs: SimpleNamespace(
                update=lambda _n: None, close=lambda: None
            ),
        ), mock.patch(
            "tools.train_progressive.Image.fromarray",
            return_value=_SavedImage(),
        ), mock.patch(
            "tools.train_progressive.tdom_to_uint8",
            new=lambda _image: np.zeros((2, 2, 3), dtype=np.uint8),
        ):
            trainer.args = SimpleNamespace(
                renders_dir=str(Path(root) / "renders"),
                metrics_dir=str(Path(root) / "metrics"),
            )
            trainer.Render_Evaluate_All_Images()
            render_path = Path(root) / "renders" / "cameras" / "image_0.jpg"
            self.assertTrue(render_path.is_file())
            metrics = (Path(root) / "metrics" / "per_camera.csv").read_text()
            self.assertIn("renders/cameras/image_0.jpg", metrics)
            self.assertIn(",2,", metrics)

        self.assertEqual(render_masks, [camera_mask] * len(cameras))
        self.assertEqual(metric_masks, [camera_mask] * len(cameras))

        self.assertLessEqual(_BoundedRenderImage.peak, 2)


if __name__ == "__main__":
    unittest.main()
