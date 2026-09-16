import tempfile
import unittest
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
    def save(self, _path):
        pass


class FinalRenderMemoryTests(unittest.TestCase):
    def setUp(self):
        _BoundedRenderImage.live = 0
        _BoundedRenderImage.peak = 0

    def test_full_render_does_not_retain_previous_cuda_frames(self):
        cameras = [
            SimpleNamespace(
                image_name=f"image_{index}",
                original_image=SimpleNamespace(cuda=lambda: object()),
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

        def fake_render(*_args, **_kwargs):
            return {
                "render": _BoundedRenderImage(),
                "visibility_filter": SimpleNamespace(shape=(1,)),
            }

        with tempfile.TemporaryDirectory() as root, mock.patch(
            "tools.train_progressive.render", new=fake_render
        ), mock.patch(
            "tools.train_progressive.psnr",
            new=lambda *_args: torch.tensor([3.0]),
        ), mock.patch(
            "tools.train_progressive.tqdm",
            new=lambda *_args, **_kwargs: SimpleNamespace(update=lambda _n: None),
        ), mock.patch(
            "tools.train_progressive.Image.fromarray",
            return_value=_SavedImage(),
        ), mock.patch(
            "tools.train_progressive.tdom_to_uint8",
            new=lambda _image: np.zeros((2, 2, 3), dtype=np.uint8),
        ):
            trainer.args = SimpleNamespace(Model_Path_Dir=root)
            trainer.Render_Evaluate_All_Images()

        self.assertLessEqual(_BoundedRenderImage.peak, 2)


if __name__ == "__main__":
    unittest.main()
