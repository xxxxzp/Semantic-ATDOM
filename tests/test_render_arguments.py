from argparse import ArgumentParser
from pathlib import Path
import unittest

from semantic_atdom.arguments import ModelParams
from semantic_atdom.utils.image_utils import tdom_to_uint8

import numpy as np
import torch


class RenderArgumentTests(unittest.TestCase):
    def test_tdom_queue_conversion_preserves_radiometric_values(self):
        tdom = torch.tensor(
            [
                [[-0.5, 0.0], [0.5, 1.0]],
                [[1.5, 2.0], [2.5, 3.0]],
                [[3.5, 4.0], [4.5, 5.0]],
            ],
            dtype=torch.float32,
        )
        expected = tdom.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        expected = (expected * 255).astype(np.uint8)

        np.testing.assert_array_equal(tdom_to_uint8(tdom), expected)

    def test_constant_tdom_converts_without_nan(self):
        converted = tdom_to_uint8(torch.ones((3, 2, 4)))
        np.testing.assert_array_equal(converted, np.full((2, 4, 3), 255, np.uint8))

    def test_model_params_preserve_external_image_folder(self):
        parser = ArgumentParser()
        model_params = ModelParams(parser, sentinel=True)
        image_folder = Path("/data/example/images")

        args = parser.parse_args(
            [
                "--source_path",
                "/data/example/source",
                "--Image_Folder_Path",
                str(image_folder),
            ]
        )

        dataset = model_params.extract(args)

        self.assertEqual(dataset.Image_Folder_Path, str(image_folder))


if __name__ == "__main__":
    unittest.main()
