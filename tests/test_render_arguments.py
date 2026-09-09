from argparse import ArgumentParser
from pathlib import Path
import unittest

from semantic_atdom.arguments import ModelParams


class RenderArgumentTests(unittest.TestCase):
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
