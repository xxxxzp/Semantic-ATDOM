from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from semantic_atdom.arguments import OptimizationParams
from semantic_atdom.utils.output_retention import (
    prune_intermediate_stage_outputs,
)
from tools.train_progressive import semantic_online_required


class OutputRetentionTests(unittest.TestCase):
    def test_periodic_model_output_is_disabled_by_default(self):
        parser = ArgumentParser()
        OptimizationParams(parser)

        args = parser.parse_args([])

        self.assertEqual(args.OutputModelInterval, 0)
        self.assertEqual(args.semantic_checkpoint_interval, 0)
        self.assertTrue(args.semantic_tdom_final_only)

    def test_final_only_rgb_outputs_defer_semantics_until_final_boundary(self):
        parser = ArgumentParser()
        OptimizationParams(parser)
        args = parser.parse_args([])
        args.enable_semantics = True
        args.render_training_progress = False

        self.assertFalse(semantic_online_required(args))

        args.render_training_progress = True
        self.assertTrue(semantic_online_required(args))
        args.render_training_progress = False
        args.semantic_checkpoint_interval = 5
        self.assertTrue(semantic_online_required(args))
        args.semantic_checkpoint_interval = 0
        args.semantic_tdom_final_only = False
        self.assertTrue(semantic_online_required(args))

    def test_final_only_retention_removes_only_numeric_stage_directories(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in (13, 454, 487):
                stage_root = root / str(stage)
                stage_root.mkdir()
                (stage_root / "sentinel").write_text(str(stage))
            cache = root / "cache"
            cache.mkdir()
            (cache / "sentinel").write_text("keep")

            removed = prune_intermediate_stage_outputs(root, 487)

            self.assertEqual(removed, (13, 454))
            self.assertFalse((root / "13").exists())
            self.assertFalse((root / "454").exists())
            self.assertEqual((root / "487" / "sentinel").read_text(), "487")
            self.assertEqual((cache / "sentinel").read_text(), "keep")

    def test_missing_final_stage_fails_before_removing_anything(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "13").mkdir()

            with self.assertRaisesRegex(RuntimeError, "final-stage directory"):
                prune_intermediate_stage_outputs(root, 487)

            self.assertTrue((root / "13").is_dir())


if __name__ == "__main__":
    unittest.main()
