from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from semantic_atdom.utils.camera_calibration import rectified_pinhole_from_colmap
from tools.train_progressive import (
    camera_image_mapping_changed,
    copy_dataset_context,
    resolve_semantic_render_palette,
    trainable_cameras,
    semantic_online_required,
)


class TrainingWorkerContextTests(unittest.TestCase):
    def test_online_finalization_updates_semantics_without_progress_video(self):
        args = Namespace(enable_semantics=True, semantic_final_rebuild=False,
            semantic_tdom_final_only=True, semantic_checkpoint_interval=0,
            render_training_progress=False)
        self.assertTrue(semantic_online_required(args))
        args.semantic_final_rebuild = True
        self.assertFalse(semantic_online_required(args))
        args.render_training_progress = True
        self.assertTrue(semantic_online_required(args))
        args.enable_semantics = False
        self.assertFalse(semantic_online_required(args))

    def test_invalid_optional_palette_has_one_consistent_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic"
            semantic.mkdir()
            (semantic / "class_spec.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ignore_id": 255,
                        "classes": [
                            {
                                "id": 0,
                                "name": "ground",
                                "color": [1, 2, 3],
                            }
                        ],
                    }
                )
            )
            palette = semantic / "tdom_palette.json"
            palette.write_text("not-json")

            with self.assertWarns(RuntimeWarning):
                class_spec, colors, source = resolve_semantic_render_palette(root)

            self.assertEqual(class_spec.num_classes, 1)
            self.assertEqual(colors, ((1, 2, 3),))
            self.assertIsNone(source)

    def test_sentinel_cameras_are_removed_from_training_pool(self):
        valid = Namespace(Tri_Mask=torch.ones((1, 2, 2), dtype=torch.bool))
        sentinel = Namespace(Tri_Mask=torch.ones(2, dtype=torch.bool))

        self.assertEqual(trainable_cameras([sentinel, valid]), [valid])
        self.assertEqual(trainable_cameras([sentinel]), [])

    def test_distortion_change_reloads_rgb_even_when_fov_is_unchanged(self):
        previous_calibration = rectified_pinhole_from_colmap(
            "RADIAL", (100.0, 50.0, 40.0, 0.01, 0.001), width=100, height=80
        )
        changed_calibration = rectified_pinhole_from_colmap(
            "RADIAL", (100.0, 50.0, 40.0, 0.02, 0.001), width=100, height=80
        )
        previous = {
            "FoVx": 1.0,
            "FoVy": 0.8,
            "calibration": previous_calibration,
        }

        self.assertTrue(
            camera_image_mapping_changed(
                Namespace(
                    FovX=1.0,
                    FovY=0.8,
                    calibration=changed_calibration,
                ),
                previous,
            )
        )
        self.assertFalse(
            camera_image_mapping_changed(
                Namespace(
                    FovX=1.0,
                    FovY=0.8,
                    calibration=previous_calibration,
                ),
                previous,
            )
        )

    def test_preflight_context_is_copied_to_independent_worker_arguments(self):
        source = Namespace(
            dataset_cache_root="/output/cache/dataset",
            dataset_preflight_path="/output/cache/dataset/preflight.json",
            dataset_preflight_sha256="a" * 64,
            semantic_authoritative_exact_complete=False,
        )
        worker = Namespace()

        copy_dataset_context(source, worker)

        self.assertEqual(worker.dataset_cache_root, source.dataset_cache_root)
        self.assertEqual(
            worker.dataset_preflight_path, source.dataset_preflight_path
        )
        self.assertEqual(
            worker.dataset_preflight_sha256,
            source.dataset_preflight_sha256,
        )
        self.assertIs(
            worker.semantic_authoritative_exact_complete,
            source.semantic_authoritative_exact_complete,
        )


if __name__ == "__main__":
    unittest.main()
