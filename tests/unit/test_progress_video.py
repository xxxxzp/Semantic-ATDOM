import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from tools.train_progressive import (
    DataSaver,
    GaussianTrainer,
    TRAINING_PROGRESS_HEIGHT,
    TRAINING_PROGRESS_WIDTH,
    select_training_progress_stages,
)


class ProgressVideoTests(unittest.TestCase):
    def _saver(self, root, enabled=True):
        saver = DataSaver()
        saver.SaveDirRootPath = str(Path(root) / ".work" / "video_frames")
        saver.MetricsRootPath = str(Path(root) / "metrics")
        saver.VideoRootPath = str(Path(root) / "videos")
        saver.RenderTrainingProgress = enabled
        saver.TrainingProgressFPS = 4.0
        saver.LegendEntries = [
            {"id": 0, "name": "background", "color": [0, 0, 0]},
            {"id": 1, "name": "roof", "color": [119, 11, 32]},
        ]
        Path(saver.MetricsRootPath).mkdir(parents=True, exist_ok=True)
        saver.MakeDirs()
        return saver

    @staticmethod
    def _metadata(
        stage,
        phase,
        min_x,
        max_x,
        frame_width,
        min_y=0,
        max_y=0,
        frame_height=64,
    ):
        return {
            "schema_version": 1,
            "stage": stage,
            "phase": phase,
            "iteration": stage * 100,
            "observed_cameras": stage,
            "start_stage": 1,
            "end_stage": 2,
            "progress": float(stage - 1),
            "semantic_through_image": stage,
            "semantic_status": "exact_final" if phase == "final_refinement" else "approx_online",
            "semantic_lag_batches": 0,
            "semantic_historical_evidence_stale": False,
            "semantic_errors": [],
            "window": {
                "grid_id": "grid-a",
                "tile_indices": {
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                },
                "width_pixels": frame_width * 8,
                "height_pixels": frame_height * 8,
                "plane_bounds": [
                    float(min_x),
                    float(max_x + 1),
                    float(min_y),
                    float(max_y + 1),
                ],
            },
            "grid": {
                "grid_id": "grid-a",
                "tile_shape": [512, 512],
                "gsd": [0.1, 0.1],
            },
            "map_downsample": 8,
            "column_plane_direction": 1,
            "row_plane_direction": 1,
            "trajectory_pixels": [[8.0, 32.0], [32.0, 32.0]],
        }

    def test_stage_selection_reserves_final_refinement_frame(self):
        stages = select_training_progress_stages(13, 487, 120)

        self.assertEqual(stages[0], 13)
        self.assertEqual(stages[-1], 487)
        self.assertEqual(len(stages), 119)
        self.assertEqual(len(stages), len(set(stages)))

    def test_disabled_video_creates_no_placeholder_directories(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory, enabled=False)

            self.assertFalse(Path(saver.SaveDirRootPath).exists())
            self.assertFalse((Path(directory) / "videos").exists())

    def test_class_legend_uses_the_available_map_height(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory)
            saver.LegendEntries = [
                {
                    "id": index,
                    "name": "class-{}".format(index),
                    "color": [255, 32 + index * 8, 16],
                }
                for index in range(8)
            ]
            canvas = np.zeros((1080, 360, 3), dtype=np.uint8)

            saver._draw_legend(canvas, 0, 78, 980)

            colored_rows = np.where(canvas[:, :, 2] > 200)[0]
            self.assertGreater(colored_rows.size, 0)
            self.assertGreater(
                int(colored_rows.max()) - int(colored_rows.min()),
                450,
            )

    def test_progress_frame_uses_striped_rgb_and_releases_before_semantics(self):
        trainer = object.__new__(GaussianTrainer)
        trainer.TDOM_Cam = object()
        trainer.TDOM_alignment = {
            "window": {"grid_id": "grid-a"},
            "grid": {
                "origin_world": [0.0, 0.0, 0.0],
                "pixel_x_axis_world": [1.0, 0.0, 0.0],
                "pixel_y_axis_world": [0.0, 1.0, 0.0],
            },
        }
        status = SimpleNamespace(
            semantic_through_image="camera-1",
            lag_batches=0,
            status="approx_online",
            historical_evidence_stale=False,
            errors=(),
        )
        trainer.semantic_hook = SimpleNamespace(
            wait_for_idle=lambda: None,
            current_status=lambda _stage: status,
        )
        trainer.last_semantic_sync_result = object()
        trainer.imagenum = 1
        trainer.AlreadyTrainingIterations = 100
        trainer.training_progress_stages = {1}
        trainer.args = SimpleNamespace(
            render_training_progress=True,
            StartFromImageNo=1,
            EndAtImageNo=2,
        )
        trainer.gaussians = object()
        trainer.background = object()
        trainer.semantic_tdom_config = object()
        trainer.scene = SimpleNamespace(getTrainCameras=lambda: [])
        trainer.DataSaver_queue = mock.Mock()
        camera = object()
        semantic_frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(
            trainer, "_training_progress_camera", return_value=camera
        ), mock.patch.object(
            trainer,
            "_project_world_points",
            side_effect=[[], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        ), mock.patch.object(
            trainer, "_semantic_progress_frame", return_value=semantic_frame
        ) as semantic_render, mock.patch(
            "tools.train_progressive.rasterize_rgb_tdom_striped",
            return_value=torch.zeros((3, 2, 2)),
        ) as striped_rgb, mock.patch(
            "tools.train_progressive.render"
        ) as full_frame_rgb, mock.patch(
            "tools.train_progressive.torch.cuda.empty_cache"
        ) as empty_cache:
            timing = trainer._capture_training_progress_tdom("On_The_Fly")

        striped_rgb.assert_called_once_with(
            trainer.gaussians,
            camera,
            trainer.background,
            trainer.semantic_tdom_config,
        )
        full_frame_rgb.assert_not_called()
        semantic_render.assert_called_once_with(camera)
        self.assertEqual(empty_cache.call_count, 2)
        trainer.DataSaver_queue.put.assert_called_once()
        self.assertTrue(timing["requested"])
        self.assertGreaterEqual(timing["semantic_wait_seconds"], 0.0)
        self.assertGreaterEqual(timing["rgb_tdom_seconds"], 0.0)
        self.assertGreaterEqual(timing["semantic_tdom_seconds"], 0.0)

    def test_rgb_and_semantic_maps_encode_one_auditable_h264_video(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory)
            first_rgb = np.full((64, 64, 3), [30, 60, 90], dtype=np.uint8)
            first_semantic = np.full((64, 64, 3), [119, 11, 32], dtype=np.uint8)
            second_rgb = np.full((128, 64, 3), [90, 120, 150], dtype=np.uint8)
            second_semantic = np.full((128, 64, 3), [0, 80, 100], dtype=np.uint8)
            saver.SaveTrainingProgress(
                first_rgb,
                first_semantic,
                self._metadata(1, "initialization", 0, 0, 64),
            )
            saver.SaveTrainingProgress(
                second_rgb,
                second_semantic,
                self._metadata(
                    2,
                    "final_refinement",
                    0,
                    0,
                    64,
                    min_y=0,
                    max_y=1,
                    frame_height=128,
                ),
            )

            saver.GetDemo()

            video = Path(directory) / "videos" / "training_progress.mp4"
            capture = cv2.VideoCapture(str(video))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.assertLess(video_width, TRAINING_PROGRESS_WIDTH)
            self.assertEqual(
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                TRAINING_PROGRESS_HEIGHT,
            )
            ok, first = capture.read()
            ok_final, final = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertTrue(ok_final)
            self.assertGreater(float(first.mean()), 1.0)
            rgb_mask = (
                (final[:, :, 0] > 130)
                & (final[:, :, 1] > 100)
                & (final[:, :, 2] > 70)
            )
            semantic_mask = (
                (final[:, :, 0] > 80)
                & (final[:, :, 1] > 60)
                & (final[:, :, 2] < 30)
            )
            rgb_x = np.where(rgb_mask)[1]
            semantic_x = np.where(semantic_mask)[1]
            self.assertGreater(rgb_x.size, 1000)
            self.assertGreater(semantic_x.size, 1000)
            map_gap = int(semantic_x.min()) - int(rgb_x.max()) - 1
            self.assertLessEqual(map_gap, 32)
            self.assertFalse(Path(saver.TrainingProgressFramePath).exists())

            metrics = json.loads(
                (Path(directory) / "metrics" / "training_progress.json").read_text()
            )
            self.assertEqual(metrics["codec"], "h264")
            self.assertEqual(metrics["width"], video_width)
            self.assertEqual(metrics["frame_count"], 2)
            self.assertEqual(metrics["frames"][-1]["phase"], "final_refinement")
            self.assertEqual(
                metrics["frames"][0]["semantic_status"], "approx_online"
            )
            self.assertNotIn("trajectory_pixels", metrics["frames"][0])


if __name__ == "__main__":
    unittest.main()
