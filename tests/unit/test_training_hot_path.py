import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from tools.train_progressive import (
    DataSaver,
    StageTrainingInputCache,
    TrainingProgressReporter,
    release_camera_sampling_buffers,
    should_retain_visibility,
    summarize_progressive_stage_timings,
)


class _RecordingImage:
    def __init__(self, name):
        self.name = name
        self.transfers = 0

    def to(self, device):
        self.transfers += 1
        return (self.name, str(device), self.transfers)


class _RecordingLoss:
    def __init__(self, value=0.25):
        self.value = value
        self.item_calls = 0

    def item(self):
        self.item_calls += 1
        return self.value


class _RecordingProgressBar:
    def __init__(self):
        self.updates = []
        self.postfixes = []
        self.close_calls = 0

    def update(self, amount):
        self.updates.append(amount)

    def set_postfix(self, value):
        self.postfixes.append(value)

    def close(self):
        self.close_calls += 1


class TrainingHotPathTests(unittest.TestCase):
    def test_stage_cache_transfers_new_inputs_once_but_does_not_retain_global_views(self):
        new_image = _RecordingImage("new")
        global_image = _RecordingImage("global")
        new_mask = _RecordingImage("new-mask")
        global_mask = _RecordingImage("global-mask")
        new_camera = SimpleNamespace(
            original_image=new_image,
            Tri_Mask=new_mask,
        )
        global_camera = SimpleNamespace(
            original_image=global_image,
            Tri_Mask=global_mask,
        )

        cache = StageTrainingInputCache([new_camera], device="cuda:0")

        first = cache.get(new_camera)
        for _ in range(99):
            self.assertIs(cache.get(new_camera), first)
            cache.get_mask(new_camera)
        cache.get(global_camera)
        cache.get(global_camera)
        cache.get_mask(global_camera)
        cache.get_mask(global_camera)

        self.assertEqual(new_image.transfers, 1)
        self.assertEqual(new_mask.transfers, 1)
        self.assertEqual(global_image.transfers, 2)
        self.assertEqual(global_mask.transfers, 2)

    def test_progress_reporter_samples_loss_without_losing_progress(self):
        loss = _RecordingLoss()
        progress = _RecordingProgressBar()
        reporter = TrainingProgressReporter(
            progress,
            interval=10,
            initial_ema_loss=0.0,
        )

        for step in range(1, 201):
            reporter.maybe_update(loss, step=step, total_steps=200)

        self.assertEqual(loss.item_calls, 20)
        self.assertEqual(sum(progress.updates), 200)
        self.assertEqual(progress.close_calls, 1)
        self.assertEqual(len(progress.postfixes), 20)

    def test_progress_reporter_always_reports_a_short_final_interval(self):
        loss = _RecordingLoss()
        progress = _RecordingProgressBar()
        reporter = TrainingProgressReporter(progress, interval=10)

        for step in range(1, 8):
            reporter.maybe_update(loss, step=step, total_steps=7)

        self.assertEqual(loss.item_calls, 1)
        self.assertEqual(progress.updates, [7])
        self.assertEqual(progress.close_calls, 1)

    def test_sampling_buffers_are_released_after_point_generation(self):
        camera = SimpleNamespace(
            GradLog=object(),
            MaxGradLog=12.5,
            LogMask=object(),
        )

        release_camera_sampling_buffers([camera])

        self.assertIsNone(camera.GradLog)
        self.assertEqual(camera.MaxGradLog, -1)
        self.assertIsNone(camera.LogMask)

    def test_visibility_is_not_retained_for_normal_progressive_training(self):
        self.assertFalse(
            should_retain_visibility("On_The_Fly", no_sampling=False)
        )
        self.assertTrue(
            should_retain_visibility("Initialization", no_sampling=False)
        )
        self.assertTrue(
            should_retain_visibility("On_The_Fly", no_sampling=True)
        )
        self.assertTrue(
            should_retain_visibility(
                "On_The_Fly",
                no_sampling=False,
                reset_pending=True,
            )
        )

    def test_progressive_timing_uses_actual_new_image_count(self):
        summary = summarize_progressive_stage_timings(
            [
                {
                    "new_image_count": 1,
                    "geometry_seconds": 2.0,
                    "online_seconds": 3.0,
                },
                {
                    "new_image_count": 2,
                    "geometry_seconds": 4.0,
                    "online_seconds": 8.0,
                },
            ]
        )

        self.assertEqual(summary["progressive_stage_count"], 2)
        self.assertEqual(summary["progressive_image_count"], 3)
        self.assertEqual(summary["geometry_seconds_per_image"], 2.0)
        self.assertAlmostEqual(summary["online_seconds_per_image"], 11.0 / 3.0)

    def test_saved_timing_has_no_stage_number_based_average(self):
        with TemporaryDirectory() as directory:
            saver = DataSaver()
            saver.MetricsRootPath = directory
            stage_records = [
                {
                    "new_image_count": 1,
                    "geometry_seconds": 2.0,
                    "online_seconds": 3.0,
                },
                {
                    "new_image_count": 2,
                    "geometry_seconds": 4.0,
                    "online_seconds": 8.0,
                },
            ]

            saver.SaveTimeCost(
                {"ProgressiveTrain": 6.0},
                imagenum=100,
                stage_records=stage_records,
            )

            payload = json.loads(
                (Path(directory) / "timing.json").read_text()
            )
            self.assertNotIn("time_per_stage_seconds", payload)
            self.assertEqual(payload["progressive_stage_count"], 2)
            self.assertEqual(payload["progressive_image_count"], 3)
            self.assertEqual(payload["geometry_seconds_per_image"], 2.0)


if __name__ == "__main__":
    unittest.main()
