import unittest

import numpy as np
import torch
from skimage.metrics import structural_similarity

from semantic_atdom.utils.metrics import lpips_distance, ssim_chw
from semantic_atdom.utils.training_schedule import (
    progressive_stage_iterations,
    total_training_iterations,
)


class _RecordingLpips:
    def __init__(self):
        self.inputs = None
        self.normalize = None

    def __call__(self, first, second, normalize=False):
        self.inputs = (first.clone(), second.clone())
        self.normalize = normalize
        return torch.tensor([0.25])


class TrainingContractTests(unittest.TestCase):
    def test_ssim_transposes_chw_to_hwc(self):
        first = torch.linspace(0.0, 1.0, 3 * 9 * 11).reshape(3, 9, 11)
        second = first.roll(1, dims=2)
        expected = structural_similarity(
            first.permute(1, 2, 0).numpy(),
            second.permute(1, 2, 0).numpy(),
            channel_axis=-1,
            data_range=1.0,
            win_size=7,
        )

        self.assertAlmostEqual(ssim_chw(first, second), expected, places=7)

    def test_lpips_normalizes_exactly_once(self):
        metric = _RecordingLpips()
        first = torch.full((3, 2, 2), 0.25)
        second = torch.full((3, 2, 2), 0.75)

        self.assertEqual(lpips_distance(first, second, metric), 0.25)
        self.assertTrue(metric.normalize)
        torch.testing.assert_close(metric.inputs[0], first.unsqueeze(0))
        torch.testing.assert_close(metric.inputs[1], second.unsqueeze(0))

    def test_global_optimization_interval_controls_actual_iterations(self):
        self.assertEqual(
            progressive_stage_iterations(
                new_image_count=3,
                transition_index=1,
                local_iterations_per_image=100,
                global_interval=2,
                global_iterations=40,
            ),
            (300, 0),
        )
        self.assertEqual(
            progressive_stage_iterations(
                new_image_count=0,
                transition_index=2,
                local_iterations_per_image=100,
                global_interval=2,
                global_iterations=40,
            ),
            (0, 40),
        )
        self.assertEqual(
            total_training_iterations(
                initial_iterations=200,
                final_iterations=60,
                stage_new_image_counts=(3, 0, 1),
                local_iterations_per_image=100,
                global_interval=2,
                global_iterations=40,
            ),
            700,
        )


if __name__ == "__main__":
    unittest.main()
