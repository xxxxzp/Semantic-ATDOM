from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from semantic_atdom.training_checkpoint import (
    capture_rng_state,
    checkpoint_stage_from_path,
    pack_camera_mask,
    restore_rng_state,
    unpack_camera_mask,
)


class TrainingCheckpointTests(unittest.TestCase):
    def test_checkpoint_stage_is_unambiguous(self):
        self.assertEqual(
            checkpoint_stage_from_path(Path("/output/487/chkpnt487.pth")), 487
        )
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            checkpoint_stage_from_path(Path("/output/latest/model.pth"))

    def test_camera_masks_round_trip_in_packed_form(self):
        mask = torch.tensor(
            [[[True, False, True], [False, False, True]]], dtype=torch.bool
        )
        payload = pack_camera_mask(mask)

        self.assertLess(payload["data"].numel(), mask.numel())
        torch.testing.assert_close(unpack_camera_mask(payload), mask)

    def test_scalar_all_valid_mask_uses_constant_encoding(self):
        payload = pack_camera_mask(torch.ones(1, dtype=torch.bool))
        self.assertEqual(payload, {"kind": "all_valid"})
        torch.testing.assert_close(
            unpack_camera_mask(payload), torch.ones(1, dtype=torch.bool)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_rng_restore_accepts_checkpoint_loaded_on_cuda(self):
        original = capture_rng_state()
        mapped = dict(original)
        mapped["torch_cpu"] = original["torch_cpu"].cuda()
        mapped["torch_cuda"] = [state.cuda() for state in original["torch_cuda"]]
        try:
            restore_rng_state(mapped)
            self.assertEqual(torch.random.get_rng_state().device.type, "cpu")
        finally:
            restore_rng_state(original)


if __name__ == "__main__":
    unittest.main()
