from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from semantic_atdom.training_checkpoint import (
    capture_rng_state,
    checkpoint_stage_from_path,
    load_training_checkpoint,
    pack_camera_mask,
    restore_rng_state,
    save_training_checkpoint,
    unpack_camera_mask,
)


class TrainingCheckpointTests(unittest.TestCase):
    def test_checkpoint_stage_is_unambiguous(self):
        self.assertEqual(
            checkpoint_stage_from_path(Path("/output/487/chkpnt487.pth")), 487
        )
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            checkpoint_stage_from_path(Path("/output/latest/model.pth"))

    def test_rolling_checkpoint_stage_comes_from_atomic_payload(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            torch.save({"schema_version": 2, "stage": 31}, path)
            path.with_suffix(".pt.json").write_text(
                '{"schema_version": 2, "stage": 20}'
            )

            self.assertEqual(checkpoint_stage_from_path(path), 31)

    def test_sidecar_replace_failure_leaves_new_checkpoint_resumable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            payload = {
                "schema_version": 2,
                "stage": 31,
                "gaussians": None,
                "iteration": 100,
                "cameras": [],
                "images_already_trained_iterations": {},
                "progressive_transition_index": 2,
                "rng_state": {},
                "data_preprocessor_rng_state": None,
                "phase": "stage_complete",
            }
            real_replace = __import__("os").replace
            metadata = path.with_suffix(".pt.json")

            def fail_sidecar(source, destination):
                if Path(destination) == metadata:
                    raise OSError("injected sidecar failure")
                return real_replace(source, destination)

            with mock.patch(
                "semantic_atdom.training_checkpoint.os.replace",
                side_effect=fail_sidecar,
            ):
                with self.assertRaisesRegex(OSError, "sidecar failure"):
                    save_training_checkpoint(path, payload)

            restored = load_training_checkpoint(path)
            self.assertEqual(restored["stage"], 31)
            self.assertEqual(restored["iteration"], 100)

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
