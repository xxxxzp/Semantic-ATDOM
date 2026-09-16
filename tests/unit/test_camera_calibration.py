import unittest

import numpy as np

from semantic_atdom.utils.camera_calibration import (
    rectify_array,
    rectified_pinhole_from_colmap,
)


class CameraCalibrationTests(unittest.TestCase):
    def test_radial_camera_is_rectified_to_centered_pinhole(self):
        calibration = rectified_pinhole_from_colmap(
            "RADIAL",
            np.array([100.0, 4.0, 3.0, 0.1, 0.02]),
            width=8,
            height=6,
        )

        self.assertTrue(calibration.requires_rectification)
        self.assertEqual(calibration.target_cx, 4.0)
        self.assertEqual(calibration.target_cy, 3.0)
        self.assertEqual(calibration.target_fx, 100.0)
        self.assertEqual(calibration.target_fy, 100.0)

    def test_rgb_and_label_rectification_use_the_same_pixel_map(self):
        calibration = rectified_pinhole_from_colmap(
            "RADIAL",
            np.array([5.0, 4.0, 3.0, 0.2, 0.03]),
            width=8,
            height=6,
        )
        labels = np.arange(48, dtype=np.uint8).reshape(6, 8)
        rgb = np.repeat(labels[:, :, None], 3, axis=2)

        rectified_labels = rectify_array(
            labels, calibration, interpolation="nearest", border_value=255
        )
        rectified_rgb = rectify_array(
            rgb, calibration, interpolation="nearest", border_value=0
        )

        valid = rectified_labels != 255
        np.testing.assert_array_equal(rectified_rgb[:, :, 0][valid], rectified_labels[valid])

    def test_centered_pinhole_is_an_identity_transform(self):
        calibration = rectified_pinhole_from_colmap(
            "PINHOLE", np.array([10.0, 11.0, 4.0, 3.0]), width=8, height=6
        )
        image = np.arange(48, dtype=np.uint8).reshape(6, 8)

        self.assertFalse(calibration.requires_rectification)
        np.testing.assert_array_equal(rectify_array(image, calibration), image)

    def test_unsupported_camera_model_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported COLMAP camera model"):
            rectified_pinhole_from_colmap(
                "FOV", np.array([10.0, 4.0, 3.0, 0.5]), width=8, height=6
            )


if __name__ == "__main__":
    unittest.main()
