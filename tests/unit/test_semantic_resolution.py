from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from semantic_atdom.semantics.label_grid import labels_on_camera_grid
from semantic_atdom.semantics.orchestrator import (
    CudaSemanticProjector, OnlineSemanticConfig, label_boundary_weights,
)
from semantic_atdom.semantics.planar_fallback import _rectified_labels
from semantic_atdom.semantics.schema import SemanticSchemaError
from semantic_atdom.training_checkpoint import camera_from_state, camera_to_state
from semantic_atdom.utils.camera_calibration import (
    _rectification_maps, rectify_points, rectified_pinhole_from_colmap,
)
from semantic_atdom.utils.camera_utils import loadCam, resize_camera_mask
from semantic_atdom.utils.pixel_grid import PIXEL_GRID_VERSION, camera_resolution, resize_nearest


class SemanticResolutionTests(unittest.TestCase):
    def test_explicit_width_matches_existing_small_grid_and_categorical_sampling(self):
        self.assertEqual(camera_resolution(4032, 3024, 1612), (1612, 1209))
        camera = self._camera(19, 15, 11)
        self.assertEqual((camera.image_width, camera.image_height), (11, 8))
        labels = np.arange(19 * 15, dtype=np.uint16).reshape(15, 19) % 8
        expected = np.array(Image.fromarray(labels).resize((11, 8), Image.Resampling.NEAREST))
        np.testing.assert_array_equal(labels_on_camera_grid(labels, camera), expected)
        mask = resize_camera_mask(torch.from_numpy(labels == 7), (11, 8))
        np.testing.assert_array_equal(mask.numpy()[0], expected == 7)
        self.assertEqual((camera.calibration.width, camera.calibration.height), (19, 15))

    def _camera(self, width=16, height=16, factor=2, radial=False):
        calibration = rectified_pinhole_from_colmap(
            "RADIAL" if radial else "PINHOLE",
            [12, width / 2, height / 2, .1, .01] if radial else
            [12, 13, width / 2, height / 2],
            width=width, height=height,
        )
        w, h = camera_resolution(width, height, factor)
        return SimpleNamespace(
            image_width=w, image_height=h, calibration=calibration,
            Tri_Mask=None, world_view_transform=None, full_proj_transform=None,
            camera_center=None, FoVx=1.0, FoVy=1.0,
        )

    def test_labels_and_masks_sample_the_same_centers_as_pillow(self):
        # Non-divisible dimensions expose the left-aligned torch nearest bug.
        labels = np.arange(15 * 19, dtype=np.uint16).reshape(15, 19) % 8
        labels[7:9, 8:10] = 255
        original = labels.copy()
        for factor in (1, 2, 4, 8):
            camera = self._camera(19, 15, factor)
            size = (camera.image_width, camera.image_height)
            expected = np.array(Image.fromarray(labels).resize(size, Image.Resampling.NEAREST))
            np.testing.assert_array_equal(labels_on_camera_grid(labels, camera), expected)
            self.assertTrue(set(np.unique(expected)).issubset(set(np.unique(labels))))
            mask = labels != 255
            resized = resize_camera_mask(torch.from_numpy(mask), size)[0].numpy()
            np.testing.assert_array_equal(resized, expected != 255)
        np.testing.assert_array_equal(labels, original)

    def test_real_two_row_mask_is_not_a_degenerate_camera_sentinel(self):
        mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
        self.assertEqual(tuple(resize_camera_mask(mask, (2, 1)).shape), (1, 1, 2))
        sentinel = torch.zeros(2)
        self.assertIs(resize_camera_mask(sentinel, (2, 1)), sentinel)

    def test_wrong_source_grid_is_rejected_even_if_it_matches_training_size(self):
        camera = self._camera()
        with self.assertRaisesRegex(SemanticSchemaError, "source calibration"):
            labels_on_camera_grid(np.zeros((8, 8), dtype=np.uint8), camera)
        camera.calibration = None
        with self.assertRaises(SemanticSchemaError):
            labels_on_camera_grid(np.zeros((16, 16), dtype=np.uint8), camera)

    def test_radial_map_and_sfm_points_agree_in_pixel_center_coordinates(self):
        camera = self._camera(19, 15, 4, radial=True)
        calibration = rectified_pinhole_from_colmap(
            "RADIAL", [12, 19 / 2 + .7, 15 / 2 - .4, .1, .01],
            width=19, height=15,
        )
        mx, my = _rectification_maps(calibration)
        # Analytic RADIAL projection of full-grid pixel centers, independent
        # of OpenCV map generation, with conversion back to array indices.
        rows, columns = np.indices((15, 19))
        x = (columns + .5 - calibration.target_cx) / calibration.target_fx
        y = (rows + .5 - calibration.target_cy) / calibration.target_fy
        radius2 = x*x + y*y
        distortion = 1 + .1 * radius2 + .01 * radius2**2
        np.testing.assert_allclose(mx, x * distortion * 12 + calibration.source_cx - .5, atol=2e-6)
        np.testing.assert_allclose(my, y * distortion * 12 + calibration.source_cy - .5, atol=2e-6)
        points = np.column_stack((mx.ravel() + .5, my.ravel() + .5))
        corrected = rectify_points(points, calibration)
        np.testing.assert_allclose(corrected, np.column_stack((columns.ravel() + .5, rows.ravel() + .5)), atol=.002)

    def test_projection_and_fallback_share_ids_and_source_boundary_scale(self):
        labels = np.zeros((16, 16), dtype=np.uint8)
        labels[:, 5:] = 7
        labels[0, :] = 255
        original = labels.copy()
        spec = SimpleNamespace(ignore_id=255, num_classes=8)
        gaussians = SimpleNamespace(get_xyz=None, get_opacity=None, get_scaling=None, get_rotation=None)
        config = OnlineSemanticConfig()
        for factor in (1, 2, 4, 8):
            camera = self._camera(factor=factor)
            size = (camera.image_width, camera.image_height)
            observation = SimpleNamespace(labels=labels)
            dataset = SimpleNamespace(class_spec=spec, load_observation=lambda _: observation)
            with mock.patch("semantic_atdom.semantics.orchestrator.rasterize_semantic_observation") as kernel:
                kernel.return_value = SimpleNamespace(distribution=None, support=None, valid_mass=None)
                CudaSemanticProjector(spec, config).project(gaussians, camera, observation)
                actual = kernel.call_args.kwargs
            expected = resize_nearest(labels, size)
            np.testing.assert_array_equal(actual["labels"].numpy(), expected)
            np.testing.assert_array_equal(_rectified_labels(dataset, camera, "test"), expected)
            expected_weights = resize_nearest(label_boundary_weights(labels), size)
            np.testing.assert_array_equal(actual["boundary_weights"].numpy(), expected_weights)
        np.testing.assert_array_equal(labels, original)

    def test_radial_ignore_border_and_training_mask_gate_projection(self):
        labels = np.full((16, 16), 7, dtype=np.uint8)
        for factor in (2, 4, 8):
            camera = self._camera(factor=factor, radial=True)
            h, w = camera.image_height, camera.image_width
            camera.Tri_Mask = torch.ones((1, h, w), dtype=torch.bool)
            camera.Tri_Mask[:, h // 2, w // 2] = False
            result = labels_on_camera_grid(labels, camera)
            self.assertEqual(result[h // 2, w // 2], 255)
            self.assertEqual(set(np.unique(result)), {7, 255})
        self.assertTrue((labels == 7).all())

    def test_camera_loading_uses_rounded_grid_with_original_calibration(self):
        camera = self._camera(19, 15, 4)
        info = SimpleNamespace(
            image=Image.new("RGB", (19, 15)), uid=1, R=None, T=None,
            FovX=1, FovY=1, Tri_Mask=torch.ones((1, 15, 19), dtype=torch.bool),
            image_name="test", image_path="test.png", calibration=camera.calibration,
        )
        with mock.patch("semantic_atdom.utils.camera_utils.Camera") as constructor:
            loadCam(SimpleNamespace(resolution=4, data_device="cpu"), 0, info, 1.0)
        values = constructor.call_args.kwargs
        self.assertEqual(tuple(values["image"].shape), (3, 4, 5))
        self.assertEqual(tuple(values["Tri_Mask"].shape), (1, 4, 5))
        self.assertIs(values["calibration"], camera.calibration)

    def test_checkpoint_preserves_source_calibration_and_rejects_grid_changes(self):
        camera = self._camera(19, 15, 4, radial=True)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (19, 15)).save(image_path)
            camera.image_path = str(image_path)
            camera.uid = camera.colmap_id = 1
            camera.R = np.eye(3)
            camera.T = camera.trans = np.zeros(3)
            camera.scale = 1.0
            camera.image_name = "test"
            camera.Tri_Mask = torch.ones((1, 4, 5), dtype=torch.bool)
            state = camera_to_state(camera)
            self.assertEqual(state["pixel_grid_version"], PIXEL_GRID_VERSION)
            with mock.patch("semantic_atdom.training_checkpoint.Camera") as constructor:
                camera_from_state(state, data_device="cpu", expected_resolution=4)
            restored = constructor.call_args.kwargs
            self.assertEqual(tuple(restored["image"].shape), (3, 4, 5))
            self.assertEqual(restored["calibration"], camera.calibration)
            with self.assertRaisesRegex(ValueError, "resolution"):
                camera_from_state(state, data_device="cpu", expected_resolution=2)
            state.pop("pixel_grid_version")
            with self.assertRaisesRegex(ValueError, "pixel mapping"):
                camera_from_state(state, data_device="cpu")

    def test_semantic_recovery_identity_records_mapping_and_boundary_units(self):
        config = OnlineSemanticConfig().to_dict()
        self.assertEqual(config["pixel_grid_version"], PIXEL_GRID_VERSION)
        self.assertEqual(config["boundary_bandwidth_units"], "rectified_source_pixels")


if __name__ == "__main__":
    unittest.main()
