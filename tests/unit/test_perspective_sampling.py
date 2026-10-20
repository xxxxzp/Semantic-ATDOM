"""Sampled points must lie on the requested camera rays and triangle plane."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from semantic_atdom.scene.dataset_readers import Get_3D_Sampled_Points, Get_3D_Sampled_Points_GPU


class PerspectiveSamplingTests(unittest.TestCase):
    def test_tensor_and_numpy_preserve_reprojection_under_arbitrary_pose(self):
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        translation = np.array([2., -3., 4.])
        camera_points = np.array([[0., 0., 1.], [2., 0., 2.], [0., 4., 4.]])
        world = (camera_points - translation) @ rotation.T
        xys = camera_points[:, :2] / camera_points[:, 2:]
        samples = np.array([[1/3, 1/3], [.2, .5]])
        triangles = np.array([[0, 1, 2]])
        ids = np.zeros(2, dtype=np.int64)
        for device in ('numpy', 'cpu', 'cuda'):
            if device == 'cuda' and not torch.cuda.is_available():
                continue
            with self.subTest(device=device):
                if device == 'numpy':
                    result, _ = Get_3D_Sampled_Points(samples, ids, SimpleNamespace(simplices=triangles), xys, world, world, None, None, camera_depths=camera_points[:, 2])
                else:
                    t = lambda value: torch.as_tensor(value, device=device)
                    result, _ = Get_3D_Sampled_Points_GPU(t(samples), t(ids), t(triangles), t(xys), t(world), t(world), None, None, camera_depths=t(camera_points[:, 2]))
                    result = result.cpu().numpy()
                projected = result @ rotation + translation
                np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], samples, atol=1e-6)
                # The triangle's camera-space plane is x/2 + 3*y/4 - z = -1.
                np.testing.assert_allclose(projected[:, 0]/2 + 3*projected[:, 1]/4 - projected[:, 2], -1, atol=1e-6)
