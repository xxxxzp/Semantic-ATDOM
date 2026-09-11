from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from semantic_atdom.semantics.state import SemanticGaussianState


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GaussianTopologyIntegrationTests(unittest.TestCase):
    def _make_model(self):
        from semantic_atdom.scene.gaussian_model import GaussianModel

        device = torch.device("cuda")
        model = GaussianModel(0)
        model.spatial_lr_scale = 1.0
        model._xyz = torch.nn.Parameter(
            torch.tensor(
                [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                device=device,
            )
        )
        model._features_dc = torch.nn.Parameter(
            torch.tensor(
                [[[10.0, 0.0, 0.0]], [[20.0, 0.0, 0.0]], [[30.0, 0.0, 0.0]]],
                device=device,
            )
        )
        model._features_rest = torch.nn.Parameter(
            torch.zeros((3, 0, 3), device=device)
        )
        model._scaling = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    [[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1]],
                    device=device,
                )
            )
        )
        rotations = torch.zeros((3, 4), device=device)
        rotations[:, 0] = 1.0
        model._rotation = torch.nn.Parameter(rotations)
        model._opacity = torch.nn.Parameter(torch.zeros((3, 1), device=device))
        model.max_radii2D = torch.zeros(3, device=device)
        model.training_setup(
            SimpleNamespace(
                percent_dense=0.2,
                position_lr_init=1e-3,
                position_lr_final=1e-4,
                position_lr_delay_mult=1.0,
                position_lr_max_steps=10,
                feature_lr=1e-3,
                opacity_lr=1e-3,
                scaling_lr=1e-3,
                rotation_lr=1e-3,
            )
        )
        model.tmp_radii = torch.zeros(3, device=device)
        return model

    def test_real_clone_split_and_prune_keep_semantic_state_aligned(self):
        model = self._make_model()
        state = SemanticGaussianState.initialize(3, 2, device=model.get_xyz.device)
        state.add_evidence(
            torch.tensor(
                [[4.0, 1.0], [0.0, 3.0], [2.0, 2.0]], device=state.device
            )
        )
        state.semantic_support[:] = torch.tensor(
            [10.0, 20.0, 30.0], device=state.device
        )
        state.group_count[:] = torch.tensor([1, 2, 3], device=state.device)
        state.mark_clean()
        events = []

        def apply_delta(delta):
            events.append(delta.kind)
            state.apply_topology(delta)

        model.set_topology_observer(apply_delta, owner=state)
        optimizer_tensor_ids = {
            id(parameter)
            for group in model.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertNotIn(id(state.alpha_raw), optimizer_tensor_ids)
        self.assertNotIn(id(state.alpha_robust), optimizer_tensor_ids)
        model.xyz_gradient_accum[:] = torch.tensor(
            [[1.0], [1.0], [0.0]], device=state.device
        )
        model.denom.fill_(1.0)
        model.densify_and_prune(
            max_grad=0.5,
            min_opacity=0.0,
            extent=1.0,
            max_screen_size=None,
            radii=torch.zeros(3, device=state.device),
        )
        self.assertEqual(model.get_xyz.shape[0], 5)
        self.assertEqual(state.num_gaussians, 5)
        self.assertEqual(events, ["clone", "split"])
        torch.testing.assert_close(
            state.alpha_raw - 0.5,
            torch.tensor(
                [[4.0, 1.0], [2.0, 2.0], [4.0, 1.0], [0.0, 3.0], [0.0, 3.0]],
                device=state.device,
            ),
        )
        self.assertEqual(state.gaussian_ids.tolist(), [0, 2, 3, 4, 5])
        torch.testing.assert_close(
            state.semantic_support,
            torch.tensor([5.0, 30.0, 5.0, 10.0, 10.0], device=state.device),
        )
        self.assertEqual(state.group_count.tolist(), [1, 3, 1, 2, 2])
        self.assertEqual(state.dirty.tolist(), [True, False, True, True, True])
        torch.testing.assert_close(
            model.get_features_dc[:, 0, 0],
            torch.tensor([10.0, 30.0, 10.0, 20.0, 20.0], device=state.device),
        )

        model.prune_points(
            torch.tensor([False, False, True, False, False], device=state.device)
        )
        self.assertEqual(model.get_xyz.shape[0], 4)
        self.assertEqual(state.num_gaussians, 4)
        self.assertEqual(events, ["clone", "split", "prune"])

    def test_progressive_expand_publishes_append_prior(self):
        from semantic_atdom.utils.graphics_utils import BasicPointCloud

        model = self._make_model()
        model.tmp_radii = None
        state = SemanticGaussianState.initialize(3, 2, device=model.get_xyz.device)
        events = []

        def apply_delta(delta):
            events.append(delta.kind)
            state.apply_topology(delta)

        model.set_topology_observer(apply_delta, owner=state)
        point_cloud = BasicPointCloud(
            points=np.array([[40.0, 0.0, 0.0]], dtype=np.float32),
            colors=np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
            normals=np.zeros((1, 3), dtype=np.float32),
        )
        with patch(
            "semantic_atdom.scene.gaussian_model.distCUDA2",
            return_value=torch.ones(1, device="cuda"),
        ):
            model.expand_from_pcd(point_cloud, spatial_lr_scale=1.0)

        self.assertEqual(events, ["append"])
        self.assertEqual(model.get_xyz.shape[0], 4)
        self.assertEqual(state.num_gaussians, 4)
        torch.testing.assert_close(
            state.alpha_raw[3], torch.tensor([0.5, 0.5], device="cuda")
        )
        self.assertEqual(state.gaussian_ids.tolist(), [0, 1, 2, 3])

    def test_observer_failure_never_interrupts_geometry_and_requires_rebuild(self):
        model = self._make_model()

        class FailingObserver:
            num_gaussians = 3
            is_valid = True
            invalid_reason = None

            def __call__(self, _delta):
                raise RuntimeError("semantic OOM")

            def invalidate(self, reason):
                self.is_valid = False
                self.invalid_reason = str(reason)

        observer = FailingObserver()
        model.set_topology_observer(observer)
        with self.assertWarnsRegex(RuntimeWarning, "must be rebuilt"):
            model.densify_and_clone(
                torch.tensor([[1.0], [0.0], [0.0]], device="cuda"),
                grad_threshold=0.5,
                scene_extent=1.0,
            )

        self.assertEqual(model.get_xyz.shape[0], 4)
        self.assertTrue(model.topology_requires_rebuild)
        self.assertIn("semantic OOM", str(model.topology_observer_error))
        self.assertFalse(observer.is_valid)
        model.prune_points(torch.tensor([False, True, False, False], device="cuda"))
        self.assertEqual(model.get_xyz.shape[0], 3)

    def test_topology_delta_construction_failure_is_also_isolated(self):
        from semantic_atdom.scene import gaussian_model

        model = self._make_model()
        state = SemanticGaussianState.initialize(3, 2, device="cuda")
        model.set_topology_observer(state.apply_topology)
        with patch.object(
            gaussian_model.TopologyDelta,
            "clone",
            side_effect=RuntimeError("delta OOM"),
        ):
            with self.assertWarnsRegex(RuntimeWarning, "must be rebuilt"):
                model.densify_and_clone(
                    torch.tensor([[1.0], [0.0], [0.0]], device="cuda"),
                    grad_threshold=0.5,
                    scene_extent=1.0,
                )

        self.assertEqual(model.get_xyz.shape[0], 4)
        self.assertTrue(model.topology_requires_rebuild)
        self.assertFalse(state.is_valid)

    def test_invalidate_failure_cannot_escape_into_geometry(self):
        model = self._make_model()

        class BrokenInvalidationObserver:
            num_gaussians = 3
            is_valid = True

            def __call__(self, _delta):
                raise RuntimeError("original sync failure")

            def invalidate(self, _reason):
                raise RuntimeError("invalidate also failed")

        model.set_topology_observer(BrokenInvalidationObserver())
        with self.assertWarnsRegex(RuntimeWarning, "original sync failure"):
            model.densify_and_clone(
                torch.tensor([[1.0], [0.0], [0.0]], device="cuda"),
                grad_threshold=0.5,
                scene_extent=1.0,
            )
        self.assertEqual(model.get_xyz.shape[0], 4)
        self.assertIn("original sync failure", str(model.topology_observer_error))

    def test_full_geometry_replacement_invalidates_bound_semantics(self):
        from semantic_atdom.utils.graphics_utils import BasicPointCloud

        model = self._make_model()
        state = SemanticGaussianState.initialize(3, 2, device="cuda")
        model.set_topology_observer(state.apply_topology)
        geometry_hash_before = model.geometry_sha256()
        point_cloud = BasicPointCloud(
            points=np.array([[99.0, 0.0, 0.0]], dtype=np.float32),
            colors=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            normals=np.zeros((1, 3), dtype=np.float32),
        )
        with patch(
            "semantic_atdom.scene.gaussian_model.distCUDA2",
            return_value=torch.ones(1, device="cuda"),
        ):
            with self.assertWarnsRegex(RuntimeWarning, "complete Gaussian model"):
                model.create_from_pcd(point_cloud, spatial_lr_scale=1.0)

        self.assertEqual(model.get_xyz.shape[0], 1)
        self.assertFalse(state.is_valid)
        self.assertTrue(model.topology_requires_rebuild)
        self.assertNotEqual(model.geometry_sha256(), geometry_hash_before)


if __name__ == "__main__":
    unittest.main()
