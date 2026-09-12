import unittest

import numpy as np
import torch

from semantic_atdom.scene.cameras import DummyCamera
from semantic_atdom.scene.cameras import DummyPipeline
from semantic_atdom.gaussian_renderer import render
from semantic_atdom.semantics.orchestrator import FrozenGaussianSnapshot
from semantic_atdom.semantics.schema import SemanticClassSpec
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    rasterize_semantic_tdom,
)


try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        _C,
    )

    HAS_SEMANTIC_TDOM_CUDA = torch.cuda.is_available() and hasattr(
        _C, "semantic_tdom"
    )
except (ImportError, OSError):
    HAS_SEMANTIC_TDOM_CUDA = False


@unittest.skipUnless(
    HAS_SEMANTIC_TDOM_CUDA,
    "requires CUDA and the rebuilt Semantic TDOM rasterizer",
)
class SemanticCudaTdomTests(unittest.TestCase):
    @staticmethod
    def _two_class_spec():
        return SemanticClassSpec.from_dict(
            {
                "schema_version": 1,
                "ignore_id": 255,
                "classes": [
                    {"id": 0, "name": "ground", "color": [0, 0, 0]},
                    {"id": 1, "name": "building", "color": [255, 0, 0]},
                ],
            }
        )

    def test_complete_frozen_snapshot_is_accepted_by_rgb_tdom_renderer(self):
        class Gaussians:
            active_sh_degree = 0
            max_sh_degree = 0
            _xyz = torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"
            )
            _features_dc = torch.zeros(
                (1, 1, 3), dtype=torch.float32, device="cuda"
            )
            _features_rest = torch.zeros(
                (1, 0, 3), dtype=torch.float32, device="cuda"
            )
            _opacity = torch.zeros(
                (1, 1), dtype=torch.float32, device="cuda"
            )
            _scaling = torch.full(
                (1, 3), -1.5, dtype=torch.float32, device="cuda"
            )
            _rotation = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )

        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            16,
            16,
        )
        snapshot = FrozenGaussianSnapshot(Gaussians())
        snapshot.wait_ready()
        result = render(
            camera,
            snapshot,
            DummyPipeline(),
            torch.zeros(3, dtype=torch.float32, device="cuda"),
            Render_Mask=torch.ones(1, dtype=torch.bool, device="cuda"),
            TDOM=True,
        )

        self.assertEqual(tuple(result["render"].shape), (3, 16, 16))
        self.assertTrue(bool(torch.isfinite(result["render"]).all()))

    def test_rgb_tdom_honors_non_default_pytorch_stream(self):
        class Gaussians:
            active_sh_degree = 0
            max_sh_degree = 0
            _xyz = torch.tensor(
                [[0.0, 0.0, 2.0]], dtype=torch.float32, device="cuda"
            )
            _features_dc = torch.ones(
                (1, 1, 3), dtype=torch.float32, device="cuda"
            )
            _features_rest = torch.zeros(
                (1, 0, 3), dtype=torch.float32, device="cuda"
            )
            _opacity = torch.full(
                (1, 1), 3.0, dtype=torch.float32, device="cuda"
            )
            _scaling = torch.zeros(
                (1, 3), dtype=torch.float32, device="cuda"
            )
            _rotation = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )

        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            256,
            256,
        )
        snapshot = FrozenGaussianSnapshot(Gaussians())
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            snapshot.wait_ready(stream)
            rendered = render(
                camera,
                snapshot,
                DummyPipeline(),
                torch.zeros(3, dtype=torch.float32, device="cuda"),
                Render_Mask=torch.ones(
                    1, dtype=torch.bool, device="cuda"
                ),
                TDOM=True,
            )["render"].clone()
        stream.synchronize()

        self.assertGreater(float(rendered.max().item()), 0.0)

    def test_high_level_renderer_enforces_memory_budget(self):
        class Gaussians:
            get_xyz = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
            get_opacity = torch.ones((1, 1), dtype=torch.float32, device="cuda")
            get_scaling = torch.ones((1, 3), dtype=torch.float32, device="cuda")
            get_rotation = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )

        spec = self._two_class_spec()
        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            16,
            16,
        )
        with self.assertRaisesRegex(ValueError, "byte budget"):
            rasterize_semantic_tdom(
                Gaussians(),
                SemanticGaussianState.initialize(1, 2, device="cuda"),
                camera,
                spec,
                config=SemanticTdomConfig(max_output_bytes=1),
            )

    def test_high_level_group_disagreement_has_local_valid_support(self):
        class Gaussians:
            get_xyz = torch.tensor(
                [[0.0, 0.0, 2.0]], dtype=torch.float32, device="cuda"
            )
            get_opacity = torch.tensor(
                [[0.9]], dtype=torch.float32, device="cuda"
            )
            get_scaling = torch.ones(
                (1, 3), dtype=torch.float32, device="cuda"
            )
            get_rotation = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )

        state = SemanticGaussianState.initialize(1, 2, device="cuda")
        state.add_evidence(torch.tensor([[4.0, 1.0]], device="cuda"))
        state.group_variance[0] = torch.tensor([0.1, 0.1], device="cuda")
        state.group_count[0] = 2
        state.group_disagreement_available = True
        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            64,
            64,
        )

        result = rasterize_semantic_tdom(
            Gaussians(),
            state,
            camera,
            self._two_class_spec(),
            config=SemanticTdomConfig(
                min_free_cuda_bytes=0, save_diagnostics=True
            ),
        )
        valid = torch.isfinite(result.group_disagreement)

        self.assertTrue(bool(valid.any()))
        torch.testing.assert_close(
            result.group_disagreement[valid],
            torch.full_like(result.group_disagreement[valid], 0.2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_high_level_exact_stripes_match_full_frame(self):
        class Gaussians:
            get_xyz = torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]],
                dtype=torch.float32,
                device="cuda",
            )
            get_opacity = torch.tensor(
                [[0.5], [0.5]], dtype=torch.float32, device="cuda"
            )
            get_scaling = torch.tensor(
                [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                dtype=torch.float32,
                device="cuda",
            )
            get_rotation = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )

        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            16,
            32,
        )
        state = SemanticGaussianState.initialize(2, 2, device="cuda")
        prior_only = rasterize_semantic_tdom(
            Gaussians(),
            state,
            camera,
            self._two_class_spec(),
            config=SemanticTdomConfig(
                min_free_cuda_bytes=0, save_diagnostics=True
            ),
        )
        self.assertEqual(float(prior_only.absolute_support.max()), 0.0)
        self.assertTrue(bool((prior_only.class_ids == 255).all()))
        self.assertTrue(bool(torch.isnan(prior_only.entropy).all()))
        self.assertTrue(bool(torch.isnan(prior_only.group_disagreement).all()))
        self.assertEqual(
            prior_only.metrics["group_disagreement_status"], "unavailable"
        )

        state.add_evidence(
            torch.tensor([[4.0, 1.0], [1.0, 4.0]], device="cuda")
        )
        full = rasterize_semantic_tdom(
            Gaussians(),
            state,
            camera,
            self._two_class_spec(),
            config=SemanticTdomConfig(
                min_free_cuda_bytes=0, save_diagnostics=True
            ),
        )
        striped = rasterize_semantic_tdom(
            Gaussians(),
            state,
            camera,
            self._two_class_spec(),
            config=SemanticTdomConfig(
                render_tile_rows=16,
                min_free_cuda_bytes=0,
                save_diagnostics=True,
            ),
        )

        self.assertEqual(full.metrics["render_mode"], "full_frame")
        self.assertEqual(striped.metrics["render_mode"], "exact_striped")
        torch.testing.assert_close(striped.class_ids, full.class_ids)
        for name in (
            "coverage",
            "absolute_support",
            "entropy",
            "mixture_disagreement",
            "group_disagreement",
            "prior_regularized_variance_proxy",
        ):
            torch.testing.assert_close(
                getattr(striped, name), getattr(full, name), equal_nan=True
            )

    def test_high_level_renderer_never_turns_prior_into_class_mass(self):
        class Gaussians:
            get_xyz = torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]],
                dtype=torch.float32,
                device="cuda",
            )
            get_opacity = torch.tensor(
                [[0.5], [0.5]], dtype=torch.float32, device="cuda"
            )
            get_scaling = torch.full(
                (2, 3), 0.2, dtype=torch.float32, device="cuda"
            )
            get_rotation = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )

        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            16,
            32,
        )
        state = SemanticGaussianState.initialize(2, 2, device="cuda")
        state.add_evidence(
            torch.tensor([[0.0, 4.0], [0.0, 4.0]], device="cuda")
        )
        result = rasterize_semantic_tdom(
            Gaussians(),
            state,
            camera,
            self._two_class_spec(),
            config=SemanticTdomConfig(
                min_free_cuda_bytes=0, save_probabilities=True
            ),
        )
        supported = result.absolute_support > 0
        classified = ~result.unknown_mask

        self.assertTrue(bool(supported.any()))
        self.assertTrue(bool(classified.any()))
        self.assertTrue(bool((result.probabilities[0, supported] == 0).all()))
        self.assertTrue(bool((result.class_ids[classified] == 1).all()))

    def _empty_call(self, channels):
        device = torch.device("cuda")
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return _C.semantic_tdom(
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 1), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 4), dtype=torch.float32, device=device),
            1.0,
            empty,
            torch.empty((0, channels), dtype=torch.float32, device=device),
            torch.eye(4, dtype=torch.float32, device=device),
            torch.eye(4, dtype=torch.float32, device=device),
            1.0,
            1.0,
            torch.zeros(3, dtype=torch.float32, device=device),
            3,
            5,
            1 << 60,
            False,
        )

    def test_dynamic_features_match_rgb_tdom_alpha_compositing(self):
        device = torch.device("cuda")
        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            16,
            32,
        )
        means = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]],
            dtype=torch.float32,
            device=device,
        )
        opacity = torch.tensor(
            [[0.5], [0.5]], dtype=torch.float32, device=device
        )
        scales = torch.tensor(
            [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
            dtype=torch.float32,
            device=device,
        )
        rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        features = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        empty = torch.empty(0, dtype=torch.float32, device=device)
        semantic, coverage, radii = _C.semantic_tdom(
            means,
            opacity,
            scales,
            rotations,
            1.0,
            empty,
            features,
            camera.world_view_transform.contiguous(),
            camera.full_proj_transform.contiguous(),
            float(torch.tan(torch.tensor(camera.FoVx * 0.5))),
            float(torch.tan(torch.tensor(camera.FoVy * 0.5))),
            camera.camera_center.contiguous(),
            camera.image_height,
            camera.image_width,
            1 << 60,
            True,
        )

        settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=float(torch.tan(torch.tensor(camera.FoVx * 0.5))),
            tanfovy=float(torch.tan(torch.tensor(camera.FoVy * 0.5))),
            bg=torch.zeros(3, dtype=torch.float32, device=device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            get_normal=False,
            debug=True,
            Render_Mask=torch.ones(1, dtype=torch.bool, device=device),
            Use_Render_Mask=False,
            TDOM=True,
        )
        rgb, rgb_radii, _, _, _, _ = GaussianRasterizer(settings)(
            means3D=means,
            means2D=torch.zeros_like(means),
            opacities=opacity,
            colors_precomp=features,
            scales=scales,
            rotations=rotations,
            all_map=torch.zeros_like(features),
        )

        torch.testing.assert_close(semantic, rgb, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            coverage, semantic.sum(dim=0), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(radii, rgb_radii)
        self.assertFalse(semantic.requires_grad)

        tile_arguments = (
            means,
            opacity,
            scales,
            rotations,
            1.0,
            empty,
            features,
            camera.world_view_transform.contiguous(),
            camera.full_proj_transform.contiguous(),
            float(torch.tan(torch.tensor(camera.FoVx * 0.5))),
            float(torch.tan(torch.tensor(camera.FoVy * 0.5))),
            camera.camera_center.contiguous(),
            camera.image_height,
            camera.image_width,
        )
        first, first_coverage, _, _ = _C.semantic_tdom_tile(
            *tile_arguments, 0, 16, -1, 1 << 60, True
        )
        second, second_coverage, _, _ = _C.semantic_tdom_tile(
            *tile_arguments, 16, 16, -1, 1 << 60, True
        )
        torch.testing.assert_close(torch.cat((first, second), dim=1), semantic)
        torch.testing.assert_close(
            torch.cat((first_coverage, second_coverage), dim=0), coverage
        )

        with self.assertRaisesRegex(RuntimeError, "binning buffer"):
            _C.semantic_tdom(
                means,
                opacity,
                scales,
                rotations,
                1.0,
                empty,
                features,
                camera.world_view_transform.contiguous(),
                camera.full_proj_transform.contiguous(),
                float(torch.tan(torch.tensor(camera.FoVx * 0.5))),
                float(torch.tan(torch.tensor(camera.FoVy * 0.5))),
                camera.camera_center.contiguous(),
                camera.image_height,
                camera.image_width,
                0,
                True,
            )

    def test_squared_semantic_moment_uses_squared_alpha_t_weight(self):
        device = torch.device("cuda")
        camera = DummyCamera(
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            1.0,
            16,
            16,
        )
        means = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        opacity = torch.tensor([[0.5]], device=device)
        scales = torch.tensor([[0.2, 0.2, 0.2]], device=device)
        rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device
        )
        features = torch.tensor([[0.7, 0.3]], device=device)
        empty = torch.empty(0, dtype=torch.float32, device=device)
        _, coverage, _, squared = _C.semantic_tdom_tile(
            means,
            opacity,
            scales,
            rotations,
            1.0,
            empty,
            features,
            camera.world_view_transform.contiguous(),
            camera.full_proj_transform.contiguous(),
            float(torch.tan(torch.tensor(camera.FoVx * 0.5))),
            float(torch.tan(torch.tensor(camera.FoVy * 0.5))),
            camera.camera_center.contiguous(),
            camera.image_height,
            camera.image_width,
            0,
            camera.image_height,
            0,
            1 << 60,
            True,
        )

        torch.testing.assert_close(
            squared, 0.7 * coverage.square(), rtol=1e-5, atol=1e-6
        )

    def test_empty_geometry_returns_unknown_ready_zero_maps(self):
        rendered, coverage, radii = self._empty_call(4)
        self.assertEqual(tuple(rendered.shape), (4, 3, 5))
        self.assertEqual(tuple(coverage.shape), (3, 5))
        self.assertEqual(radii.numel(), 0)
        self.assertEqual(float(rendered.sum()), 0.0)

    def test_channel_boundaries_cover_current_and_maximum_class_counts(self):
        for channels in (9, 16, 20):
            rendered, coverage, _ = self._empty_call(channels)
            self.assertEqual(tuple(rendered.shape), (channels, 3, 5))
            self.assertEqual(tuple(coverage.shape), (3, 5))

    def test_more_than_twenty_channels_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "1..20"):
            self._empty_call(21)


if __name__ == "__main__":
    unittest.main()
