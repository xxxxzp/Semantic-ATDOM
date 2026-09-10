import unittest

import numpy as np
import torch

from semantic_atdom.scene.cameras import DummyCamera
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

        spec = SemanticClassSpec.from_dict(
            {
                "schema_version": 1,
                "ignore_id": 255,
                "classes": [
                    {"id": 0, "name": "ground", "color": [0, 0, 0]},
                    {"id": 1, "name": "building", "color": [255, 0, 0]},
                ],
            }
        )
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
            16,
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

    def test_empty_geometry_returns_unknown_ready_zero_maps(self):
        rendered, coverage, radii = self._empty_call(4)
        self.assertEqual(tuple(rendered.shape), (4, 3, 5))
        self.assertEqual(tuple(coverage.shape), (3, 5))
        self.assertEqual(radii.numel(), 0)
        self.assertEqual(float(rendered.sum()), 0.0)

    def test_channel_boundaries_cover_current_and_maximum_class_counts(self):
        for channels in (9, 16):
            rendered, coverage, _ = self._empty_call(channels)
            self.assertEqual(tuple(rendered.shape), (channels, 3, 5))
            self.assertEqual(tuple(coverage.shape), (3, 5))

    def test_more_than_sixteen_channels_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "1..16"):
            self._empty_call(17)


if __name__ == "__main__":
    unittest.main()
