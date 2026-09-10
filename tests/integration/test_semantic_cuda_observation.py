import math
import unittest

import torch

from semantic_atdom.semantics.cuda_observation import (
    rasterize_semantic_observation,
)
from semantic_atdom.semantics.reference import footprint_observation
from semantic_atdom.utils.graphics_utils import getProjectionMatrix


try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        _C,
    )

    HAS_SEMANTIC_CUDA = torch.cuda.is_available() and hasattr(
        _C, "semantic_observation"
    )
except (ImportError, OSError):
    HAS_SEMANTIC_CUDA = False


@unittest.skipUnless(
    HAS_SEMANTIC_CUDA,
    "requires CUDA and the rebuilt semantic rasterizer extension",
)
class SemanticCudaObservationTests(unittest.TestCase):
    height = 16
    width = 16
    num_classes = 2

    def setUp(self):
        self.device = torch.device("cuda")
        fov = math.pi / 2.0
        self.viewmatrix = torch.eye(
            4, dtype=torch.float32, device=self.device
        )
        self.projmatrix = (
            getProjectionMatrix(0.01, 100.0, fov, fov)
            .transpose(0, 1)
            .contiguous()
            .to(self.device)
        )
        self.campos = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.means3d = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.opacities = torch.tensor(
            [[0.5], [0.5]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.scales = torch.tensor(
            [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.labels = torch.zeros(
            (self.height, self.width), dtype=torch.int64
        )
        self.labels[:, self.width // 2 :] = 1
        self.labels[self.height // 2, :] = 255
        self.boundary = torch.ones(
            (self.height, self.width), dtype=torch.float32
        )
        self.boundary[:, self.width // 2 - 1 : self.width // 2 + 1] = 0.25

    def _observe(self, **overrides):
        arguments = dict(
            means3d=self.means3d,
            opacities=self.opacities,
            labels=self.labels,
            viewmatrix=self.viewmatrix,
            projmatrix=self.projmatrix,
            tan_fovx=1.0,
            tan_fovy=1.0,
            campos=self.campos,
            num_classes=self.num_classes,
            scales=self.scales,
            rotations=self.rotations,
            boundary_weights=self.boundary,
            return_dense_contributions=True,
            debug=True,
        )
        arguments.update(overrides)
        return rasterize_semantic_observation(**arguments)

    def test_cuda_accumulation_matches_wp0_reference(self):
        result = self._observe()
        reference = footprint_observation(
            result.dense_contributions,
            self.labels.flatten(),
            num_classes=self.num_classes,
            boundary_weights=self.boundary.flatten(),
        )

        for field in (
            "evidence",
            "all_mass",
            "valid_mass",
            "distribution",
            "support",
        ):
            torch.testing.assert_close(
                getattr(result, field),
                getattr(reference, field),
                rtol=1e-4,
                atol=1e-5,
            )
        self.assertTrue(bool((result.radii > 0).all()))
        self.assertTrue(bool((result.evidence > 0).all()))
        self.assertTrue(bool((result.support >= 0).all()))
        self.assertTrue(bool((result.support <= 1).all()))
        self.assertGreater(
            float(result.all_mass[0]), float(result.all_mass[1])
        )

    def test_alpha_t_contributions_match_existing_rgb_rasterizer(self):
        result = self._observe()
        settings = GaussianRasterizationSettings(
            image_height=self.height,
            image_width=self.width,
            tanfovx=1.0,
            tanfovy=1.0,
            bg=torch.zeros(3, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=self.viewmatrix,
            projmatrix=self.projmatrix,
            sh_degree=0,
            campos=self.campos,
            prefiltered=False,
            get_normal=False,
            debug=True,
            Render_Mask=torch.ones(
                (self.height, self.width),
                dtype=torch.bool,
                device=self.device,
            ),
            Use_Render_Mask=False,
            TDOM=False,
        )
        rasterizer = GaussianRasterizer(settings)
        color, _, _, _, _, _ = rasterizer(
            means3D=self.means3d.detach(),
            means2D=torch.zeros_like(self.means3d),
            opacities=self.opacities.detach(),
            colors_precomp=torch.ones(
                (2, 3), dtype=torch.float32, device=self.device
            ),
            scales=self.scales.detach(),
            rotations=self.rotations.detach(),
            all_map=torch.zeros(
                (2, 3), dtype=torch.float32, device=self.device
            ),
        )

        semantic_coverage = result.dense_contributions.sum(dim=1)
        torch.testing.assert_close(
            semantic_coverage,
            color[0].flatten(),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(bool((semantic_coverage <= 1.0).all()))

    def test_forward_only_path_does_not_attach_geometry_gradients(self):
        result = self._observe()

        for output in result[:-1]:
            self.assertFalse(output.requires_grad)
        self.assertFalse(result.dense_contributions.requires_grad)
        for geometry in (
            self.means3d,
            self.opacities,
            self.scales,
            self.rotations,
        ):
            self.assertIsNone(geometry.grad)

    def test_covariance_and_scale_rotation_paths_agree(self):
        covariance = torch.tensor(
            [
                [0.04, 0.0, 0.0, 0.04, 0.0, 0.04],
                [0.04, 0.0, 0.0, 0.04, 0.0, 0.04],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        scale_result = self._observe()
        covariance_result = self._observe(
            scales=None,
            rotations=None,
            cov3d_precomp=covariance,
        )

        torch.testing.assert_close(
            covariance_result.dense_contributions,
            scale_result.dense_contributions,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            covariance_result.evidence,
            scale_result.evidence,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_production_path_omits_dense_pixel_gaussian_matrix(self):
        result = self._observe(return_dense_contributions=False)

        self.assertIsNone(result.dense_contributions)
        self.assertEqual(result.evidence.shape, (2, self.num_classes))
        self.assertEqual(result.all_mass.shape, (2,))

    def test_dense_test_output_is_size_guarded(self):
        with self.assertRaisesRegex(ValueError, "test-only"):
            self._observe(dense_element_limit=1)

    def test_alpha_t_early_termination_matches_rgb_rule(self):
        gaussian_count = 4
        result = self._observe(
            means3d=torch.tensor(
                [[0.0, 0.0, 1.0]] * gaussian_count,
                dtype=torch.float32,
                device=self.device,
            ),
            opacities=torch.ones(
                (gaussian_count, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            scales=torch.ones(
                (gaussian_count, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            rotations=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * gaussian_count,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        center_pixel = (self.height // 2) * self.width + self.width // 2
        center_contributions = result.dense_contributions[center_pixel]

        self.assertEqual(int((center_contributions > 0).sum()), 1)
        torch.testing.assert_close(
            center_contributions.sum(),
            torch.tensor(0.99, device=self.device),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_empty_gaussian_set_returns_aligned_empty_statistics(self):
        result = self._observe(
            means3d=torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            ),
            opacities=torch.empty(
                (0, 1), dtype=torch.float32, device=self.device
            ),
            scales=torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            ),
            rotations=torch.empty(
                (0, 4), dtype=torch.float32, device=self.device
            ),
        )

        self.assertEqual(result.evidence.shape, (0, self.num_classes))
        self.assertEqual(result.all_mass.shape, (0,))
        self.assertEqual(
            result.dense_contributions.shape,
            (self.height * self.width, 0),
        )

        covariance_result = self._observe(
            means3d=torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            ),
            opacities=torch.empty(
                (0, 1), dtype=torch.float32, device=self.device
            ),
            scales=None,
            rotations=None,
            cov3d_precomp=torch.empty(
                (0, 6), dtype=torch.float32, device=self.device
            ),
        )
        self.assertEqual(
            covariance_result.evidence.shape, (0, self.num_classes)
        )


if __name__ == "__main__":
    unittest.main()
