"""Forward-only CUDA accumulation of semantic observations over Gaussian footprints."""

from typing import NamedTuple, Optional

import torch


_MAX_DENSE_ELEMENTS = 16 * 1024 * 1024


class CudaObservationStats(NamedTuple):
    evidence: torch.Tensor
    all_mass: torch.Tensor
    valid_mass: torch.Tensor
    distribution: torch.Tensor
    support: torch.Tensor
    radii: torch.Tensor
    dense_contributions: Optional[torch.Tensor]


def _require_cuda_float32(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if not value.is_cuda:
        raise ValueError("{} must be a CUDA tensor".format(name))
    if value.dtype != torch.float32:
        raise ValueError("{} must use float32".format(name))
    return value.contiguous()


@torch.no_grad()
def rasterize_semantic_observation(
    means3d: torch.Tensor,
    opacities: torch.Tensor,
    labels: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    campos: torch.Tensor,
    num_classes: int,
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    cov3d_precomp: Optional[torch.Tensor] = None,
    boundary_weights: Optional[torch.Tensor] = None,
    scale_modifier: float = 1.0,
    ignore_id: int = 255,
    return_dense_contributions: bool = False,
    dense_element_limit: int = 16 * 1024 * 1024,
    eps: float = 1e-8,
    debug: bool = False,
) -> CudaObservationStats:
    """Accumulate per-Gaussian alpha-T evidence without an autograd graph.

    The production path returns only O(N*K) statistics. Dense pixel-by-Gaussian
    contributions are guarded and intended exclusively for numerical tests.
    """
    from diff_gaussian_rasterization import _C

    if not hasattr(_C, "semantic_observation"):
        raise RuntimeError(
            "diff_gaussian_rasterization must be rebuilt with the Semantic "
            "A-TDOM observation kernel"
        )
    means3d = _require_cuda_float32("means3d", means3d)
    opacities = _require_cuda_float32("opacities", opacities)
    viewmatrix = _require_cuda_float32("viewmatrix", viewmatrix)
    projmatrix = _require_cuda_float32("projmatrix", projmatrix)
    campos = _require_cuda_float32("campos", campos)
    if means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError("means3d must have shape [N, 3]")
    gaussian_count = means3d.shape[0]
    if opacities.numel() != gaussian_count:
        raise ValueError("opacities must contain one value per Gaussian")
    if viewmatrix.numel() != 16 or projmatrix.numel() != 16:
        raise ValueError("viewmatrix and projmatrix must contain 16 values")
    if campos.numel() != 3:
        raise ValueError("campos must contain three values")
    if any(
        value.device != means3d.device
        for value in (opacities, viewmatrix, projmatrix, campos)
    ):
        raise ValueError("all geometry and camera tensors must share one device")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if ignore_id != 255:
        raise ValueError("Semantic A-TDOM V1 reserves ignore_id 255")
    if tan_fovx <= 0 or tan_fovy <= 0:
        raise ValueError("field-of-view tangents must be positive")
    if scale_modifier <= 0:
        raise ValueError("scale_modifier must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    using_covariance = cov3d_precomp is not None
    if using_covariance and (scales is not None or rotations is not None):
        raise ValueError(
            "provide either cov3d_precomp or both scales and rotations"
        )
    if not using_covariance and (scales is None or rotations is None):
        raise ValueError(
            "provide either cov3d_precomp or both scales and rotations"
        )
    empty = torch.empty(0, dtype=torch.float32, device=means3d.device)
    if using_covariance:
        covariance = _require_cuda_float32("cov3d_precomp", cov3d_precomp)
        if covariance.shape != (gaussian_count, 6):
            raise ValueError("cov3d_precomp must have shape [N, 6]")
        scales_value = empty
        rotations_value = empty
    else:
        scales_value = _require_cuda_float32("scales", scales)
        rotations_value = _require_cuda_float32("rotations", rotations)
        if scales_value.shape != (gaussian_count, 3):
            raise ValueError("scales must have shape [N, 3]")
        if rotations_value.shape != (gaussian_count, 4):
            raise ValueError("rotations must have shape [N, 4]")
        covariance = empty
    if any(
        value.device != means3d.device
        for value in (scales_value, rotations_value, covariance)
    ):
        raise ValueError("all Gaussian tensors must share one device")

    labels = labels.detach()
    integer_types = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if labels.ndim != 2 or labels.dtype not in integer_types:
        raise ValueError("labels must be a two-dimensional integer tensor")
    if labels.shape[0] < 1 or labels.shape[1] < 1:
        raise ValueError("labels must have non-empty height and width")
    labels = labels.to(device=means3d.device, dtype=torch.long).contiguous()
    valid = labels != ignore_id
    invalid = valid & ((labels < 0) | (labels >= num_classes))
    if bool(invalid.any()):
        raise ValueError("labels contain class IDs outside [0, num_classes)")

    if boundary_weights is None:
        boundary_weights = torch.ones(
            labels.shape, dtype=torch.float32, device=means3d.device
        )
    else:
        boundary_weights = boundary_weights.detach().to(
            device=means3d.device, dtype=torch.float32
        )
        if boundary_weights.shape != labels.shape:
            raise ValueError("boundary_weights must match labels")
        if not bool(torch.isfinite(boundary_weights).all()):
            raise ValueError("boundary_weights must be finite")
        if bool((boundary_weights < 0).any()) or bool(
            (boundary_weights > 1).any()
        ):
            raise ValueError("boundary_weights must be in [0, 1]")
        boundary_weights = boundary_weights.contiguous()

    if not isinstance(dense_element_limit, int) or dense_element_limit < 0:
        raise ValueError("dense_element_limit must be a non-negative integer")
    dense_elements = labels.numel() * gaussian_count
    effective_dense_limit = min(dense_element_limit, _MAX_DENSE_ELEMENTS)
    if return_dense_contributions and dense_elements > effective_dense_limit:
        raise ValueError(
            "dense contribution output is test-only and exceeds its element limit"
        )

    with torch.cuda.device(means3d.device):
        evidence, all_mass, valid_mass, radii, dense = _C.semantic_observation(
            means3d,
            opacities,
            scales_value,
            rotations_value,
            float(scale_modifier),
            covariance,
            labels,
            boundary_weights,
            viewmatrix,
            projmatrix,
            float(tan_fovx),
            float(tan_fovy),
            campos,
            int(num_classes),
            int(ignore_id),
            bool(return_dense_contributions),
            bool(debug),
        )
    evidence_mass = evidence.sum(dim=1)
    distribution = evidence / (evidence_mass.unsqueeze(1) + eps)
    support = (
        (1.0 - torch.exp(-all_mass)) * valid_mass / (all_mass + eps)
    ).clamp_(min=0.0, max=1.0)
    return CudaObservationStats(
        evidence=evidence,
        all_mass=all_mass,
        valid_mass=valid_mass,
        distribution=distribution,
        support=support,
        radii=radii,
        dense_contributions=(dense if return_dense_contributions else None),
    )
