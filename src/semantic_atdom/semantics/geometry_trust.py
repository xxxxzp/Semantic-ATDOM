"""One detached geometry eligibility rule for maps and planar fallback."""
from typing import Dict, Tuple
import torch


def trusted_geometry_mask(
    scales: torch.Tensor, max_scale_ratio: float
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Exclude detached scale outliers from Semantic TDOM support.

    Progressive geometry can contain a small number of extremely large
    Gaussians whose dark SH appearance is nearly invisible in RGB while their
    opacity spreads one semantic class across the map border.  This mask only
    bounds the read-only semantic render; it never mutates geometry or feeds
    back into training.
    """
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("Gaussian scales must have shape [N, 3]")
    maximum_scales = scales.detach().max(dim=1).values
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("Gaussian scales must be finite")
    median_scale = (
        float(maximum_scales.median().item())
        if maximum_scales.numel()
        else 0.0
    )
    enabled = max_scale_ratio > 0 and maximum_scales.numel() > 0
    scale_limit = median_scale * max_scale_ratio if enabled else None
    trusted = torch.ones_like(maximum_scales, dtype=torch.bool)
    if enabled:
        trusted &= maximum_scales <= scale_limit
    input_count = int(maximum_scales.numel())
    trusted_count = int(trusted.sum().item())
    return trusted, {
        "method": "max_world_scale_over_scene_median",
        "enabled": bool(enabled),
        "max_gaussian_scale_ratio": float(max_scale_ratio),
        "median_max_world_scale": median_scale,
        "max_trusted_world_scale": (
            None if scale_limit is None else float(scale_limit)
        ),
        "input_gaussian_count": input_count,
        "trusted_gaussian_count": trusted_count,
        "excluded_gaussian_count": input_count - trusted_count,
    }


