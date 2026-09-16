"""Metric adapters with explicit image layout and range contracts."""

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _validate_chw(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.ndim != 3 or first.shape[0] != 3:
        raise ValueError("metric images must have shape [3, H, W]")
    if first.shape != second.shape:
        raise ValueError("metric images must have identical shapes")


def ssim_chw(first: torch.Tensor, second: torch.Tensor) -> float:
    _validate_chw(first, second)
    first_hwc = first.detach().to("cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
    second_hwc = second.detach().to("cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
    min_dim = min(first_hwc.shape[:2])
    if min_dim < 3:
        raise ValueError("SSIM requires image height and width of at least 3")
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    return float(
        structural_similarity(
            first_hwc,
            second_hwc,
            channel_axis=-1,
            data_range=1.0,
            win_size=win_size,
        )
    )


def lpips_distance(
    first: torch.Tensor, second: torch.Tensor, lpips_model
) -> float:
    _validate_chw(first, second)
    # lpips==0.1.4 maps [0, 1] to [-1, 1] internally when normalize=True.
    result = lpips_model(
        first.unsqueeze(0), second.unsqueeze(0), normalize=True
    )
    return float(result.item())
