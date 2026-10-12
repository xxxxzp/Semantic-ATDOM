"""Pixel-center mapping between a rectified source grid and a training grid.

COLMAP coordinates measure from the upper-left image edge (center = i + .5).
Array coordinates measure from the first pixel center (center = i).  Resizing
maps array centers as (i + .5) * source_size / target_size - .5.  The nearest
source index is therefore floor((i + .5) * source_size / target_size).
"""

import numpy as np


PIXEL_GRID_VERSION = "colmap-center-nearest-v1"


def nearest_indices(source_size, target_size):
    source_size, target_size = int(source_size), int(target_size)
    if source_size < 1 or target_size < 1:
        raise ValueError("pixel grid dimensions must be positive")
    # Integer arithmetic also makes exact half-way ties deterministic.
    return ((2 * np.arange(target_size, dtype=np.int64) + 1) * source_size) // (
        2 * target_size
    )


def resize_nearest(value, resolution):
    """Sample an HxW[xC] array without converting or mixing its values."""
    array = np.asarray(value)
    width, height = map(int, resolution)
    if array.ndim < 2:
        raise ValueError("pixel array must have at least two dimensions")
    rows = nearest_indices(array.shape[0], height)
    columns = nearest_indices(array.shape[1], width)
    return array[rows[:, None], columns[None, :]].copy()


def camera_resolution(width, height, resolution, resolution_scale=1.0):
    """Resolve actual rounded dimensions once, for initial and added cameras."""
    if resolution in (1, 2, 4, 8):
        scale = float(resolution) * float(resolution_scale)
        target = (round(width / scale), round(height / scale))
    else:
        down = max(1.0, width / 1600.0) if resolution == -1 else width / float(resolution)
        scale = down * float(resolution_scale)
        target = (int(width / scale), int(height / scale))
    if min(target) < 1:
        raise ValueError("resolution produces an empty camera grid")
    return target
