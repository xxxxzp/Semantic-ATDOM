"""COLMAP camera calibration normalization for the pinhole rasterizer.

The Gaussian rasterizer accepts a centered pinhole projection.  Raw COLMAP
images may instead use an offset principal point and/or lens distortion.  This
module defines the single rectification contract used by RGB training and
semantic projection so both modalities remain pixel aligned.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import cv2
import numpy as np


_PARAMETER_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
}


@dataclass(frozen=True)
class RectifiedPinhole:
    source_model: str
    source_params: tuple
    width: int
    height: int
    source_fx: float
    source_fy: float
    source_cx: float
    source_cy: float
    distortion: tuple
    target_fx: float
    target_fy: float
    target_cx: float
    target_cy: float

    @property
    def requires_rectification(self) -> bool:
        values = self.distortion + (
            self.source_cx - self.target_cx,
            self.source_cy - self.target_cy,
        )
        return any(abs(float(value)) > 1e-12 for value in values)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        payload = dict(value)
        payload["source_params"] = tuple(payload["source_params"])
        payload["distortion"] = tuple(payload["distortion"])
        return cls(**payload)


def rectified_pinhole_from_colmap(
    model: str, params: Sequence[float], *, width: int, height: int
) -> RectifiedPinhole:
    model = str(model).upper()
    if model not in _PARAMETER_COUNTS:
        raise ValueError("unsupported COLMAP camera model: {}".format(model))
    values = tuple(float(value) for value in params)
    expected = _PARAMETER_COUNTS[model]
    if len(values) != expected:
        raise ValueError(
            "COLMAP {} camera requires {} parameters, got {}".format(
                model, expected, len(values)
            )
        )
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("camera dimensions must be positive")

    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = values[0]
        cx, cy = values[1:3]
    else:
        fx, fy, cx, cy = values[:4]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        distortion = (0.0, 0.0, 0.0, 0.0, 0.0)
    elif model == "SIMPLE_RADIAL":
        distortion = (values[3], 0.0, 0.0, 0.0, 0.0)
    elif model == "RADIAL":
        distortion = (values[3], values[4], 0.0, 0.0, 0.0)
    else:
        distortion = (values[4], values[5], values[6], values[7], 0.0)

    # getProjectionMatrix assumes a centered principal point.  Keeping focal
    # length and output dimensions fixed makes the conversion deterministic
    # and avoids hidden cropping or zooming.
    target_cx = float(width) * 0.5
    target_cy = float(height) * 0.5
    return RectifiedPinhole(
        source_model=model,
        source_params=values,
        width=int(width),
        height=int(height),
        source_fx=fx,
        source_fy=fy,
        source_cx=cx,
        source_cy=cy,
        distortion=distortion,
        target_fx=fx,
        target_fy=fy,
        target_cx=target_cx,
        target_cy=target_cy,
    )


@lru_cache(maxsize=32)
def _rectification_maps(calibration: RectifiedPinhole):
    source = np.array(
        [
            [calibration.source_fx, 0.0, calibration.source_cx],
            [0.0, calibration.source_fy, calibration.source_cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [calibration.target_fx, 0.0, calibration.target_cx],
            [0.0, calibration.target_fy, calibration.target_cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return cv2.initUndistortRectifyMap(
        source,
        np.asarray(calibration.distortion, dtype=np.float64),
        None,
        target,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )


def rectify_array(
    value: np.ndarray,
    calibration: RectifiedPinhole,
    *,
    interpolation: str = "linear",
    border_value=0,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:2] != (calibration.height, calibration.width):
        raise ValueError(
            "image shape {} does not match calibration {}".format(
                array.shape[:2], (calibration.height, calibration.width)
            )
        )
    if not calibration.requires_rectification:
        return array.copy()
    modes = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    if interpolation not in modes:
        raise ValueError("interpolation must be 'linear' or 'nearest'")
    map_x, map_y = _rectification_maps(calibration)
    return cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=modes[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def rectification_valid_mask(calibration: RectifiedPinhole) -> np.ndarray:
    source = np.ones((calibration.height, calibration.width), dtype=np.uint8)
    return rectify_array(
        source, calibration, interpolation="nearest", border_value=0
    ).astype(bool)


def rectify_points(points: np.ndarray, calibration: RectifiedPinhole) -> np.ndarray:
    """Map distorted source pixels into the rectified pinhole image."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if not calibration.requires_rectification:
        return value.copy()
    source = np.array(
        [
            [calibration.source_fx, 0.0, calibration.source_cx],
            [0.0, calibration.source_fy, calibration.source_cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [calibration.target_fx, 0.0, calibration.target_cx],
            [0.0, calibration.target_fy, calibration.target_cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return cv2.undistortPoints(
        value.reshape(-1, 1, 2),
        source,
        np.asarray(calibration.distortion, dtype=np.float64),
        P=target,
    ).reshape(-1, 2)
