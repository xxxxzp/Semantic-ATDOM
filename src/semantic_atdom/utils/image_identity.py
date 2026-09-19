"""Canonical image identity handling shared by RGB and semantic inputs."""

from pathlib import Path


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def normalize_image_identity(value: object) -> str:
    """Strip known image suffixes without truncating decimal timestamps."""
    path = Path(str(value))
    result = path.stem if path.suffix.lower() in IMAGE_SUFFIXES else path.name
    if not result:
        raise ValueError("image identity must not be empty")
    return result
