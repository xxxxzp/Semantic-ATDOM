"""Validated acquisition-order and overlap inputs for semantic processing."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import io
from pathlib import Path
import re
from typing import Tuple

import numpy as np

from semantic_atdom.semantics.schema import (
    SemanticSchemaError,
    normalize_image_id,
)


def image_id(value: object) -> str:
    return normalize_image_id(value)


def acquisition_key(value: str):
    value = image_id(value)
    try:
        return (0, Decimal(value), value)
    except InvalidOperation:
        match = re.search(r"(\d+)$", value)
        if match is None:
            return (2, value)
        return (1, value[: match.start()], int(match.group(1)), value)


_INLINE_RECORD_SEPARATOR = re.compile(r"0000[1-9A-Fa-f][0-9A-Fa-f]{11}")


def _read_exported_text(path: Path) -> str:
    """Read vendor text exports that may contain an inline record marker."""

    content = path.read_text(encoding="utf-8")
    content_without_trailing_whitespace = content.rstrip()
    marker_length = 16
    if len(content_without_trailing_whitespace) < marker_length:
        return content
    marker = content_without_trailing_whitespace[-marker_length:]
    if _INLINE_RECORD_SEPARATOR.fullmatch(marker) is None:
        return content
    return content.replace(marker, "\n")


@dataclass(frozen=True)
class AcquisitionOverlap:
    """COLMAP match counts addressed by image id, never by time-order row."""

    image_ids: Tuple[str, ...]
    matrix: np.ndarray

    @staticmethod
    def input_paths(snapshot_root: Path) -> Tuple[Path, Path]:
        snapshot_root = Path(snapshot_root)
        legacy_root = snapshot_root / "sparse" / "0"
        candidates = (
            (
                legacy_root / "imagesNames.txt",
                legacy_root / "imageMatchMatrix.txt",
            ),
            (
                snapshot_root / "match_matrix_image_name.txt",
                snapshot_root / "match_matrix_value.txt",
            ),
        )
        for names_path, matrix_path in candidates:
            if names_path.is_file() and matrix_path.is_file():
                return names_path, matrix_path
        raise SemanticSchemaError(
            "stage is missing overlap inputs; expected either {} and {}, or {} and {}".format(
                *[str(path) for pair in candidates for path in pair]
            )
        )

    @classmethod
    def load(cls, snapshot_root: Path) -> "AcquisitionOverlap":
        try:
            names_path, matrix_path = cls.input_paths(snapshot_root)
            raw_names = _read_exported_text(names_path)
            names = tuple(
                image_id(item)
                for item in re.split(r"[,\s]+", raw_names.strip())
                if item
            )
            matrix = np.loadtxt(
                io.StringIO(_read_exported_text(matrix_path)),
                delimiter=",",
                dtype=np.float64,
            )
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to load COLMAP overlap inputs from {}: {}".format(
                    snapshot_root, error
                )
            )
        if len(names) == 0 or len(names) != len(set(names)):
            raise SemanticSchemaError(
                "overlap names must contain unique image ids"
            )
        if matrix.ndim == 0 and len(names) == 1:
            matrix = matrix.reshape(1, 1)
        elif matrix.ndim == 1 and len(names) == 1:
            matrix = matrix.reshape(1, 1)
        if matrix.shape != (len(names), len(names)):
            raise SemanticSchemaError(
                "overlap matrix shape {} does not match {} names".format(
                    matrix.shape, len(names)
                )
            )
        if not bool(np.isfinite(matrix).all()) or bool((matrix < 0).any()):
            raise SemanticSchemaError(
                "overlap matrix must contain finite non-negative counts"
            )
        return cls(names, matrix)

    @property
    def acquisition_order(self) -> Tuple[str, ...]:
        return tuple(sorted(self.image_ids, key=acquisition_key))

    def normalized(self, first: str, second: str) -> float:
        positions = {name: index for index, name in enumerate(self.image_ids)}
        first = image_id(first)
        second = image_id(second)
        try:
            i = positions[first]
            j = positions[second]
        except KeyError as error:
            raise SemanticSchemaError(
                "image {} is absent from overlap names".format(error.args[0])
            )
        denominator = min(float(self.matrix[i, i]), float(self.matrix[j, j]))
        if denominator <= 0:
            return 0.0
        return min(1.0, max(0.0, float(self.matrix[i, j]) / denominator))
