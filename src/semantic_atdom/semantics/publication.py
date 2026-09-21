"""Fixed-grid and atomic paired-map publication contracts."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from semantic_atdom.semantics.provenance import SEMANTIC_METHOD_VERSION
from semantic_atdom.utils.atomic_io import write_json_atomic
from semantic_atdom.semantics.schema import SemanticClassSpec, sha256_file
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    SemanticTdomResult,
    write_semantic_tdom,
)


_PIXEL_AXIS_CONVENTION = (
    "pixel centers; columns increase along pixel_x_axis_world; "
    "rows increase along pixel_y_axis_world"
)


def _finite_tuple(values: Sequence[float], length: int, name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError("{} must contain {} finite values".format(name, length))
    return result


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_save_json_atomic = write_json_atomic


@dataclass(frozen=True)
class MapGridWindow:
    grid_id: str
    tile_min_x: int
    tile_max_x: int
    tile_min_y: int
    tile_max_y: int
    width_pixels: int
    height_pixels: int
    plane_bounds: Tuple[float, float, float, float]
    pixel_axis_convention: str = _PIXEL_AXIS_CONVENTION

    @property
    def center_plane(self) -> Tuple[float, float]:
        xmin, xmax, ymin, ymax = self.plane_bounds
        return ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)

    @property
    def width_world(self) -> float:
        return self.plane_bounds[1] - self.plane_bounds[0]

    @property
    def height_world(self) -> float:
        return self.plane_bounds[3] - self.plane_bounds[2]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": 1,
            "grid_id": self.grid_id,
            "tile_indices": {
                "min_x": self.tile_min_x,
                "max_x": self.tile_max_x,
                "min_y": self.tile_min_y,
                "max_y": self.tile_max_y,
            },
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "plane_bounds": list(self.plane_bounds),
            "pixel_axis_convention": self.pixel_axis_convention,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], grid: "MapGridSpec"
    ) -> "MapGridWindow":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("unsupported map grid window schema")
        indices = payload.get("tile_indices")
        if not isinstance(indices, Mapping):
            raise ValueError("map grid window requires tile_indices")
        values = tuple(
            indices.get(name) for name in ("min_x", "max_x", "min_y", "max_y")
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("map grid window tile indices must be integers")
        min_x, max_x, min_y, max_y = values
        if max_x < min_x or max_y < min_y:
            raise ValueError("map grid window tile range is invalid")
        tile_width_world = grid.tile_shape[0] * grid.gsd[0]
        tile_height_world = grid.tile_shape[1] * grid.gsd[1]
        expected = cls(
            grid_id=grid.grid_id,
            tile_min_x=min_x,
            tile_max_x=max_x,
            tile_min_y=min_y,
            tile_max_y=max_y,
            width_pixels=(max_x - min_x + 1) * grid.tile_shape[0],
            height_pixels=(max_y - min_y + 1) * grid.tile_shape[1],
            plane_bounds=(
                grid.origin_plane[0] + min_x * tile_width_world,
                grid.origin_plane[0] + (max_x + 1) * tile_width_world,
                grid.origin_plane[1] + min_y * tile_height_world,
                grid.origin_plane[1] + (max_y + 1) * tile_height_world,
            ),
        )
        candidate = cls(
            grid_id=str(payload.get("grid_id", "")),
            tile_min_x=min_x,
            tile_max_x=max_x,
            tile_min_y=min_y,
            tile_max_y=max_y,
            width_pixels=int(payload.get("width_pixels", 0)),
            height_pixels=int(payload.get("height_pixels", 0)),
            plane_bounds=_finite_tuple(payload.get("plane_bounds"), 4, "plane_bounds"),
            pixel_axis_convention=str(payload.get("pixel_axis_convention", "")),
        )
        if candidate != expected:
            raise ValueError("map grid window is inconsistent with its fixed grid")
        return candidate


@dataclass(frozen=True)
class MapGridSpec:
    """Task-level local grid whose identity is independent of covered extent."""

    grid_id: str
    plane_normal_world: Tuple[float, float, float]
    pixel_x_axis_world: Tuple[float, float, float]
    pixel_y_axis_world: Tuple[float, float, float]
    plane_offset_world: float
    origin_plane: Tuple[float, float]
    origin_world: Tuple[float, float, float]
    gsd: Tuple[float, float]
    tile_shape: Tuple[int, int]
    crs: Optional[str] = None
    pixel_axis_convention: str = _PIXEL_AXIS_CONVENTION

    @classmethod
    def create(
        cls,
        *,
        plane_normal_world: Sequence[float],
        pixel_x_axis_world: Sequence[float],
        pixel_y_axis_world: Sequence[float],
        plane_offset_world: float,
        first_bounds: Sequence[float],
        requested_gsd: Sequence[float],
        tile_shape: Sequence[int] = (512, 512),
        default_width_pixels: int = 4000,
        crs: Optional[str] = None,
    ) -> "MapGridSpec":
        normal = _finite_tuple(plane_normal_world, 3, "plane_normal_world")
        x_axis = _finite_tuple(pixel_x_axis_world, 3, "pixel_x_axis_world")
        y_axis = _finite_tuple(pixel_y_axis_world, 3, "pixel_y_axis_world")
        bounds = _finite_tuple(first_bounds, 4, "first_bounds")
        requested = _finite_tuple(requested_gsd, 2, "requested_gsd")
        plane_offset = float(plane_offset_world)
        if not math.isfinite(plane_offset):
            raise ValueError("plane_offset_world must be finite")
        xmin, xmax, ymin, ymax = bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("first_bounds must have positive width and height")
        tile = tuple(int(value) for value in tile_shape)
        if len(tile) != 2 or any(value <= 0 for value in tile):
            raise ValueError("tile_shape must contain two positive integers")
        if default_width_pixels <= 0:
            raise ValueError("default_width_pixels must be positive")
        if requested == (0.0, 0.0):
            default_gsd = (xmax - xmin) / float(default_width_pixels)
            gsd = (default_gsd, default_gsd)
        elif requested[0] > 0 and requested[1] > 0:
            gsd = requested
        else:
            raise ValueError("requested_gsd must be both zero or both positive")
        cls._validate_basis(normal, x_axis, y_axis)
        origin = (xmin, ymin)
        origin_world = tuple(
            origin[0] * x_axis[index]
            + origin[1] * y_axis[index]
            + plane_offset * normal[index]
            for index in range(3)
        )
        identity = {
            "schema_version": 1,
            "plane_normal_world": list(normal),
            "pixel_x_axis_world": list(x_axis),
            "pixel_y_axis_world": list(y_axis),
            "plane_offset_world": plane_offset,
            "origin_plane": list(origin),
            "origin_world": list(origin_world),
            "gsd": list(gsd),
            "tile_shape": list(tile),
            "crs": crs,
            "pixel_axis_convention": _PIXEL_AXIS_CONVENTION,
        }
        return cls(
            grid_id=_canonical_sha256(identity),
            plane_normal_world=normal,
            pixel_x_axis_world=x_axis,
            pixel_y_axis_world=y_axis,
            plane_offset_world=plane_offset,
            origin_plane=origin,
            origin_world=origin_world,
            gsd=gsd,
            tile_shape=tile,
            crs=crs,
        )

    @staticmethod
    def _validate_basis(normal, x_axis, y_axis) -> None:
        def dot(first, second):
            return sum(a * b for a, b in zip(first, second))

        for name, axis in (
            ("plane_normal_world", normal),
            ("pixel_x_axis_world", x_axis),
            ("pixel_y_axis_world", y_axis),
        ):
            if abs(dot(axis, axis) - 1.0) > 1e-4:
                raise ValueError("{} must be unit length".format(name))
        if any(
            abs(dot(first, second)) > 1e-4
            for first, second in (
                (normal, x_axis),
                (normal, y_axis),
                (x_axis, y_axis),
            )
        ):
            raise ValueError("map grid axes must be mutually orthogonal")

    def _identity_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": 1,
            "plane_normal_world": list(self.plane_normal_world),
            "pixel_x_axis_world": list(self.pixel_x_axis_world),
            "pixel_y_axis_world": list(self.pixel_y_axis_world),
            "plane_offset_world": self.plane_offset_world,
            "origin_plane": list(self.origin_plane),
            "origin_world": list(self.origin_world),
            "gsd": list(self.gsd),
            "tile_shape": list(self.tile_shape),
            "crs": self.crs,
            "pixel_axis_convention": self.pixel_axis_convention,
        }

    def to_dict(self) -> Mapping[str, object]:
        payload = dict(self._identity_dict())
        payload["grid_id"] = self.grid_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MapGridSpec":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("unsupported map grid schema")
        result = cls(
            grid_id=str(payload.get("grid_id", "")),
            plane_normal_world=_finite_tuple(
                payload["plane_normal_world"], 3, "plane_normal_world"
            ),
            pixel_x_axis_world=_finite_tuple(
                payload["pixel_x_axis_world"], 3, "pixel_x_axis_world"
            ),
            pixel_y_axis_world=_finite_tuple(
                payload["pixel_y_axis_world"], 3, "pixel_y_axis_world"
            ),
            plane_offset_world=float(payload["plane_offset_world"]),
            origin_plane=_finite_tuple(payload["origin_plane"], 2, "origin_plane"),
            origin_world=_finite_tuple(payload["origin_world"], 3, "origin_world"),
            gsd=_finite_tuple(payload["gsd"], 2, "gsd"),
            tile_shape=tuple(int(value) for value in payload["tile_shape"]),
            crs=payload.get("crs"),
            pixel_axis_convention=str(payload.get("pixel_axis_convention", "")),
        )
        if not math.isfinite(result.plane_offset_world):
            raise ValueError("plane_offset_world must be finite")
        cls._validate_basis(
            result.plane_normal_world,
            result.pixel_x_axis_world,
            result.pixel_y_axis_world,
        )
        if len(result.tile_shape) != 2 or any(
            value <= 0 for value in result.gsd + result.tile_shape
        ):
            raise ValueError("map grid GSD and tile shape must be positive")
        if result.pixel_axis_convention != _PIXEL_AXIS_CONVENTION:
            raise ValueError("unsupported pixel_axis_convention")
        expected_origin_world = tuple(
            result.origin_plane[0] * result.pixel_x_axis_world[index]
            + result.origin_plane[1] * result.pixel_y_axis_world[index]
            + result.plane_offset_world * result.plane_normal_world[index]
            for index in range(3)
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8)
            for actual, expected in zip(result.origin_world, expected_origin_world)
        ):
            raise ValueError("origin_world is inconsistent with the map plane")
        if _canonical_sha256(result._identity_dict()) != result.grid_id:
            raise ValueError("map grid grid_id does not match its content")
        return result

    @classmethod
    def load(cls, path: Path) -> "MapGridSpec":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.is_file():
            existing = self.load(path)
            if existing != self:
                raise ValueError("refusing to replace an existing map grid")
            return
        _save_json_atomic(path, self.to_dict())

    def window_for_bounds(self, bounds: Sequence[float]) -> MapGridWindow:
        xmin, xmax, ymin, ymax = _finite_tuple(bounds, 4, "bounds")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("bounds must have positive width and height")
        tile_width_world = self.tile_shape[0] * self.gsd[0]
        tile_height_world = self.tile_shape[1] * self.gsd[1]
        min_x = math.floor((xmin - self.origin_plane[0]) / tile_width_world)
        max_x = math.ceil((xmax - self.origin_plane[0]) / tile_width_world) - 1
        min_y = math.floor((ymin - self.origin_plane[1]) / tile_height_world)
        max_y = math.ceil((ymax - self.origin_plane[1]) / tile_height_world) - 1
        max_x = max(min_x, max_x)
        max_y = max(min_y, max_y)
        plane_bounds = (
            self.origin_plane[0] + min_x * tile_width_world,
            self.origin_plane[0] + (max_x + 1) * tile_width_world,
            self.origin_plane[1] + min_y * tile_height_world,
            self.origin_plane[1] + (max_y + 1) * tile_height_world,
        )
        return MapGridWindow(
            grid_id=self.grid_id,
            tile_min_x=min_x,
            tile_max_x=max_x,
            tile_min_y=min_y,
            tile_max_y=max_y,
            width_pixels=(max_x - min_x + 1) * self.tile_shape[0],
            height_pixels=(max_y - min_y + 1) * self.tile_shape[1],
            plane_bounds=plane_bounds,
        )


@dataclass(frozen=True)
class MapPublicationPolicy:
    """Independent map/checkpoint cadence and bounded overload behavior."""

    publication_stage_interval: int = 1
    checkpoint_stage_interval: int = 0
    min_free_cuda_bytes: int = 512 * 1024**2
    max_pair_output_bytes: int = 8 * 1024**3
    online_time_budget_seconds: float = 30.0
    cuda_stream_priority: int = 0
    queue_policy: str = "latest-wins"
    overload_policy: str = "degrade-diagnostics-then-skip"

    def __post_init__(self) -> None:
        if self.publication_stage_interval < 1:
            raise ValueError("publication_stage_interval must be positive")
        if self.checkpoint_stage_interval < 0:
            raise ValueError("checkpoint_stage_interval must be non-negative")
        if self.min_free_cuda_bytes < 0 or self.max_pair_output_bytes < 1:
            raise ValueError("map publication byte budgets are invalid")
        if self.online_time_budget_seconds < 0:
            raise ValueError("online_time_budget_seconds must be non-negative")
        if self.queue_policy != "latest-wins":
            raise ValueError("only latest-wins map publication is supported")
        if self.overload_policy != "degrade-diagnostics-then-skip":
            raise ValueError("unsupported map overload policy")

    def should_publish(self, geometry_stage: int, final: bool) -> bool:
        return bool(final) or int(geometry_stage) % self.publication_stage_interval == 0

    def should_checkpoint(
        self,
        geometry_stage: int,
        final: bool,
        exists: bool,
    ) -> bool:
        del exists
        return bool(final) or (
            self.checkpoint_stage_interval > 0
            and int(geometry_stage) % self.checkpoint_stage_interval == 0
        )

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": 1,
            "publication_stage_interval": self.publication_stage_interval,
            "checkpoint_stage_interval": self.checkpoint_stage_interval,
            "min_free_cuda_bytes": self.min_free_cuda_bytes,
            "max_pair_output_bytes": self.max_pair_output_bytes,
            "online_time_budget_seconds": self.online_time_budget_seconds,
            "cuda_stream_priority": self.cuda_stream_priority,
            "queue_policy": self.queue_policy,
            "overload_policy": self.overload_policy,
        }


def _validate_shared_manifest(shared: Mapping[str, object]) -> None:
    required = (
        "product_key",
        "geometry_stage",
        "projection_geometry_hash",
        "rgb_model_hash",
        "semantic_state_hash",
        "semantic_status",
        "alignment",
    )
    missing = [name for name in required if name not in shared]
    if missing:
        raise ValueError("paired map manifest is missing {}".format(", ".join(missing)))
    product_key = shared["product_key"]
    if not isinstance(product_key, str) or Path(product_key).name != product_key:
        raise ValueError("paired map product_key must be one safe path component")
    for name in (
        "projection_geometry_hash",
        "rgb_model_hash",
        "semantic_state_hash",
    ):
        value = shared[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("paired map {} must be a SHA-256 digest".format(name))
    alignment = shared["alignment"]
    if not isinstance(alignment, Mapping):
        raise ValueError("paired map alignment must be an object")
    grid = alignment.get("grid")
    window = alignment.get("window")
    if not isinstance(grid, Mapping) or not isinstance(window, Mapping):
        raise ValueError("paired map alignment requires grid and window")
    grid_spec = MapGridSpec.from_dict(grid)
    grid_window = MapGridWindow.from_dict(window, grid_spec)
    if grid_spec.grid_id != grid_window.grid_id:
        raise ValueError("paired map grid/window identity mismatch")
    for alignment_name, expected in (
        ("image_width", grid_window.width_pixels),
        ("image_height", grid_window.height_pixels),
    ):
        value = alignment.get(alignment_name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                "paired map {} does not match its grid window".format(
                    alignment_name
                )
            )


def publish_tdom_pair(
    *,
    output_root: Path,
    rgb_image: np.ndarray,
    semantic_result: SemanticTdomResult,
    class_spec: SemanticClassSpec,
    semantic_config: SemanticTdomConfig,
    shared_manifest: Mapping[str, object],
) -> Path:
    """Publish RGB and Semantic TDOM as one directory-level transaction."""
    _validate_shared_manifest(shared_manifest)
    shared_manifest = dict(shared_manifest)
    alignment_sha256 = _canonical_sha256(shared_manifest["alignment"])
    supplied_alignment_hash = shared_manifest.get("alignment_sha256")
    if (
        supplied_alignment_hash is not None
        and supplied_alignment_hash != alignment_sha256
    ):
        raise ValueError("paired map alignment hash does not match")
    shared_manifest["alignment_sha256"] = alignment_sha256
    rgb = np.asarray(rgb_image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_image must be uint8 [H, W, 3]")
    semantic_shape = tuple(int(value) for value in semantic_result.class_ids.shape)
    if rgb.shape[:2] != semantic_shape:
        raise ValueError("RGB and Semantic TDOM dimensions do not match")
    alignment = shared_manifest["alignment"]
    if (
        int(alignment["image_width"]) != int(rgb.shape[1])
        or int(alignment["image_height"]) != int(rgb.shape[0])
    ):
        raise ValueError("paired TDOM dimensions do not match map alignment")

    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(output_root.name + ".publishing")
    backup = output_root.with_name(output_root.name + ".previous")
    if backup.exists() and not output_root.exists():
        os.replace(str(backup), str(output_root))
    if staging.exists():
        shutil.rmtree(str(staging))
    if backup.exists():
        shutil.rmtree(str(backup))
    staging.mkdir()
    try:
        rgb_root = staging / "rgb"
        rgb_root.mkdir()
        rgb_path = rgb_root / "tdom.png"
        temporary_rgb = rgb_path.with_name(rgb_path.name + ".tmp")
        Image.fromarray(rgb, mode="RGB").save(temporary_rgb, format="PNG")
        with temporary_rgb.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(str(temporary_rgb), str(rgb_path))
        rgb_payload = dict(shared_manifest)
        rgb_payload.update(
            {
                "schema_version": 1,
                "product_type": "rgb_tdom",
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "files": {"tdom.png": sha256_file(rgb_path)},
                "file_sizes_bytes": {"tdom.png": int(rgb_path.stat().st_size)},
            }
        )
        rgb_manifest = rgb_root / "manifest.json"
        _save_json_atomic(rgb_manifest, rgb_payload)

        semantic_manifest = write_semantic_tdom(
            semantic_result,
            staging / "semantic",
            class_spec,
            dict(shared_manifest, product_type="semantic_tdom"),
            config=semantic_config,
        )
        with semantic_manifest.open("r", encoding="utf-8") as stream:
            semantic_payload = json.load(stream)
        product_files = {
            "rgb/{}".format(name): digest
            for name, digest in rgb_payload["files"].items()
        }
        product_files.update(
            {
                "semantic/{}".format(name): digest
                for name, digest in semantic_payload["files"].items()
            }
        )
        publication_policy = shared_manifest.get("publication_policy", {})
        maximum_bytes = (
            publication_policy.get("max_pair_output_bytes")
            if isinstance(publication_policy, Mapping)
            else None
        )
        actual_bytes = sum(
            path.stat().st_size
            for path in staging.rglob("*")
            if path.is_file()
        )
        if maximum_bytes is not None and actual_bytes > int(maximum_bytes):
            raise ValueError(
                "paired TDOM output requires {} bytes, budget is {}".format(
                    actual_bytes, maximum_bytes
                )
            )
        pair_payload = dict(shared_manifest)
        pair_payload.update(
            {
                "schema_version": 1,
                "semantic_method_version": SEMANTIC_METHOD_VERSION,
                "publication_status": "complete",
                "grid_id": shared_manifest["alignment"]["grid"]["grid_id"],
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "product_files": product_files,
                "product_files_bytes": int(actual_bytes),
                "rgb_manifest": {
                    "path": "rgb/manifest.json",
                    "sha256": sha256_file(rgb_manifest),
                },
                "semantic_manifest": {
                    "path": "semantic/manifest.json",
                    "sha256": sha256_file(semantic_manifest),
                },
            }
        )
        pair_manifest = staging / "pair_manifest.json"
        _save_json_atomic(pair_manifest, pair_payload)
        if output_root.exists():
            os.replace(str(output_root), str(backup))
        try:
            os.replace(str(staging), str(output_root))
        except Exception:
            if backup.exists() and not output_root.exists():
                os.replace(str(backup), str(output_root))
            raise
        published_manifest = output_root / "pair_manifest.json"
        latest_root = output_root.parent
        if latest_root.name == "releases":
            latest_root = latest_root.parent
        manifest_reference = published_manifest.relative_to(latest_root).as_posix()
        try:
            _save_json_atomic(
                latest_root / "latest.json",
                {
                    "schema_version": 1,
                    "product_key": shared_manifest["product_key"],
                    "pair_manifest": manifest_reference,
                    "pair_manifest_sha256": sha256_file(published_manifest),
                },
            )
        except Exception:
            if output_root.exists():
                shutil.rmtree(str(output_root))
            if backup.exists():
                os.replace(str(backup), str(output_root))
            raise
        if backup.exists():
            shutil.rmtree(str(backup))
        return published_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise
