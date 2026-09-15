"""Stage-synchronous online semantic fusion for progressive A-TDOM."""

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from semantic_atdom.semantics.cuda_observation import (
    rasterize_semantic_observation,
)
from semantic_atdom.semantics.reference import aggregate_observation_group
from semantic_atdom.semantics.rebuild import FinalSemanticRebuilder
from semantic_atdom.semantics.schema import (
    LedgerEntry,
    SemanticClassSpec,
    SemanticDataset,
    SemanticLedger,
    SemanticObservation,
    SemanticSchemaError,
    normalize_image_id,
    sha256_file,
)
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.provenance import (
    RGB_MODEL_ATTRIBUTE_NAMES,
    SEMANTIC_PROJECTION_ATTRIBUTE_NAMES,
    SEMANTIC_METHOD_VERSION,
    rgb_model_sha256,
    semantic_projection_sha256,
    semantic_state_sha256,
)
from semantic_atdom.semantics.publication import (
    MapPublicationPolicy,
    publish_tdom_pair,
)
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    rasterize_rgb_tdom_striped,
    rasterize_semantic_tdom,
)
from semantic_atdom.utils.image_utils import tdom_to_uint8


def _image_id(value: object) -> str:
    return normalize_image_id(value)


def _acquisition_key(image_id: str):
    try:
        return (0, Decimal(image_id), image_id)
    except InvalidOperation:
        match = re.search(r"(\d+)$", image_id)
        if match is None:
            return (2, image_id)
        return (1, image_id[: match.start()], int(match.group(1)), image_id)


_INLINE_RECORD_SEPARATOR = re.compile(r"0000[1-9A-Fa-f][0-9A-Fa-f]{11}")


def _read_exported_text(path: Path) -> str:
    """Read vendor text exports that may contain an inline record marker."""
    content = path.read_text(encoding="utf-8")
    return _INLINE_RECORD_SEPARATOR.sub("\n", content)


_GEOMETRY_ATTRIBUTE_NAMES = RGB_MODEL_ATTRIBUTE_NAMES


def _model_attributes(
    gaussians, names: Sequence[str]
) -> Dict[str, torch.Tensor]:
    cached = getattr(gaussians, "geometry_attributes_cpu", None)
    if callable(cached):
        values = cached()
        return {name: values[name] for name in names}
    if hasattr(gaussians, "_values"):
        source = gaussians._values
        return {
            name: source[name].detach().cpu().contiguous()
            for name in names
        }
    return {
        name: getattr(gaussians, "_" + name).detach().cpu().contiguous()
        for name in names
    }


def _projection_attributes(gaussians) -> Dict[str, torch.Tensor]:
    return _model_attributes(gaussians, SEMANTIC_PROJECTION_ATTRIBUTE_NAMES)


def _rgb_attributes(gaussians) -> Dict[str, torch.Tensor]:
    return _model_attributes(gaussians, RGB_MODEL_ATTRIBUTE_NAMES)


def _model_snapshot_nbytes(gaussians, names: Sequence[str]) -> int:
    return sum(
        int(getattr(gaussians, "_" + name).numel())
        * int(getattr(gaussians, "_" + name).element_size())
        for name in names
    )


def _estimated_pair_output_bytes(
    request: "MapPublicationRequest", num_classes: int
) -> int:
    """Conservative uncompressed product estimate used before GPU cloning."""
    pixels = int(request.camera.image_width) * int(request.camera.image_height)
    config = request.semantic_config
    diagnostic_bytes = 2 if config.quantize_diagnostics else 4
    # RGB, semantic id/color, unknown mask/reason and two mandatory rasters.
    bytes_per_pixel = 3 + 1 + 3 + 1 + 1 + 2 * diagnostic_bytes
    if config.save_diagnostics:
        bytes_per_pixel += 8 * diagnostic_bytes
    if config.save_probabilities:
        bytes_per_pixel += 2 * int(num_classes)
    return pixels * bytes_per_pixel + 1024**2


def _tdom_alignment(camera, supplied: Optional[Mapping[str, object]] = None):
    """Return JSON-safe camera data shared by RGB and semantic TDOM products."""
    alignment = dict(supplied or {})
    alignment.setdefault("schema_version", 1)
    alignment.update(
        {
            "image_width": int(camera.image_width),
            "image_height": int(camera.image_height),
            "fov_x_radians": float(camera.FoVx),
            "fov_y_radians": float(camera.FoVy),
            "world_view_transform": camera.world_view_transform.detach()
            .cpu()
            .tolist(),
            "full_projection_transform": camera.full_proj_transform.detach()
            .cpu()
            .tolist(),
            "camera_center": camera.camera_center.detach().cpu().tolist(),
        }
    )
    return alignment


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
                _image_id(item)
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
        return tuple(sorted(self.image_ids, key=_acquisition_key))

    def normalized(self, first: str, second: str) -> float:
        positions = {name: index for index, name in enumerate(self.image_ids)}
        first = _image_id(first)
        second = _image_id(second)
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


@dataclass(frozen=True)
class OnlineSemanticConfig:
    overlap_threshold: float = 0.4
    max_group_size: int = 4
    reliability_floor: float = 0.1
    temperature: float = 0.1
    min_consensus_support: float = 1.0
    prior_total: float = 1.0
    boundary_bandwidth: float = 2.0
    boundary_floor: float = 0.25
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if self.max_group_size < 1 or self.max_group_size > 4:
            raise ValueError("max_group_size must be in [1, 4]")
        if not 0 <= self.reliability_floor <= 1:
            raise ValueError("reliability_floor must be in [0, 1]")
        if self.temperature <= 0 or self.prior_total <= 0 or self.eps <= 0:
            raise ValueError("temperature, prior_total, and eps must be positive")
        if self.min_consensus_support < 0:
            raise ValueError("min_consensus_support must be non-negative")
        if self.boundary_bandwidth < 0:
            raise ValueError("boundary_bandwidth must be non-negative")
        if not 0 <= self.boundary_floor <= 1:
            raise ValueError("boundary_floor must be in [0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 2,
            "semantic_method_version": SEMANTIC_METHOD_VERSION,
            "overlap_threshold": self.overlap_threshold,
            "max_group_size": self.max_group_size,
            "reliability_floor": self.reliability_floor,
            "temperature": self.temperature,
            "min_consensus_support": self.min_consensus_support,
            "prior_total": self.prior_total,
            "boundary_bandwidth": self.boundary_bandwidth,
            "boundary_floor": self.boundary_floor,
            "eps": self.eps,
        }


class ProjectedObservation(NamedTuple):
    distribution: torch.Tensor
    support: torch.Tensor
    semantic_support: torch.Tensor


class SemanticSyncResult(NamedTuple):
    geometry_stage: int
    semantic_through_image: Optional[str]
    processed_image_ids: Tuple[str, ...]
    deferred_image_ids: Tuple[str, ...]
    lag_batches: int
    within_lag_budget: bool
    errors: Tuple[str, ...]
    status: str = "approx_online"
    historical_evidence_stale: bool = False
    publication_status: str = "not_requested"
    published_product_key: Optional[str] = None


@dataclass(frozen=True)
class MapPublicationRequest:
    """A map event independent from the geometry-stage synchronization event."""

    product_key: str
    output_root: Path
    camera: object
    alignment: Mapping[str, object]
    rgb_renderer: object
    rgb_pipeline: object
    rgb_background: torch.Tensor
    semantic_config: SemanticTdomConfig
    publication_due: bool = True
    diagnostics_degraded: bool = False

    def __post_init__(self) -> None:
        if not self.product_key or Path(self.product_key).name != self.product_key:
            raise ValueError("product_key must be one safe path component")
        if not callable(self.rgb_renderer):
            raise ValueError("rgb_renderer must be callable")
        if not isinstance(self.publication_due, bool):
            raise ValueError("publication_due must be a boolean")
        if not isinstance(self.diagnostics_degraded, bool):
            raise ValueError("diagnostics_degraded must be a boolean")
        if not isinstance(self.alignment, Mapping):
            raise ValueError("map alignment must be an object")
        grid = self.alignment.get("grid")
        window = self.alignment.get("window")
        if not isinstance(grid, Mapping) or not isinstance(window, Mapping):
            raise ValueError("map alignment requires a fixed grid and window")
        if grid.get("grid_id") != window.get("grid_id"):
            raise ValueError("map grid/window identity mismatch")
        if getattr(self.rgb_pipeline, "compute_cov3D_python", False):
            raise ValueError("paired RGB TDOM requires CUDA covariance computation")
        if getattr(self.rgb_pipeline, "GetNormal", False):
            raise ValueError("paired RGB TDOM does not render normal maps")


@dataclass(frozen=True)
class _PendingStageRequest:
    args: Tuple[object, ...]
    publication_status_override: Optional[str] = None


class SemanticCheckpoint(NamedTuple):
    state_path: Path
    state_sha256: str
    projection_geometry_hash: str
    rgb_model_hash: Optional[str]


class FrozenGaussianSnapshot:
    """Detached complete RGB model for one paired map publication."""

    _ATTRIBUTE_NAMES = _GEOMETRY_ATTRIBUTE_NAMES

    def __init__(self, gaussians):
        self._values = {
            name: getattr(gaussians, "_" + name).detach().clone()
            for name in self._ATTRIBUTE_NAMES
        }
        self._cpu_values = None
        self._projection_hash = None
        self._rgb_model_hash = None
        self.active_sh_degree = int(getattr(gaussians, "active_sh_degree", 0))
        self.max_sh_degree = int(
            getattr(gaussians, "max_sh_degree", self.active_sh_degree)
        )
        self._ready_event = None
        if self.get_xyz.is_cuda:
            self._ready_event = torch.cuda.Event()
            self._ready_event.record(torch.cuda.current_stream(self.get_xyz.device))

    @property
    def get_xyz(self):
        return self._values["xyz"]

    @property
    def get_opacity(self):
        return torch.sigmoid(self._values["opacity"])

    @property
    def get_scaling(self):
        return torch.exp(self._values["scaling"])

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._values["rotation"])

    @property
    def get_features_dc(self):
        return self._values["features_dc"]

    @property
    def get_features_rest(self):
        return self._values["features_rest"]

    @property
    def get_features(self):
        return torch.cat((self.get_features_dc, self.get_features_rest), dim=1)

    def wait_ready(self, stream=None) -> None:
        if self._ready_event is not None:
            (stream or torch.cuda.current_stream(self.get_xyz.device)).wait_event(
                self._ready_event
            )

    def semantic_projection_sha256(self) -> str:
        if self._projection_hash is None:
            self._projection_hash = semantic_projection_sha256(
                self.geometry_attributes_cpu()
            )
        return self._projection_hash

    def rgb_model_sha256(self) -> str:
        if self._rgb_model_hash is None:
            self._rgb_model_hash = rgb_model_sha256(
                self.geometry_attributes_cpu()
            )
        return self._rgb_model_hash

    def geometry_sha256(self) -> str:
        """Deprecated complete-model hash alias kept for external callers."""
        return self.rgb_model_sha256()

    def geometry_attributes_cpu(self) -> Dict[str, torch.Tensor]:
        if self._cpu_values is None:
            self._cpu_values = {
                name: value.detach().cpu().contiguous()
                for name, value in self._values.items()
            }
        return self._cpu_values


class FrozenSemanticSnapshot:
    """Projection-only snapshot; appearance is intentionally not copied."""

    _ATTRIBUTE_NAMES = SEMANTIC_PROJECTION_ATTRIBUTE_NAMES

    def __init__(self, gaussians):
        self._values = {
            name: getattr(gaussians, "_" + name).detach().clone()
            for name in self._ATTRIBUTE_NAMES
        }
        self._cpu_values = None
        self._projection_hash = None
        self._ready_event = None
        if self.get_xyz.is_cuda:
            self._ready_event = torch.cuda.Event()
            self._ready_event.record(torch.cuda.current_stream(self.get_xyz.device))

    @property
    def get_xyz(self):
        return self._values["xyz"]

    @property
    def get_opacity(self):
        return torch.sigmoid(self._values["opacity"])

    @property
    def get_scaling(self):
        return torch.exp(self._values["scaling"])

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._values["rotation"])

    def wait_ready(self, stream=None) -> None:
        if self._ready_event is not None:
            (stream or torch.cuda.current_stream(self.get_xyz.device)).wait_event(
                self._ready_event
            )

    def semantic_projection_sha256(self) -> str:
        if self._projection_hash is None:
            self._projection_hash = semantic_projection_sha256(
                self.geometry_attributes_cpu()
            )
        return self._projection_hash

    def geometry_attributes_cpu(self) -> Dict[str, torch.Tensor]:
        if self._cpu_values is None:
            self._cpu_values = {
                name: value.detach().cpu().contiguous()
                for name, value in self._values.items()
            }
        return self._cpu_values


class FrozenCameraSnapshot:
    """Only the camera fields used by semantic projection."""

    def __init__(self, camera):
        self.image_name = getattr(camera, "image_name", "semantic_tdom")
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.FoVx = camera.FoVx
        self.FoVy = camera.FoVy
        self.world_view_transform = camera.world_view_transform.detach().clone()
        self.full_proj_transform = camera.full_proj_transform.detach().clone()
        self.camera_center = camera.camera_center.detach().clone()
        tri_mask = getattr(camera, "Tri_Mask", None)
        self.Tri_Mask = (
            None if tri_mask is None else tri_mask.detach().clone()
        )


def label_boundary_weights(
    labels: np.ndarray,
    ignore_id: int = 255,
    bandwidth: float = 2.0,
    floor: float = 0.25,
) -> np.ndarray:
    """Create a continuous reliability ramp around hard-label boundaries."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    valid = labels != ignore_id
    boundary = np.zeros(labels.shape, dtype=bool)
    vertical = labels[1:, :] != labels[:-1, :]
    horizontal = labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    if bandwidth <= 0 or not bool(boundary.any()):
        weights = np.ones(labels.shape, dtype=np.float32)
    else:
        from scipy.ndimage import distance_transform_edt

        distance = distance_transform_edt(~boundary)
        ramp = np.clip(distance / float(bandwidth), 0.0, 1.0)
        weights = (floor + (1.0 - floor) * ramp).astype(np.float32)
    weights[~valid] = 0.0
    return weights


class CudaSemanticProjector:
    """Adapt a camera, Gaussian model, and validated label to the WP2 kernel."""

    def __init__(self, class_spec: SemanticClassSpec, config: OnlineSemanticConfig):
        self.class_spec = class_spec
        self.config = config

    def project(self, gaussians, camera, observation: SemanticObservation):
        labels = observation.labels
        expected_shape = (camera.image_height, camera.image_width)
        if tuple(labels.shape) != expected_shape:
            raise SemanticSchemaError(
                "label {} has shape {}, camera expects {}; full-resolution "
                "semantic projection does not resize labels".format(
                    observation.key.image_id, labels.shape, expected_shape
                )
            )
        labels = labels.copy()
        tri_mask = getattr(camera, "Tri_Mask", None)
        if tri_mask is not None and tuple(tri_mask.shape[-2:]) == expected_shape:
            valid_image = tri_mask.detach().to("cpu").numpy().astype(bool)
            labels[~valid_image.reshape(expected_shape)] = self.class_spec.ignore_id
        boundary = label_boundary_weights(
            labels,
            ignore_id=self.class_spec.ignore_id,
            bandwidth=self.config.boundary_bandwidth,
            floor=self.config.boundary_floor,
        )
        result = rasterize_semantic_observation(
            means3d=gaussians.get_xyz,
            opacities=gaussians.get_opacity,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            labels=torch.from_numpy(labels.astype(np.int64, copy=False)),
            boundary_weights=torch.from_numpy(boundary),
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            tan_fovx=math.tan(float(camera.FoVx) * 0.5),
            tan_fovy=math.tan(float(camera.FoVy) * 0.5),
            campos=camera.camera_center,
            num_classes=self.class_spec.num_classes,
            ignore_id=self.class_spec.ignore_id,
            eps=self.config.eps,
        )
        return ProjectedObservation(
            result.distribution, result.support, result.valid_mass
        )


class OnlineSemanticOrchestrator:
    """Maintain G0/G1 while an open correlated-view group is replaced in place."""

    def __init__(
        self,
        state: SemanticGaussianState,
        class_spec: SemanticClassSpec,
        config: OnlineSemanticConfig,
        ledger: Optional[SemanticLedger] = None,
        label_provenance: Optional[Mapping[str, object]] = None,
        open_group: Optional[Mapping[str, object]] = None,
    ):
        if state.num_classes != class_spec.num_classes:
            raise ValueError("semantic state and class spec disagree")
        if state.prior_total != config.prior_total:
            raise ValueError("semantic state and config prior_total disagree")
        self.state = state
        self.class_spec = class_spec
        self.config = config
        self.ledger = ledger or SemanticLedger()
        self.label_provenance = dict(
            label_provenance
            or {"schema_version": 1, "versions": {}}
        )
        self.open_group = (
            None if open_group is None else dict(open_group)
        )
        self.current_geometry_stage = 0
        self._validate_runtime_state()

    @classmethod
    def from_state(
        cls,
        state: SemanticGaussianState,
        class_spec: SemanticClassSpec,
        config: OnlineSemanticConfig,
    ) -> "OnlineSemanticOrchestrator":
        context = state.recovery_context
        entries = (
            LedgerEntry.from_dict(item)
            for item in context["ledger"]["observations"]
        )
        return cls(
            state,
            class_spec,
            config,
            ledger=SemanticLedger(entries),
            label_provenance=context["label_provenance"],
            open_group=context["open_group"],
        )

    def _validate_runtime_state(self) -> None:
        representatives = ()
        if self.open_group is not None:
            representatives = tuple(
                self.open_group.get("representative_image_ids", ())
            )
        if len(representatives) != self.state.open_distributions.shape[0]:
            raise ValueError(
                "open group metadata and cached observations disagree"
            )
        if self.open_group is None and bool(
            (self.state.open_group_count > 0).any()
        ):
            raise ValueError("provisional tensors require open group metadata")
        versions = {entry.label_version for entry in self.ledger.entries}
        if self.open_group is not None:
            versions.add(str(self.open_group["label_version"]))
        if len(versions) > 1:
            raise SemanticSchemaError(
                "online fusion cannot mix label versions; rebuild is required"
            )

    @property
    def open_member_ids(self) -> Tuple[str, ...]:
        if self.open_group is None:
            return ()
        return tuple(self.open_group["member_image_ids"])

    @property
    def label_version(self) -> Optional[str]:
        versions = {entry.label_version for entry in self.ledger.entries}
        if self.open_group is not None:
            versions.add(str(self.open_group["label_version"]))
        return next(iter(versions)) if versions else None

    def _provenance_candidate(self, observation: SemanticObservation):
        version = observation.key.label_version
        candidate = {
            "model": observation.metadata["model"],
            "checkpoint_sha256": observation.metadata["checkpoint_sha256"],
            "prompt_config": observation.metadata["prompt_config"],
            "thresholds": dict(observation.metadata["thresholds"]),
        }
        versions = self.label_provenance.setdefault("versions", {})
        previous = versions.get(version)
        if previous is not None and previous != candidate:
            raise SemanticSchemaError(
                "label provenance changed within version {}".format(version)
            )
        return version, candidate

    def ingest(
        self,
        observation: SemanticObservation,
        projection: ProjectedObservation,
        overlap: AcquisitionOverlap,
    ) -> bool:
        entry = LedgerEntry(
            image_id=observation.key.image_id,
            label_version=observation.key.label_version,
            label_sha256=observation.metadata["label_sha256"],
            score_sha256=observation.metadata.get("score_sha256"),
        )
        previous = self.ledger.get(entry.image_id, entry.label_version)
        if previous is not None:
            self.ledger.record(entry)
            return False
        if self.label_version not in (None, entry.label_version):
            raise SemanticSchemaError(
                "a new label version requires rebuilding online semantics"
            )
        provenance_version, provenance = self._provenance_candidate(observation)

        image_id = entry.image_id
        members = self.open_member_ids
        starts_new_group = not members
        if members:
            starts_new_group = (
                len(members) >= self.config.max_group_size
                or overlap.normalized(members[0], image_id)
                < self.config.overlap_threshold
            )
        if starts_new_group:
            next_members = [image_id]
            next_open_group = {
                "schema_version": 1,
                "anchor_image_id": image_id,
                "member_image_ids": next_members,
                "representative_image_ids": next_members,
                "label_version": entry.label_version,
                "projection_topology_version": self.state.topology_version,
                "projection_geometry_stage": self.current_geometry_stage,
            }
            distributions = projection.distribution.detach().unsqueeze(0)
            supports = projection.support.detach().unsqueeze(0)
            semantic_supports = projection.semantic_support.detach().unsqueeze(0)
        else:
            next_members = list(members) + [image_id]
            representatives = [next_members[0]]
            if next_members[-1] != next_members[0]:
                representatives.append(next_members[-1])
            next_open_group = {
                "schema_version": 1,
                "anchor_image_id": next_members[0],
                "member_image_ids": next_members,
                "representative_image_ids": representatives,
                "label_version": entry.label_version,
                "projection_topology_version": self.state.topology_version,
                "projection_geometry_stage": self.current_geometry_stage,
            }
            # Retain only the anchor and newest keyframe.  Keeping an older
            # middle view would mix geometry versions after a later refresh.
            if self.state.open_distributions.shape[0] == 1:
                distributions = torch.cat(
                    (
                        self.state.open_distributions,
                        projection.distribution.unsqueeze(0),
                    ),
                    dim=0,
                )
                supports = torch.cat(
                    (self.state.open_supports, projection.support.unsqueeze(0)),
                    dim=0,
                )
                semantic_supports = torch.cat(
                    (
                        self.state.open_semantic_supports,
                        projection.semantic_support.unsqueeze(0),
                    ),
                    dim=0,
                )
            else:
                distributions = torch.stack(
                    (
                        self.state.open_distributions[0],
                        projection.distribution,
                    )
                )
                supports = torch.stack(
                    (self.state.open_supports[0], projection.support)
                )
                semantic_supports = torch.stack(
                    (
                        self.state.open_semantic_supports[0],
                        projection.semantic_support,
                    )
                )

        group = aggregate_observation_group(
            distributions, supports, eps=self.config.eps
        )
        raw_evidence = group.support.unsqueeze(1) * group.distribution
        if starts_new_group and members:
            closed_raw = self.state.alpha_raw
            closed_count = self.state.group_count
        else:
            closed_raw = self.state.alpha_raw - self.state.open_raw_evidence
            closed_count = self.state.group_count - self.state.open_group_count
        closed_evidence = (closed_raw - self.state.prior()).clamp_min(0.0)
        closed_support = closed_evidence.sum(dim=1)
        consensus = closed_evidence / (
            closed_support.unsqueeze(1) + self.config.eps
        )
        js = self._js_divergence(group.distribution, consensus)
        reliability = self.config.reliability_floor + (
            1.0 - self.config.reliability_floor
        ) * torch.exp(-js / self.config.temperature)
        eligible = (
            (closed_count >= 2)
            & (closed_support >= self.config.min_consensus_support)
            & (group.support > self.config.eps)
        )
        reliability = torch.where(
            eligible, reliability, torch.ones_like(reliability)
        )
        robust_evidence = reliability.unsqueeze(1) * raw_evidence
        increment = (group.support > self.config.eps).to(dtype=torch.long)
        self.state.replace_open_group(
            raw_evidence,
            robust_evidence,
            increment,
            distributions,
            supports,
            semantic_supports,
            semantic_support_increment=projection.semantic_support,
            commit_existing=bool(starts_new_group and members),
        )
        self.open_group = next_open_group
        self.label_provenance["versions"][provenance_version] = provenance
        self.ledger.record(entry)
        if len(self.open_member_ids) >= self.config.max_group_size:
            self.close_group()
        return True

    def _js_divergence(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        eps = self.config.eps
        first = first / (first.sum(dim=1, keepdim=True) + eps)
        second = second / (second.sum(dim=1, keepdim=True) + eps)
        midpoint = 0.5 * (first + second)
        first_term = first * (
            torch.log(first.clamp_min(eps))
            - torch.log(midpoint.clamp_min(eps))
        )
        second_term = second * (
            torch.log(second.clamp_min(eps))
            - torch.log(midpoint.clamp_min(eps))
        )
        return 0.5 * (first_term.sum(dim=1) + second_term.sum(dim=1))

    def close_group(self) -> None:
        if self.open_group is None:
            return
        self.state.close_open_group()
        self.open_group = None

    def refresh_open_group(
        self, projections: Mapping[str, ProjectedObservation]
    ) -> bool:
        """Replace active-keyframe rows after a Gaussian topology change."""
        if self.open_group is None:
            return False
        projected_version = self.open_group.get(
            "projection_topology_version", self.state.topology_version
        )
        projected_stage = self.open_group.get("projection_geometry_stage", 0)
        if (
            projected_version == self.state.topology_version
            and projected_stage == self.current_geometry_stage
        ):
            return False
        representatives = tuple(
            self.open_group.get("representative_image_ids", ())
        )
        if not representatives or any(
            image_id not in projections for image_id in representatives
        ):
            return False
        distributions = self.state.open_distributions.clone()
        supports = self.state.open_supports.clone()
        semantic_supports = self.state.open_semantic_supports.clone()
        dirty = self.state.active_projection_dirty
        for index, image_id in enumerate(representatives):
            projection = projections[image_id]
            distribution = projection.distribution.detach().to(
                device=self.state.device, dtype=self.state.dtype
            )
            support = projection.support.detach().to(
                device=self.state.device, dtype=self.state.dtype
            )
            semantic_support = projection.semantic_support.detach().to(
                device=self.state.device, dtype=self.state.dtype
            )
            if distribution.shape != (self.state.num_gaussians, self.state.num_classes):
                raise ValueError("refreshed distribution shape is invalid")
            if support.shape != (self.state.num_gaussians,):
                raise ValueError("refreshed support shape is invalid")
            if semantic_support.shape != (self.state.num_gaussians,):
                raise ValueError("refreshed semantic support shape is invalid")
            distributions[index, dirty] = distribution[dirty]
            supports[index, dirty] = support[dirty]
            semantic_supports[index, dirty] = semantic_support[dirty]
        group = aggregate_observation_group(
            distributions, supports, eps=self.config.eps
        )
        raw_evidence = group.support.unsqueeze(1) * group.distribution
        closed_raw = self.state.alpha_raw - self.state.open_raw_evidence
        closed_count = self.state.group_count - self.state.open_group_count
        closed_evidence = (closed_raw - self.state.prior()).clamp_min(0.0)
        closed_support = closed_evidence.sum(dim=1)
        consensus = closed_evidence / (
            closed_support.unsqueeze(1) + self.config.eps
        )
        reliability = self.config.reliability_floor + (
            1.0 - self.config.reliability_floor
        ) * torch.exp(
            -self._js_divergence(group.distribution, consensus)
            / self.config.temperature
        )
        eligible = (
            (closed_count >= 2)
            & (closed_support >= self.config.min_consensus_support)
            & (group.support > self.config.eps)
        )
        reliability = torch.where(
            eligible, reliability, torch.ones_like(reliability)
        )
        self.state.replace_open_group(
            raw_evidence,
            reliability.unsqueeze(1) * raw_evidence,
            (group.support > self.config.eps).to(dtype=torch.long),
            distributions,
            supports,
            semantic_supports,
        )
        self.open_group["projection_topology_version"] = self.state.topology_version
        self.open_group["projection_geometry_stage"] = self.current_geometry_stage
        self.state.mark_active_projection_current(dirty)
        return True

    def recovery_context(self) -> Dict[str, object]:
        return {
            "schema_version": 2,
            "semantic_method_version": SEMANTIC_METHOD_VERSION,
            "ledger": {
                "schema_version": 1,
                "observations": [
                    entry.to_dict() for entry in self.ledger.entries
                ],
            },
            "label_provenance": self.label_provenance,
            "open_group": self.open_group,
            "ontology": self.class_spec.ontology_dict(),
            "semantic_config": self.config.to_dict(),
        }


class ProgressiveSemanticHook:
    """Consume ready labels at geometry stage boundaries and checkpoint state."""

    def __init__(
        self,
        dataset: SemanticDataset,
        orchestrator: OnlineSemanticOrchestrator,
        projector,
        output_root: Path,
        publication_policy: Optional[MapPublicationPolicy] = None,
        final_authoritative_complete: bool = True,
    ):
        self.dataset = dataset
        self.orchestrator = orchestrator
        self.projector = projector
        self.output_root = Path(output_root)
        self.publication_policy = publication_policy or MapPublicationPolicy()
        self.final_authoritative_complete = bool(final_authoritative_complete)
        self._stage_boundaries = []
        self._verified_recorded_files = {}
        self._last_publication_seconds = 0.0

    @classmethod
    def create(
        cls,
        dataset_root: Path,
        output_root: Path,
        gaussians,
        config: Optional[OnlineSemanticConfig] = None,
        projector=None,
        semantic_dataset_root: Optional[Path] = None,
        publication_policy: Optional[MapPublicationPolicy] = None,
        resume: bool = True,
        final_authoritative_complete: bool = True,
    ) -> "ProgressiveSemanticHook":
        config = config or OnlineSemanticConfig()
        dataset = SemanticDataset(semantic_dataset_root or dataset_root)
        output_root = Path(output_root)
        latest_path = output_root / "latest.json"
        if resume and latest_path.is_file():
            try:
                with latest_path.open("r", encoding="utf-8") as stream:
                    latest = json.load(stream)
                if (
                    latest.get("schema_version") != 5
                    or latest.get("semantic_method_version")
                    != SEMANTIC_METHOD_VERSION
                    or latest.get("ontology_hash")
                    != dataset.class_spec.ontology_fingerprint
                    or latest.get("slot") not in (0, 1)
                ):
                    raise ValueError("invalid latest semantic slot")
                slot = latest["slot"]
                state_path = output_root / "semantic_state.{}.pt".format(slot)
                context_path = output_root / "recovery_context.{}.json".format(slot)
                geometry_path = output_root / "geometry_checkpoint.{}.pt".format(slot)
                with context_path.open("r", encoding="utf-8") as stream:
                    context = json.load(stream)
            except (OSError, ValueError, AttributeError, KeyError, TypeError) as error:
                raise SemanticSchemaError(
                    "unable to load semantic recovery context: {}".format(error)
                )
            if not state_path.is_file():
                raise SemanticSchemaError("latest semantic state file is missing")
            if latest.get("semantic_state") != state_path.name:
                raise SemanticSchemaError(
                    "latest semantic state name does not match its slot"
                )
            if latest.get("semantic_state_sha256") != sha256_file(state_path):
                raise SemanticSchemaError(
                    "latest semantic state content hash does not match"
                )
            if not geometry_path.is_file():
                raise SemanticSchemaError("latest geometry checkpoint is missing")
            if latest.get("geometry_checkpoint") != geometry_path.name:
                raise SemanticSchemaError(
                    "latest geometry checkpoint name does not match its slot"
                )
            if latest.get("geometry_checkpoint_sha256") != sha256_file(
                geometry_path
            ):
                raise SemanticSchemaError(
                    "latest geometry checkpoint content hash does not match"
                )
            try:
                geometry_payload = torch.load(
                    str(geometry_path), map_location="cpu"
                )
                attributes = geometry_payload["attributes"]
                checkpoint_projection_hash = semantic_projection_sha256(
                    attributes
                )
                checkpoint_rgb_hash = geometry_payload.get("rgb_model_hash")
                if checkpoint_rgb_hash is not None and (
                    not isinstance(checkpoint_rgb_hash, str)
                    or re.fullmatch(r"[0-9a-f]{64}", checkpoint_rgb_hash) is None
                ):
                    raise ValueError("invalid optional RGB model hash")
                current_attributes = _projection_attributes(gaussians)
                current_projection_hash = semantic_projection_sha256(
                    current_attributes
                )
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                raise SemanticSchemaError(
                    "unable to validate geometry checkpoint: {}".format(error)
                )
            if (
                geometry_payload.get("schema_version") != 3
                or geometry_payload.get("semantic_method_version")
                != SEMANTIC_METHOD_VERSION
                or geometry_payload.get("projection_geometry_hash")
                != checkpoint_projection_hash
                or latest.get("projection_geometry_hash")
                != checkpoint_projection_hash
                or latest.get("rgb_model_hash") != checkpoint_rgb_hash
                or current_projection_hash != checkpoint_projection_hash
            ):
                raise SemanticSchemaError(
                    "semantic recovery does not match the persisted geometry checkpoint"
                )
            if context.get("semantic_config") != config.to_dict():
                raise SemanticSchemaError(
                    "semantic configuration changed; explicit rebuild is required"
                )
            recovery_hash = SemanticGaussianState.validated_recovery_context_sha256(
                context,
                ontology_hash=dataset.class_spec.ontology_fingerprint,
                prior_total=config.prior_total,
                num_classes=dataset.class_spec.num_classes,
            )
            if latest.get("recovery_context_sha256") != recovery_hash:
                raise SemanticSchemaError(
                    "latest semantic recovery context hash does not match"
                )
            state = SemanticGaussianState.load(
                state_path,
                expected_projection_geometry_hash=current_projection_hash,
                expected_ontology_hash=dataset.class_spec.ontology_fingerprint,
                expected_gaussian_count=gaussians.get_xyz.shape[0],
                expected_recovery_context_sha256=recovery_hash,
                map_location=gaussians.get_xyz.device,
            )
            orchestrator = OnlineSemanticOrchestrator.from_state(
                state, dataset.class_spec, config
            )
        elif resume and any(
            (output_root / name).exists()
            for name in (
                "semantic_state.0.pt",
                "semantic_state.1.pt",
                "recovery_context.0.json",
                "recovery_context.1.json",
                "geometry_checkpoint.0.pt",
                "geometry_checkpoint.1.pt",
            )
        ):
            raise SemanticSchemaError(
                "semantic slot files exist without an atomic latest pointer"
            )
        else:
            state = SemanticGaussianState.initialize(
                gaussians.get_xyz.shape[0],
                dataset.class_spec.num_classes,
                prior_total=config.prior_total,
                device=gaussians.get_xyz.device,
            )
            orchestrator = OnlineSemanticOrchestrator(
                state, dataset.class_spec, config
            )
        if projector is None:
            projector = CudaSemanticProjector(dataset.class_spec, config)
        hook = cls(
            dataset,
            orchestrator,
            projector,
            output_root,
            publication_policy=publication_policy,
            final_authoritative_complete=final_authoritative_complete,
        )
        status_path = output_root / "sync_status.json"
        if resume and status_path.is_file():
            try:
                with status_path.open("r", encoding="utf-8") as stream:
                    status = json.load(stream)
                if (
                    status.get("schema_version") != 3
                    or status.get("semantic_method_version")
                    != SEMANTIC_METHOD_VERSION
                ):
                    raise ValueError("unsupported semantic sync status")
                boundaries = status.get("stage_boundaries", [])
                if (
                    not isinstance(boundaries, list)
                    or any(not isinstance(item, str) or not item for item in boundaries)
                ):
                    raise ValueError("invalid semantic stage boundaries")
                hook._stage_boundaries = list(boundaries)
            except (OSError, ValueError, AttributeError) as error:
                raise SemanticSchemaError(
                    "unable to load semantic sync status: {}".format(error)
                )
        return hook

    def checkpoint_required(
        self,
        geometry_stage: int,
        final: bool,
        publication_requested: bool,
    ) -> bool:
        return self.publication_policy.should_checkpoint(
            geometry_stage,
            final,
            exists=(self.output_root / "latest.json").is_file(),
        )

    def synchronize(
        self,
        geometry_stage: int,
        cameras: Sequence[object],
        gaussians,
        snapshot_root: Path,
        final: bool = False,
        stage_boundaries: Optional[Sequence[str]] = None,
        publication_request: Optional[MapPublicationRequest] = None,
    ) -> SemanticSyncResult:
        geometry_stage = int(geometry_stage)
        online_usable = self.orchestrator.state.is_valid
        if (
            online_usable
            and geometry_stage != self.orchestrator.current_geometry_stage
        ):
            closed_count = (
                self.orchestrator.state.group_count
                - self.orchestrator.state.open_group_count
            )
            if bool((closed_count > 0).any()):
                self.orchestrator.state.mark_historical_evidence_stale()
            self.orchestrator.state.mark_dirty()
            if self.orchestrator.open_group is not None:
                self.orchestrator.state.mark_active_projection_dirty()
        self.orchestrator.current_geometry_stage = geometry_stage
        overlap = AcquisitionOverlap.load(snapshot_root)
        camera_by_id = {
            _image_id(camera.image_name): camera for camera in cameras
        }
        all_ordered = tuple(overlap.acquisition_order)
        ordered = tuple(
            image_id
            for image_id in all_ordered
            if image_id in camera_by_id
        )
        if stage_boundaries is not None:
            self._stage_boundaries = list(stage_boundaries)
        elif ordered and (
            not self._stage_boundaries
            or self._stage_boundaries[-1] != ordered[-1]
        ):
            self._stage_boundaries.append(ordered[-1])
        processed = []
        errors = []
        deferred = []
        if not online_usable:
            errors.append(
                "online semantic state unavailable: {}".format(
                    self.orchestrator.state.invalid_reason or "unknown failure"
                )
            )
        if final:
            missing_cameras = tuple(
                image_id
                for image_id in all_ordered
                if image_id not in camera_by_id
            )
            if missing_cameras:
                errors.append(
                    "final exact rebuild missing cameras: {}".format(
                        ", ".join(missing_cameras)
                    )
                )
                deferred.extend(missing_cameras)
        if online_usable:
            try:
                self._refresh_active_window(camera_by_id, gaussians)
            except Exception as error:
                errors.append("active-window refresh: {}".format(error))
        ledger_images = {
            entry.image_id for entry in self.orchestrator.ledger.entries
        }
        for position, image_id in enumerate(ordered if online_usable else ()):
            if image_id in ledger_images:
                try:
                    self._verify_recorded_metadata(image_id)
                except Exception as error:
                    errors.append("{}: {}".format(image_id, error))
                    deferred.extend(ordered[position:])
                    break
                continue
            label_path = self.dataset.semantic_root / "labels" / (image_id + ".png")
            metadata_path = (
                self.dataset.semantic_root / "metadata" / (image_id + ".json")
            )
            if not label_path.is_file() or not metadata_path.is_file():
                deferred.extend(ordered[position:])
                break
            try:
                observation = self.dataset.load_observation(image_id)
                projection = self.projector.project(
                    gaussians, camera_by_id[image_id], observation
                )
                if self.orchestrator.ingest(observation, projection, overlap):
                    processed.append(image_id)
                    ledger_images.add(image_id)
            except Exception as error:
                errors.append("{}: {}".format(image_id, error))
                deferred.extend(ordered[position:])
                break
        exact_rebuilt = False
        exact_complete = False
        if final:
            if online_usable:
                self.orchestrator.close_group()
            final_ready = []
            final_missing = []
            for image_id in all_ordered:
                label_path = (
                    self.dataset.semantic_root / "labels" / (image_id + ".png")
                )
                metadata_path = (
                    self.dataset.semantic_root
                    / "metadata"
                    / (image_id + ".json")
                )
                if image_id not in camera_by_id:
                    final_missing.append(image_id)
                elif label_path.is_file() and metadata_path.is_file():
                    final_ready.append(image_id)
                else:
                    final_missing.append(image_id)
            deferred = list(dict.fromkeys(deferred + final_missing))
            if final_ready:
                try:
                    rebuilt = FinalSemanticRebuilder(
                        self.dataset,
                        self.projector,
                        self.orchestrator.config,
                    ).rebuild(
                        tuple(final_ready),
                        camera_by_id,
                        gaussians,
                        overlap,
                        online_state=self.orchestrator.state,
                    )
                    replacement = OnlineSemanticOrchestrator(
                        rebuilt.state,
                        self.dataset.class_spec,
                        self.orchestrator.config,
                        ledger=rebuilt.ledger,
                        label_provenance=rebuilt.label_provenance,
                    )
                    replacement.current_geometry_stage = geometry_stage
                    self.output_root.mkdir(parents=True, exist_ok=True)
                    self._save_json_atomic(
                        self.output_root / "final_rebuild_metrics.json",
                        {
                            "schema_version": 2,
                            "semantic_method_version": SEMANTIC_METHOD_VERSION,
                            "geometry_stage": geometry_stage,
                            "projection_geometry_hash": semantic_projection_sha256(
                                _projection_attributes(gaussians)
                            ),
                            "rgb_model_hash": (
                                rgb_model_sha256(_rgb_attributes(gaussians))
                                if isinstance(gaussians, FrozenGaussianSnapshot)
                                else None
                            ),
                            "groups": [list(group) for group in rebuilt.image_groups],
                            "metrics": rebuilt.metrics.to_dict(),
                        },
                    )
                    # Swap only after every fallible rebuild/metrics operation
                    # completed, so failure always leaves the online state live.
                    self.orchestrator = replacement
                    ledger_images = {
                        entry.image_id for entry in rebuilt.ledger.entries
                    }
                    exact_rebuilt = True
                    exact_complete = (
                        not final_missing and self.final_authoritative_complete
                    )
                except Exception as error:
                    errors.append("final exact rebuild: {}".format(error))
            else:
                errors.append("final exact rebuild has no valid observations")

        through_index = 0
        for image_id in ordered:
            if image_id not in ledger_images:
                break
            through_index += 1
        semantic_through = ordered[through_index - 1] if through_index else None
        if semantic_through is None:
            lag = len(self._stage_boundaries)
        else:
            through_key = _acquisition_key(semantic_through)
            lag = sum(
                _acquisition_key(boundary) > through_key
                for boundary in self._stage_boundaries
            )
        historical_stale = (
            self.orchestrator.state.historical_evidence_stale
            if self.orchestrator.state.is_valid
            else True
        )
        if exact_rebuilt:
            status = "exact_final" if exact_complete else "exact_partial"
        elif final:
            status = (
                "online_fallback"
                if self.orchestrator.state.is_valid
                else "failed"
            )
        elif not self.orchestrator.state.is_valid:
            status = "failed"
        elif historical_stale or deferred or errors or lag > 1:
            status = "stale_online"
        else:
            status = "approx_online"
        result = SemanticSyncResult(
            int(geometry_stage),
            semantic_through,
            tuple(processed),
            tuple(deferred),
            lag,
            lag <= 1,
            tuple(errors),
            status,
            historical_stale,
        )
        checkpoint = None
        if self.orchestrator.state.is_valid and self.checkpoint_required(
            geometry_stage,
            final,
            publication_request is not None,
        ):
            try:
                checkpoint = self._persist(gaussians, result)
            except Exception as error:
                errors.append("semantic checkpoint: {}".format(error))
                result = result._replace(errors=tuple(errors), status="failed")
        elif self.orchestrator.state.is_valid:
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.orchestrator.ledger.save(self.output_root / "ledger.json")
            self._write_sync_status(result)

        if result.status != "failed" and publication_request is not None:
            result = self._publish_pair(
                gaussians,
                result,
                publication_request,
                checkpoint,
                exact_rebuilt,
                final,
            )
        else:
            if not self.orchestrator.state.is_valid:
                self.output_root.mkdir(parents=True, exist_ok=True)
                self._write_sync_status(result)
        if self.orchestrator.state.is_valid:
            self._write_sync_status(result)
        return result

    def _publish_pair(
        self,
        gaussians,
        result: SemanticSyncResult,
        request: MapPublicationRequest,
        checkpoint: Optional[SemanticCheckpoint],
        exact_rebuilt: bool,
        final: bool,
    ) -> SemanticSyncResult:
        if not isinstance(gaussians, FrozenGaussianSnapshot):
            raise SemanticSchemaError(
                "paired publication requires one complete frozen RGB snapshot"
            )
        policy = self.publication_policy
        if (
            not final
            and policy.online_time_budget_seconds > 0
            and self._last_publication_seconds
            > policy.online_time_budget_seconds
        ):
            self._last_publication_seconds = 0.0
            return result._replace(publication_status="skipped_time_budget")

        config = request.semantic_config
        free_bytes = None
        if gaussians.get_xyz.is_cuda:
            free_bytes, _ = torch.cuda.mem_get_info(gaussians.get_xyz.device)
            if not final and free_bytes < policy.min_free_cuda_bytes:
                return result._replace(publication_status="skipped_memory_headroom")
            if (
                not final
                and free_bytes < 2 * policy.min_free_cuda_bytes
                and (config.save_diagnostics or config.save_probabilities)
            ):
                config = replace(
                    config,
                    save_diagnostics=False,
                    save_probabilities=False,
                )
        estimated_output = _estimated_pair_output_bytes(
            replace(request, semantic_config=config),
            self.dataset.class_spec.num_classes,
        )
        if (
            not final
            and estimated_output > policy.max_pair_output_bytes
            and (config.save_diagnostics or config.save_probabilities)
        ):
            config = replace(
                config,
                save_diagnostics=False,
                save_probabilities=False,
            )
            estimated_output = _estimated_pair_output_bytes(
                replace(request, semantic_config=config),
                self.dataset.class_spec.num_classes,
            )
        if not final and estimated_output > policy.max_pair_output_bytes:
            return result._replace(publication_status="skipped_output_budget")
        config = replace(
            config,
            max_output_bytes=min(
                config.max_output_bytes, policy.max_pair_output_bytes
            ),
            min_free_cuda_bytes=max(
                config.min_free_cuda_bytes, policy.min_free_cuda_bytes
            ),
        )

        started = time.perf_counter()
        errors = list(result.errors)
        try:
            projection_hash = gaussians.semantic_projection_sha256()
            rgb_hash = gaussians.rgb_model_sha256()
            if checkpoint is not None:
                if projection_hash != checkpoint.projection_geometry_hash:
                    raise SemanticSchemaError(
                        "map snapshot does not match its semantic checkpoint"
                    )
                if (
                    checkpoint.rgb_model_hash is not None
                    and rgb_hash != checkpoint.rgb_model_hash
                ):
                    raise SemanticSchemaError(
                        "map RGB snapshot does not match its semantic checkpoint"
                    )
            state_hash = semantic_state_sha256(
                self.orchestrator.state,
                self.orchestrator.recovery_context(),
                projection_hash,
                self.dataset.class_spec.ontology_fingerprint,
            )
            semantic_result = rasterize_semantic_tdom(
                gaussians,
                self.orchestrator.state,
                request.camera,
                self.dataset.class_spec,
                config=config,
            )
            elapsed = time.perf_counter() - started
            if (
                not final
                and policy.online_time_budget_seconds > 0
                and elapsed > policy.online_time_budget_seconds
            ):
                self._last_publication_seconds = elapsed
                return result._replace(
                    publication_status="skipped_time_budget"
                )
            if config.render_tile_rows:
                rgb_result = rasterize_rgb_tdom_striped(
                    gaussians,
                    request.camera,
                    request.rgb_background,
                    config,
                    debug=bool(getattr(request.rgb_pipeline, "debug", False)),
                )
            else:
                render_mask = torch.ones(
                    1, dtype=torch.bool, device=request.rgb_background.device
                )
                rgb_result = request.rgb_renderer(
                    request.camera,
                    gaussians,
                    request.rgb_pipeline,
                    request.rgb_background,
                    Render_Mask=render_mask,
                    TDOM=True,
                )["render"]
            rgb_image = tdom_to_uint8(rgb_result)
            del rgb_result
            alignment = _tdom_alignment(request.camera, request.alignment)
            shared_manifest = {
                "product_key": request.product_key,
                "geometry_stage": result.geometry_stage,
                "semantic_through_image": result.semantic_through_image,
                "label_version": self.orchestrator.label_version,
                "projection_geometry_hash": projection_hash,
                "rgb_model_hash": rgb_hash,
                "semantic_state_hash": state_hash,
                "semantic_checkpoint_hash": (
                    checkpoint.state_sha256 if checkpoint is not None else None
                ),
                "semantic_status": result.status,
                "posterior": (
                    "robust_exact_final" if exact_rebuilt else "robust_online"
                ),
                "semantic_quality": {
                    "final_requested": bool(final),
                    "exact_rebuild_succeeded": bool(exact_rebuilt),
                    "status": result.status,
                    "historical_evidence_stale": (
                        result.historical_evidence_stale
                    ),
                    "deferred_image_ids": list(result.deferred_image_ids),
                    "errors_before_publication": list(result.errors),
                    "gaussian_ids_sha256": (
                        self.orchestrator.state.gaussian_ids_sha256
                    ),
                    "topology_version": int(
                        self.orchestrator.state.topology_version
                    ),
                    "num_gaussians": int(
                        self.orchestrator.state.num_gaussians
                    ),
                },
                "alignment": alignment,
                "publication_policy": policy.to_dict(),
                "publication_resources": {
                    "free_cuda_bytes_before_render": free_bytes,
                    "diagnostics_degraded": (
                        request.diagnostics_degraded
                        or config != request.semantic_config
                    ),
                },
            }
            publish_tdom_pair(
                output_root=request.output_root,
                rgb_image=rgb_image,
                semantic_result=semantic_result,
                class_spec=self.dataset.class_spec,
                semantic_config=config,
                shared_manifest=shared_manifest,
            )
            self.orchestrator.state.mark_clean()
            self._last_publication_seconds = time.perf_counter() - started
            return result._replace(
                publication_status="complete",
                published_product_key=request.product_key,
            )
        except Exception as error:
            self.orchestrator.state.mark_dirty()
            self._last_publication_seconds = time.perf_counter() - started
            errors.append("paired TDOM publication: {}".format(error))
            return result._replace(
                errors=tuple(errors),
                status="failed",
                publication_status="failed",
            )

    def begin_final(self) -> None:
        """Close provisional evidence before final geometry refinement starts."""
        self.orchestrator.close_group()

    def _refresh_active_window(self, camera_by_id, gaussians) -> None:
        group = self.orchestrator.open_group
        if group is None or group.get(
            "projection_topology_version",
            self.orchestrator.state.topology_version,
        ) == self.orchestrator.state.topology_version and group.get(
            "projection_geometry_stage", 0
        ) == self.orchestrator.current_geometry_stage:
            return
        representatives = tuple(group.get("representative_image_ids", ()))
        projections = {}
        for image_id in representatives:
            if image_id not in camera_by_id:
                raise SemanticSchemaError(
                    "active semantic keyframe {} has no camera".format(image_id)
                )
            observation = self.dataset.load_observation(image_id)
            projections[image_id] = self.projector.project(
                gaussians, camera_by_id[image_id], observation
            )
        self.orchestrator.refresh_open_group(projections)

    def _verify_recorded_metadata(self, image_id: str) -> None:
        label_path = self.dataset.semantic_root / "labels" / (image_id + ".png")
        metadata_path = (
            self.dataset.semantic_root / "metadata" / (image_id + ".json")
        )
        score_path = self.dataset.semantic_root / "scores" / (image_id + ".npy")
        paths = [label_path, metadata_path]
        if score_path.is_file():
            paths.append(score_path)
        try:
            signature = tuple(
                (str(path), path.stat().st_size, path.stat().st_mtime_ns)
                for path in paths
            )
        except OSError as error:
            raise SemanticSchemaError(
                "unable to stat recorded semantic files: {}".format(error)
            )
        if self._verified_recorded_files.get(image_id) == signature:
            return
        observation = self.dataset.load_observation(image_id)
        metadata = observation.metadata
        version = observation.key.label_version
        previous = self.orchestrator.ledger.get(image_id, version)
        if previous is None:
            raise SemanticSchemaError(
                "label version changed; explicit semantic rebuild is required"
            )
        candidate = LedgerEntry(
            image_id=image_id,
            label_version=version,
            label_sha256=metadata.get("label_sha256"),
            score_sha256=metadata.get("score_sha256"),
        )
        self.orchestrator.ledger.record(candidate)
        self._verified_recorded_files[image_id] = signature

    def _persist(
        self, gaussians, result: SemanticSyncResult
    ) -> SemanticCheckpoint:
        self.output_root.mkdir(parents=True, exist_ok=True)
        attributes = _projection_attributes(gaussians)
        projection_geometry_hash = semantic_projection_sha256(attributes)
        rgb_hash = (
            gaussians.rgb_model_sha256()
            if isinstance(gaussians, FrozenGaussianSnapshot)
            else None
        )
        current_attributes = _projection_attributes(gaussians)
        if projection_geometry_hash != semantic_projection_sha256(
            current_attributes
        ):
            raise SemanticSchemaError(
                "frozen geometry changed while semantic state was persisted"
            )
        context = self.orchestrator.recovery_context()
        latest_path = self.output_root / "latest.json"
        current_slot = 1
        if latest_path.is_file():
            try:
                with latest_path.open("r", encoding="utf-8") as stream:
                    current_slot = int(json.load(stream)["slot"])
            except (OSError, ValueError, KeyError, TypeError):
                current_slot = 1
        slot = 1 - current_slot if current_slot in (0, 1) else 0
        state_path = self.output_root / "semantic_state.{}.pt".format(slot)
        context_path = self.output_root / "recovery_context.{}.json".format(slot)
        geometry_path = self.output_root / "geometry_checkpoint.{}.pt".format(slot)
        self.orchestrator.state.save(
            state_path,
            projection_geometry_hash=projection_geometry_hash,
            rgb_model_hash=rgb_hash,
            ontology_hash=self.dataset.class_spec.ontology_fingerprint,
            palette_hash=self.dataset.class_spec.palette_fingerprint,
            recovery_context=context,
        )
        state_hash = sha256_file(state_path)
        context_hash = SemanticGaussianState.validated_recovery_context_sha256(
            context,
            ontology_hash=self.dataset.class_spec.ontology_fingerprint,
            prior_total=self.orchestrator.config.prior_total,
            num_classes=self.dataset.class_spec.num_classes,
        )
        self._save_json_atomic(context_path, context)
        self._save_torch_atomic(
            geometry_path,
            {
                "schema_version": 3,
                "semantic_method_version": SEMANTIC_METHOD_VERSION,
                "projection_geometry_hash": projection_geometry_hash,
                "rgb_model_hash": rgb_hash,
                "attributes": attributes,
            },
        )
        geometry_checkpoint_hash = sha256_file(geometry_path)
        self._save_json_atomic(
            latest_path,
            {
                "schema_version": 5,
                "semantic_method_version": SEMANTIC_METHOD_VERSION,
                "slot": slot,
                "projection_geometry_hash": projection_geometry_hash,
                "rgb_model_hash": rgb_hash,
                "ontology_hash": self.dataset.class_spec.ontology_fingerprint,
                "palette_hash": self.dataset.class_spec.palette_fingerprint,
                "geometry_checkpoint": geometry_path.name,
                "geometry_checkpoint_sha256": geometry_checkpoint_hash,
                "semantic_state": state_path.name,
                "semantic_state_sha256": state_hash,
                "recovery_context_sha256": context_hash,
            },
        )
        self.orchestrator.state.recovery_context = context
        self.orchestrator.ledger.save(self.output_root / "ledger.json")
        self._write_sync_status(result)
        return SemanticCheckpoint(
            state_path,
            state_hash,
            projection_geometry_hash,
            rgb_hash,
        )

    def _write_sync_status(self, result: SemanticSyncResult) -> None:
        payload = {
            "schema_version": 3,
            "semantic_method_version": SEMANTIC_METHOD_VERSION,
            "geometry_stage": result.geometry_stage,
            "semantic_through_image": result.semantic_through_image,
            "label_version": self.orchestrator.label_version,
            "processed_image_ids": list(result.processed_image_ids),
            "deferred_image_ids": list(result.deferred_image_ids),
            "lag_batches": result.lag_batches,
            "within_lag_budget": result.within_lag_budget,
            "errors": list(result.errors),
            "status": result.status,
            "historical_evidence_stale": result.historical_evidence_stale,
            "publication_status": result.publication_status,
            "published_product_key": result.published_product_key,
            "publication_policy": self.publication_policy.to_dict(),
            "stage_boundaries": list(self._stage_boundaries),
        }
        self._save_json_atomic(
            self.output_root / "sync_status.json", payload
        )

    @staticmethod
    def _save_torch_atomic(path: Path, payload: Mapping[str, object]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))

    @staticmethod
    def _save_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))


class AsyncProgressiveSemanticHook:
    """Serialize semantic work off-thread while geometry keeps running."""

    def __init__(self, core: ProgressiveSemanticHook):
        self.core = core
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="semantic-atdom"
        )
        self._stage_pending = False
        self._pending_stage_count = 0
        self._latest_stage_token = 0
        self._latest_stage_request = None
        self._last_result = None
        self._error = None
        self._is_valid = True
        self._expected_gaussian_count = core.orchestrator.state.num_gaussians
        self._stage_boundaries = []
        self._closed = False
        self._lock = threading.Lock()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._semantic_stream = None
        state_device = getattr(self.orchestrator.state, "device", torch.device("cpu"))
        if torch.device(state_device).type == "cuda":
            self._semantic_stream = torch.cuda.Stream(
                device=state_device,
                priority=core.publication_policy.cuda_stream_priority,
            )

    @classmethod
    def create(cls, *args, **kwargs) -> "AsyncProgressiveSemanticHook":
        return cls(ProgressiveSemanticHook.create(*args, **kwargs))

    @property
    def orchestrator(self):
        return self.core.orchestrator

    @property
    def num_gaussians(self) -> int:
        return self._expected_gaussian_count

    @property
    def is_valid(self) -> bool:
        return self._is_valid and self.orchestrator.state.is_valid

    def invalidate(self, reason: object) -> None:
        self._is_valid = False
        self.orchestrator.state.invalidate(reason)

    def apply_topology(self, delta) -> None:
        """Queue topology in the same order as stage snapshots without waiting."""
        if self._closed:
            raise RuntimeError("asynchronous semantic hook is closed")
        if not self.is_valid:
            raise RuntimeError("asynchronous semantic state requires rebuild")
        delta.validate()
        if delta.old_count != self._expected_gaussian_count:
            raise ValueError(
                "semantic topology expected {}, received {}".format(
                    self._expected_gaussian_count, delta.old_count
                )
            )
        self._expected_gaussian_count = delta.new_count
        future = self._executor.submit(
            self._execute_callable, self.orchestrator.state.apply_topology, delta
        )
        future.add_done_callback(self._background_done)

    def begin_final(self) -> None:
        if self._closed:
            raise RuntimeError("asynchronous semantic hook is closed")
        future = self._executor.submit(self._execute_callable, self.core.begin_final)
        future.add_done_callback(self._background_done)

    def close(self, reason: Optional[object] = None, wait: bool = True) -> None:
        """Idempotently release the worker and optionally invalidate its state."""
        if reason is not None:
            self.invalidate(reason)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._latest_stage_request = None
        self._executor.shutdown(wait=wait)

    def synchronize(
        self,
        geometry_stage: int,
        cameras: Sequence[object],
        gaussians,
        snapshot_root: Path,
        final: bool = False,
        publication_request: Optional[MapPublicationRequest] = None,
    ) -> SemanticSyncResult:
        if self._closed:
            raise RuntimeError("asynchronous semantic hook is closed")
        if not final:
            self._raise_background_error()
        camera_ids = tuple(_image_id(item.image_name) for item in cameras)
        if camera_ids:
            boundary = max(
                camera_ids, key=_acquisition_key
            )
            if not self._stage_boundaries or self._stage_boundaries[-1] != boundary:
                self._stage_boundaries.append(boundary)

        if not final:
            with self._lock:
                pending_publication = (
                    None
                    if self._latest_stage_request is None
                    else self._latest_stage_request.args[6]
                )
                if publication_request is None and pending_publication is not None:
                    frozen_cameras = tuple(
                        FrozenCameraSnapshot(item) for item in cameras
                    )
                    request = _PendingStageRequest(
                        (
                            geometry_stage,
                            frozen_cameras,
                            FrozenSemanticSnapshot(gaussians),
                            snapshot_root,
                            False,
                            tuple(self._stage_boundaries),
                            None,
                        )
                    )
                    token = None
                    queued_product_key = pending_publication.product_key
                elif (
                    publication_request is not None
                    and not publication_request.publication_due
                ):
                    publication_request = (
                        replace(publication_request, publication_due=True)
                        if pending_publication is not None
                        else None
                    )
                if not (
                    publication_request is None
                    and pending_publication is not None
                ):
                    frozen_publication, publication_status_override = (
                        self._freeze_publication_request(
                            publication_request, gaussians, final=False
                        )
                    )
                    frozen_cameras = tuple(
                        FrozenCameraSnapshot(item) for item in cameras
                    )
                    snapshot = (
                        FrozenGaussianSnapshot(gaussians)
                        if frozen_publication is not None
                        else FrozenSemanticSnapshot(gaussians)
                    )
                    request = _PendingStageRequest(
                        (
                            geometry_stage,
                            frozen_cameras,
                            snapshot,
                            snapshot_root,
                            False,
                            tuple(self._stage_boundaries),
                            frozen_publication,
                        ),
                        publication_status_override,
                    )
                    # Submit an ordered marker for every replaceable snapshot.
                    # Its queue position prevents crossing later topology events.
                    self._latest_stage_token += 1
                    token = self._latest_stage_token
                    self._latest_stage_request = request
                    queued_product_key = (
                        None
                        if frozen_publication is None
                        else frozen_publication.product_key
                    )
                self._pending_stage_count += 1
                self._stage_pending = True
                self._idle_event.clear()
            future = self._executor.submit(
                self._run_stage_marker if token is not None else self._run_stage,
                token if token is not None else request,
            )
            future.add_done_callback(self._stage_done)
            status = self._current_status(geometry_stage)
            if queued_product_key is not None:
                status = status._replace(
                    publication_status="queued",
                    published_product_key=queued_product_key,
                )
            elif request.publication_status_override is not None:
                status = status._replace(
                    publication_status=request.publication_status_override
                )
            return status

        if final:
            frozen_publication, publication_status_override = (
                self._freeze_publication_request(
                    publication_request, gaussians, final=True
                )
            )
            frozen_cameras = tuple(FrozenCameraSnapshot(item) for item in cameras)
            snapshot = (
                FrozenGaussianSnapshot(gaussians)
                if frozen_publication is not None
                else FrozenSemanticSnapshot(gaussians)
            )
            try:
                self._idle_event.wait()
                self._executor.submit(lambda: None).result()
                result = self._executor.submit(
                    self._execute_core,
                    geometry_stage,
                    frozen_cameras,
                    snapshot,
                    snapshot_root,
                    True,
                    tuple(self._stage_boundaries),
                    frozen_publication,
                ).result()
                if publication_status_override is not None:
                    result = result._replace(
                        publication_status=publication_status_override
                    )
                    status_writer = getattr(self.core, "_write_sync_status", None)
                    if callable(status_writer):
                        status_writer(result)
                self._last_result = result
                self._stage_pending = False
                return result
            except Exception as error:
                self.invalidate(error)
                raise
            finally:
                self.close(wait=True)

    def _freeze_publication_request(
        self,
        request: Optional[MapPublicationRequest],
        gaussians,
        final: bool,
    ) -> Tuple[Optional[MapPublicationRequest], Optional[str]]:
        if request is None:
            return None, None
        config = request.semantic_config
        estimated_output = _estimated_pair_output_bytes(
            request, self.core.dataset.class_spec.num_classes
        )
        if (
            estimated_output > self.core.publication_policy.max_pair_output_bytes
            and (config.save_diagnostics or config.save_probabilities)
        ):
            config = replace(
                config,
                save_diagnostics=False,
                save_probabilities=False,
            )
            request = replace(
                request,
                semantic_config=config,
                diagnostics_degraded=True,
            )
            estimated_output = _estimated_pair_output_bytes(
                request, self.core.dataset.class_spec.num_classes
            )
        if estimated_output > self.core.publication_policy.max_pair_output_bytes:
            return None, "skipped_output_budget"
        if gaussians.get_xyz.is_cuda:
            snapshot_bytes = _model_snapshot_nbytes(
                gaussians, RGB_MODEL_ATTRIBUTE_NAMES
            )
            free_bytes, _ = torch.cuda.mem_get_info(gaussians.get_xyz.device)
            if (
                free_bytes
                < self.core.publication_policy.min_free_cuda_bytes
                + snapshot_bytes
            ):
                return None, "skipped_memory_headroom"
        if isinstance(request.camera, FrozenCameraSnapshot):
            return request, None
        return (
            replace(
                request,
                camera=FrozenCameraSnapshot(request.camera),
                rgb_background=request.rgb_background.detach().clone(),
                alignment=dict(request.alignment),
            ),
            None,
        )

    def _run_stage_marker(self, token):
        with self._lock:
            if token != self._latest_stage_token:
                return None
            request = self._latest_stage_request
            self._latest_stage_request = None
        if request is None:
            return None
        return self._run_stage(request)

    def _run_stage(self, request: _PendingStageRequest):
        result = self._execute_core(*request.args)
        if request.publication_status_override is not None:
            result = result._replace(
                publication_status=request.publication_status_override
            )
            status_writer = getattr(self.core, "_write_sync_status", None)
            if callable(status_writer):
                status_writer(result)
        return result

    def _execute_core(self, *request):
        snapshot = request[2]
        if self._semantic_stream is None:
            snapshot.wait_ready()
            return self.core.synchronize(*request)
        with torch.cuda.stream(self._semantic_stream):
            snapshot.wait_ready(self._semantic_stream)
            result = self.core.synchronize(*request)
        self._semantic_stream.synchronize()
        return result

    def _execute_callable(self, function, *args):
        if self._semantic_stream is None:
            return function(*args)
        with torch.cuda.stream(self._semantic_stream):
            result = function(*args)
        self._semantic_stream.synchronize()
        return result

    def _stage_done(self, future) -> None:
        try:
            result = future.result()
            if result is not None:
                self._last_result = result
        except Exception as error:
            self._error = error
            self.invalidate(error)
        with self._lock:
            self._pending_stage_count -= 1
            if self._pending_stage_count == 0:
                self._stage_pending = False
                self._idle_event.set()

    def _background_done(self, future) -> None:
        try:
            future.result()
        except Exception as error:
            self._error = error
            self.invalidate(error)

    def _raise_background_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "asynchronous semantic worker failed: {}".format(self._error)
            )

    def _current_status(self, geometry_stage: int) -> SemanticSyncResult:
        through = (
            None
            if self._last_result is None
            else self._last_result.semantic_through_image
        )
        if through is None:
            lag = len(self._stage_boundaries)
        else:
            through_key = _acquisition_key(through)
            lag = sum(
                _acquisition_key(boundary) > through_key
                for boundary in self._stage_boundaries
            )
        errors = (
            () if self._last_result is None else self._last_result.errors
        )
        if self._error is not None:
            errors = errors + (str(self._error),)
        historical_stale = self.orchestrator.state.historical_evidence_stale
        if not self.is_valid:
            status = "failed"
        elif historical_stale or errors or lag > 1:
            status = "stale_online"
        else:
            status = "approx_online"
        return SemanticSyncResult(
            int(geometry_stage),
            through,
            (),
            (),
            lag,
            lag <= 1,
            errors,
            status,
            historical_stale,
        )
