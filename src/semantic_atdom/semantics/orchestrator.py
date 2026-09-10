"""Stage-synchronous online semantic fusion for progressive A-TDOM."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
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
    sha256_file,
)
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    rasterize_semantic_tdom,
    write_semantic_tdom,
)


def _image_id(value: object) -> str:
    result = Path(str(value)).stem
    if not result:
        raise ValueError("image id must not be empty")
    return result


def _acquisition_key(image_id: str):
    match = re.search(r"(\d+)$", image_id)
    if match is None:
        return (1, image_id)
    return (0, int(match.group(1)), image_id)


_GEOMETRY_ATTRIBUTE_NAMES = (
    "xyz",
    "features_dc",
    "features_rest",
    "opacity",
    "scaling",
    "rotation",
)


def _geometry_attributes(gaussians) -> Dict[str, torch.Tensor]:
    cached = getattr(gaussians, "geometry_attributes_cpu", None)
    if callable(cached):
        return cached()
    if hasattr(gaussians, "_values"):
        source = gaussians._values
        return {
            name: source[name].detach().cpu().contiguous()
            for name in _GEOMETRY_ATTRIBUTE_NAMES
        }
    return {
        name: getattr(gaussians, "_" + name).detach().cpu().contiguous()
        for name in _GEOMETRY_ATTRIBUTE_NAMES
    }


def _geometry_attributes_sha256(attributes: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"semantic-atdom-geometry-v1\0")
    for name in _GEOMETRY_ATTRIBUTE_NAMES:
        value = attributes[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tdom_alignment(camera, supplied: Optional[Mapping[str, object]] = None):
    """Return JSON-safe camera data shared by RGB and semantic TDOM products."""
    alignment = dict(supplied or {})
    alignment.update(
        {
            "schema_version": 1,
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

    @classmethod
    def load(cls, snapshot_root: Path) -> "AcquisitionOverlap":
        sparse_root = Path(snapshot_root) / "sparse" / "0"
        names_path = sparse_root / "imagesNames.txt"
        matrix_path = sparse_root / "imageMatchMatrix.txt"
        try:
            raw_names = names_path.read_text(encoding="utf-8")
            names = tuple(
                _image_id(item)
                for item in re.split(r"[,\s]+", raw_names.strip())
                if item
            )
            matrix = np.loadtxt(str(matrix_path), delimiter=",", dtype=np.float64)
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to load COLMAP overlap inputs from {}: {}".format(
                    sparse_root, error
                )
            )
        if len(names) == 0 or len(names) != len(set(names)):
            raise SemanticSchemaError(
                "imagesNames.txt must contain unique image ids"
            )
        if matrix.ndim == 0 and len(names) == 1:
            matrix = matrix.reshape(1, 1)
        elif matrix.ndim == 1 and len(names) == 1:
            matrix = matrix.reshape(1, 1)
        if matrix.shape != (len(names), len(names)):
            raise SemanticSchemaError(
                "imageMatchMatrix shape {} does not match {} names".format(
                    matrix.shape, len(names)
                )
            )
        if not bool(np.isfinite(matrix).all()) or bool((matrix < 0).any()):
            raise SemanticSchemaError(
                "imageMatchMatrix must contain finite non-negative counts"
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
                "image {} is absent from imagesNames.txt".format(error.args[0])
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
        if self.boundary_bandwidth < 0:
            raise ValueError("boundary_bandwidth must be non-negative")
        if not 0 <= self.boundary_floor <= 1:
            raise ValueError("boundary_floor must be in [0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "overlap_threshold": self.overlap_threshold,
            "max_group_size": self.max_group_size,
            "reliability_floor": self.reliability_floor,
            "temperature": self.temperature,
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


class FrozenGaussianSnapshot:
    """Detached geometry copy that remains stable while RGB training continues."""

    _ATTRIBUTE_NAMES = _GEOMETRY_ATTRIBUTE_NAMES

    def __init__(self, gaussians):
        self._values = {
            name: getattr(gaussians, "_" + name).detach().clone()
            for name in self._ATTRIBUTE_NAMES
        }
        self._cpu_values = None
        self._geometry_hash = None

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

    def geometry_sha256(self) -> str:
        if self._geometry_hash is None:
            self._geometry_hash = _geometry_attributes_sha256(
                self.geometry_attributes_cpu()
            )
        return self._geometry_hash

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
        consensus = closed_raw / (
            closed_raw.sum(dim=1, keepdim=True) + self.config.eps
        )
        js = self._js_divergence(group.distribution, consensus)
        reliability = self.config.reliability_floor + (
            1.0 - self.config.reliability_floor
        ) * torch.exp(-js / self.config.temperature)
        eligible = (closed_count >= 2) & (group.support > self.config.eps)
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
        dirty = self.state.dirty
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
        consensus = closed_raw / (
            closed_raw.sum(dim=1, keepdim=True) + self.config.eps
        )
        reliability = self.config.reliability_floor + (
            1.0 - self.config.reliability_floor
        ) * torch.exp(
            -self._js_divergence(group.distribution, consensus)
            / self.config.temperature
        )
        eligible = (closed_count >= 2) & (group.support > self.config.eps)
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
        self.state.mark_clean(dirty)
        return True

    def recovery_context(self) -> Dict[str, object]:
        return {
            "ledger": {
                "schema_version": 1,
                "observations": [
                    entry.to_dict() for entry in self.ledger.entries
                ],
            },
            "label_provenance": self.label_provenance,
            "open_group": self.open_group,
            "class_spec": self.class_spec.to_dict(),
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
    ):
        self.dataset = dataset
        self.orchestrator = orchestrator
        self.projector = projector
        self.output_root = Path(output_root)
        self._stage_boundaries = []
        self._verified_recorded_files = {}

    @classmethod
    def create(
        cls,
        dataset_root: Path,
        output_root: Path,
        gaussians,
        config: Optional[OnlineSemanticConfig] = None,
        projector=None,
        semantic_dataset_root: Optional[Path] = None,
    ) -> "ProgressiveSemanticHook":
        config = config or OnlineSemanticConfig()
        dataset = SemanticDataset(semantic_dataset_root or dataset_root)
        output_root = Path(output_root)
        latest_path = output_root / "latest.json"
        if latest_path.is_file():
            try:
                with latest_path.open("r", encoding="utf-8") as stream:
                    latest = json.load(stream)
                if (
                    latest.get("schema_version") != 2
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
                checkpoint_geometry_hash = _geometry_attributes_sha256(attributes)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                raise SemanticSchemaError(
                    "unable to validate geometry checkpoint: {}".format(error)
                )
            if (
                geometry_payload.get("schema_version") != 1
                or geometry_payload.get("geometry_hash")
                != checkpoint_geometry_hash
                or latest.get("geometry_hash") != checkpoint_geometry_hash
                or gaussians.geometry_sha256() != checkpoint_geometry_hash
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
                class_spec_hash=dataset.class_spec.fingerprint,
                prior_total=config.prior_total,
                num_classes=dataset.class_spec.num_classes,
            )
            if latest.get("recovery_context_sha256") != recovery_hash:
                raise SemanticSchemaError(
                    "latest semantic recovery context hash does not match"
                )
            state = SemanticGaussianState.load(
                state_path,
                expected_geometry_hash=gaussians.geometry_sha256(),
                expected_class_spec_hash=dataset.class_spec.fingerprint,
                expected_gaussian_count=gaussians.get_xyz.shape[0],
                expected_recovery_context_sha256=recovery_hash,
                map_location=gaussians.get_xyz.device,
            )
            orchestrator = OnlineSemanticOrchestrator.from_state(
                state, dataset.class_spec, config
            )
        elif any(
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
        hook = cls(dataset, orchestrator, projector, output_root)
        status_path = output_root / "sync_status.json"
        if status_path.is_file():
            try:
                with status_path.open("r", encoding="utf-8") as stream:
                    status = json.load(stream)
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

    def synchronize(
        self,
        geometry_stage: int,
        cameras: Sequence[object],
        gaussians,
        snapshot_root: Path,
        final: bool = False,
        stage_boundaries: Optional[Sequence[str]] = None,
        tdom_camera: Optional[object] = None,
        tdom_output_root: Optional[Path] = None,
        tdom_config: Optional[SemanticTdomConfig] = None,
        tdom_alignment: Optional[Mapping[str, object]] = None,
    ) -> SemanticSyncResult:
        geometry_stage = int(geometry_stage)
        if geometry_stage != self.orchestrator.current_geometry_stage:
            self.orchestrator.state.mark_dirty()
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
        try:
            self._refresh_active_window(camera_by_id, gaussians)
        except Exception as error:
            errors.append("active-window refresh: {}".format(error))
        ledger_images = {
            entry.image_id for entry in self.orchestrator.ledger.entries
        }
        for position, image_id in enumerate(ordered):
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
        if final:
            self.orchestrator.close_group()
            if ordered and not deferred and not errors:
                try:
                    rebuilt = FinalSemanticRebuilder(
                        self.dataset,
                        self.projector,
                        self.orchestrator.config,
                    ).rebuild(
                        ordered,
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
                            "schema_version": 1,
                            "geometry_stage": geometry_stage,
                            "geometry_hash": gaussians.geometry_sha256(),
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
                except Exception as error:
                    errors.append("final exact rebuild: {}".format(error))

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
        result = SemanticSyncResult(
            int(geometry_stage),
            semantic_through,
            tuple(processed),
            tuple(deferred),
            lag,
            lag <= 1,
            tuple(errors),
        )
        if tdom_camera is not None and tdom_output_root is not None:
            try:
                tdom_result = rasterize_semantic_tdom(
                    gaussians,
                    self.orchestrator.state,
                    tdom_camera,
                    self.dataset.class_spec,
                    config=tdom_config,
                )
                write_semantic_tdom(
                    tdom_result,
                    tdom_output_root,
                    self.dataset.class_spec,
                    {
                        "geometry_stage": result.geometry_stage,
                        "semantic_through_image": result.semantic_through_image,
                        "label_version": self.orchestrator.label_version,
                        "geometry_hash": gaussians.geometry_sha256(),
                        "posterior": (
                            "robust_exact_final"
                            if exact_rebuilt
                            else "robust_online"
                        ),
                        "semantic_quality": {
                            "final_requested": bool(final),
                            "exact_rebuild_succeeded": bool(exact_rebuilt),
                            "status": (
                                "exact_final"
                                if exact_rebuilt
                                else (
                                    "online_fallback"
                                    if final
                                    else "online"
                                )
                            ),
                            "deferred_image_ids": list(deferred),
                            "errors_before_tdom": list(errors),
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
                        "alignment": _tdom_alignment(
                            tdom_camera, tdom_alignment
                        ),
                    },
                    config=tdom_config,
                )
                self.orchestrator.state.mark_clean()
            except Exception as error:
                errors.append("semantic TDOM render: {}".format(error))
                result = result._replace(errors=tuple(errors))
        self._persist(gaussians, result)
        return result

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

    def _persist(self, gaussians, result: SemanticSyncResult) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        attributes = _geometry_attributes(gaussians)
        geometry_hash = _geometry_attributes_sha256(attributes)
        if geometry_hash != gaussians.geometry_sha256():
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
            geometry_hash=geometry_hash,
            class_spec_hash=self.dataset.class_spec.fingerprint,
            recovery_context=context,
        )
        context_hash = SemanticGaussianState.validated_recovery_context_sha256(
            context,
            class_spec_hash=self.dataset.class_spec.fingerprint,
            prior_total=self.orchestrator.config.prior_total,
            num_classes=self.dataset.class_spec.num_classes,
        )
        self._save_json_atomic(context_path, context)
        self._save_torch_atomic(
            geometry_path,
            {
                "schema_version": 1,
                "geometry_hash": geometry_hash,
                "attributes": attributes,
            },
        )
        geometry_checkpoint_hash = sha256_file(geometry_path)
        self._save_json_atomic(
            latest_path,
            {
                "schema_version": 2,
                "slot": slot,
                "geometry_hash": geometry_hash,
                "geometry_checkpoint": geometry_path.name,
                "geometry_checkpoint_sha256": geometry_checkpoint_hash,
                "recovery_context_sha256": context_hash,
            },
        )
        self.orchestrator.state.recovery_context = context
        self.orchestrator.ledger.save(self.output_root / "ledger.json")
        payload = {
            "schema_version": 1,
            "geometry_stage": result.geometry_stage,
            "semantic_through_image": result.semantic_through_image,
            "label_version": self.orchestrator.label_version,
            "processed_image_ids": list(result.processed_image_ids),
            "deferred_image_ids": list(result.deferred_image_ids),
            "lag_batches": result.lag_batches,
            "within_lag_budget": result.within_lag_budget,
            "errors": list(result.errors),
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
            self.orchestrator.state.apply_topology, delta
        )
        future.add_done_callback(self._background_done)

    def begin_final(self) -> None:
        if self._closed:
            raise RuntimeError("asynchronous semantic hook is closed")
        future = self._executor.submit(self.core.begin_final)
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
        tdom_camera: Optional[object] = None,
        tdom_output_root: Optional[Path] = None,
        tdom_config: Optional[SemanticTdomConfig] = None,
        tdom_alignment: Optional[Mapping[str, object]] = None,
    ) -> SemanticSyncResult:
        if self._closed:
            raise RuntimeError("asynchronous semantic hook is closed")
        self._raise_background_error()
        camera_ids = tuple(_image_id(item.image_name) for item in cameras)
        if camera_ids:
            boundary = max(
                camera_ids, key=_acquisition_key
            )
            if not self._stage_boundaries or self._stage_boundaries[-1] != boundary:
                self._stage_boundaries.append(boundary)

        frozen_cameras = tuple(FrozenCameraSnapshot(item) for item in cameras)
        frozen_tdom_camera = (
            None if tdom_camera is None else FrozenCameraSnapshot(tdom_camera)
        )
        snapshot = FrozenGaussianSnapshot(gaussians)
        request = (
            geometry_stage,
            frozen_cameras,
            snapshot,
            snapshot_root,
            False,
            tuple(self._stage_boundaries),
            frozen_tdom_camera,
            tdom_output_root,
            tdom_config,
            tdom_alignment,
        )
        if not final:
            with self._lock:
                # Submit an ordered marker for every snapshot.  The marker may
                # skip obsolete semantic work, but its position in the single
                # worker queue prevents a frozen snapshot from crossing later
                # topology events.
                self._latest_stage_token += 1
                token = self._latest_stage_token
                # Retain only the newest heavyweight GPU snapshot.  Ordered
                # markers carry a token, not a second reference to old tensors.
                self._latest_stage_request = request
                self._pending_stage_count += 1
                self._stage_pending = True
                self._idle_event.clear()
            future = self._executor.submit(
                self._run_stage_marker, token
            )
            future.add_done_callback(self._stage_done)
            return self._current_status(geometry_stage)

        if final:
            try:
                self._idle_event.wait()
                self._executor.submit(lambda: None).result()
                self._raise_background_error()
                result = self._executor.submit(
                    self.core.synchronize,
                    geometry_stage,
                    frozen_cameras,
                    snapshot,
                    snapshot_root,
                    True,
                    tuple(self._stage_boundaries),
                    frozen_tdom_camera,
                    tdom_output_root,
                    tdom_config,
                    tdom_alignment,
                ).result()
                self._last_result = result
                self._stage_pending = False
                return result
            except Exception as error:
                self.invalidate(error)
                raise
            finally:
                self.close(wait=True)

    def _run_stage_marker(self, token):
        with self._lock:
            if token != self._latest_stage_token:
                return None
            request = self._latest_stage_request
        if request is None:
            return None
        try:
            return self.core.synchronize(*request)
        finally:
            with self._lock:
                if (
                    token == self._latest_stage_token
                    and self._latest_stage_request is request
                ):
                    self._latest_stage_request = None

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
        return SemanticSyncResult(
            int(geometry_stage),
            through,
            (),
            (),
            lag,
            lag <= 1,
            errors,
        )
