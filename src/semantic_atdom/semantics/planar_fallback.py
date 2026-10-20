"""Conservative multi-view fallback for planar semantic surfaces.

The Gaussian TDOM remains authoritative wherever trusted geometry passes its
support gates.  This module only considers bounded holes in that map and only
publishes a fallback class after independent source-observation groups agree.
Boundary-connected no-data can therefore never be converted into a class.
"""

from dataclasses import asdict, dataclass
import math
import time
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence, Tuple

import cv2
import numpy as np
import torch

from semantic_atdom.semantics.label_grid import labels_on_camera_grid


@dataclass(frozen=True)
class PlanarFallbackConfig:
    class_name: str = ""
    min_component_pixels: int = 1024
    ring_radius_pixels: int = 15
    min_boundary_consensus: float = 0.98
    plane_context_pixels: int = 512
    min_plane_gaussians: int = 32
    max_plane_mad_world: float = 1.0
    min_views: int = 5
    min_agreement: float = 0.95
    min_groups: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str):
            raise ValueError("class_name must be a string")
        for name in (
            "min_component_pixels",
            "ring_radius_pixels",
            "plane_context_pixels",
            "min_plane_gaussians",
            "min_views",
            "min_groups",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        for name in ("min_boundary_consensus", "min_agreement"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0 or value > 1:
                raise ValueError("{} must be in (0, 1]".format(name))
        if (
            not math.isfinite(self.max_plane_mad_world)
            or self.max_plane_mad_world <= 0
        ):
            raise ValueError("max_plane_mad_world must be positive")

    @property
    def enabled(self) -> bool:
        return bool(self.class_name)

    def to_dict(self):
        return asdict(self)


class PlanarCandidate(NamedTuple):
    component_id: int
    area: int
    bbox: Tuple[int, int, int, int]
    boundary_consensus: float
    rows: np.ndarray
    columns: np.ndarray


def find_bounded_class_candidates(
    class_ids: np.ndarray,
    *,
    ignore_id: int,
    target_class_id: int,
    config: PlanarFallbackConfig,
) -> Tuple[PlanarCandidate, ...]:
    """Return internal unknown components enclosed by one target class.

    A component touching any image edge is rejected before semantic boundary
    inspection.  This topological invariant is the hard guard preventing a
    large Gaussian from filling the unobserved outer canvas.
    """
    values = np.asarray(class_ids)
    if values.ndim != 2:
        raise ValueError("class_ids must be two-dimensional")
    if not config.enabled:
        return ()
    unknown = (values == int(ignore_id)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        unknown, connectivity=8
    )
    height, width = values.shape
    candidates = []
    radius = int(config.ring_radius_pixels)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    for component_id in range(1, count):
        x, y, box_width, box_height, area = (
            int(item) for item in stats[component_id]
        )
        if area < config.min_component_pixels:
            continue
        if (
            x == 0
            or y == 0
            or x + box_width == width
            or y + box_height == height
        ):
            continue
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(width, x + box_width + radius)
        y1 = min(height, y + box_height + radius)
        local_component = (
            labels[y:y + box_height, x:x + box_width] == component_id
        )
        expanded = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        expanded[
            y - y0:y - y0 + box_height,
            x - x0:x - x0 + box_width,
        ] = local_component
        ring = cv2.dilate(expanded, kernel) > 0
        ring &= expanded == 0
        ring_values = values[y0:y1, x0:x1][ring]
        ring_values = ring_values[ring_values != int(ignore_id)]
        if ring_values.size == 0:
            continue
        consensus = float(np.mean(ring_values == int(target_class_id)))
        if consensus < config.min_boundary_consensus:
            continue
        local_rows, local_columns = np.nonzero(local_component)
        candidates.append(
            PlanarCandidate(
                component_id=component_id,
                area=area,
                bbox=(x, y, box_width, box_height),
                boundary_consensus=consensus,
                rows=(local_rows + y).astype(np.int32, copy=False),
                columns=(local_columns + x).astype(np.int32, copy=False),
            )
        )
    return tuple(candidates)


def accept_planar_votes(
    target_votes: np.ndarray,
    valid_votes: np.ndarray,
    target_groups: np.ndarray,
    observed_groups: np.ndarray,
    config: PlanarFallbackConfig,
):
    """Apply the auditable per-pixel multi-view acceptance gate."""
    target = np.asarray(target_votes)
    valid = np.asarray(valid_votes)
    target_group_count = np.asarray(target_groups)
    observed_group_count = np.asarray(observed_groups)
    if not (
        target.shape
        == valid.shape
        == target_group_count.shape
        == observed_group_count.shape
    ):
        raise ValueError("planar vote arrays must have identical shapes")
    agreement = np.divide(
        target,
        valid,
        out=np.zeros(target.shape, dtype=np.float64),
        where=valid > 0,
    )
    accepted = (
        (valid >= config.min_views)
        & (agreement >= config.min_agreement)
        & (observed_group_count >= config.min_groups)
        & (target_group_count >= config.min_groups)
    )
    return accepted, agreement


class PlanarFallbackOutcome(NamedTuple):
    result: object
    metrics: Mapping[str, object]


def _image_id(value: object) -> str:
    return Path(str(value)).stem


def _project_points(camera, world_points: np.ndarray):
    matrix = camera.full_proj_transform.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    homogeneous = np.concatenate(
        (
            np.asarray(world_points, dtype=np.float64),
            np.ones((len(world_points), 1), dtype=np.float64),
        ),
        axis=1,
    )
    clip = homogeneous @ matrix
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 1e-8)
    ndc = np.zeros((len(world_points), 2), dtype=np.float64)
    ndc[valid] = clip[valid, :2] / clip[valid, 3:4]
    columns = ((ndc[:, 0] + 1.0) * int(camera.image_width) - 1.0) * 0.5
    rows = ((ndc[:, 1] + 1.0) * int(camera.image_height) - 1.0) * 0.5
    valid &= (
        (columns >= 0)
        & (columns < int(camera.image_width))
        & (rows >= 0)
        & (rows < int(camera.image_height))
    )
    return columns, rows, valid


def _unproject_pixels_to_plane(
    camera,
    rows: np.ndarray,
    columns: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    width = int(camera.image_width)
    height = int(camera.image_height)
    inverse = np.linalg.inv(
        camera.full_proj_transform.detach().cpu().numpy().astype(np.float64)
    )
    ndc_x = (2.0 * np.asarray(columns, dtype=np.float64) + 1.0) / width - 1.0
    ndc_y = (2.0 * np.asarray(rows, dtype=np.float64) + 1.0) / height - 1.0
    near_clip = np.stack(
        (ndc_x, ndc_y, np.full_like(ndc_x, -1.0), np.ones_like(ndc_x)),
        axis=1,
    )
    far_clip = near_clip.copy()
    far_clip[:, 2] = 1.0
    near = near_clip @ inverse
    far = far_clip @ inverse
    near = near[:, :3] / near[:, 3:4]
    far = far[:, :3] / far[:, 3:4]
    direction = far - near
    denominator = direction @ plane_normal
    if bool((np.abs(denominator) <= 1e-10).any()):
        raise ValueError("TDOM rays are parallel to the planar fallback surface")
    distance = (float(plane_offset) - near @ plane_normal) / denominator
    return near + distance[:, None] * direction


def _rectified_labels(dataset, camera, image_id: str) -> np.ndarray:
    observation = dataset.load_observation(image_id)
    return labels_on_camera_grid(observation.labels, camera, dataset.class_spec.ignore_id)


def apply_planar_multiview_fallback(
    *,
    gaussians,
    semantic_state,
    tdom_camera,
    alignment: Mapping[str, object],
    semantic_result,
    class_spec,
    dataset,
    source_cameras: Sequence[object],
    observation_groups: Sequence[Sequence[str]],
    geometry_scale_ratio: float,
    config: PlanarFallbackConfig,
) -> PlanarFallbackOutcome:
    """Fill only bounded, multi-view-proven planar semantic holes.

    The function is intentionally final-product-only. It does not mutate the
    Gaussian model or semantic state, and the returned provenance raster keeps
    fallback pixels distinguishable from trusted Gaussian pixels.
    """
    started = time.perf_counter()
    if not config.enabled:
        return PlanarFallbackOutcome(semantic_result, {"enabled": False})
    by_name = {item.name: item for item in class_spec.classes}
    if config.class_name not in by_name:
        raise ValueError(
            "planar fallback class is absent from the ontology: {}".format(
                config.class_name
            )
        )
    target_id = int(by_name[config.class_name].class_id)
    class_ids = semantic_result.class_ids.detach().cpu().clone()
    unknown_mask = semantic_result.unknown_mask.detach().cpu().clone()
    unknown_reason = semantic_result.unknown_reason.detach().cpu().clone()
    entropy = semantic_result.entropy.detach().cpu().clone()
    probabilities_output = (
        None
        if semantic_result.probabilities is None
        else semantic_result.probabilities.detach().cpu().clone()
    )
    normalized_entropy = (
        None
        if semantic_result.normalized_entropy is None
        else semantic_result.normalized_entropy.detach().cpu().clone()
    )
    top1_probability = (
        None
        if semantic_result.top1_probability is None
        else semantic_result.top1_probability.detach().cpu().clone()
    )
    margin = (
        None
        if semantic_result.margin is None
        else semantic_result.margin.detach().cpu().clone()
    )
    provenance = torch.zeros_like(class_ids, dtype=torch.uint8)
    provenance[~unknown_mask] = 1
    candidates = find_bounded_class_candidates(
        class_ids.numpy(),
        ignore_id=class_spec.ignore_id,
        target_class_id=target_id,
        config=config,
    )

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64, copy=False)
    scales = gaussians.get_scaling.detach().cpu().numpy()
    maximum_scales = scales.max(axis=1)
    median_scale = float(np.median(maximum_scales)) if len(maximum_scales) else 0.0
    scale_limit = (
        median_scale * float(geometry_scale_ratio)
        if geometry_scale_ratio > 0
        else None
    )
    trusted = np.ones(len(xyz), dtype=bool)
    if scale_limit is not None:
        trusted &= maximum_scales <= scale_limit
    alpha = semantic_state.alpha_robust.detach().cpu().numpy()
    prior = float(semantic_state.prior_total) / int(semantic_state.num_classes)
    evidence = np.maximum(alpha - prior, 0.0)
    evidence_total = evidence.sum(axis=1)
    gaussian_classes = evidence.argmax(axis=1)
    gaussian_columns, gaussian_rows, gaussian_visible = _project_points(
        tdom_camera, xyz
    )

    grid = alignment.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("planar fallback requires map grid alignment")
    plane_normal = np.asarray(grid.get("plane_normal_world"), dtype=np.float64)
    if plane_normal.shape != (3,) or not np.isfinite(plane_normal).all():
        raise ValueError("planar fallback requires a finite plane normal")
    norm = float(np.linalg.norm(plane_normal))
    if norm <= 0:
        raise ValueError("planar fallback plane normal must be non-zero")
    plane_normal /= norm
    camera_by_id = {
        _image_id(camera.image_name): camera for camera in source_cameras
    }

    component_metrics = []
    filled_pixels = 0
    for candidate in candidates:
        x, y, width, height = candidate.bbox
        context = int(config.plane_context_pixels)
        nearby = (
            gaussian_visible
            & trusted
            & (gaussian_classes == target_id)
            & (evidence_total > 0)
            & (gaussian_columns >= x - context)
            & (gaussian_columns < x + width + context)
            & (gaussian_rows >= y - context)
            & (gaussian_rows < y + height + context)
        )
        offsets = xyz[nearby] @ plane_normal
        if offsets.size < config.min_plane_gaussians:
            component_metrics.append(
                {
                    "component_id": candidate.component_id,
                    "area": candidate.area,
                    "bbox": list(candidate.bbox),
                    "status": "rejected_insufficient_plane_gaussians",
                    "plane_gaussian_count": int(offsets.size),
                }
            )
            continue
        initial_median = float(np.median(offsets))
        initial_mad = float(np.median(np.abs(offsets - initial_median)))
        clip_radius = 3.0 * max(initial_mad, 1e-6)
        clipped = offsets[np.abs(offsets - initial_median) <= clip_radius]
        plane_offset = float(np.median(clipped))
        plane_mad = float(np.median(np.abs(clipped - plane_offset)))
        if (
            clipped.size < config.min_plane_gaussians
            or plane_mad > config.max_plane_mad_world
        ):
            component_metrics.append(
                {
                    "component_id": candidate.component_id,
                    "area": candidate.area,
                    "bbox": list(candidate.bbox),
                    "status": "rejected_nonplanar_context",
                    "plane_gaussian_count": int(clipped.size),
                    "plane_mad_world": plane_mad,
                }
            )
            continue
        world = _unproject_pixels_to_plane(
            tdom_camera,
            candidate.rows,
            candidate.columns,
            plane_normal,
            plane_offset,
        )
        class_votes = np.zeros(
            (class_spec.num_classes, candidate.area), dtype=np.uint32
        )
        observed_groups = np.zeros(candidate.area, dtype=np.uint16)
        target_groups = np.zeros(candidate.area, dtype=np.uint16)
        used_cameras = set()
        for group in observation_groups:
            group_valid = np.zeros(candidate.area, dtype=np.uint16)
            group_target = np.zeros(candidate.area, dtype=np.uint16)
            for raw_image_id in group:
                image_id = _image_id(raw_image_id)
                camera = camera_by_id.get(image_id)
                if camera is None:
                    continue
                columns, rows, valid = _project_points(camera, world)
                if not bool(valid.any()):
                    continue
                labels = _rectified_labels(dataset, camera, image_id)
                indices = np.flatnonzero(valid)
                pixel_columns = np.clip(
                    np.rint(columns[indices]).astype(np.int64),
                    0,
                    int(camera.image_width) - 1,
                )
                pixel_rows = np.clip(
                    np.rint(rows[indices]).astype(np.int64),
                    0,
                    int(camera.image_height) - 1,
                )
                sampled = labels[pixel_rows, pixel_columns]
                keep = sampled != class_spec.ignore_id
                indices = indices[keep]
                sampled = sampled[keep]
                if indices.size == 0:
                    continue
                used_cameras.add(image_id)
                group_valid[indices] += 1
                group_target[indices] += (sampled == target_id).astype(np.uint8)
                for value in np.unique(sampled):
                    hits = indices[sampled == value]
                    class_votes[int(value), hits] += 1
            present = group_valid > 0
            observed_groups += present.astype(np.uint16)
            target_groups += (
                present & (group_target * 2 > group_valid)
            ).astype(np.uint16)
        valid_votes = class_votes.sum(axis=0)
        target_votes = class_votes[target_id]
        accepted, agreement = accept_planar_votes(
            target_votes,
            valid_votes,
            target_groups,
            observed_groups,
            config,
        )
        accepted_rows = candidate.rows[accepted]
        accepted_columns = candidate.columns[accepted]
        class_ids[accepted_rows, accepted_columns] = target_id
        unknown_mask[accepted_rows, accepted_columns] = False
        unknown_reason[accepted_rows, accepted_columns] = 0
        provenance[accepted_rows, accepted_columns] = 2
        probabilities = np.divide(
            class_votes,
            valid_votes[None, :],
            out=np.zeros(class_votes.shape, dtype=np.float64),
            where=valid_votes[None, :] > 0,
        )
        direct_entropy = -(
            probabilities
            * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
        ).sum(axis=0)
        entropy[accepted_rows, accepted_columns] = torch.from_numpy(
            direct_entropy[accepted].astype(np.float32, copy=False)
        )
        if probabilities_output is not None:
            probabilities_output[
                :, accepted_rows, accepted_columns
            ] = torch.from_numpy(
                probabilities[:, accepted].astype(np.float32, copy=False)
            ).to(dtype=probabilities_output.dtype)
        if normalized_entropy is not None:
            denominator = math.log(max(2, int(class_spec.num_classes)))
            normalized_entropy[accepted_rows, accepted_columns] = (
                torch.from_numpy(
                    direct_entropy[accepted].astype(np.float32, copy=False)
                )
                / denominator
            )
        if top1_probability is not None or margin is not None:
            sorted_probabilities = np.sort(probabilities, axis=0)
            direct_top1 = sorted_probabilities[-1]
            direct_margin = (
                direct_top1 - sorted_probabilities[-2]
                if class_spec.num_classes > 1
                else direct_top1
            )
            if top1_probability is not None:
                top1_probability[accepted_rows, accepted_columns] = (
                    torch.from_numpy(
                        direct_top1[accepted].astype(np.float32, copy=False)
                    )
                )
            if margin is not None:
                margin[accepted_rows, accepted_columns] = torch.from_numpy(
                    direct_margin[accepted].astype(np.float32, copy=False)
                )
        accepted_count = int(accepted.sum())
        filled_pixels += accepted_count
        component_metrics.append(
            {
                "component_id": candidate.component_id,
                "area": candidate.area,
                "bbox": list(candidate.bbox),
                "status": (
                    "accepted" if accepted_count == candidate.area else "partial"
                ),
                "boundary_consensus": candidate.boundary_consensus,
                "plane_gaussian_count": int(clipped.size),
                "plane_offset_world": plane_offset,
                "plane_mad_world": plane_mad,
                "source_camera_count": len(used_cameras),
                "filled_pixel_count": accepted_count,
                "rejected_pixel_count": candidate.area - accepted_count,
                "valid_views_min_median_max": [
                    int(valid_votes.min()),
                    float(np.median(valid_votes)),
                    int(valid_votes.max()),
                ],
                "agreement_min_median_max": [
                    float(agreement.min()),
                    float(np.median(agreement)),
                    float(agreement.max()),
                ],
                "observed_groups_min_median_max": [
                    int(observed_groups.min()),
                    float(np.median(observed_groups)),
                    int(observed_groups.max()),
                ],
            }
        )

    metrics = {
        "enabled": True,
        "class_name": config.class_name,
        "class_id": target_id,
        "candidate_component_count": len(candidates),
        "candidate_pixel_count": int(sum(item.area for item in candidates)),
        "filled_pixel_count": int(filled_pixels),
        "rejected_pixel_count": int(
            sum(item.area for item in candidates) - filled_pixels
        ),
        "boundary_connected_filled_pixel_count": 0,
        "config": config.to_dict(),
        "components": component_metrics,
        "seconds": float(time.perf_counter() - started),
        "support_raster_semantics": (
            "Gaussian-only; consult class_provenance for planar fallback pixels"
        ),
    }
    updated_metrics = dict(semantic_result.metrics)
    updated_metrics["planar_multiview_fallback"] = metrics
    updated_result = semantic_result._replace(
        class_ids=class_ids,
        colors=None,
        unknown_mask=unknown_mask,
        unknown_reason=unknown_reason,
        entropy=entropy,
        probabilities=probabilities_output,
        normalized_entropy=normalized_entropy,
        top1_probability=top1_probability,
        margin=margin,
        metrics=updated_metrics,
        class_provenance=provenance,
    )
    return PlanarFallbackOutcome(updated_result, metrics)
