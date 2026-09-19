"""Forward-only Semantic TDOM rendering and product serialization."""

from dataclasses import asdict, dataclass
from enum import IntFlag
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from semantic_atdom.semantics.schema import SemanticClassSpec, sha256_file
from semantic_atdom.semantics.provenance import SEMANTIC_METHOD_VERSION
from semantic_atdom.semantics.reference import evidence_conditioned_belief


MAX_SEMANTIC_TDOM_CLASSES = 15


class UnknownReason(IntFlag):
    """Independent reasons why a pixel is withheld from the hard class map."""

    LOW_COVERAGE = 1
    LOW_ABSOLUTE_SUPPORT = 2
    LOW_SUPPORT_RATIO = 4
    LOW_MARGIN = 8
    LOW_TOP1_PROBABILITY = 16
    HIGH_NORMALIZED_ENTROPY = 32
    HIGH_GROUP_DISAGREEMENT = 64
    NO_SEMANTIC_EVIDENCE = 128


@dataclass(frozen=True)
class SemanticTdomConfig:
    """Configurable unknown gate; defaults preserve the V1 hard-map behavior."""

    min_coverage: float = 0.05
    min_absolute_support: float = 0.01
    min_support_ratio: float = 0.05
    min_margin: float = 0.05
    min_top1_probability: float = 0.0
    max_normalized_entropy: float = 1.0
    max_group_disagreement: float = 2.0
    eps: float = 1e-8
    save_probabilities: bool = False
    save_diagnostics: bool = False
    quantize_diagnostics: bool = True
    render_tile_rows: int = 0
    finalize_tile_rows: int = 256
    validate_rendered_inputs: bool = False
    max_output_bytes: int = 8 * 1024**3
    min_free_cuda_bytes: int = 512 * 1024**2
    max_gaussian_scale_ratio: float = 14.0
    render_palette: Optional[Tuple[Tuple[int, int, int], ...]] = None
    render_palette_source: Optional[str] = None

    def __post_init__(self):
        for name in (
            "min_coverage",
            "min_absolute_support",
            "min_support_ratio",
            "min_margin",
            "min_top1_probability",
            "max_normalized_entropy",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or value > 1:
                raise ValueError("{} must be in [0, 1]".format(name))
        if (
            not math.isfinite(self.max_group_disagreement)
            or self.max_group_disagreement < 0
            or self.max_group_disagreement > 2
        ):
            raise ValueError("max_group_disagreement must be in [0, 2]")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be positive")
        if (
            not isinstance(self.render_tile_rows, int)
            or self.render_tile_rows < 0
        ):
            raise ValueError("render_tile_rows must be a non-negative integer")
        if self.render_tile_rows and self.render_tile_rows % 16:
            raise ValueError("render_tile_rows must be a multiple of 16")
        if (
            not isinstance(self.finalize_tile_rows, int)
            or self.finalize_tile_rows <= 0
        ):
            raise ValueError("finalize_tile_rows must be a positive integer")
        if not isinstance(self.max_output_bytes, int) or self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")
        if (
            not isinstance(self.min_free_cuda_bytes, int)
            or self.min_free_cuda_bytes < 0
        ):
            raise ValueError("min_free_cuda_bytes must be a non-negative integer")
        if (
            not math.isfinite(self.max_gaussian_scale_ratio)
            or self.max_gaussian_scale_ratio < 0
            or 0 < self.max_gaussian_scale_ratio <= 1
        ):
            raise ValueError(
                "max_gaussian_scale_ratio must be zero (disabled) or greater "
                "than one"
            )
        if self.render_palette is not None:
            if not self.render_palette:
                raise ValueError("render_palette must not be empty")
            for color in self.render_palette:
                if (
                    not isinstance(color, (tuple, list))
                    or len(color) != 3
                    or any(
                        isinstance(channel, bool)
                        or not isinstance(channel, int)
                        or channel < 0
                        or channel > 255
                        for channel in color
                    )
                ):
                    raise ValueError(
                        "render_palette colors must contain three integers in [0, 255]"
                    )
        if self.render_palette_source is not None and not isinstance(
            self.render_palette_source, str
        ):
            raise ValueError("render_palette_source must be a string or None")

    def to_dict(self):
        payload = asdict(self)
        payload.pop("render_palette")
        payload.pop("render_palette_source")
        for name in (
            "min_margin",
            "min_top1_probability",
            "max_normalized_entropy",
            "max_group_disagreement",
        ):
            payload.pop(name)
        payload["semantic_method_version"] = SEMANTIC_METHOD_VERSION
        return payload


def load_semantic_tdom_palette(
    path: Path, class_spec: SemanticClassSpec
) -> Tuple[Tuple[int, int, int], ...]:
    """Load visualization-only class-color overrides.

    The label class specification remains the authority for IDs and names.  A
    render palette may change colors only, so changing map styling never
    invalidates SAM label metadata or semantic evidence.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(
            "unable to read Semantic TDOM palette {}: {}".format(path, error)
        )
    schema_version = (
        payload.get("schema_version") if isinstance(payload, Mapping) else None
    )
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("Semantic TDOM palette requires schema_version 1")
    overrides = payload.get("overrides")
    if not isinstance(overrides, Mapping):
        raise ValueError("Semantic TDOM palette overrides must be an object")

    classes_by_name = {item.name: item for item in class_spec.classes}
    unknown_names = sorted(set(overrides) - set(classes_by_name))
    if unknown_names:
        raise ValueError(
            "Semantic TDOM palette contains unknown classes: {}".format(
                ", ".join(unknown_names)
            )
        )
    colors = [tuple(item.color) for item in class_spec.classes]
    for name, color in overrides.items():
        if (
            not isinstance(color, (tuple, list))
            or len(color) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or channel < 0
                or channel > 255
                for channel in color
            )
        ):
            raise ValueError(
                "Semantic TDOM palette color for {} must contain three "
                "integers in [0, 255]".format(name)
            )
        colors[classes_by_name[name].class_id] = tuple(color)
    return tuple(colors)


def _effective_render_palette(
    class_spec: SemanticClassSpec, config: SemanticTdomConfig
) -> Tuple[Tuple[int, int, int], ...]:
    colors = config.render_palette
    if colors is None:
        colors = tuple(tuple(item.color) for item in class_spec.classes)
    if len(colors) != class_spec.num_classes:
        raise ValueError(
            "render_palette has {} colors, expected {}".format(
                len(colors), class_spec.num_classes
            )
        )
    return tuple(tuple(color) for color in colors)


def _render_palette_payload(
    class_spec: SemanticClassSpec, config: SemanticTdomConfig
) -> Dict[str, object]:
    colors = _effective_render_palette(class_spec, config)
    return {
        "schema_version": 1,
        "class_spec_sha256": class_spec.fingerprint,
        "source": config.render_palette_source,
        "classes": [
            {
                "id": item.class_id,
                "name": item.name,
                "color": list(colors[item.class_id]),
            }
            for item in class_spec.classes
        ],
        "ignore_id": class_spec.ignore_id,
        "ignore_color": [0, 0, 0],
    }


class SemanticTdomResult(NamedTuple):
    class_ids: torch.Tensor
    colors: Optional[torch.Tensor]
    unknown_mask: torch.Tensor
    unknown_reason: torch.Tensor
    probabilities: Optional[torch.Tensor]
    coverage: Optional[torch.Tensor]
    absolute_support: torch.Tensor
    support_ratio: Optional[torch.Tensor]
    top1_probability: Optional[torch.Tensor]
    entropy: torch.Tensor
    normalized_entropy: Optional[torch.Tensor]
    margin: Optional[torch.Tensor]
    mixture_disagreement: Optional[torch.Tensor]
    group_disagreement: Optional[torch.Tensor]
    prior_regularized_variance_proxy: Optional[torch.Tensor]
    radii: Optional[torch.Tensor]
    metrics: Mapping[str, object]

    @property
    def posterior_variance(self) -> Optional[torch.Tensor]:
        """Deprecated schema-2 alias for downstream Python callers."""
        return self.prior_regularized_variance_proxy


def _empty_uncertainty(reference: torch.Tensor) -> torch.Tensor:
    return torch.full_like(reference, float("nan"))


def _trusted_geometry_mask(
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
    if not bool(torch.isfinite(maximum_scales).all()):
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


@torch.no_grad()
def rasterize_rgb_tdom_striped(
    gaussians,
    camera,
    background: torch.Tensor,
    config: SemanticTdomConfig,
    scale_modifier: float = 1.0,
    debug: bool = False,
) -> torch.Tensor:
    """Render the paired RGB TDOM in exact horizontal stripes.

    The regular RGB rasterizer allocates binning and image buffers for the
    complete orthographic map.  Once a progressive flight expands beyond the
    initial local block, those buffers can exceed the available CUDA memory.
    This path uses the same front-to-back alpha compositing as the Semantic
    TDOM tile kernel and transfers each completed stripe to CPU immediately.
    """
    from diff_gaussian_rasterization import _C
    from semantic_atdom.utils.sh_utils import eval_sh

    if not hasattr(_C, "semantic_tdom_tile"):
        raise RuntimeError(
            "diff_gaussian_rasterization must be rebuilt with exact tiled "
            "TDOM support"
        )
    means3d = gaussians.get_xyz.detach().to(dtype=torch.float32).contiguous()
    if not means3d.is_cuda:
        raise ValueError("RGB TDOM geometry must be on CUDA")
    if background.numel() != 3:
        raise ValueError("RGB TDOM background must contain three channels")
    background = background.detach().to(
        device=means3d.device, dtype=torch.float32
    ).reshape(3)

    image_height = int(camera.image_height)
    image_width = int(camera.image_width)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("RGB TDOM dimensions must be positive")

    features = gaussians.get_features
    shs_view = features.transpose(1, 2).reshape(
        -1, 3, (int(gaussians.max_sh_degree) + 1) ** 2
    )
    directions = means3d - camera.camera_center.reshape(1, 3)
    directions = directions / directions.norm(dim=1, keepdim=True)
    colors = torch.clamp_min(
        eval_sh(int(gaussians.active_sh_degree), shs_view, directions) + 0.5,
        0.0,
    ).to(dtype=torch.float32).contiguous()

    opacities = gaussians.get_opacity.detach().to(
        dtype=torch.float32
    ).contiguous()
    scales = gaussians.get_scaling.detach().to(dtype=torch.float32).contiguous()
    rotations = gaussians.get_rotation.detach().to(
        dtype=torch.float32
    ).contiguous()
    empty = torch.empty(0, dtype=torch.float32, device=means3d.device)
    viewmatrix = camera.world_view_transform.detach().to(
        device=means3d.device, dtype=torch.float32
    ).contiguous()
    projmatrix = camera.full_proj_transform.detach().to(
        device=means3d.device, dtype=torch.float32
    ).contiguous()
    campos = camera.camera_center.detach().to(
        device=means3d.device, dtype=torch.float32
    ).contiguous()

    # Release cached full-frame publication buffers before sizing the striped
    # renderer.  Live tensors are unaffected.
    with torch.cuda.device(means3d.device):
        torch.cuda.empty_cache()
        free_cuda_bytes, _ = torch.cuda.mem_get_info()
    capacity_bytes = min(
        int(config.max_output_bytes),
        int(free_cuda_bytes) - int(config.min_free_cuda_bytes),
    )
    gaussian_bytes = means3d.shape[0] * (3 + 32) * 4
    binning_reserve_bytes = max(128 * 1024**2, means3d.shape[0] * 128)
    fixed_bytes = gaussian_bytes + binning_reserve_bytes
    bytes_per_output_row = 5 * image_width * 4
    max_rows_by_budget = int(
        (capacity_bytes - fixed_bytes) // bytes_per_output_row
    )
    max_rows_by_budget = (max_rows_by_budget // 16) * 16
    requested_rows = config.render_tile_rows or image_height
    tile_rows = min(requested_rows, image_height, max_rows_by_budget)
    if tile_rows < image_height:
        tile_rows = (tile_rows // 16) * 16
    if tile_rows <= 0:
        raise ValueError(
            "RGB TDOM has insufficient memory for one 16-row render tile"
        )

    output = torch.empty(
        (3, image_height, image_width), dtype=torch.float32, device="cpu"
    )
    tan_fovx = float(math.tan(camera.FoVx * 0.5))
    tan_fovy = float(math.tan(camera.FoVy * 0.5))
    for output_y_start in range(0, image_height, tile_rows):
        output_height = min(tile_rows, image_height - output_y_start)
        tile_output_bytes = 5 * output_height * image_width * 4
        max_binning_bytes = max(
            0, capacity_bytes - gaussian_bytes - tile_output_bytes
        )
        rendered, coverage, radii, squared_moment = _C.semantic_tdom_tile(
            means3d,
            opacities,
            scales,
            rotations,
            float(scale_modifier),
            empty,
            colors,
            viewmatrix,
            projmatrix,
            tan_fovx,
            tan_fovy,
            campos,
            image_height,
            image_width,
            int(output_y_start),
            int(output_height),
            -1,
            int(max_binning_bytes),
            bool(debug),
        )
        rendered.add_((1.0 - coverage).unsqueeze(0) * background[:, None, None])
        output[:, output_y_start : output_y_start + output_height].copy_(
            rendered.cpu()
        )
        del rendered, coverage, radii, squared_moment
    return output


def _semantic_render_features(
    semantic_state,
    device: torch.device,
    use_robust: bool,
    eps: float,
    gaussian_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the K+5 per-Gaussian values consumed by the CUDA renderer."""
    alpha = semantic_state.alpha_robust if use_robust else semantic_state.alpha_raw
    alpha = alpha.detach().to(device=device, dtype=torch.float32)
    if gaussian_mask is not None:
        alpha = alpha[gaussian_mask]
    belief = evidence_conditioned_belief(
        alpha, prior_total=semantic_state.prior_total, eps=eps
    )
    concentration = belief.concentration
    evidence_ratio = belief.evidence_ratio
    distribution = belief.distribution
    probability_second_moment = distribution.square().sum(dim=1)
    # Squared distance between two categorical distributions can reach 2
    # (opposite one-hot vectors), so the flight-group trace must retain [0, 2].
    group_disagreement = semantic_state.group_variance.detach().to(
        device=device, dtype=torch.float32
    )
    group_count = semantic_state.group_count.detach().to(device=device)
    if gaussian_mask is not None:
        group_disagreement = group_disagreement[gaussian_mask]
        group_count = group_count[gaussian_mask]
    group_disagreement = group_disagreement.sum(dim=1).clamp(min=0.0, max=2.0)
    group_eligible = group_count >= 2
    if not semantic_state.group_disagreement_available:
        group_eligible = torch.zeros_like(group_eligible)
    group_support = evidence_ratio * group_eligible.to(dtype=torch.float32)
    prior_regularized_mean = alpha / (concentration.unsqueeze(1) + eps)
    prior_regularized_variance_proxy = (
        (1.0 - prior_regularized_mean.square().sum(dim=1)).clamp_min(0.0)
        / (concentration + 1.0)
    )
    auxiliary = torch.stack(
        (
            evidence_ratio,
            evidence_ratio * probability_second_moment,
            group_support * group_disagreement,
            group_support,
            evidence_ratio.square() * prior_regularized_variance_proxy,
        ),
        dim=1,
    )
    return torch.cat(
        (belief.evidence / concentration.unsqueeze(1), auxiliary), dim=1
    ).contiguous()


def finalize_semantic_tdom(
    rendered_features: torch.Tensor,
    coverage: torch.Tensor,
    class_spec: SemanticClassSpec,
    config: Optional[SemanticTdomConfig] = None,
    radii: Optional[torch.Tensor] = None,
    retain_probabilities: bool = True,
    create_colors: bool = True,
) -> SemanticTdomResult:
    """Normalize composited evidence and apply an explainable unknown gate.

    ``rendered_features`` accepts the legacy ``K + 1`` layout, the deprecated
    ``K + 4`` enhanced layout, or the current ``K + 5`` enhanced layout. The
    current layout adds an independent valid-support denominator for
    flight-group disagreement.
    """
    config = config or SemanticTdomConfig()
    rendered_features = rendered_features.detach()
    coverage = coverage.detach().to(
        device=rendered_features.device, dtype=rendered_features.dtype
    )
    class_count = class_spec.num_classes
    valid_channel_counts = (
        class_count + 1,
        class_count + 4,
        class_count + 5,
    )
    if (
        rendered_features.ndim != 3
        or rendered_features.shape[0] not in valid_channel_counts
    ):
        raise ValueError(
            "rendered_features must have shape [num_classes + 1, H, W] "
            "or an enhanced [num_classes + 4|5, H, W] layout"
        )
    if coverage.shape != rendered_features.shape[1:]:
        raise ValueError("coverage must have shape [H, W]")
    if config.validate_rendered_inputs:
        if not bool(torch.isfinite(rendered_features).all()) or bool(
            (rendered_features < 0).any()
        ):
            raise ValueError(
                "rendered semantic features must be finite and non-negative"
            )
        if not bool(torch.isfinite(coverage).all()) or bool((coverage < 0).any()):
            raise ValueError("coverage must be finite and non-negative")

    numerator = rendered_features[:class_count]
    # A view would retain the complete enhanced allocation after finalization.
    absolute_support = rendered_features[class_count].clone()
    probabilities = numerator / (absolute_support.unsqueeze(0) + config.eps)
    probability_sum = probabilities.sum(dim=0, keepdim=True)
    probabilities = torch.where(
        probability_sum > config.eps,
        probabilities / (probability_sum + config.eps),
        torch.zeros_like(probabilities),
    ).clamp(min=0.0, max=1.0)
    support_ratio = absolute_support / (coverage + config.eps)
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(config.eps))
    ).sum(dim=0)
    if class_count > 1:
        top_two = torch.topk(probabilities, k=2, dim=0).values
        top1_probability = top_two[0]
        margin = top_two[0] - top_two[1]
    else:
        top1_probability = probabilities[0]
        margin = probabilities[0]
    entropy_scale = math.log(class_count) if class_count > 1 else 1.0
    normalized_entropy = entropy / entropy_scale

    enhanced_channels = rendered_features.shape[0] - class_count
    enhanced = enhanced_channels in (4, 5)
    if enhanced:
        weighted_second_moment = rendered_features[class_count + 1]
        mixture_disagreement = (
            weighted_second_moment / (absolute_support + config.eps)
            - probabilities.square().sum(dim=0)
        ).clamp(min=0.0, max=1.0)
        if enhanced_channels == 5:
            group_support = rendered_features[class_count + 3]
            group_disagreement = (
                rendered_features[class_count + 2]
                / (group_support + config.eps)
            ).clamp(min=0.0, max=2.0)
            group_evidence = group_support > config.eps
            variance_index = class_count + 4
        else:
            group_disagreement = (
                rendered_features[class_count + 2]
                / (absolute_support + config.eps)
            ).clamp(min=0.0, max=2.0)
            group_evidence = absolute_support > config.eps
            variance_index = class_count + 3
        prior_regularized_variance_proxy = (
            rendered_features[variance_index]
            / (absolute_support.square() + config.eps)
        ).clamp(min=0.0, max=1.0)
    else:
        mixture_disagreement = _empty_uncertainty(absolute_support)
        group_disagreement = _empty_uncertainty(absolute_support)
        prior_regularized_variance_proxy = _empty_uncertainty(absolute_support)

    reason = torch.zeros_like(absolute_support, dtype=torch.uint8)

    def add_reason(mask: torch.Tensor, flag: UnknownReason) -> None:
        reason.bitwise_or_(mask.to(dtype=torch.uint8) * int(flag))

    semantic_evidence = absolute_support > config.eps
    add_reason(~semantic_evidence, UnknownReason.NO_SEMANTIC_EVIDENCE)
    add_reason(coverage < config.min_coverage, UnknownReason.LOW_COVERAGE)
    add_reason(
        absolute_support < config.min_absolute_support,
        UnknownReason.LOW_ABSOLUTE_SUPPORT,
    )
    add_reason(
        support_ratio < config.min_support_ratio,
        UnknownReason.LOW_SUPPORT_RATIO,
    )
    valid = reason == 0
    class_ids = probabilities.argmax(dim=0).to(dtype=torch.uint8)
    class_ids = torch.where(
        valid,
        class_ids,
        torch.full_like(class_ids, class_spec.ignore_id),
    )
    # Entropy describes ambiguity, including supported pixels rejected by the
    # hard gate. It is undefined only where no semantic evidence exists.
    entropy = torch.where(
        semantic_evidence, entropy, torch.full_like(entropy, float("nan"))
    )
    normalized_entropy = torch.where(
        semantic_evidence,
        normalized_entropy,
        torch.full_like(normalized_entropy, float("nan")),
    )
    top1_probability = torch.where(
        semantic_evidence,
        top1_probability,
        torch.full_like(top1_probability, float("nan")),
    )
    margin = torch.where(
        semantic_evidence,
        margin,
        torch.full_like(margin, float("nan")),
    )
    if enhanced:
        undefined = torch.full_like(absolute_support, float("nan"))
        mixture_disagreement = torch.where(
            semantic_evidence, mixture_disagreement, undefined
        )
        group_disagreement = torch.where(
            group_evidence, group_disagreement, undefined
        )
        prior_regularized_variance_proxy = torch.where(
            semantic_evidence,
            prior_regularized_variance_proxy,
            undefined,
        )
    colors = None
    if create_colors:
        palette = torch.tensor(
            _effective_render_palette(class_spec, config),
            dtype=torch.uint8,
            device=class_ids.device,
        )
        safe_ids = class_ids.long().clamp(max=class_count - 1)
        colors = palette[safe_ids]
        colors = torch.where(
            valid.unsqueeze(2), colors, torch.zeros_like(colors)
        ).permute(2, 0, 1)
    returned_probabilities = probabilities if retain_probabilities else None
    return SemanticTdomResult(
        class_ids=class_ids,
        colors=colors,
        unknown_mask=~valid,
        unknown_reason=reason,
        probabilities=returned_probabilities,
        coverage=coverage,
        absolute_support=absolute_support,
        support_ratio=support_ratio,
        top1_probability=top1_probability,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        margin=margin,
        mixture_disagreement=mixture_disagreement,
        group_disagreement=group_disagreement,
        prior_regularized_variance_proxy=prior_regularized_variance_proxy,
        radii=radii,
        metrics={},
    )


def _finalize_semantic_tdom_in_tiles(
    rendered_features: torch.Tensor,
    coverage: torch.Tensor,
    class_spec: SemanticClassSpec,
    config: SemanticTdomConfig,
    radii: Optional[torch.Tensor],
    metrics: Mapping[str, object],
    outputs: Optional[Dict[str, torch.Tensor]] = None,
    output_row_start: int = 0,
    total_height: Optional[int] = None,
) -> SemanticTdomResult:
    """Finalize row tiles and immediately transfer compact products to CPU."""
    height, width = coverage.shape
    core_float_names = (
        "absolute_support",
        "entropy",
    )
    diagnostic_float_names = (
        "coverage",
        "support_ratio",
        "top1_probability",
        "normalized_entropy",
        "margin",
        "mixture_disagreement",
        "group_disagreement",
        "prior_regularized_variance_proxy",
    )
    float_names = core_float_names + (
        diagnostic_float_names if config.save_diagnostics else ()
    )
    if outputs is None:
        output_height = height if total_height is None else total_height
        outputs = {
            "class_ids": torch.empty((output_height, width), dtype=torch.uint8),
            "unknown_mask": torch.empty((output_height, width), dtype=torch.bool),
            "unknown_reason": torch.empty((output_height, width), dtype=torch.uint8),
        }
        for name in float_names:
            outputs[name] = torch.empty(
                (output_height, width), dtype=torch.float32
            )
        if config.save_probabilities:
            outputs["probabilities"] = torch.empty(
                (class_spec.num_classes, output_height, width),
                dtype=torch.float16,
            )
    probabilities = outputs.get("probabilities")

    for row_start in range(0, height, config.finalize_tile_rows):
        row_end = min(row_start + config.finalize_tile_rows, height)
        tile = finalize_semantic_tdom(
            rendered_features[:, row_start:row_end],
            coverage[row_start:row_end],
            class_spec,
            config=config,
            retain_probabilities=config.save_probabilities,
            create_colors=False,
        )
        destination_start = output_row_start + row_start
        destination_end = output_row_start + row_end
        outputs["class_ids"][destination_start:destination_end].copy_(
            tile.class_ids.cpu()
        )
        outputs["unknown_mask"][destination_start:destination_end].copy_(
            tile.unknown_mask.cpu()
        )
        outputs["unknown_reason"][destination_start:destination_end].copy_(
            tile.unknown_reason.cpu()
        )
        for name in float_names:
            outputs[name][destination_start:destination_end].copy_(
                getattr(tile, name).cpu()
            )
        if probabilities is not None:
            probabilities[:, destination_start:destination_end].copy_(
                tile.probabilities.cpu().to(dtype=torch.float16)
            )

    return SemanticTdomResult(
        class_ids=outputs["class_ids"],
        colors=None,
        unknown_mask=outputs["unknown_mask"],
        unknown_reason=outputs["unknown_reason"],
        probabilities=probabilities,
        coverage=outputs.get("coverage"),
        absolute_support=outputs["absolute_support"],
        support_ratio=outputs.get("support_ratio"),
        top1_probability=outputs.get("top1_probability"),
        entropy=outputs["entropy"],
        normalized_entropy=outputs.get("normalized_entropy"),
        margin=outputs.get("margin"),
        mixture_disagreement=outputs.get("mixture_disagreement"),
        group_disagreement=outputs.get("group_disagreement"),
        prior_regularized_variance_proxy=outputs.get(
            "prior_regularized_variance_proxy"
        ),
        radii=None if radii is None else radii.detach().cpu(),
        metrics=dict(metrics),
    )


@torch.no_grad()
def rasterize_semantic_tdom(
    gaussians,
    semantic_state,
    camera,
    class_spec: SemanticClassSpec,
    config: Optional[SemanticTdomConfig] = None,
    use_robust: bool = True,
    scale_modifier: float = 1.0,
    debug: bool = False,
) -> SemanticTdomResult:
    """Render soft semantics with the orthographic RGB TDOM camera."""
    from diff_gaussian_rasterization import _C

    config = config or SemanticTdomConfig()
    if not hasattr(_C, "semantic_tdom"):
        raise RuntimeError(
            "diff_gaussian_rasterization must be rebuilt with the Semantic "
            "TDOM kernel"
        )
    semantic_state._require_valid()
    means3d = gaussians.get_xyz.detach().to(dtype=torch.float32).contiguous()
    if not means3d.is_cuda:
        raise ValueError("Semantic TDOM geometry must be on CUDA")
    if semantic_state.num_gaussians != means3d.shape[0]:
        raise ValueError("semantic state and geometry Gaussian counts differ")
    if semantic_state.num_classes != class_spec.num_classes:
        raise ValueError("semantic state and class spec widths differ")
    opacities = gaussians.get_opacity.detach().to(dtype=torch.float32).contiguous()
    scales = gaussians.get_scaling.detach().to(dtype=torch.float32).contiguous()
    rotations = gaussians.get_rotation.detach().to(dtype=torch.float32).contiguous()
    trusted_mask, geometry_validity = _trusted_geometry_mask(
        scales, config.max_gaussian_scale_ratio
    )
    input_gaussian_count = int(means3d.shape[0])
    filter_applied = geometry_validity["excluded_gaussian_count"] > 0
    if filter_applied:
        means3d = means3d[trusted_mask].contiguous()
        opacities = opacities[trusted_mask].contiguous()
        scales = scales[trusted_mask].contiguous()
        rotations = rotations[trusted_mask].contiguous()
    # K class numerators plus support, two mixture/group numerators, a local
    # group-validity denominator and the squared-weight variance feature.
    feature_channels = class_spec.num_classes + 5
    if feature_channels > MAX_SEMANTIC_TDOM_CLASSES + 5:
        raise ValueError(
            "Semantic TDOM currently supports at most {} classes".format(
                MAX_SEMANTIC_TDOM_CLASSES
            )
        )
    image_height = int(camera.image_height)
    image_width = int(camera.image_width)
    pixel_count = image_height * image_width
    tile_pixel_count = min(
        pixel_count,
        int(config.finalize_tile_rows) * image_width,
    )
    with torch.cuda.device(means3d.device):
        # Capacity must be measured after releasing allocator cache left by a
        # completed paired RGB render. Live model/state tensors are unaffected.
        torch.cuda.empty_cache()
        free_cuda_bytes, total_cuda_bytes = torch.cuda.mem_get_info()
        allocated_before = torch.cuda.memory_allocated()
    capacity_bytes = min(
        config.max_output_bytes,
        free_cuda_bytes - config.min_free_cuda_bytes,
    )
    gaussian_bytes = means3d.shape[0] * (feature_channels + 32) * 4
    finalize_bytes = (
        (2 * class_spec.num_classes + 14) * tile_pixel_count * 4
    )
    # Binning scales with the number of Gaussian/tile intersections rather
    # than just image dimensions. Keep an explicit reserve and let C++ reject
    # an unexpectedly larger list with a descriptive error.
    binning_reserve_bytes = max(128 * 1024**2, means3d.shape[0] * 128)
    fixed_bytes = gaussian_bytes + finalize_bytes + binning_reserve_bytes
    if capacity_bytes <= fixed_bytes:
        raise ValueError(
            "Semantic TDOM fixed working set is {} bytes, but the byte budget "
            "or CUDA capacity after headroom is {} bytes".format(
                fixed_bytes, capacity_bytes
            )
        )
    max_rows_by_budget = int(
        (capacity_bytes - fixed_bytes)
        // (feature_channels * image_width * 4)
    )
    max_rows_by_budget = (max_rows_by_budget // 16) * 16
    if config.render_tile_rows:
        render_tile_rows = min(
            config.render_tile_rows, image_height, max_rows_by_budget
        )
    elif max_rows_by_budget >= image_height:
        render_tile_rows = image_height
    else:
        render_tile_rows = max_rows_by_budget
    if render_tile_rows <= 0:
        raise ValueError(
            "Semantic TDOM has insufficient memory for one 16-row render tile"
        )
    render_pixel_count = min(render_tile_rows, image_height) * image_width
    estimated_bytes = fixed_bytes + feature_channels * render_pixel_count * 4
    max_binning_bytes = min(
        config.max_output_bytes - gaussian_bytes - finalize_bytes
        - feature_channels * render_pixel_count * 4,
        free_cuda_bytes - config.min_free_cuda_bytes - gaussian_bytes
        - finalize_bytes - feature_channels * render_pixel_count * 4,
    )

    features = _semantic_render_features(
        semantic_state,
        device=means3d.device,
        use_robust=use_robust,
        eps=config.eps,
        gaussian_mask=(trusted_mask if filter_applied else None),
    )
    empty = torch.empty(0, dtype=torch.float32, device=means3d.device)
    viewmatrix = camera.world_view_transform.detach().to(
        device=means3d.device, dtype=torch.float32
    ).contiguous()
    projmatrix = camera.full_proj_transform.detach().to(
        device=means3d.device, dtype=torch.float32
    ).contiguous()
    campos = camera.camera_center.detach().to(
        device=means3d.device, dtype=torch.float32
    ).contiguous()
    if not hasattr(_C, "semantic_tdom_tile"):
        raise RuntimeError(
            "diff_gaussian_rasterization must be rebuilt with exact tiled "
            "Semantic TDOM and uncertainty-moment support"
        )
    pipeline_start = time.perf_counter()
    raster_cuda_seconds = 0.0
    raster_wall_seconds = 0.0
    finalize_seconds = 0.0
    allocated_after_raster = allocated_before
    result = None
    output_buffers = None
    render_calls = 0
    for output_y_start in range(0, image_height, render_tile_rows):
        output_height = min(
            render_tile_rows, image_height - output_y_start
        )
        raster_call_start = time.perf_counter()
        with torch.cuda.device(means3d.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            arguments = (
                means3d,
                opacities,
                scales,
                rotations,
                float(scale_modifier),
                empty,
                features,
                viewmatrix,
                projmatrix,
                float(math.tan(camera.FoVx * 0.5)),
                float(math.tan(camera.FoVy * 0.5)),
                campos,
                image_height,
                image_width,
            )
            rendered, coverage, radii, variance_numerator = (
                _C.semantic_tdom_tile(
                    *arguments,
                    int(output_y_start),
                    int(output_height),
                    int(class_spec.num_classes + 4),
                    int(max_binning_bytes),
                    bool(debug),
                )
            )
            rendered[class_spec.num_classes + 4].copy_(variance_numerator)
            end_event.record()
            end_event.synchronize()
            raster_cuda_seconds += (
                start_event.elapsed_time(end_event) / 1000.0
            )
            allocated_after_raster = max(
                allocated_after_raster, torch.cuda.memory_allocated()
            )
        raster_wall_seconds += time.perf_counter() - raster_call_start
        render_calls += 1
        finalize_start = time.perf_counter()
        result = _finalize_semantic_tdom_in_tiles(
            rendered,
            coverage,
            class_spec,
            config,
            radii,
            {},
            outputs=output_buffers,
            output_row_start=output_y_start,
            total_height=image_height,
        )
        if output_buffers is None:
            output_buffers = {
                name: getattr(result, name)
                for name in (
                    "class_ids",
                    "unknown_mask",
                    "unknown_reason",
                    "coverage",
                    "absolute_support",
                    "support_ratio",
                    "top1_probability",
                    "entropy",
                    "normalized_entropy",
                    "margin",
                    "mixture_disagreement",
                    "group_disagreement",
                    "prior_regularized_variance_proxy",
                    "probabilities",
                )
                if getattr(result, name) is not None
            }
        finalize_seconds += time.perf_counter() - finalize_start
        del rendered, coverage, variance_numerator
    pipeline_seconds = time.perf_counter() - pipeline_start
    metrics = {
        "estimated_cuda_bytes": int(estimated_bytes),
        "cuda_free_bytes_before": int(free_cuda_bytes),
        "cuda_total_bytes": int(total_cuda_bytes),
        "cuda_allocated_bytes_before": int(allocated_before),
        "cuda_allocated_bytes_after_raster": int(allocated_after_raster),
        "process_cuda_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(means3d.device)
        ),
        "raster_cuda_seconds": float(raster_cuda_seconds),
        "raster_wall_seconds": float(raster_wall_seconds),
        "render_finalize_transfer_seconds": float(pipeline_seconds),
        "render_calls": int(render_calls),
        "render_tile_rows": int(render_tile_rows),
        "render_mode": (
            "full_frame" if render_tile_rows >= image_height else "exact_striped"
        ),
        "feature_channels": int(feature_channels),
        "finalize_tile_rows": int(config.finalize_tile_rows),
        "geometry_validity_filter": geometry_validity,
    }
    if filter_applied and result is not None and result.radii is not None:
        full_radii = torch.zeros(input_gaussian_count, dtype=result.radii.dtype)
        full_radii[trusted_mask.detach().cpu()] = result.radii
        result = result._replace(radii=full_radii)
    del features
    with torch.cuda.device(means3d.device):
        allocated_after_finalize = torch.cuda.memory_allocated()
    updated_metrics = dict(metrics)
    updated_metrics.update(
        {
            "finalize_and_transfer_seconds": float(finalize_seconds),
            "cuda_allocated_bytes_after_finalize": int(
                allocated_after_finalize
            ),
        }
    )
    group_disagreement = result.group_disagreement
    if (
        group_disagreement is not None
        and not semantic_state.group_disagreement_available
    ):
        group_disagreement = torch.full_like(
            group_disagreement, float("nan")
        )
    updated_metrics["group_disagreement_status"] = (
        "exact_rebuild"
        if semantic_state.group_disagreement_available
        else "unavailable"
    )
    return result._replace(
        group_disagreement=group_disagreement,
        metrics=updated_metrics,
    )


def _save_image_atomic(path: Path, array: np.ndarray, image_format: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(str(temporary), format=image_format)
    os.replace(str(temporary), str(path))


def _save_tiff_atomic(path: Path, array: np.ndarray) -> str:
    """Write a lossless tiled TIFF when tifffile is available."""
    temporary = path.with_name(path.name + ".tmp")
    writer = "pillow-deflate"
    try:
        import tifffile

        tifffile.imwrite(
            str(temporary),
            array,
            dtype=array.dtype,
            tile=(256, 256),
            compression="zlib",
            metadata=None,
        )
        writer = "tifffile-tiled-zlib"
    except ImportError:
        Image.fromarray(array).save(
            str(temporary), format="TIFF", compression="tiff_adobe_deflate"
        )
    os.replace(str(temporary), str(path))
    return writer


def _encode_diagnostic(
    array: np.ndarray,
    maximum: float,
    quantize: bool,
):
    if not quantize:
        return array.astype(np.float32, copy=False), {
            "dtype": "float32",
            "scale": 1.0,
            "offset": 0.0,
            "nodata": "NaN",
        }
    nodata = np.uint16(65535)
    finite = np.isfinite(array)
    encoded = np.full(array.shape, nodata, dtype=np.uint16)
    normalized = np.clip(array[finite] / maximum, 0.0, 1.0)
    encoded[finite] = np.rint(normalized * 65534.0).astype(np.uint16)
    return encoded, {
        "dtype": "uint16",
        "scale": float(maximum / 65534.0),
        "offset": 0.0,
        "nodata": int(nodata),
        "decode": "value = encoded * scale; encoded == nodata is NaN",
    }


def _unknown_reason_counts(reason: torch.Tensor) -> Dict[str, int]:
    histogram = torch.bincount(
        reason.detach().cpu().long().reshape(-1), minlength=256
    )
    counts = {}
    for item in UnknownReason:
        counts[item.name.lower()] = sum(
            int(histogram[value].item())
            for value in range(256)
            if value & int(item)
        )
    return counts


def write_semantic_tdom(
    result: SemanticTdomResult,
    output_root: Path,
    class_spec: SemanticClassSpec,
    manifest: Mapping[str, object],
    config: Optional[SemanticTdomConfig] = None,
) -> Path:
    """Atomically publish one aligned, self-describing Semantic TDOM."""
    config = config or SemanticTdomConfig()
    write_start = time.perf_counter()
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

    files = {}

    def record(path: Path) -> None:
        files[path.name] = sha256_file(path)

    class_ids = result.class_ids.detach().cpu().contiguous().numpy()
    id_path = staging / "semantic_id.png"
    _save_image_atomic(id_path, class_ids, "PNG")
    record(id_path)

    palette_payload = _render_palette_payload(class_spec, config)
    palette = np.asarray(
        [item["color"] for item in palette_payload["classes"]],
        dtype=np.uint8,
    )
    safe_ids = np.minimum(class_ids, class_spec.num_classes - 1)
    color = palette[safe_ids]
    color[class_ids == class_spec.ignore_id] = 0
    color_path = staging / "semantic_color.png"
    _save_image_atomic(color_path, color, "PNG")
    record(color_path)
    del color, safe_ids

    unknown = result.unknown_mask.detach().cpu().contiguous().numpy()
    unknown_path = staging / "unknown_mask.png"
    _save_image_atomic(unknown_path, unknown.astype(np.uint8) * 255, "PNG")
    record(unknown_path)
    del unknown

    reason = result.unknown_reason.detach().cpu().contiguous()
    reason_path = staging / "unknown_reason.png"
    _save_image_atomic(reason_path, reason.numpy(), "PNG")
    record(reason_path)

    tiff_writer = None
    diagnostic_tensors = {
        "pixel_evidence_support.tif": result.absolute_support,
        "semantic_entropy.tif": result.entropy,
    }
    if config.save_diagnostics:
        diagnostic_tensors.update(
            {
                "geometry_coverage.tif": result.coverage,
                "pixel_support_ratio.tif": result.support_ratio,
                "top1_probability.tif": result.top1_probability,
                "margin.tif": result.margin,
                "mixture_disagreement.tif": result.mixture_disagreement,
                "group_disagreement.tif": result.group_disagreement,
                "prior_regularized_variance_proxy.tif": (
                    result.prior_regularized_variance_proxy
                ),
            }
        )
    raster_encodings = {}
    for name, tensor in diagnostic_tensors.items():
        path = staging / name
        array = tensor.detach().cpu().contiguous().numpy()
        maximum = (
            math.log(class_spec.num_classes)
            if name == "semantic_entropy.tif" and class_spec.num_classes > 1
            else (2.0 if name == "group_disagreement.tif" else 1.0)
        )
        encoded, encoding = _encode_diagnostic(
            array, maximum, config.quantize_diagnostics
        )
        tiff_writer = _save_tiff_atomic(path, encoded)
        raster_encodings[name] = encoding
        record(path)
        del array, encoded

    class_spec_path = staging / "class_spec.json"
    with class_spec_path.open("w", encoding="utf-8") as stream:
        json.dump(
            class_spec.to_dict(),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    record(class_spec_path)

    render_palette_path = staging / "render_palette.json"
    with render_palette_path.open("w", encoding="utf-8") as stream:
        json.dump(
            palette_payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    record(render_palette_path)

    if config.save_probabilities:
        if result.probabilities is None:
            raise ValueError(
                "save_probabilities requires retained probability values"
            )
        probability_path = staging / "semantic_probabilities.npy"
        temporary = probability_path.with_name(probability_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(
                stream,
                result.probabilities.detach().cpu().numpy().astype(
                    np.float16, copy=False
                ),
                allow_pickle=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(probability_path))
        record(probability_path)

    write_seconds = time.perf_counter() - write_start
    class_counts = torch.bincount(
        result.class_ids.detach().cpu().long().reshape(-1), minlength=256
    )
    payload = dict(manifest)
    render_metrics = dict(result.metrics)
    render_metrics["product_files_write_and_hash_seconds"] = float(
        write_seconds
    )
    payload.update(
        {
            "schema_version": 3,
            "semantic_method_version": SEMANTIC_METHOD_VERSION,
            "width": int(result.class_ids.shape[1]),
            "height": int(result.class_ids.shape[0]),
            "ontology_hash": class_spec.ontology_fingerprint,
            "palette_hash": files["render_palette.json"],
            "class_spec_sha256": class_spec.fingerprint,
            "class_spec": class_spec.to_dict(),
            "render_palette_sha256": files["render_palette.json"],
            "render_palette": palette_payload,
            "unknown_gate": config.to_dict(),
            "unknown_reason_bits": {
                item.name.lower(): int(item)
                for item in (
                    UnknownReason.LOW_COVERAGE,
                    UnknownReason.LOW_ABSOLUTE_SUPPORT,
                    UnknownReason.LOW_SUPPORT_RATIO,
                    UnknownReason.NO_SEMANTIC_EVIDENCE,
                )
            },
            "uncertainty_definitions": {
                "semantic_entropy": "categorical entropy in nats; range [0, ln(K)]",
                "normalized_entropy": "semantic_entropy / ln(K); range [0, 1]",
                "mixture_disagreement": "weighted E[||mu_g||^2] - ||E[mu_g]||^2",
                "group_disagreement": "weighted mean trace of per-Gaussian flight-group variance",
                "prior_regularized_variance_proxy": "uncalibrated squared semantic-weight propagation of independent Dirichlet trace variance",
            },
            "product_statistics": {
                "class_pixel_counts": {
                    str(item.class_id): int(class_counts[item.class_id].item())
                    for item in class_spec.classes
                },
                "unknown_pixel_count": int(
                    class_counts[class_spec.ignore_id].item()
                ),
                "unknown_reason_pixel_counts": _unknown_reason_counts(reason),
            },
            "render_metrics": render_metrics,
            "tiff_writer": tiff_writer,
            "raster_encodings": raster_encodings,
            "file_sizes_bytes": {
                name: int((staging / name).stat().st_size) for name in files
            },
            "files": files,
        }
    )
    manifest_path = staging / "manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(manifest_path))
    if output_root.exists():
        os.replace(str(output_root), str(backup))
    try:
        os.replace(str(staging), str(output_root))
    except Exception:
        if backup.exists() and not output_root.exists():
            os.replace(str(backup), str(output_root))
        raise
    if backup.exists():
        shutil.rmtree(str(backup))
    return output_root / "manifest.json"
