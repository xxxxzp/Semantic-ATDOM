"""Forward-only Semantic TDOM rendering and product serialization."""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping, NamedTuple, Optional

import numpy as np
from PIL import Image
import torch

from semantic_atdom.semantics.schema import SemanticClassSpec, sha256_file


@dataclass(frozen=True)
class SemanticTdomConfig:
    """Configurable unknown gate; defaults are conservative, not calibrated."""

    min_coverage: float = 0.05
    min_absolute_support: float = 0.01
    min_support_ratio: float = 0.05
    min_margin: float = 0.05
    eps: float = 1e-8
    save_probabilities: bool = False
    max_output_bytes: int = 8 * 1024**3
    min_free_cuda_bytes: int = 512 * 1024**2

    def __post_init__(self):
        for name in (
            "min_coverage",
            "min_absolute_support",
            "min_support_ratio",
            "min_margin",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or value > 1:
                raise ValueError("{} must be in [0, 1]".format(name))
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be positive")
        if not isinstance(self.max_output_bytes, int) or self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")
        if (
            not isinstance(self.min_free_cuda_bytes, int)
            or self.min_free_cuda_bytes < 0
        ):
            raise ValueError("min_free_cuda_bytes must be a non-negative integer")

    def to_dict(self):
        return asdict(self)


class SemanticTdomResult(NamedTuple):
    class_ids: torch.Tensor
    colors: torch.Tensor
    unknown_mask: torch.Tensor
    probabilities: torch.Tensor
    coverage: torch.Tensor
    absolute_support: torch.Tensor
    support_ratio: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    radii: Optional[torch.Tensor]


def finalize_semantic_tdom(
    rendered_features: torch.Tensor,
    coverage: torch.Tensor,
    class_spec: SemanticClassSpec,
    config: Optional[SemanticTdomConfig] = None,
    radii: Optional[torch.Tensor] = None,
) -> SemanticTdomResult:
    """Normalize composited soft evidence and apply the pixel unknown gate."""
    config = config or SemanticTdomConfig()
    rendered_features = rendered_features.detach()
    coverage = coverage.detach().to(
        device=rendered_features.device, dtype=rendered_features.dtype
    )
    expected_channels = class_spec.num_classes + 1
    if rendered_features.ndim != 3 or rendered_features.shape[0] != expected_channels:
        raise ValueError(
            "rendered_features must have shape [num_classes + 1, H, W]"
        )
    if coverage.shape != rendered_features.shape[1:]:
        raise ValueError("coverage must have shape [H, W]")
    if not bool(torch.isfinite(rendered_features).all()) or bool(
        (rendered_features < 0).any()
    ):
        raise ValueError("rendered semantic features must be finite and non-negative")
    if not bool(torch.isfinite(coverage).all()) or bool((coverage < 0).any()):
        raise ValueError("coverage must be finite and non-negative")

    numerator = rendered_features[:-1]
    absolute_support = rendered_features[-1]
    probabilities = numerator / (absolute_support.unsqueeze(0) + config.eps)
    probability_sum = probabilities.sum(dim=0, keepdim=True)
    probabilities = torch.where(
        probability_sum > config.eps,
        probabilities / (probability_sum + config.eps),
        torch.zeros_like(probabilities),
    )
    support_ratio = absolute_support / (coverage + config.eps)
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(config.eps))
    ).sum(dim=0)
    if class_spec.num_classes > 1:
        top_two = torch.topk(probabilities, k=2, dim=0).values
        margin = top_two[0] - top_two[1]
    else:
        margin = probabilities[0]
    valid = (
        (coverage >= config.min_coverage)
        & (absolute_support >= config.min_absolute_support)
        & (support_ratio >= config.min_support_ratio)
        & (margin >= config.min_margin)
        & (absolute_support > config.eps)
    )
    class_ids = probabilities.argmax(dim=0).to(dtype=torch.uint8)
    class_ids = torch.where(
        valid,
        class_ids,
        torch.full_like(class_ids, class_spec.ignore_id),
    )
    entropy = torch.where(
        valid, entropy, torch.full_like(entropy, float("nan"))
    )
    palette = torch.tensor(
        [item.color for item in class_spec.classes],
        dtype=torch.uint8,
        device=class_ids.device,
    )
    safe_ids = class_ids.long().clamp(max=class_spec.num_classes - 1)
    colors = palette[safe_ids]
    colors = torch.where(
        valid.unsqueeze(2), colors, torch.zeros_like(colors)
    ).permute(2, 0, 1)
    return SemanticTdomResult(
        class_ids,
        colors,
        ~valid,
        probabilities,
        coverage,
        absolute_support,
        support_ratio,
        entropy,
        margin,
        radii,
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
    """Render K-channel semantics with the orthographic RGB TDOM camera."""
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
    if class_spec.num_classes + 1 > 16:
        raise ValueError("Semantic TDOM currently supports at most 15 classes")
    # The forward result, normalization intermediates and serialization copies
    # coexist briefly.  Reject unsafe requests explicitly until tiled rendering
    # is implemented instead of relying on an opaque CUDA OOM.
    estimated_bytes = (
        (3 * class_spec.num_classes + 10)
        * int(camera.image_height)
        * int(camera.image_width)
        * 4
    ) + means3d.shape[0] * (class_spec.num_classes + 32) * 4
    if estimated_bytes > config.max_output_bytes:
        raise ValueError(
            "Semantic TDOM requires an estimated {} bytes, exceeding the "
            "configured {} byte budget".format(
                estimated_bytes, config.max_output_bytes
            )
        )
    with torch.cuda.device(means3d.device):
        free_cuda_bytes, _ = torch.cuda.mem_get_info()
    if estimated_bytes + config.min_free_cuda_bytes > free_cuda_bytes:
        raise ValueError(
            "Semantic TDOM estimated allocation plus headroom is {} bytes, "
            "but only {} CUDA bytes are free".format(
                estimated_bytes + config.min_free_cuda_bytes,
                free_cuda_bytes,
            )
        )
    max_binning_bytes = min(
        config.max_output_bytes - estimated_bytes,
        free_cuda_bytes - config.min_free_cuda_bytes - estimated_bytes,
    )

    alpha = semantic_state.alpha_robust if use_robust else semantic_state.alpha_raw
    alpha = alpha.detach().to(device=means3d.device, dtype=torch.float32)
    concentration = alpha.sum(dim=1)
    means = alpha / (concentration.unsqueeze(1) + config.eps)
    evidence_ratio = (
        (concentration - semantic_state.prior_total)
        / (concentration + config.eps)
    ).clamp(min=0.0, max=1.0)
    features = torch.cat(
        (means * evidence_ratio.unsqueeze(1), evidence_ratio.unsqueeze(1)),
        dim=1,
    ).contiguous()
    opacities = gaussians.get_opacity.detach().to(dtype=torch.float32).contiguous()
    scales = gaussians.get_scaling.detach().to(dtype=torch.float32).contiguous()
    rotations = gaussians.get_rotation.detach().to(dtype=torch.float32).contiguous()
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
    with torch.cuda.device(means3d.device):
        rendered, coverage, radii = _C.semantic_tdom(
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
            int(camera.image_height),
            int(camera.image_width),
            int(max_binning_bytes),
            bool(debug),
        )
    return finalize_semantic_tdom(
        rendered, coverage, class_spec, config=config, radii=radii
    )


def write_semantic_tdom(
    result: SemanticTdomResult,
    output_root: Path,
    class_spec: SemanticClassSpec,
    manifest: Mapping[str, object],
    config: Optional[SemanticTdomConfig] = None,
) -> Path:
    """Atomically publish one aligned Semantic TDOM product directory."""
    config = config or SemanticTdomConfig()
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
    arrays = {
        "semantic_id.png": result.class_ids.detach().cpu().numpy().astype(np.uint8),
        "semantic_color.png": result.colors.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8),
        "unknown_mask.png": (result.unknown_mask.detach().cpu().numpy().astype(np.uint8) * 255),
        "semantic_support.tif": result.absolute_support.detach().cpu().numpy().astype(np.float32),
        "semantic_entropy.tif": result.entropy.detach().cpu().numpy().astype(np.float32),
    }
    formats = {
        "semantic_id.png": "PNG",
        "semantic_color.png": "PNG",
        "unknown_mask.png": "PNG",
        "semantic_support.tif": "TIFF",
        "semantic_entropy.tif": "TIFF",
    }
    for name, array in arrays.items():
        path = staging / name
        temporary = path.with_name(path.name + ".tmp")
        Image.fromarray(array).save(str(temporary), format=formats[name])
        os.replace(str(temporary), str(path))

    files = {name: sha256_file(staging / name) for name in arrays}
    if config.save_probabilities:
        probability_path = staging / "semantic_probabilities.npy"
        temporary = probability_path.with_name(probability_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(
                stream,
                result.probabilities.detach().cpu().numpy().astype(np.float16),
                allow_pickle=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(probability_path))
        files[probability_path.name] = sha256_file(probability_path)

    payload = dict(manifest)
    payload.update(
        {
            "schema_version": 1,
            "width": int(result.class_ids.shape[1]),
            "height": int(result.class_ids.shape[0]),
            "class_spec_sha256": class_spec.fingerprint,
            "unknown_gate": config.to_dict(),
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
