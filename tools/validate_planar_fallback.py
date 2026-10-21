#!/usr/bin/env python3
"""Validate bounded planar semantic recovery against a frozen final run."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from semantic_atdom.scene.colmap_loader import read_intrinsics_binary
from semantic_atdom.semantics.planar_fallback import (
    PlanarFallbackConfig,
    apply_planar_multiview_fallback,
    find_bounded_class_candidates,
)
from semantic_atdom.semantics.schema import SemanticDataset
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.tdom import SemanticTdomResult
from semantic_atdom.utils.atomic_io import write_json_atomic
from semantic_atdom.utils.camera_calibration import (
    rectified_pinhole_from_colmap,
)
from semantic_atdom.utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
    getWorld2View2,
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


class _FrozenGeometry:
    def __init__(self, ply_path: Path):
        vertex = PlyData.read(str(ply_path)).elements[0]
        self._xyz = torch.from_numpy(
            np.stack(
                (
                    np.asarray(vertex["x"]),
                    np.asarray(vertex["y"]),
                    np.asarray(vertex["z"]),
                ),
                axis=1,
            ).astype(np.float32, copy=False)
        )
        scale_names = sorted(
            (
                item.name
                for item in vertex.properties
                if item.name.startswith("scale_")
            ),
            key=lambda item: int(item.rsplit("_", 1)[1]),
        )
        raw_scales = np.stack(
            tuple(np.asarray(vertex[name]) for name in scale_names), axis=1
        ).astype(np.float32, copy=False)
        self._scaling = torch.from_numpy(raw_scales)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)


def _source_camera(record, calibration):
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(record["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(record["position"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3].T
    translation = w2c[:3, 3]
    fov_x = focal2fov(float(record["fx"]), int(record["width"]))
    fov_y = focal2fov(float(record["fy"]), int(record["height"]))
    world_view = torch.from_numpy(
        getWorld2View2(rotation, translation).T.copy()
    )
    projection = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=fov_x,
        fovY=fov_y,
    ).T
    return SimpleNamespace(
        image_name=record["img_name"],
        image_width=int(record["width"]),
        image_height=int(record["height"]),
        FoVx=fov_x,
        FoVy=fov_y,
        world_view_transform=world_view,
        full_proj_transform=world_view @ projection,
        camera_center=world_view.inverse()[3, :3],
        calibration=calibration,
        Tri_Mask=None,
    )


def _tdom_camera(alignment):
    world_view = torch.tensor(
        alignment["world_view_transform"], dtype=torch.float32
    )
    full_projection = torch.tensor(
        alignment["full_projection_transform"], dtype=torch.float32
    )
    return SimpleNamespace(
        image_width=int(alignment["image_width"]),
        image_height=int(alignment["image_height"]),
        FoVx=float(alignment["fov_x_radians"]),
        FoVy=float(alignment["fov_y_radians"]),
        world_view_transform=world_view,
        full_proj_transform=full_projection,
        camera_center=world_view.inverse()[3, :3],
    )


def _boundary_connected(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.uint8)
    count, labels = cv2.connectedComponents(values, connectivity=8)
    if count <= 1:
        return np.zeros_like(values, dtype=bool)
    border_ids = np.unique(
        np.concatenate(
            (labels[0], labels[-1], labels[:, 0], labels[:, -1])
        )
    )
    border_ids = border_ids[border_ids != 0]
    return np.isin(labels, border_ids)


def _palette(path: Path) -> np.ndarray:
    payload = _read_json(path)
    return np.asarray(
        [item["color"] for item in payload["classes"]], dtype=np.uint8
    )


def _colorize(class_ids: np.ndarray, palette: np.ndarray, ignore_id: int):
    safe = np.minimum(class_ids, len(palette) - 1)
    result = palette[safe]
    result[class_ids == ignore_id] = 0
    return result


def _save_png(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(value).save(str(temporary), format="PNG")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-name", default="water")
    parser.add_argument("--expected-main-area", type=int, default=0)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    run_config = _read_json(run_root / "provenance" / "config.json")
    arguments = run_config["arguments"]
    pair_key = _read_json(run_root / "products" / "maps" / "latest.json")[
        "product_key"
    ]
    release = run_root / "products" / "maps" / "releases" / pair_key
    pair_manifest = _read_json(release / "pair_manifest.json")
    alignment = pair_manifest["alignment"]
    semantic_root = Path(arguments["semantic_data_root"]).resolve()
    if (semantic_root / "class_spec.json").is_file():
        dataset = SemanticDataset(semantic_root.parent, semantic_root.name)
    elif (semantic_root / "semantic" / "class_spec.json").is_file():
        dataset = SemanticDataset(semantic_root)
    else:
        raise ValueError(
            "semantic_data_root contains neither a direct nor semantic/ "
            "class_spec.json: {}".format(semantic_root)
        )
    by_name = {item.name: item for item in dataset.class_spec.classes}
    if args.class_name not in by_name:
        raise ValueError("class is absent from ontology: {}".format(args.class_name))
    target_id = int(by_name[args.class_name].class_id)

    source_root = Path(arguments["Source_Path_Dir"]).resolve()
    stage = int(pair_manifest["geometry_stage"])
    intrinsics = read_intrinsics_binary(
        str(source_root / str(stage) / "bin" / "cameras.bin")
    )
    if len(intrinsics) != 1:
        raise ValueError("validator requires one shared source calibration")
    intrinsic = next(iter(intrinsics.values()))
    calibration = rectified_pinhole_from_colmap(
        intrinsic.model,
        intrinsic.params,
        width=intrinsic.width,
        height=intrinsic.height,
    )
    source_cameras = tuple(
        _source_camera(item, calibration)
        for item in _read_json(run_root / "model" / "cameras.json")
    )

    geometry = _FrozenGeometry(run_root / "model" / "gaussians.ply")
    semantic_manifest = _read_json(run_root / "model" / "semantic" / "manifest.json")
    state = SemanticGaussianState.load(
        run_root / "model" / "semantic" / "state.pt",
        expected_projection_geometry_hash=(
            semantic_manifest["projection_geometry_hash"]
        ),
        expected_ontology_hash=dataset.class_spec.ontology_fingerprint,
        expected_gaussian_count=len(geometry.get_xyz),
        expected_recovery_context_sha256=(
            semantic_manifest["recovery_context_sha256"]
        ),
        map_location="cpu",
    )
    if not (run_root / "model/semantic/final_rebuild_metrics.json").is_file():
        raise ValueError("planar validation requires a run with --semantic_final_rebuild")
    if not all((release / "semantic" / name).is_file()
               for name in ("semantic_id.png", "unknown_reason.png")):
        raise ValueError(
            "planar validation requires --semantic_tdom_save_class_ids and --semantic_tdom_save_masks"
        )
    groups = _read_json(
        run_root / "model" / "semantic" / "final_rebuild_metrics.json"
    )["groups"]

    semantic_dir = release / "semantic"
    before = np.asarray(Image.open(semantic_dir / "semantic_id.png")).copy()
    unknown_reason = np.asarray(
        Image.open(semantic_dir / "unknown_reason.png")
    ).copy()
    before_unknown = before == dataset.class_spec.ignore_id
    base_result = SemanticTdomResult(
        class_ids=torch.from_numpy(before),
        colors=None,
        unknown_mask=torch.from_numpy(before_unknown),
        unknown_reason=torch.from_numpy(unknown_reason),
        probabilities=None,
        coverage=None,
        absolute_support=torch.zeros(before.shape, dtype=torch.float32),
        support_ratio=None,
        top1_probability=None,
        entropy=torch.full(before.shape, float("nan"), dtype=torch.float32),
        normalized_entropy=None,
        margin=None,
        mixture_disagreement=None,
        group_disagreement=None,
        prior_regularized_variance_proxy=None,
        radii=None,
        metrics={"validation_source": str(semantic_dir)},
    )
    config = PlanarFallbackConfig(class_name=args.class_name)
    candidates = find_bounded_class_candidates(
        before,
        ignore_id=dataset.class_spec.ignore_id,
        target_class_id=target_id,
        config=config,
    )
    if not candidates:
        raise RuntimeError("no bounded target-class semantic hole was found")
    main_candidate = max(candidates, key=lambda item: item.area)

    outcome = apply_planar_multiview_fallback(
        gaussians=geometry,
        semantic_state=state,
        tdom_camera=_tdom_camera(alignment),
        alignment=alignment,
        semantic_result=base_result,
        class_spec=dataset.class_spec,
        dataset=dataset,
        source_cameras=source_cameras,
        observation_groups=groups,
        geometry_scale_ratio=float(
            arguments["semantic_tdom_max_gaussian_scale_ratio"]
        ),
        config=config,
    )
    after = outcome.result.class_ids.numpy()
    provenance = outcome.result.class_provenance.numpy()
    changed = before_unknown & (after != dataset.class_spec.ignore_id)
    boundary_unknown = _boundary_connected(before_unknown)
    main_mask = np.zeros(before.shape, dtype=bool)
    main_mask[main_candidate.rows, main_candidate.columns] = True
    main_restored = int(np.count_nonzero((after == target_id) & main_mask))
    boundary_filled = int(np.count_nonzero(changed & boundary_unknown))
    changed_outside_candidates = int(
        np.count_nonzero(changed & (provenance != 2))
    )
    expected_area_ok = (
        args.expected_main_area <= 0
        or main_candidate.area == args.expected_main_area
    )
    checks = {
        "main_component_area_matches_expected": expected_area_ok,
        "main_component_fully_restored_as_target": (
            main_restored == main_candidate.area
        ),
        "boundary_connected_newly_classified_pixels_is_zero": (
            boundary_filled == 0
        ),
        "every_new_class_has_fallback_provenance": (
            changed_outside_candidates == 0
        ),
    }
    metrics = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "run_root": str(run_root),
        "product_key": pair_key,
        "class_name": args.class_name,
        "class_id": target_id,
        "main_component": {
            "area": main_candidate.area,
            "bbox": list(main_candidate.bbox),
            "restored_as_target_pixels": main_restored,
        },
        "newly_classified_pixels": int(np.count_nonzero(changed)),
        "boundary_connected_newly_classified_pixels": boundary_filled,
        "newly_classified_without_fallback_provenance": (
            changed_outside_candidates
        ),
        "fallback": outcome.metrics,
    }
    palette = _palette(semantic_dir / "render_palette.json")
    _save_png(
        output_root / "semantic_after.png",
        _colorize(after, palette, dataset.class_spec.ignore_id),
    )
    _save_png(output_root / "class_provenance.png", provenance)
    x, y, width, height = main_candidate.bbox
    pad = 32
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(before.shape[1], x + width + pad), min(
        before.shape[0], y + height + pad
    )
    before_color = _colorize(before, palette, dataset.class_spec.ignore_id)
    after_color = _colorize(after, palette, dataset.class_spec.ignore_id)
    comparison = np.concatenate(
        (before_color[y0:y1, x0:x1], after_color[y0:y1, x0:x1]), axis=1
    )
    _save_png(output_root / "main_component_before_after.png", comparison)
    write_json_atomic(output_root / "acceptance.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
