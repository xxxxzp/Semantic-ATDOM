#!/usr/bin/env python3
"""Migrate one validated legacy progressive run into output schema v2."""

from __future__ import annotations

from argparse import ArgumentParser
import ast
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

from semantic_atdom.utils.run_output import (
    RUN_OUTPUT_SCHEMA_VERSION,
    sha256_file,
    write_json_atomic,
)


def _inventory(root: Path, excluded: Path):
    records = []
    for path in sorted(root.rglob("*")):
        if excluded == path or excluded in path.parents or not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    if (
        source.stat().st_size != destination.stat().st_size
        or sha256_file(source) != sha256_file(destination)
    ):
        raise RuntimeError("hard-linked artifact failed verification: {}".format(source))


def _link_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link(path, target)


def _parse_namespace(path: Path):
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "Namespace"
        or expression.args
    ):
        raise ValueError("legacy cfg_args is not a Namespace keyword call")
    return {
        item.arg: ast.literal_eval(item.value)
        for item in expression.keywords
        if item.arg is not None
    }


def _legacy_metrics(root: Path, staging: Path):
    metrics = staging / "metrics"
    metrics.mkdir(parents=True)
    evaluation_source = root / "Intermediate_Results" / "eval.txt"
    timing_source = root / "Intermediate_Results" / "TimeCost.txt"
    final_evaluation = None
    if evaluation_source.is_file():
        _link(evaluation_source, metrics / "evaluations_legacy.txt")
        pattern = re.compile(
            r"\[ITER (?P<iteration>\d+)\] Evaluating (?P<split>\w+): "
            r"L1 (?P<l1>\S+) Gaussians (?P<gaussians>\d+) "
            r"PSNR (?P<psnr>\S+) SSIM (?P<ssim>\S+) LPIPS (?P<lpips>\S+)"
        )
        for line in evaluation_source.read_text(encoding="utf-8").splitlines():
            match = pattern.fullmatch(line)
            if match and float(match.group("ssim")) != 0.0:
                final_evaluation = {
                    "schema_version": 1,
                    "iteration": int(match.group("iteration")),
                    "split": match.group("split"),
                    "l1": float(match.group("l1")),
                    "gaussians": int(match.group("gaussians")),
                    "psnr": float(match.group("psnr")),
                    "ssim": float(match.group("ssim")),
                    "lpips": float(match.group("lpips")),
                    "complete_metrics": True,
                }
        if final_evaluation is not None:
            write_json_atomic(metrics / "final_evaluation.json", final_evaluation)
    timing = {}
    if timing_source.is_file():
        _link(timing_source, metrics / "timing_legacy.txt")
        for line in timing_source.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            timing[key] = float(value.rstrip("s"))
        write_json_atomic(metrics / "timing.json", {"schema_version": 1, **timing})
    return final_evaluation, timing


def _legacy_camera_renders(root: Path, staging: Path):
    source = root / "OutputImages"
    if not source.is_dir():
        return 0
    destination = staging / "renders" / "cameras"
    destination.mkdir(parents=True)
    rows = []
    for camera in sorted(path for path in source.iterdir() if path.is_dir()):
        images = sorted(camera.glob("PredictionImages*.jpg"))
        if len(images) != 1:
            raise RuntimeError("legacy camera render is ambiguous: {}".format(camera))
        target = destination / (camera.name + ".jpg")
        _link(images[0], target)
        metric_path = camera / "PSNR.txt"
        match = re.fullmatch(
            r"(\d+): ([^,]+), Visible_Gaussians_Num: (\d+)\n?",
            metric_path.read_text(encoding="utf-8"),
        )
        if match is None:
            raise RuntimeError("legacy per-camera metric is invalid: {}".format(metric_path))
        rows.append(
            {
                "camera_id": camera.name,
                "iteration": int(match.group(1)),
                "psnr": float(match.group(2)),
                "visible_gaussians_legacy_tensor_length": int(match.group(3)),
                "render": "renders/cameras/{}.jpg".format(camera.name),
            }
        )
    with (staging / "metrics" / "per_camera.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def migrate(root: Path, final_stage: int, apply: bool) -> None:
    root = root.expanduser().resolve()
    stage = root / str(final_stage)
    legacy_maps = root / "Map_Products"
    legacy_semantic = root / "semantic"
    if (root / "manifest.json").exists():
        raise FileExistsError("run already has a published manifest")
    for required in (stage, legacy_maps, legacy_semantic):
        if not required.exists():
            raise FileNotFoundError(required)
    staging = root.with_name(root.name + ".migration-output-v2")
    backup = root.with_name(root.name + ".migration-backup")
    if staging.exists():
        raise FileExistsError("stale migration staging directory: {}".format(staging))
    if backup.exists():
        raise FileExistsError("stale migration backup directory: {}".format(backup))

    inventory = _inventory(root, staging)
    print(
        "Validated legacy inventory: {} files, {} bytes".format(
            len(inventory), sum(item["size_bytes"] for item in inventory)
        )
    )
    if not apply:
        print("Dry run only; pass --apply to migrate in place")
        return

    staging.mkdir()
    try:
        provenance = staging / "provenance"
        provenance.mkdir()
        write_json_atomic(
            provenance / "migration_inventory.json",
            {
                "schema_version": 1,
                "source_layout": "legacy-progressive",
                "files": inventory,
            },
        )

        point_cloud = stage / "point_cloud" / "iteration_{}".format(final_stage) / "point_cloud.ply"
        _link(point_cloud, staging / "model" / "gaussians.ply")
        _link(stage / "chkpnt{}.pth".format(final_stage), staging / "model" / "training_state.pt")
        for source_name, destination in (
            ("cameras.json", staging / "model" / "cameras.json"),
            ("cfg_args", staging / "model" / "cfg_args"),
            ("input.ply", provenance / "final_input.ply"),
        ):
            source = stage / source_name
            if source.is_file():
                _link(source, destination)
        config = _parse_namespace(stage / "cfg_args")
        config["model_path"] = "."
        config["Model_Path_Dir"] = "."
        if config.get("Source_Path_Dir"):
            config["source_path"] = str(
                Path(config["Source_Path_Dir"]) / str(final_stage)
            )
        for key, value in tuple(config.items()):
            if not isinstance(value, str) or not os.path.isabs(value):
                continue
            try:
                config[key] = Path(value).resolve().relative_to(root).as_posix()
            except ValueError:
                pass
        write_json_atomic(
            provenance / "config.json",
            {"schema_version": 1, "arguments": config, "migrated": True},
        )

        legacy_latest = json.loads((legacy_maps / "latest.json").read_text())
        legacy_pair = legacy_maps / legacy_latest["pair_manifest"]
        if sha256_file(legacy_pair) != legacy_latest["pair_manifest_sha256"]:
            raise RuntimeError("legacy map latest hash is invalid")
        release = legacy_pair.parent
        release_destination = staging / "products" / "maps" / "releases" / release.name
        _link_tree(release, release_destination)
        grid_source = legacy_maps / "map_grid.json"
        if grid_source.is_file():
            _link(grid_source, staging / "products" / "maps" / "grid.json")
        write_json_atomic(
            staging / "products" / "maps" / "latest.json",
            {
                "schema_version": 1,
                "product_key": legacy_latest["product_key"],
                "pair_manifest": "releases/{}/pair_manifest.json".format(release.name),
                "pair_manifest_sha256": legacy_latest["pair_manifest_sha256"],
            },
        )

        semantic_latest = json.loads((legacy_semantic / "latest.json").read_text())
        slot = int(semantic_latest["slot"])
        state_source = legacy_semantic / "semantic_state.{}.pt".format(slot)
        geometry_source = legacy_semantic / "geometry_checkpoint.{}.pt".format(slot)
        if sha256_file(state_source) != semantic_latest["semantic_state_sha256"]:
            raise RuntimeError("legacy active semantic state hash is invalid")
        if sha256_file(geometry_source) != semantic_latest["geometry_checkpoint_sha256"]:
            raise RuntimeError("legacy active semantic geometry hash is invalid")
        semantic_model = staging / "model" / "semantic"
        _link(state_source, semantic_model / "state.pt")
        _link(geometry_source, semantic_model / "geometry.pt")
        _link(
            legacy_semantic / "recovery_context.{}.json".format(slot),
            semantic_model / "recovery_context.json",
        )
        for name in ("ledger.json", "sync_status.json", "final_rebuild_metrics.json"):
            source = legacy_semantic / name
            if source.is_file():
                _link(source, semantic_model / name)
        _link(release / "semantic" / "class_spec.json", semantic_model / "schema.json")
        semantic_manifest = dict(semantic_latest)
        semantic_manifest.update(
            {
                "schema_version": 1,
                "source_slot_schema_version": semantic_latest.get("schema_version"),
                "state": "state.pt",
                "state_sha256": sha256_file(semantic_model / "state.pt"),
                "geometry": "geometry.pt",
                "geometry_sha256": sha256_file(semantic_model / "geometry.pt"),
                "recovery_context": "recovery_context.json",
                "migrated_from_slot": slot,
            }
        )
        write_json_atomic(semantic_model / "manifest.json", semantic_manifest)

        final_evaluation, timing = _legacy_metrics(root, staging)
        if final_evaluation is None:
            raise RuntimeError(
                "legacy run has no complete final evaluation metadata"
            )
        camera_count = _legacy_camera_renders(root, staging)
        logs = staging / "logs"
        logs.mkdir()
        diary = root / "Diary.txt"
        if diary.is_file() and diary.stat().st_size:
            _link(diary, logs / "training.log")
        for old_name, new_name in (
            ("environment_fingerprint.json", "environment_start.json"),
            ("final_environment_fingerprint.json", "environment_end.json"),
        ):
            source = root / old_name
            if source.is_file():
                _link(source, provenance / new_name)
        cache_candidates = list((root / "cache").glob("*/preflight.json"))
        if len(cache_candidates) == 1:
            _link(cache_candidates[0], provenance / "dataset_preflight.json")

        pair_payload = json.loads((release_destination / "pair_manifest.json").read_text())
        sync_payload = json.loads((semantic_model / "sync_status.json").read_text())
        summary = {
            "schema_version": 1,
            "final_stage": final_stage,
            "evaluation": final_evaluation,
            "timing": timing,
            "semantic": sync_payload,
            "map": pair_payload,
            "per_camera_count": camera_count,
            "migrated": True,
        }
        write_json_atomic(staging / "metrics" / "summary.json", summary)
        artifacts = {
            "model": "model",
            "maps": "products/maps",
            "metrics": "metrics",
            "logs": "logs",
            "provenance": "provenance",
        }
        if camera_count:
            artifacts["renders"] = {
                "cameras": {
                    "path": "renders/cameras",
                    "count": camera_count,
                }
            }
        write_json_atomic(
            staging / "manifest.json",
            {
                "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
                "run_id": root.name,
                "status": "succeeded",
                "started_at": None,
                "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source_dataset": config.get("Source_Path_Dir"),
                "git": {"commit": None, "branch": None, "dirty": None},
                "command": None,
                "options": {"migrated": True, "legacy_output": True},
                "final_stage": int(final_stage),
                "optimizer_iteration": int(final_evaluation["iteration"]),
                "artifacts": artifacts,
                "failure": None,
            },
        )

        # Commit the complete schema-v2 tree with one directory exchange.  A
        # failure before the second rename restores the untouched legacy run;
        # consumers therefore never observe a partially installed layout.
        os.replace(root, backup)
        try:
            os.replace(staging, root)
        except Exception:
            os.replace(backup, root)
            raise
        try:
            shutil.rmtree(backup)
        except OSError as error:
            print(
                "Migration committed; legacy backup cleanup is incomplete: {}".format(
                    error
                ),
                file=sys.stderr,
            )
        print("Migration complete: {}".format(root))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not root.exists():
            os.replace(backup, root)
        raise


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--final-stage", type=int, default=487)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    migrate(arguments.run_root, arguments.final_stage, arguments.apply)


if __name__ == "__main__":
    sys.exit(main())
