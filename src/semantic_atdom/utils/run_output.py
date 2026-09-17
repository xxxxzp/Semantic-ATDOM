"""Self-describing output contract for progressive training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, Mapping, Optional


RUN_OUTPUT_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_value(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_revision(repository_root: Path) -> Dict[str, Any]:
    repository_root = Path(repository_root)

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=str(repository_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def normalized_run_name(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip())
    result = result.strip("-._")
    if not result or result in {"latest", ".work"}:
        raise ValueError("run name must contain a safe descriptive token")
    return result


def allocate_run_root(
    output_root: Path,
    run_name: str,
    *,
    repository_root: Path,
    timestamp: Optional[datetime] = None,
) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    revision = git_revision(repository_root).get("commit") or "unknown"
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_id = "{}_{}_{}".format(
        normalized_run_name(run_name), stamp, str(revision)[:7]
    )
    return output_root / "runs" / run_id


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def from_root(cls, root: Path) -> "RunLayout":
        return cls(Path(root).expanduser().resolve())

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def semantic_model(self) -> Path:
        return self.model / "semantic"

    @property
    def products(self) -> Path:
        return self.root / "products"

    @property
    def maps(self) -> Path:
        return self.products / "maps"

    @property
    def map_releases(self) -> Path:
        return self.maps / "releases"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def work(self) -> Path:
        return self.root / ".work"

    @property
    def stages(self) -> Path:
        return self.work / "stages"

    @property
    def semantic_work(self) -> Path:
        return self.work / "semantic"

    @property
    def recovery(self) -> Path:
        return self.work / "recovery"

    @property
    def cache(self) -> Path:
        return self.work / "cache"

    @property
    def video_frames(self) -> Path:
        return self.work / "video_frames"

    def relative(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def prepare(self) -> None:
        if self.manifest.exists():
            raise FileExistsError("run output already has a manifest: {}".format(self.root))
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                "run output must not reuse a non-empty directory: {}".format(
                    self.root
                )
            )
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.metrics,
            self.logs,
            self.provenance,
            self.map_releases,
            self.stages,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def stage(self, stage: int) -> Path:
        return self.stages / str(int(stage))

    def read_manifest(self) -> Dict[str, Any]:
        with self.manifest.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def begin(
        self,
        *,
        source_dataset: Path,
        repository_root: Path,
        command: list[str],
        options: Mapping[str, Any],
    ) -> None:
        self.prepare()
        write_json_atomic(
            self.manifest,
            {
                "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
                "run_id": self.root.name,
                "status": "running",
                "started_at": _utc_now(),
                "finished_at": None,
                "source_dataset": str(Path(source_dataset).expanduser().resolve()),
                "git": git_revision(repository_root),
                "command": list(command),
                "options": dict(options),
                "artifacts": {},
                "failure": None,
            },
        )

    def set_status(
        self,
        status: str,
        *,
        artifacts: Optional[Mapping[str, Any]] = None,
        failure: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if status not in {"running", "succeeded", "failed", "interrupted"}:
            raise ValueError("unsupported run status: {}".format(status))
        payload = self.read_manifest()
        payload["status"] = status
        payload["finished_at"] = None if status == "running" else _utc_now()
        if artifacts is not None:
            payload["artifacts"] = dict(artifacts)
        payload["failure"] = None if failure is None else dict(failure)
        write_json_atomic(self.manifest, payload)

    def write_config(self, arguments: Mapping[str, Any]) -> Path:
        internal_keys = {
            "Model_Path_Dir",
            "run_root",
            "work_dir",
            "stage_output_dir",
            "semantic_work_dir",
            "map_output_dir",
            "metrics_dir",
            "logs_dir",
            "provenance_dir",
            "renders_dir",
            "videos_dir",
        }
        payload = {}
        for key, value in arguments.items():
            if value and (
                key in internal_keys
                or (isinstance(value, str) and os.path.isabs(value))
            ):
                try:
                    value = self.relative(Path(value))
                except (OSError, ValueError):
                    value = str(value)
            payload[key] = _json_value(value)
        path = self.provenance / "config.json"
        write_json_atomic(path, {"schema_version": 1, "arguments": payload})
        return path

    def publish_latest(self) -> Path:
        outputs_root = self.root.parent.parent if self.root.parent.name == "runs" else self.root.parent
        latest = outputs_root / "latest"
        temporary = outputs_root / ".latest.tmp"
        target = os.path.relpath(self.root, outputs_root)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, latest)
        return latest


def move_file_verified(source: Path, destination: Path) -> Dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    size = source.stat().st_size
    digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    if destination.stat().st_size != size or sha256_file(destination) != digest:
        raise RuntimeError("moved artifact failed verification: {}".format(destination))
    return {"path": destination, "size_bytes": size, "sha256": digest}


def finalize_stage_model(
    layout: RunLayout,
    final_stage: int,
    *,
    include_training_state: bool = False,
) -> Dict[str, Any]:
    stage_root = layout.stage(final_stage)
    if not stage_root.is_dir():
        raise RuntimeError("final stage output is missing: {}".format(stage_root))
    point_cloud = stage_root / "point_cloud" / "iteration_{}".format(final_stage) / "point_cloud.ply"
    records = {
        "gaussians": move_file_verified(point_cloud, layout.model / "gaussians.ply"),
    }
    checkpoint = stage_root / "chkpnt{}.pth".format(final_stage)
    checkpoint_metadata = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if include_training_state:
        records["training_state"] = move_file_verified(
            checkpoint, layout.model / "training_state.pt"
        )
        if checkpoint_metadata.is_file():
            records["training_state_metadata"] = move_file_verified(
                checkpoint_metadata, layout.model / "training_state.json"
            )
    optional = {
        "cameras": (stage_root / "cameras.json", layout.model / "cameras.json"),
        "config_compatibility": (stage_root / "cfg_args", layout.model / "cfg_args"),
        "sparse_input": (stage_root / "input.ply", layout.provenance / "final_input.ply"),
    }
    for name, (source, destination) in optional.items():
        if source.is_file():
            records[name] = move_file_verified(source, destination)
    for event in stage_root.glob("events.out.tfevents.*"):
        records.setdefault("tensorboard", []).append(
            move_file_verified(event, layout.logs / "tensorboard" / event.name)
        )
    shutil.rmtree(stage_root)
    return records


def archive_semantic_state(layout: RunLayout) -> Dict[str, Any]:
    source = layout.semantic_work
    latest_path = source / "latest.json"
    if not latest_path.is_file():
        raise RuntimeError("final semantic pointer is missing: {}".format(latest_path))
    with latest_path.open("r", encoding="utf-8") as stream:
        latest = json.load(stream)
    slot = int(latest["slot"])
    state_source = source / "semantic_state.{}.pt".format(slot)
    context_source = source / "recovery_context.{}.json".format(slot)
    geometry_source = source / "geometry_checkpoint.{}.pt".format(slot)
    if sha256_file(state_source) != latest["semantic_state_sha256"]:
        raise RuntimeError("active semantic state hash does not match latest pointer")
    if sha256_file(geometry_source) != latest["geometry_checkpoint_sha256"]:
        raise RuntimeError("active semantic geometry hash does not match latest pointer")
    records = {
        "state": move_file_verified(state_source, layout.semantic_model / "state.pt"),
        "recovery_context": move_file_verified(
            context_source, layout.semantic_model / "recovery_context.json"
        ),
        "geometry": move_file_verified(
            geometry_source, layout.semantic_model / "geometry.pt"
        ),
    }
    for name in ("ledger.json", "sync_status.json", "final_rebuild_metrics.json"):
        candidate = source / name
        if candidate.is_file():
            records[name.removesuffix(".json")] = move_file_verified(
                candidate, layout.semantic_model / name
            )
    archive = dict(latest)
    archive.update(
        {
            "schema_version": 1,
            "source_slot_schema_version": latest.get("schema_version"),
            "state": "state.pt",
            "state_sha256": records["state"]["sha256"],
            "recovery_context": "recovery_context.json",
            "geometry": "geometry.pt",
            "geometry_sha256": records["geometry"]["sha256"],
        }
    )
    write_json_atomic(layout.semantic_model / "manifest.json", archive)
    records["manifest"] = {
        "path": layout.semantic_model / "manifest.json",
        "size_bytes": (layout.semantic_model / "manifest.json").stat().st_size,
        "sha256": sha256_file(layout.semantic_model / "manifest.json"),
    }
    return records


def relative_artifact_records(layout: RunLayout, value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "path" and isinstance(item, Path):
                result[key] = layout.relative(item)
            else:
                result[key] = relative_artifact_records(layout, item)
        return result
    if isinstance(value, list):
        return [relative_artifact_records(layout, item) for item in value]
    return value
