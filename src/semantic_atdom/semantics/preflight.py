"""Read-only validation for progressive RGB/SfM and semantic inputs."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from semantic_atdom.semantics.orchestrator import AcquisitionOverlap
from semantic_atdom.semantics.schema import SemanticDataset, SemanticSchemaError, sha256_file


_IMAGE_ID = re.compile(r"^[^/\\\x00]+$")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _save_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_colmap_image_names(path: Path) -> Tuple[str, ...]:
    """Read image names while skipping large points2D payloads."""
    names = []
    try:
        with path.open("rb") as stream:
            count_raw = stream.read(8)
            if len(count_raw) != 8:
                raise ValueError("truncated image count")
            count = struct.unpack("<Q", count_raw)[0]
            for _ in range(count):
                if len(stream.read(64)) != 64:
                    raise ValueError("truncated image record")
                encoded = bytearray()
                while True:
                    value = stream.read(1)
                    if not value:
                        raise ValueError("unterminated image name")
                    if value == b"\0":
                        break
                    encoded.extend(value)
                name = encoded.decode("utf-8")
                points_raw = stream.read(8)
                if len(points_raw) != 8:
                    raise ValueError("truncated points2D count")
                point_count = struct.unpack("<Q", points_raw)[0]
                stream.seek(24 * point_count, os.SEEK_CUR)
                image_id = Path(name).stem
                if _IMAGE_ID.fullmatch(image_id) is None:
                    raise ValueError("image name has an invalid stem: {}".format(name))
                names.append((image_id, name))
    except (OSError, UnicodeError, struct.error, ValueError) as error:
        raise SemanticSchemaError(
            "unable to read COLMAP image identities from {}: {}".format(path, error)
        )
    image_ids = tuple(item[0] for item in names)
    if len(image_ids) != len(set(image_ids)):
        raise SemanticSchemaError("COLMAP image identities must be unique")
    return tuple(names)


def _stage_sparse_root(stage_root: Path) -> Path:
    sparse = stage_root / "sparse" / "0"
    if sparse.is_dir():
        return sparse
    binary = stage_root / "bin"
    if binary.is_dir():
        return binary
    raise SemanticSchemaError("stage has neither sparse/0 nor bin: {}".format(stage_root))


def _index_rgb_images(root: Path) -> Dict[str, Path]:
    try:
        paths = tuple(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
    except OSError as error:
        raise SemanticSchemaError(
            "unable to read RGB image directory {}: {}".format(root, error)
        )
    index = {}
    for path in paths:
        image_id = path.stem
        if image_id in index:
            raise SemanticSchemaError(
                "RGB image id {} is ambiguous in {}".format(image_id, root)
            )
        index[image_id] = path
    return index


@dataclass(frozen=True)
class DatasetPreflightResult:
    report_path: Path
    report_sha256: str
    selected_stages: Tuple[int, ...]
    final_image_ids: Tuple[str, ...]
    labels_complete: bool
    source_binding_complete: bool

    @property
    def authoritative_exact_complete(self) -> bool:
        return self.labels_complete and self.source_binding_complete


def run_dataset_preflight(
    *,
    dataset_root: Path,
    semantic_data_root: Path,
    output_root: Path,
    start_stage: int,
    end_stage: int,
    image_folder_path: Optional[Path] = None,
) -> DatasetPreflightResult:
    """Validate selected stages and semantic observations without writing inputs."""
    if semantic_data_root is None or not str(semantic_data_root).strip():
        raise SemanticSchemaError(
            "--semantic_data_root is required when semantics are enabled"
        )
    dataset_root = Path(dataset_root).expanduser().resolve()
    semantic_data_root = Path(semantic_data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SemanticSchemaError("dataset root does not exist: {}".format(dataset_root))
    if not semantic_data_root.is_dir():
        raise SemanticSchemaError(
            "semantic data root does not exist: {}".format(semantic_data_root)
        )
    if _is_within(output_root, dataset_root) or _is_within(output_root, semantic_data_root):
        raise SemanticSchemaError("output root must be outside every read-only input root")

    available = sorted(
        int(path.name)
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    effective_end = max(available) if int(end_stage) < int(start_stage) else int(end_stage)
    selected = tuple(stage for stage in available if int(start_stage) <= stage <= effective_end)
    expected = tuple(range(int(start_stage), effective_end + 1))
    if selected != expected:
        missing = sorted(set(expected) - set(selected))
        raise SemanticSchemaError(
            "progressive stage range is incomplete; missing {}".format(missing)
        )

    stage_records = []
    previous_ids = set()
    final_names = ()
    shared_image_root = (
        Path(image_folder_path).expanduser().resolve()
        if image_folder_path is not None and str(image_folder_path).strip()
        else None
    )
    shared_rgb_index = (
        _index_rgb_images(shared_image_root) if shared_image_root is not None else None
    )
    for stage in selected:
        stage_root = dataset_root / str(stage)
        sparse_root = _stage_sparse_root(stage_root)
        names_path, matrix_path = AcquisitionOverlap.input_paths(stage_root)
        required = (
            sparse_root / "images.bin",
            sparse_root / "cameras.bin",
            sparse_root / "points3D.bin",
            names_path,
            matrix_path,
        )
        missing_files = [str(path) for path in required if not path.is_file()]
        if missing_files:
            raise SemanticSchemaError(
                "stage {} is missing required files: {}".format(stage, missing_files)
            )
        snapshot_image_names = _read_colmap_image_names(sparse_root / "images.bin")
        snapshot_ids = {item[0] for item in snapshot_image_names}
        overlap = AcquisitionOverlap.load(stage_root)
        image_ids = overlap.image_ids
        current_ids = set(image_ids)
        identity_set_is_cumulative = previous_ids.issubset(current_ids)
        if not snapshot_ids.issubset(current_ids):
            raise SemanticSchemaError(
                "stage {} local COLMAP identities are absent from cumulative overlap identities".format(
                    stage
                )
            )
        stage_image_root = (
            shared_image_root if shared_image_root is not None else stage_root / "images"
        )
        rgb_index = (
            shared_rgb_index
            if shared_rgb_index is not None
            else _index_rgb_images(stage_image_root)
        )
        absent_rgb = sorted(current_ids - set(rgb_index))
        if absent_rgb:
            raise SemanticSchemaError(
                "stage {} is missing RGB images: {}".format(stage, absent_rgb[:10])
            )
        stage_records.append(
            {
                "stage": stage,
                "camera_count": len(image_ids),
                "snapshot_camera_count": len(snapshot_ids),
                "stage_number_matches_camera_count": len(image_ids) == stage,
                "identity_set_is_cumulative": identity_set_is_cumulative,
                "new_image_ids": sorted(current_ids - previous_ids),
                "dropped_image_ids": sorted(previous_ids - current_ids),
                "images_bin_sha256": sha256_file(sparse_root / "images.bin"),
                "cameras_bin_sha256": sha256_file(sparse_root / "cameras.bin"),
                "images_names_sha256": sha256_file(names_path),
                "overlap_matrix_sha256": sha256_file(matrix_path),
            }
        )
        previous_ids = current_ids
        final_names = tuple((image_id, rgb_index[image_id].name) for image_id in image_ids)

    semantic_dataset = SemanticDataset(semantic_data_root)
    class_counts = np.zeros(semantic_dataset.class_spec.num_classes, dtype=np.int64)
    ignore_count = 0
    source_verified = []
    source_unverified = []
    labels = []
    final_stage_root = dataset_root / str(selected[-1])
    final_image_root = shared_image_root or final_stage_root / "images"
    for image_id, image_name in sorted(final_names):
        observation = semantic_dataset.load_observation(image_id)
        rgb_path = final_image_root / image_name
        with Image.open(str(rgb_path)) as image:
            rgb_size = image.size
        label_size = (observation.labels.shape[1], observation.labels.shape[0])
        if rgb_size != label_size:
            raise SemanticSchemaError(
                "RGB/label dimensions differ for {}: {} != {}".format(
                    image_id, rgb_size, label_size
                )
            )
        valid = observation.labels != semantic_dataset.class_spec.ignore_id
        if bool(valid.any()):
            class_counts += np.bincount(
                observation.labels[valid].astype(np.int64),
                minlength=semantic_dataset.class_spec.num_classes,
            )
        ignore_count += int((~valid).sum())
        rgb_hash = sha256_file(rgb_path)
        expected_source_hash = observation.metadata.get("source_image_sha256")
        if expected_source_hash is None:
            source_unverified.append(image_id)
        elif expected_source_hash != rgb_hash:
            raise SemanticSchemaError(
                "metadata source_image_sha256 does not match RGB for {}".format(image_id)
            )
        else:
            source_verified.append(image_id)
        labels.append(
            {
                "image_id": image_id,
                "label_version": observation.key.label_version,
                "label_sha256": observation.metadata["label_sha256"],
                "rgb_sha256": rgb_hash,
                "source_binding": (
                    "verified" if image_id in source_verified else "unverified"
                ),
            }
        )

    report = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "semantic_data_root": str(semantic_data_root),
        "selected_stages": list(selected),
        "stage_records": stage_records,
        "final_image_ids": [item[0] for item in sorted(final_names)],
        "labels_complete": len(labels) == len(final_names),
        "source_binding_complete": not source_unverified,
        "authoritative_exact_complete": (
            len(labels) == len(final_names) and not source_unverified
        ),
        "source_binding_verified_image_ids": source_verified,
        "source_binding_unverified_image_ids": source_unverified,
        "class_spec_sha256": semantic_dataset.class_spec.fingerprint,
        "ontology_sha256": semantic_dataset.class_spec.ontology_fingerprint,
        "class_pixel_counts": class_counts.tolist(),
        "ignore_pixel_count": ignore_count,
        "observations": labels,
    }
    dataset_id = hashlib.sha256(str(dataset_root).encode("utf-8")).hexdigest()[:16]
    report_path = output_root / "cache" / dataset_id / "preflight.json"
    _save_json_atomic(report_path, report)
    report_hash = sha256_file(report_path)
    _save_json_atomic(
        report_path.with_name("preflight.sha256.json"),
        {"schema_version": 1, "preflight_sha256": report_hash},
    )
    return DatasetPreflightResult(
        report_path=report_path,
        report_sha256=report_hash,
        selected_stages=selected,
        final_image_ids=tuple(item[0] for item in sorted(final_names)),
        labels_complete=len(labels) == len(final_names),
        source_binding_complete=not source_unverified,
    )
