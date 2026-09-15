"""Prepare raw integer label rasters for the canonical semantic contract."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Tuple

import numpy as np
from PIL import Image

from semantic_atdom.semantics.schema import (
    SemanticClassSpec,
    SemanticSchemaError,
    sha256_file,
)


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class IdLabelAdapterResult:
    dataset_root: Path
    image_count: int
    adapter_spec_sha256: str
    manifest_path: Path


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_adapter_spec(path: Path) -> Tuple[Mapping[str, object], SemanticClassSpec, Dict[int, int]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SemanticSchemaError(
            "unable to read semantic adapter spec {}: {}".format(path, error)
        )
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise SemanticSchemaError("unsupported semantic adapter spec")
    class_payload = payload.get("class_spec")
    if not isinstance(class_payload, Mapping):
        raise SemanticSchemaError("adapter spec requires class_spec")
    class_spec = SemanticClassSpec.from_dict(class_payload)
    raw_mapping = payload.get("source_id_to_class_id")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise SemanticSchemaError("adapter spec requires source_id_to_class_id")
    try:
        mapping = {int(source): int(target) for source, target in raw_mapping.items()}
    except (TypeError, ValueError):
        raise SemanticSchemaError("adapter source and target ids must be integers")
    valid_targets = set(range(class_spec.num_classes)) | {class_spec.ignore_id}
    if any(source < 0 or source > 255 for source in mapping):
        raise SemanticSchemaError("adapter source ids must be in [0, 255]")
    if any(target not in valid_targets for target in mapping.values()):
        raise SemanticSchemaError("adapter target ids are outside the class spec")
    label_version = payload.get("label_version")
    producer = payload.get("producer")
    if not isinstance(label_version, str) or not label_version.strip():
        raise SemanticSchemaError("adapter spec requires label_version")
    if not isinstance(producer, str) or not producer.strip():
        raise SemanticSchemaError("adapter spec requires producer")
    return payload, class_spec, mapping


def _image_index(root: Path) -> Dict[str, Path]:
    index = {}
    try:
        paths = tuple(root.iterdir())
    except OSError as error:
        raise SemanticSchemaError("unable to read image root {}: {}".format(root, error))
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if path.stem in index:
            raise SemanticSchemaError(
                "duplicate image id {} in {}".format(path.stem, root)
            )
        index[path.stem] = path
    return index


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def prepare_id_label_dataset(
    *,
    label_root: Path,
    image_root: Path,
    adapter_spec_path: Path,
    output_root: Path,
) -> IdLabelAdapterResult:
    """Convert one explicitly described uint8 ID-label layout atomically."""
    label_root = Path(label_root).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    adapter_spec_path = Path(adapter_spec_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not label_root.is_dir() or not image_root.is_dir():
        raise SemanticSchemaError("label_root and image_root must exist")
    if output_root.exists():
        raise SemanticSchemaError("semantic adapter output already exists: {}".format(output_root))

    payload, class_spec, mapping = _load_adapter_spec(adapter_spec_path)
    adapter_hash = _canonical_sha256(payload)
    images = _image_index(image_root)
    label_paths = tuple(
        sorted(
            path
            for path in label_root.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
    )
    if not label_paths:
        raise SemanticSchemaError("no PNG ID labels found in {}".format(label_root))
    missing_images = [path.stem for path in label_paths if path.stem not in images]
    if missing_images:
        raise SemanticSchemaError(
            "raw labels are missing source RGB images: {}".format(missing_images[:10])
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    semantic = staging / "semantic"
    labels_dir = semantic / "labels"
    metadata_dir = semantic / "metadata"
    labels_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    _write_json(semantic / "class_spec.json", class_spec.to_dict())

    lookup = np.full(256, class_spec.ignore_id, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        lookup[source_id] = target_id
    records = []
    observed_source_ids = set()
    for source_path in label_paths:
        try:
            with Image.open(str(source_path)) as image:
                source_labels = np.asarray(image).copy()
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to read raw label {}: {}".format(source_path, error)
            )
        if source_labels.ndim != 2 or source_labels.dtype != np.uint8:
            raise SemanticSchemaError(
                "raw ID label must be a single-channel uint8 PNG: {}".format(source_path)
            )
        source_ids = set(int(value) for value in np.unique(source_labels))
        unmapped = source_ids - set(mapping)
        if unmapped:
            raise SemanticSchemaError(
                "raw label {} contains unmapped source ids {}".format(
                    source_path, sorted(unmapped)
                )
            )
        observed_source_ids.update(source_ids)
        labels = lookup[source_labels]
        label_path = labels_dir / source_path.name
        Image.fromarray(labels, mode="L").save(str(label_path), format="PNG")
        rgb_path = images[source_path.stem]
        height, width = labels.shape
        metadata = {
            "schema_version": 1,
            "image_id": source_path.stem,
            "label_version": payload["label_version"],
            "model": payload["producer"],
            "checkpoint_sha256": adapter_hash,
            "prompt_config": {
                "adapter": "explicit-id-map-v1",
                "adapter_spec_sha256": adapter_hash,
                "source_id_to_class_id": payload["source_id_to_class_id"],
            },
            "thresholds": {},
            "width": width,
            "height": height,
            "class_spec_sha256": class_spec.fingerprint,
            "label_sha256": sha256_file(label_path),
            "source_image_sha256": sha256_file(rgb_path),
        }
        metadata_path = metadata_dir / (source_path.stem + ".json")
        _write_json(metadata_path, metadata)
        records.append(
            {
                "image_id": source_path.stem,
                "source_label_sha256": sha256_file(source_path),
                "label_sha256": metadata["label_sha256"],
                "source_image_sha256": metadata["source_image_sha256"],
            }
        )

    manifest = {
        "schema_version": 1,
        "adapter_spec_path": str(adapter_spec_path),
        "adapter_spec_sha256": adapter_hash,
        "class_spec_sha256": class_spec.fingerprint,
        "source_label_root": str(label_root),
        "source_image_root": str(image_root),
        "image_count": len(records),
        "observed_source_ids": sorted(observed_source_ids),
        "records": records,
    }
    manifest_path = staging / "adapter_manifest.json"
    _write_json(manifest_path, manifest)
    os.replace(str(staging), str(output_root))
    return IdLabelAdapterResult(
        dataset_root=output_root,
        image_count=len(records),
        adapter_spec_sha256=adapter_hash,
        manifest_path=output_root / "adapter_manifest.json",
    )
