"""Versioned file contract between semantic label producers and A-TDOM."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


class SemanticSchemaError(ValueError):
    """Raised when semantic inputs violate the data contract."""


@dataclass(frozen=True)
class SemanticClass:
    class_id: int
    name: str
    color: Tuple[int, int, int]
    is_background: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SemanticClass":
        if not isinstance(value, Mapping):
            raise SemanticSchemaError("each class must be a JSON object")
        class_id = value.get("id")
        name = value.get("name")
        color = value.get("color")
        is_background = value.get("is_background", False)
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise SemanticSchemaError("class id must be an integer")
        if not isinstance(name, str) or not name.strip():
            raise SemanticSchemaError("class name must be a non-empty string")
        if (
            not isinstance(color, Sequence)
            or isinstance(color, (str, bytes))
            or len(color) != 3
            or any(
                isinstance(channel, bool) or not isinstance(channel, int)
                for channel in color
            )
            or any(channel < 0 or channel > 255 for channel in color)
        ):
            raise SemanticSchemaError(
                "class color must contain three integers in [0, 255]"
            )
        if not isinstance(is_background, bool):
            raise SemanticSchemaError("is_background must be a boolean")
        return cls(
            class_id=class_id,
            name=name.strip(),
            color=tuple(color),
            is_background=is_background,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.class_id,
            "name": self.name,
            "color": list(self.color),
            "is_background": self.is_background,
        }


@dataclass(frozen=True)
class SemanticClassSpec:
    schema_version: int
    ignore_id: int
    classes: Tuple[SemanticClass, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SemanticClassSpec":
        schema_version = value.get("schema_version")
        ignore_id = value.get("ignore_id")
        raw_classes = value.get("classes")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 1
        ):
            raise SemanticSchemaError("schema_version must be a positive integer")
        if ignore_id != 255:
            raise SemanticSchemaError("Semantic A-TDOM V1 reserves ignore_id 255")
        if (
            not isinstance(raw_classes, Sequence)
            or isinstance(raw_classes, (str, bytes))
            or not raw_classes
        ):
            raise SemanticSchemaError("classes must be a non-empty list")

        classes = tuple(SemanticClass.from_dict(item) for item in raw_classes)
        class_ids = [item.class_id for item in classes]
        if class_ids != list(range(len(classes))):
            raise SemanticSchemaError(
                "class ids must be contiguous, ordered, and start at zero"
            )
        names = [item.name for item in classes]
        if len(names) != len(set(names)):
            raise SemanticSchemaError("class names must be unique")
        if sum(item.is_background for item in classes) > 1:
            raise SemanticSchemaError("at most one class may be background")
        return cls(schema_version, ignore_id, classes)

    @classmethod
    def load(cls, path: Path) -> "SemanticClassSpec":
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to read class spec {}: {}".format(path, error)
            )
        if not isinstance(value, Mapping):
            raise SemanticSchemaError("class spec root must be a JSON object")
        return cls.from_dict(value)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def background_id(self) -> Optional[int]:
        for item in self.classes:
            if item.is_background:
                return item.class_id
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ignore_id": self.ignore_id,
            "classes": [item.to_dict() for item in self.classes],
        }

    def ontology_dict(self) -> Dict[str, object]:
        """Return class meaning without presentation-only colors."""
        return {
            "schema_version": self.schema_version,
            "ignore_id": self.ignore_id,
            "classes": [
                {
                    "id": item.class_id,
                    "name": item.name,
                    "is_background": item.is_background,
                }
                for item in self.classes
            ],
        }

    def palette_dict(self) -> Dict[str, object]:
        """Return the display mapping independently from class meaning."""
        return {
            "schema_version": self.schema_version,
            "classes": [
                {"id": item.class_id, "color": list(item.color)}
                for item in self.classes
            ],
        }

    @staticmethod
    def _fingerprint(value: Mapping[str, object]) -> str:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint(self.to_dict())

    @property
    def ontology_fingerprint(self) -> str:
        return self._fingerprint(self.ontology_dict())

    @property
    def palette_fingerprint(self) -> str:
        return self._fingerprint(self.palette_dict())


@dataclass(frozen=True)
class ObservationKey:
    image_id: str
    label_version: str


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def normalize_image_id(value: object) -> str:
    """Return an image identity while preserving decimal timestamp suffixes."""
    text = str(value)
    path = Path(text)
    result = path.stem if path.suffix.lower() in _IMAGE_SUFFIXES else path.name
    if not result:
        raise SemanticSchemaError("image_id must not be empty")
    return result


@dataclass(frozen=True)
class SemanticObservation:
    key: ObservationKey
    labels: np.ndarray
    scores: Optional[np.ndarray]
    metadata: Mapping[str, object]
    label_path: Path
    score_path: Optional[Path]


class SemanticDataset:
    """Read and validate one dataset-level semantic directory."""

    def __init__(self, dataset_root: Path, semantic_directory: str = "semantic"):
        self.dataset_root = Path(dataset_root)
        self.semantic_root = self.dataset_root / semantic_directory
        self.class_spec = SemanticClassSpec.load(
            self.semantic_root / "class_spec.json"
        )

    def load_observation(
        self, image_id: str, require_scores: bool = False
    ) -> SemanticObservation:
        image_id = normalize_image_id(image_id)
        label_path = self.semantic_root / "labels" / (image_id + ".png")
        metadata_path = self.semantic_root / "metadata" / (image_id + ".json")
        score_path = self.semantic_root / "scores" / (image_id + ".npy")
        if not label_path.is_file():
            raise SemanticSchemaError("missing semantic label: {}".format(label_path))
        if not metadata_path.is_file():
            raise SemanticSchemaError(
                "missing semantic metadata: {}".format(metadata_path)
            )

        try:
            with Image.open(str(label_path)) as image:
                labels = np.asarray(image).copy()
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to read label {}: {}".format(label_path, error)
            )
        self._validate_labels(labels)

        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to read metadata {}: {}".format(metadata_path, error)
            )
        self._validate_metadata(metadata, image_id, labels.shape, label_path)

        scores = None
        resolved_score_path = None
        if score_path.is_file():
            try:
                scores = np.load(str(score_path), mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise SemanticSchemaError(
                    "unable to read scores {}: {}".format(score_path, error)
                )
            self._validate_scores(scores, labels.shape)
            expected_score_hash = metadata.get("score_sha256")
            if not self._is_sha256(expected_score_hash):
                raise SemanticSchemaError(
                    "metadata requires score_sha256 when scores exist"
                )
            if sha256_file(score_path) != expected_score_hash:
                raise SemanticSchemaError("metadata score_sha256 does not match")
            resolved_score_path = score_path
        elif require_scores:
            raise SemanticSchemaError("missing semantic scores: {}".format(score_path))
        elif metadata.get("score_sha256") is not None:
            raise SemanticSchemaError(
                "metadata score_sha256 is set but the score file is missing"
            )

        return SemanticObservation(
            key=ObservationKey(image_id, metadata["label_version"]),
            labels=labels,
            scores=scores,
            metadata=metadata,
            label_path=label_path,
            score_path=resolved_score_path,
        )

    def _validate_labels(self, labels: np.ndarray) -> None:
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise SemanticSchemaError(
                "semantic label must be a single-channel integer image"
            )
        valid = (labels == self.class_spec.ignore_id) | (
            (labels >= 0) & (labels < self.class_spec.num_classes)
        )
        if not bool(np.all(valid)):
            raise SemanticSchemaError(
                "semantic label contains invalid class IDs: {}".format(
                    np.unique(labels[~valid]).tolist()
                )
            )

    def _validate_metadata(
        self,
        metadata: object,
        image_id: str,
        label_shape: Tuple[int, int],
        label_path: Path,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise SemanticSchemaError("metadata root must be a JSON object")
        if metadata.get("schema_version") != 1:
            raise SemanticSchemaError("unsupported semantic metadata schema")
        if str(metadata.get("image_id")) != image_id:
            raise SemanticSchemaError("metadata image_id does not match file name")
        label_version = metadata.get("label_version")
        if not isinstance(label_version, str) or not label_version.strip():
            raise SemanticSchemaError("metadata requires a label_version")
        model = metadata.get("model")
        if not isinstance(model, str) or not model.strip():
            raise SemanticSchemaError("metadata requires a model identifier")
        if not self._is_sha256(metadata.get("checkpoint_sha256")):
            raise SemanticSchemaError("metadata requires checkpoint_sha256")
        if "prompt_config" not in metadata:
            raise SemanticSchemaError("metadata requires prompt_config")
        try:
            json.dumps(metadata["prompt_config"], ensure_ascii=False)
        except (TypeError, ValueError):
            raise SemanticSchemaError("metadata prompt_config must be JSON-compatible")
        if not isinstance(metadata.get("thresholds"), Mapping):
            raise SemanticSchemaError("metadata thresholds must be a JSON object")
        if metadata.get("class_spec_sha256") != self.class_spec.fingerprint:
            raise SemanticSchemaError("metadata class_spec_sha256 does not match")
        expected_label_hash = metadata.get("label_sha256")
        if not self._is_sha256(expected_label_hash):
            raise SemanticSchemaError("metadata requires label_sha256")
        if sha256_file(label_path) != expected_label_hash:
            raise SemanticSchemaError("metadata label_sha256 does not match")
        height, width = label_shape
        if metadata.get("width") != width or metadata.get("height") != height:
            raise SemanticSchemaError("metadata dimensions do not match label")

    def _validate_scores(
        self, scores: np.ndarray, label_shape: Tuple[int, int]
    ) -> None:
        expected = (self.class_spec.num_classes,) + tuple(label_shape)
        if scores.shape != expected:
            raise SemanticSchemaError(
                "scores have shape {}, expected {}".format(scores.shape, expected)
            )
        if not np.issubdtype(scores.dtype, np.floating):
            raise SemanticSchemaError("scores must use a floating dtype")
        if not bool(np.isfinite(scores).all()):
            raise SemanticSchemaError("scores must contain only finite values")

    @staticmethod
    def _is_sha256(value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        return all(character in "0123456789abcdef" for character in value.lower())


@dataclass(frozen=True)
class LedgerEntry:
    image_id: str
    label_version: str
    label_sha256: str
    score_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.image_id, str)
            or not self.image_id
            or not isinstance(self.label_version, str)
            or not self.label_version
        ):
            raise SemanticSchemaError(
                "ledger image_id and label_version must be non-empty"
            )
        if not SemanticDataset._is_sha256(self.label_sha256):
            raise SemanticSchemaError("ledger label_sha256 is invalid")
        if self.score_sha256 is not None and not SemanticDataset._is_sha256(
            self.score_sha256
        ):
            raise SemanticSchemaError("ledger score_sha256 is invalid")

    @property
    def key(self) -> ObservationKey:
        return ObservationKey(self.image_id, self.label_version)

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_id": self.image_id,
            "label_version": self.label_version,
            "label_sha256": self.label_sha256,
            "score_sha256": self.score_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LedgerEntry":
        if not isinstance(value, Mapping):
            raise SemanticSchemaError("each ledger entry must be a JSON object")
        required = ("image_id", "label_version", "label_sha256")
        if any(
            not isinstance(value.get(name), str) or not value.get(name)
            for name in required
        ):
            raise SemanticSchemaError("ledger entry has missing string fields")
        score_sha = value.get("score_sha256")
        if score_sha is not None and not isinstance(score_sha, str):
            raise SemanticSchemaError("score_sha256 must be a string or null")
        return cls(
            image_id=value["image_id"],
            label_version=value["label_version"],
            label_sha256=value["label_sha256"],
            score_sha256=score_sha,
        )


class SemanticLedger:
    """Idempotency ledger keyed by image id and label version."""

    def __init__(self, entries: Iterable[LedgerEntry] = ()):
        self._entries = {}
        for entry in entries:
            self.record(entry)

    def __contains__(self, key: ObservationKey) -> bool:
        return key in self._entries

    def contains(self, image_id: str, label_version: str) -> bool:
        return ObservationKey(image_id, label_version) in self

    def get(self, image_id: str, label_version: str) -> Optional[LedgerEntry]:
        return self._entries.get(ObservationKey(image_id, label_version))

    def record(self, entry: LedgerEntry) -> bool:
        previous = self._entries.get(entry.key)
        if previous is None:
            self._entries[entry.key] = entry
            return True
        if previous != entry:
            raise SemanticSchemaError(
                "ledger key {} has conflicting hashes".format(entry.key)
            )
        return False

    @property
    def entries(self) -> Tuple[LedgerEntry, ...]:
        keys = sorted(
            self._entries, key=lambda value: (value.image_id, value.label_version)
        )
        return tuple(self._entries[key] for key in keys)

    @classmethod
    def load(cls, path: Path) -> "SemanticLedger":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, ValueError) as error:
            raise SemanticSchemaError(
                "unable to read ledger {}: {}".format(path, error)
            )
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise SemanticSchemaError("unsupported semantic ledger schema")
        observations = value.get("observations")
        if not isinstance(observations, list):
            raise SemanticSchemaError("ledger observations must be a list")
        return cls(LedgerEntry.from_dict(item) for item in observations)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        payload = {
            "schema_version": 1,
            "observations": [entry.to_dict() for entry in self.entries],
        }
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
