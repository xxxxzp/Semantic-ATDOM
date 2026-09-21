"""Immutable input binding for authoritative final semantic rebuilds."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from semantic_atdom.semantics.schema import (
    SemanticDataset,
    SemanticSchemaError,
    sha256_file,
)


@dataclass(frozen=True)
class FinalPreflightBinding:
    """Validated identity of every input consumed by an exact rebuild."""

    schema_version: int
    report_path: str
    report_sha256: str
    geometry_stage: int
    image_ids: tuple
    class_spec_sha256: str
    ontology_sha256: str
    images_bin_sha256: str
    cameras_bin_sha256: str
    images_names_sha256: str
    overlap_matrix_sha256: str

    def to_dict(self):
        return asdict(self)

    @property
    def binding_sha256(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _load_report(path: Path, expected_sha256: str) -> Mapping[str, object]:
    path = Path(path).expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise SemanticSchemaError(
            "dataset preflight report content hash changed before final rebuild"
        )
    try:
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, ValueError) as error:
        raise SemanticSchemaError(
            "unable to read dataset preflight report: {}".format(error)
        )
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        raise SemanticSchemaError("unsupported dataset preflight report")
    return report


def _stage_input_paths(snapshot_root: Path):
    root = Path(snapshot_root).expanduser().resolve()
    sparse = root / "sparse" / "0"
    if not sparse.is_dir():
        sparse = root / "bin"
    overlap_candidates = (
        (sparse / "imagesNames.txt", sparse / "imageMatchMatrix.txt"),
        (
            root / "match_matrix_image_name.txt",
            root / "match_matrix_value.txt",
        ),
    )
    for names_path, matrix_path in overlap_candidates:
        if names_path.is_file() and matrix_path.is_file():
            return sparse, names_path, matrix_path
    raise SemanticSchemaError("final stage overlap inputs are missing")


def validate_final_preflight_binding(
    *,
    report_path: Path,
    expected_report_sha256: str,
    semantic_dataset: SemanticDataset,
    image_ids: Sequence[str],
    camera_by_id: Mapping[str, object],
    snapshot_root: Path,
    geometry_stage: int,
) -> FinalPreflightBinding:
    """Revalidate preflight identities at the final exact boundary.

    This intentionally hashes source files again.  A final result is exact
    only relative to the same immutable RGB, label, camera and overlap inputs
    that were admitted before training began.
    """

    report_path = Path(report_path).expanduser().resolve()
    report = _load_report(report_path, expected_report_sha256)
    if not report.get("authoritative_exact_complete"):
        raise SemanticSchemaError(
            "dataset preflight did not establish authoritative complete inputs"
        )
    expected_ids = tuple(str(item) for item in report.get("final_image_ids", ()))
    actual_ids = tuple(str(item) for item in image_ids)
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise SemanticSchemaError(
            "final camera identities changed since preflight: {} != {}".format(
                actual_ids, expected_ids
            )
        )
    class_spec = semantic_dataset.class_spec
    if report.get("class_spec_sha256") != class_spec.fingerprint:
        raise SemanticSchemaError("class specification changed since preflight")
    if report.get("ontology_sha256") != class_spec.ontology_fingerprint:
        raise SemanticSchemaError("semantic ontology changed since preflight")
    expected_semantic_root = Path(
        str(report.get("semantic_data_root", ""))
    ).expanduser().resolve()
    if expected_semantic_root != semantic_dataset.dataset_root.resolve():
        raise SemanticSchemaError(
            "semantic dataset root changed since preflight"
        )

    selected_stages = tuple(report.get("selected_stages", ()))
    if not selected_stages or int(selected_stages[-1]) != int(geometry_stage):
        raise SemanticSchemaError(
            "final geometry stage does not match dataset preflight"
        )
    records = {
        int(item["stage"]): item
        for item in report.get("stage_records", ())
        if isinstance(item, Mapping) and "stage" in item
    }
    stage_record = records.get(int(geometry_stage))
    if stage_record is None:
        raise SemanticSchemaError("final stage is absent from dataset preflight")
    sparse, names_path, matrix_path = _stage_input_paths(snapshot_root)
    stage_inputs = {
        "images_bin_sha256": sparse / "images.bin",
        "cameras_bin_sha256": sparse / "cameras.bin",
        "images_names_sha256": names_path,
        "overlap_matrix_sha256": matrix_path,
    }
    actual_stage_hashes = {}
    for field, path in stage_inputs.items():
        if not path.is_file() or sha256_file(path) != stage_record.get(field):
            raise SemanticSchemaError(
                "final stage {} changed since preflight".format(field)
            )
        actual_stage_hashes[field] = stage_record[field]

    observation_records = {
        str(item["image_id"]): item
        for item in report.get("observations", ())
        if isinstance(item, Mapping) and "image_id" in item
    }
    if tuple(sorted(observation_records)) != tuple(sorted(actual_ids)):
        raise SemanticSchemaError(
            "preflight observation inventory does not match final cameras"
        )
    for image_id in actual_ids:
        expected = observation_records[image_id]
        observation = semantic_dataset.load_observation(image_id)
        if (
            observation.key.label_version != expected.get("label_version")
            or observation.metadata.get("label_sha256")
            != expected.get("label_sha256")
        ):
            raise SemanticSchemaError(
                "label content changed since preflight: {}".format(image_id)
            )
        camera = camera_by_id.get(image_id)
        image_path = None if camera is None else getattr(camera, "image_path", None)
        if not image_path:
            raise SemanticSchemaError(
                "final camera has no source RGB path for preflight binding: {}".format(
                    image_id
                )
            )
        rgb_hash = sha256_file(Path(image_path))
        if rgb_hash != expected.get("rgb_sha256"):
            raise SemanticSchemaError(
                "RGB content changed since preflight: {}".format(image_id)
            )
        if observation.metadata.get("source_image_sha256") != rgb_hash:
            raise SemanticSchemaError(
                "semantic source binding changed since preflight: {}".format(image_id)
            )

    return FinalPreflightBinding(
        schema_version=1,
        report_path=str(report_path),
        report_sha256=expected_report_sha256,
        geometry_stage=int(geometry_stage),
        image_ids=actual_ids,
        class_spec_sha256=class_spec.fingerprint,
        ontology_sha256=class_spec.ontology_fingerprint,
        **actual_stage_hashes,
    )
