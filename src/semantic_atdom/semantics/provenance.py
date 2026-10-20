"""Stable identities for semantic projection and complete RGB snapshots."""

import hashlib
import json
from typing import Mapping, Sequence

import torch


SEMANTIC_METHOD_VERSION = "g1-evidence-v2"

SEMANTIC_PROJECTION_ATTRIBUTE_NAMES = (
    "xyz",
    "opacity",
    "scaling",
    "rotation",
)
RGB_MODEL_ATTRIBUTE_NAMES = (
    "xyz",
    "features_dc",
    "features_rest",
    "opacity",
    "scaling",
    "rotation",
)


def _tensor_attributes_sha256(
    attributes: Mapping[str, torch.Tensor],
    names: Sequence[str],
    domain: bytes,
) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for name in names:
        try:
            value = attributes[name].detach().cpu().contiguous()
        except KeyError as error:
            raise ValueError("missing model attribute: {}".format(name)) from error
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def semantic_projection_sha256(
    attributes: Mapping[str, torch.Tensor],
) -> str:
    """Hash only Gaussian fields that can change semantic projection."""
    return _tensor_attributes_sha256(
        attributes,
        SEMANTIC_PROJECTION_ATTRIBUTE_NAMES,
        b"semantic-atdom-projection-v2",
    )


def rgb_model_sha256(attributes: Mapping[str, torch.Tensor]) -> str:
    """Hash the complete row-aligned Gaussian model, including appearance."""
    return _tensor_attributes_sha256(
        attributes,
        RGB_MODEL_ATTRIBUTE_NAMES,
        b"semantic-atdom-rgb-model-v2",
    )


def semantic_state_sha256(
    state,
    recovery_context: Mapping[str, object],
    projection_geometry_hash: str,
    ontology_hash: str,
) -> str:
    """Hash semantic evidence/state content without creating a checkpoint."""
    digest = hashlib.sha256(b"semantic-atdom-state-content-v1\0")
    metadata = {
        "schema_version": int(state.SCHEMA_VERSION),
        "semantic_method_version": SEMANTIC_METHOD_VERSION,
        "projection_geometry_hash": projection_geometry_hash,
        "ontology_hash": ontology_hash,
        "prior_total": float(state.prior_total),
        "topology_version": int(state.topology_version),
        "next_gaussian_id": int(state.next_gaussian_id),
        "historical_evidence_stale": bool(state.historical_evidence_stale),
        "group_disagreement_source": state.group_disagreement_source,
        "group_disagreement_available": bool(
            state.group_disagreement_available
        ),
        "recovery_context": recovery_context,
    }
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in (
        "alpha_raw",
        "alpha_robust",
        "group_variance",
        "group_count",
        "semantic_support",
        "open_raw_evidence",
        "open_robust_evidence",
        "open_group_count",
        "open_distributions",
        "open_supports",
        "open_semantic_supports",
        "dirty",
        "active_projection_dirty",
        "gaussian_ids",
    ):
        value = getattr(state, name).detach().cpu().contiguous()
        digest.update(b"\0" + name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
