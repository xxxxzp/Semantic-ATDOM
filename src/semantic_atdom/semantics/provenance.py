"""Stable identities for semantic projection and complete RGB snapshots."""

import hashlib
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
