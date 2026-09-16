"""Versioned, resumable checkpoints for progressive RGB training."""

import os
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch

from semantic_atdom.scene.cameras import Camera
from semantic_atdom.scene.dataset_readers import load_rectified_image
from semantic_atdom.utils.camera_calibration import RectifiedPinhole
from semantic_atdom.utils.general_utils import PILtoTorch


CHECKPOINT_SCHEMA_VERSION = 2


def checkpoint_stage_from_path(path: Path) -> int:
    path = Path(path)
    candidates = (path.parent.name, path.stem)
    for candidate in candidates:
        match = re.fullmatch(r"(?:chkpnt)?(\d+)", candidate)
        if match is not None:
            return int(match.group(1))
    raise ValueError("cannot infer progressive stage from checkpoint path {}".format(path))


def pack_camera_mask(mask: torch.Tensor):
    value = mask.detach().to("cpu", dtype=torch.bool).contiguous()
    if value.numel() == 1 and bool(value.item()):
        return {"kind": "all_valid"}
    flat = value.numpy().reshape(-1).astype(np.uint8)
    packed = np.packbits(flat)
    return {
        "kind": "packed_bool",
        "shape": tuple(value.shape),
        "count": int(flat.size),
        "data": torch.from_numpy(packed.copy()),
    }


def unpack_camera_mask(payload):
    if payload["kind"] == "all_valid":
        return torch.ones(1, dtype=torch.bool)
    if payload["kind"] != "packed_bool":
        raise ValueError("unsupported camera mask encoding")
    packed = payload["data"].detach().to("cpu").numpy()
    unpacked = np.unpackbits(packed, count=int(payload["count"])).astype(bool)
    return torch.from_numpy(unpacked.reshape(tuple(payload["shape"])).copy())


def camera_to_state(camera):
    calibration = getattr(camera, "calibration", None)
    image_path = getattr(camera, "image_path", None)
    if not image_path:
        raise ValueError("camera {} has no source image path".format(camera.image_name))
    return {
        "uid": int(camera.uid),
        "colmap_id": int(camera.colmap_id),
        "R": np.asarray(camera.R),
        "T": np.asarray(camera.T),
        "FoVx": float(camera.FoVx),
        "FoVy": float(camera.FoVy),
        "image_name": str(camera.image_name),
        "image_path": str(Path(image_path).resolve()),
        "image_width": int(camera.image_width),
        "image_height": int(camera.image_height),
        "trans": np.asarray(camera.trans),
        "scale": float(camera.scale),
        "mask": pack_camera_mask(camera.Tri_Mask),
        "calibration": calibration.to_dict() if calibration is not None else None,
    }


def camera_from_state(state, *, data_device):
    calibration_payload = state.get("calibration")
    calibration = (
        RectifiedPinhole.from_dict(calibration_payload)
        if calibration_payload is not None
        else None
    )
    if calibration is None:
        with Image.open(state["image_path"]) as source:
            source.load()
            image = source.copy()
    else:
        image = load_rectified_image(state["image_path"], calibration)
    resolution = (int(state["image_width"]), int(state["image_height"]))
    resized = PILtoTorch(image, resolution)
    return Camera(
        colmap_id=int(state["colmap_id"]),
        R=np.asarray(state["R"]),
        T=np.asarray(state["T"]),
        FoVx=float(state["FoVx"]),
        FoVy=float(state["FoVy"]),
        image=resized[:3, ...],
        gt_alpha_mask=(resized[3:4, ...] if resized.shape[0] == 4 else None),
        image_name=str(state["image_name"]),
        uid=int(state["uid"]),
        trans=np.asarray(state["trans"]),
        scale=float(state["scale"]),
        Tri_Mask=unpack_camera_mask(state["mask"]),
        data_device=data_device,
        image_path=state["image_path"],
        calibration=calibration,
    )


def capture_rng_state():
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload):
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"].detach().cpu())
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() for state in payload["torch_cuda"]]
        )


def save_training_checkpoint(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_training_checkpoint(path: Path, *, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint is not a resumable Semantic A-TDOM v{} checkpoint".format(
                CHECKPOINT_SCHEMA_VERSION
            )
        )
    required = {
        "stage",
        "gaussians",
        "iteration",
        "cameras",
        "images_already_trained_iterations",
        "progressive_transition_index",
        "rng_state",
        "data_preprocessor_rng_state",
        "phase",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("checkpoint is missing fields: {}".format(missing))
    inferred = checkpoint_stage_from_path(path)
    if int(payload["stage"]) != inferred:
        raise ValueError("checkpoint stage does not match its path")
    return payload
