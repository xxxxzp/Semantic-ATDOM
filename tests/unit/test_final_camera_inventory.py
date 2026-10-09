import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest import mock

from semantic_atdom.scene.cameras import Camera
from tools.train_progressive import GaussianTrainer


class _Scene:
    def __init__(self, root, cameras):
        self.model_path = str(root)
        self._cameras = cameras

    def getTrainCameras(self):
        return list(self._cameras)


def _camera(name, x):
    return SimpleNamespace(
        image_name=name,
        R=np.eye(3, dtype=np.float32),
        T=np.array([x, 0.0, 0.0], dtype=np.float32),
        width=64,
        height=48,
        FovX=1.0,
        FovY=0.8,
    )


def test_final_camera_inventory_uses_sorted_cumulative_trainer_state(tmp_path):
    trainer = GaussianTrainer.__new__(GaussianTrainer)
    trainer.scene = _Scene(
        tmp_path,
        [_camera("000003", 3.0), _camera("000001", 1.0), _camera("000002", 2.0)],
    )

    trainer._write_final_camera_inventory()

    payload = json.loads((Path(tmp_path) / "cameras.json").read_text())
    assert [item["img_name"] for item in payload] == [
        "000001",
        "000002",
        "000003",
    ]
    assert [item["id"] for item in payload] == [0, 1, 2]


def test_final_camera_inventory_rejects_duplicate_identities(tmp_path):
    trainer = GaussianTrainer.__new__(GaussianTrainer)
    trainer.scene = _Scene(
        tmp_path,
        [_camera("000001", 1.0), _camera("000001", 2.0)],
    )

    try:
        trainer._write_final_camera_inventory()
    except RuntimeError as error:
        assert "duplicate identities" in str(error)
    else:
        raise AssertionError("duplicate camera identities must fail publication")


def test_final_camera_inventory_serializes_real_runtime_camera(tmp_path):
    # Only device placement is replaced; constructor-owned field names and
    # image dimensions come from the real Camera class, not a schema mock.
    with mock.patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
        camera = Camera(
            colmap_id=7,
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=1.0,
            FoVy=0.8,
            image=torch.zeros((3, 48, 64)),
            gt_alpha_mask=None,
            image_name="000007",
            uid=0,
            data_device="cpu",
        )
    trainer = GaussianTrainer.__new__(GaussianTrainer)
    trainer.scene = _Scene(tmp_path, [camera])

    trainer._write_final_camera_inventory()

    record = json.loads((tmp_path / "cameras.json").read_text())[0]
    assert record["img_name"] == "000007"
    assert record["width"] == 64 and record["height"] == 48
    assert record["fx"] == pytest.approx(64 / (2 * np.tan(0.5)))
    assert record["fy"] == pytest.approx(48 / (2 * np.tan(0.4)))


def test_camera_metadata_failure_keeps_completed_gaussians_on_disk(tmp_path):
    trainer = GaussianTrainer.__new__(GaussianTrainer)
    trainer.args = SimpleNamespace(save_final_training_state=False)
    trainer.imagenum = 487
    saved_model = tmp_path / "point_cloud.ply"
    trainer.scene = SimpleNamespace(
        save=mock.Mock(side_effect=lambda stage: saved_model.write_text(str(stage)))
    )

    with mock.patch.object(
        trainer, "_write_final_camera_inventory", side_effect=RuntimeError("metadata")
    ), pytest.raises(RuntimeError, match="metadata"):
        trainer.GS_Save(final=True)

    trainer.scene.save.assert_called_once_with(487)
    assert saved_model.read_text() == "487"
