import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
