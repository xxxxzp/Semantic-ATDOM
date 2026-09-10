from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import torch

from semantic_atdom.semantics.orchestrator import (
    AcquisitionOverlap,
    OnlineSemanticConfig,
    ProjectedObservation,
)
from semantic_atdom.semantics.rebuild import FinalSemanticRebuilder
from semantic_atdom.semantics.schema import (
    ObservationKey,
    SemanticClassSpec,
    SemanticObservation,
    SemanticSchemaError,
)


CLASS_SPEC = SemanticClassSpec.from_dict(
    {
        "schema_version": 1,
        "ignore_id": 255,
        "classes": [
            {"id": 0, "name": "ground", "color": [0, 0, 0]},
            {"id": 1, "name": "building", "color": [255, 0, 0]},
        ],
    }
)


class FakeDataset:
    class_spec = CLASS_SPEC

    def load_observation(self, image_id):
        return SemanticObservation(
            ObservationKey(image_id, "sam3-v1"),
            np.zeros((1, 1), dtype=np.uint8),
            None,
            {
                "model": "sam3",
                "checkpoint_sha256": "a" * 64,
                "prompt_config": {"classes": ["ground", "building"]},
                "thresholds": {"confidence": 0.5},
                "label_sha256": image_id[-1] * 64,
                "score_sha256": None,
            },
            Path(image_id + ".png"),
            None,
        )


class FakeProjector:
    def __init__(self):
        self.calls = 0

    def project(self, gaussians, camera, observation):
        self.calls += 1
        distribution = {
            "000001": [1.0, 0.0],
            "000002": [1.0, 0.0],
            "000003": [0.0, 1.0],
        }[observation.key.image_id]
        return ProjectedObservation(
            torch.tensor([distribution]),
            torch.ones(1),
            torch.ones(1),
        )


class Gaussians:
    get_xyz = torch.zeros((1, 3))


class FinalSemanticRebuilderTests(unittest.TestCase):
    def test_second_pass_rejects_changed_label_identity(self):
        class ChangingDataset(FakeDataset):
            def __init__(self):
                self.calls = 0

            def load_observation(self, image_id):
                self.calls += 1
                item = super().load_observation(image_id)
                if self.calls > 3:
                    metadata = dict(item.metadata)
                    metadata["label_sha256"] = "f" * 64
                    return replace(item, metadata=metadata)
                return item

        rebuilder = FinalSemanticRebuilder(
            ChangingDataset(),
            FakeProjector(),
            OnlineSemanticConfig(max_group_size=1),
        )
        ids = ("000001", "000002", "000003")

        with self.assertRaisesRegex(
            SemanticSchemaError, "changed between final rebuild passes"
        ):
            rebuilder.rebuild(
                ids,
                {image_id: object() for image_id in ids},
                Gaussians(),
                AcquisitionOverlap(ids, np.eye(3) * 100.0),
            )

    def test_exact_rebuild_reprojects_every_image_twice(self):
        projector = FakeProjector()
        rebuilder = FinalSemanticRebuilder(
            FakeDataset(),
            projector,
            OnlineSemanticConfig(max_group_size=1),
        )
        ids = ("000001", "000002", "000003")
        overlap = AcquisitionOverlap(ids, np.eye(3) * 100.0)

        result = rebuilder.rebuild(
            ids,
            {image_id: object() for image_id in ids},
            Gaussians(),
            overlap,
        )

        self.assertEqual(projector.calls, 6)
        self.assertEqual(result.image_groups, tuple((item,) for item in ids))
        torch.testing.assert_close(
            result.state.alpha_raw, torch.tensor([[2.5, 1.5]])
        )
        self.assertEqual(result.state.group_count.item(), 3)
        self.assertLess(result.state.alpha_robust[0, 1].item(), 1.5)
        self.assertGreater(result.state.alpha_robust[0, 1].item(), 0.5)
        self.assertEqual(len(result.ledger.entries), 3)
        self.assertEqual(result.metrics.image_count, 3)


if __name__ == "__main__":
    unittest.main()
