"""Online semantics must follow persistent cameras through local SfM snapshots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from semantic_atdom.semantics.orchestrator import (
    OnlineSemanticConfig,
    OnlineSemanticOrchestrator,
    ProgressiveSemanticHook,
    ProjectedObservation,
)
from semantic_atdom.semantics.publication import MapPublicationPolicy
from semantic_atdom.semantics.schema import SemanticDataset, sha256_file
from semantic_atdom.semantics.state import SemanticGaussianState


class PersistentOverlapTests(unittest.TestCase):
    def _hook(self, root, max_group_size=4):
        semantic = root / "semantic"
        (semantic / "labels").mkdir(parents=True)
        (semantic / "metadata").mkdir()
        (semantic / "class_spec.json").write_text(json.dumps({
            "schema_version": 1,
            "ignore_id": 255,
            "classes": [{"id": 0, "name": "ground", "color": [0, 0, 0]}],
        }))
        dataset = SemanticDataset(root)
        for image_id in ("000001", "000002", "000003"):
            label = semantic / "labels" / (image_id + ".png")
            Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(label)
            metadata = {
                "schema_version": 1,
                "image_id": image_id,
                "label_version": "test-v1",
                "model": "test",
                "checkpoint_sha256": "a" * 64,
                "prompt_config": {},
                "thresholds": {},
                "class_spec_sha256": dataset.class_spec.fingerprint,
                "label_sha256": sha256_file(label),
                "width": 1,
                "height": 1,
            }
            (semantic / "metadata" / (image_id + ".json")).write_text(
                json.dumps(metadata)
            )

        class Projector:
            def project(self, *_args):
                return ProjectedObservation(
                    torch.ones((1, 1)), torch.ones(1), torch.ones(1)
                )

        return ProgressiveSemanticHook(
            dataset,
            OnlineSemanticOrchestrator(
                SemanticGaussianState.initialize(1, 1),
                dataset.class_spec,
                OnlineSemanticConfig(max_group_size=max_group_size),
            ),
            Projector(),
            root / "state",
            publication_policy=MapPublicationPolicy(checkpoint_stage_interval=0),
        )

    def _stage(self, root, stage, image_ids):
        path = root / str(stage) / "sparse" / "0"
        path.mkdir(parents=True)
        (path / "imagesNames.txt").write_text(
            ",".join(image_id + ".jpg" for image_id in image_ids)
        )
        np.savetxt(
            path / "imageMatchMatrix.txt",
            np.full((len(image_ids), len(image_ids)), 100),
            fmt="%d", delimiter=",",
        )
        return root / str(stage)

    def _sync(self, hook, stage, image_ids, snapshot):
        return hook.synchronize(
            stage,
            [SimpleNamespace(image_name=image_id) for image_id in image_ids],
            SimpleNamespace(get_xyz=torch.zeros((1, 3))),
            snapshot,
        )

    def test_partial_final_rebuild_uses_valid_views_after_invalid_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hook = self._hook(root, max_group_size=1)
            image_ids = ("000001", "000002", "000003")
            stage = self._stage(root, 3, image_ids)
            (root / "semantic" / "metadata" / "000002.json").write_text("{broken")
            calls = []
            class Projector:
                def project(self, gaussians, camera, observation):
                    calls.append(observation.key.image_id)
                    return ProjectedObservation(torch.ones((1, 1)), torch.ones(1), torch.ones(1))
            hook.projector = Projector()
            result = hook.synchronize(3, [SimpleNamespace(image_name=i) for i in image_ids],
                                      SimpleNamespace(get_xyz=torch.zeros(1, 3),
                                          _xyz=torch.zeros(1, 3), _opacity=torch.zeros(1, 1),
                                          _scaling=torch.zeros(1, 3), _rotation=torch.tensor([[1., 0., 0., 0.]])),
                                      stage, final=True)
            self.assertEqual(result.status, "exact_partial")
            self.assertIn("000003", calls)
            self.assertEqual(result.deferred_image_ids, ("000002",))
            self.assertTrue(any("000002" in error for error in result.errors))
            self.assertEqual({entry.image_id for entry in hook.orchestrator.ledger.entries},
                             {"000001", "000003"})

    def test_partial_overlap_does_not_block_unrelated_persistent_camera(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hook = self._hook(root)
            first = self._stage(root, 1, ("000001", "000003"))
            second = self._stage(root, 2, ("000001", "000002"))
            self._sync(hook, 1, ("000001", "000003"), first)

            result = self._sync(
                hook, 2, ("000001", "000002", "000003"), second
            )

            self.assertEqual(result.errors, ())
            self.assertEqual(result.processed_image_ids, ("000002",))
            self.assertEqual(result.semantic_through_image, "000003")
            self.assertEqual(result.lag_batches, 0)
            self.assertEqual(result.deferred_image_ids, ())
            self.assertEqual(len(hook.orchestrator.ledger.entries), 3)

    def test_no_new_observation_preserves_boundary_when_last_camera_is_omitted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hook = self._hook(root)
            first = self._stage(root, 1, ("000001", "000003"))
            second = self._stage(root, 2, ("000001",))
            self._sync(hook, 1, ("000001", "000003"), first)
            evidence = hook.orchestrator.state.alpha_raw.clone()

            result = self._sync(hook, 2, ("000001", "000003"), second)

            self.assertEqual(result.errors, ())
            self.assertEqual(result.processed_image_ids, ())
            self.assertEqual(result.semantic_through_image, "000003")
            self.assertEqual(result.lag_batches, 0)
            self.assertEqual(len(hook.orchestrator.ledger.entries), 2)
            torch.testing.assert_close(hook.orchestrator.state.alpha_raw, evidence)

    def test_missing_required_anchor_overlap_defers_until_it_is_observed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hook = self._hook(root)
            first = self._stage(root, 1, ("000001",))
            second = self._stage(root, 2, ("000002",))
            third = self._stage(root, 3, ("000001", "000002"))
            self._sync(hook, 1, ("000001",), first)
            evidence = hook.orchestrator.state.alpha_raw.clone()

            deferred = self._sync(hook, 2, ("000001", "000002"), second)

            self.assertTrue(deferred.errors)
            self.assertIn("absent from overlap names", deferred.errors[0])
            self.assertEqual(deferred.deferred_image_ids, ("000002",))
            self.assertEqual(deferred.semantic_through_image, "000001")
            self.assertEqual(deferred.lag_batches, 1)
            self.assertEqual(len(hook.orchestrator.ledger.entries), 1)
            torch.testing.assert_close(hook.orchestrator.state.alpha_raw, evidence)

            accepted = self._sync(hook, 3, ("000001", "000002"), third)

            self.assertEqual(accepted.errors, ())
            self.assertEqual(accepted.processed_image_ids, ("000002",))
            self.assertEqual(accepted.semantic_through_image, "000002")
            self.assertEqual(accepted.lag_batches, 0)
            self.assertEqual(len(hook.orchestrator.ledger.entries), 2)

    def test_full_group_can_close_without_querying_an_unneeded_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hook = self._hook(root, max_group_size=1)
            first = self._stage(root, 1, ("000001",))
            second = self._stage(root, 2, ("000002",))
            self._sync(hook, 1, ("000001",), first)

            result = self._sync(hook, 2, ("000001", "000002"), second)

            self.assertEqual(result.errors, ())
            self.assertEqual(result.processed_image_ids, ("000002",))
            self.assertEqual(result.lag_batches, 0)
            self.assertEqual(hook.orchestrator.state.group_count.item(), 2)


if __name__ == "__main__":
    unittest.main()
