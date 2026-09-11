from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from semantic_atdom.semantics.state import (
    SemanticGaussianState,
    SemanticStateError,
)
from semantic_atdom.semantics.schema import SemanticClassSpec
from semantic_atdom.semantics.provenance import SEMANTIC_METHOD_VERSION
from semantic_atdom.semantics.reference import render_semantic_tdom
from semantic_atdom.semantics.topology import TopologyDelta


STATE_CLASS_SPEC = {
    "schema_version": 1,
    "ignore_id": 255,
    "classes": [
        {"id": 0, "name": "ground", "color": [0, 0, 0]},
        {"id": 1, "name": "building", "color": [255, 0, 0]},
        {"id": 2, "name": "vegetation", "color": [0, 255, 0]},
    ],
}
STATE_SPEC = SemanticClassSpec.from_dict(STATE_CLASS_SPEC)
ONTOLOGY_HASH = STATE_SPEC.ontology_fingerprint
PALETTE_HASH = STATE_SPEC.palette_fingerprint
PROJECTION_GEOMETRY_HASH = "a" * 64
RGB_MODEL_HASH = "b" * 64
RECOVERY_CONTEXT = {
    "schema_version": 2,
    "semantic_method_version": SEMANTIC_METHOD_VERSION,
    "ledger": {"schema_version": 1, "observations": []},
    "label_provenance": {
        "schema_version": 1,
        "versions": {
            "sam3-v1": {
                "model": "sam3",
                "checkpoint_sha256": "b" * 64,
                "prompt_config": {"classes": ["ground", "building"]},
                "thresholds": {"min_score": 0.2},
            }
        },
    },
    "open_group": None,
    "ontology": STATE_SPEC.ontology_dict(),
    "semantic_config": {
        "schema_version": 2,
        "semantic_method_version": SEMANTIC_METHOD_VERSION,
        "prior_total": 1.0,
    },
}


class SemanticGaussianStateTests(unittest.TestCase):
    def test_initial_state_is_non_parameter_prior(self):
        state = SemanticGaussianState.initialize(3, 2)

        torch.testing.assert_close(
            state.alpha_raw, torch.full((3, 2), 0.5)
        )
        torch.testing.assert_close(state.alpha_robust, state.alpha_raw)
        self.assertFalse(state.alpha_raw.requires_grad)
        self.assertFalse(isinstance(state.alpha_raw, torch.nn.Parameter))
        self.assertEqual(state.gaussian_ids.tolist(), [0, 1, 2])

    def test_clone_copies_parent_belief_instead_of_diluting_it(self):
        state = SemanticGaussianState.initialize(2, 2)
        state.add_evidence(
            torch.tensor([[4.0, 1.0], [0.0, 3.0]]),
            support=torch.tensor([2.0, 1.0]),
        )
        state.mark_clean()
        parent_evidence = state.alpha_raw[0] - 0.5

        state.apply_topology(TopologyDelta.clone(2, torch.tensor([0])))

        self.assertEqual(state.num_gaussians, 3)
        torch.testing.assert_close(state.alpha_raw[0] - 0.5, parent_evidence)
        torch.testing.assert_close(state.alpha_raw[2], state.alpha_raw[0])
        self.assertEqual(state.gaussian_ids.tolist()[:2], [0, 1])
        self.assertNotIn(state.gaussian_ids[2].item(), [0, 1])
        self.assertTrue(state.dirty[0])
        self.assertTrue(state.dirty[2])

    def test_split_replaces_parent_and_preserves_output_order(self):
        state = SemanticGaussianState.initialize(3, 2)
        state.add_evidence(torch.tensor([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]]))
        parent_evidence = state.alpha_raw[1] - 0.5

        state.apply_topology(
            TopologyDelta.split(
                old_count=3,
                parent_indices=torch.tensor([1]),
                children_per_parent=2,
            )
        )

        self.assertEqual(state.num_gaussians, 4)
        self.assertEqual(state.gaussian_ids[:2].tolist(), [0, 2])
        torch.testing.assert_close(state.alpha_raw[2] - 0.5, parent_evidence)
        torch.testing.assert_close(state.alpha_raw[3] - 0.5, parent_evidence)
        self.assertTrue(bool(state.dirty[2:].all()))

    def test_split_copy_preserves_rendered_semantic_field(self):
        state = SemanticGaussianState.initialize(1, 2)
        state.add_evidence(torch.tensor([[4.0, 1.0]]))
        before = render_semantic_tdom(
            torch.tensor([[0.6]]), state.alpha_robust
        )

        state.apply_topology(
            TopologyDelta.split(
                old_count=1,
                parent_indices=torch.tensor([0]),
                children_per_parent=2,
            )
        )
        after = render_semantic_tdom(
            torch.tensor([[0.3, 0.3]]), state.alpha_robust
        )

        for name in (
            "probabilities",
            "pixel_evidence_support",
            "support_ratio",
            "entropy",
            "class_ids",
        ):
            torch.testing.assert_close(
                getattr(after, name), getattr(before, name), equal_nan=True
            )

    def test_prune_and_append_prior_follow_explicit_deltas(self):
        state = SemanticGaussianState.initialize(3, 2)
        state.add_evidence(torch.ones((3, 2)))
        state.apply_topology(
            TopologyDelta.prune(torch.tensor([True, False, True]))
        )
        self.assertEqual(state.gaussian_ids.tolist(), [0, 2])

        state.apply_topology(TopologyDelta.append_prior(2, 2))
        self.assertEqual(state.num_gaussians, 4)
        torch.testing.assert_close(
            state.alpha_raw[2:], torch.full((2, 2), 0.5)
        )
        self.assertTrue(bool(state.dirty[2:].all()))

    def test_sidecar_validates_geometry_and_class_spec(self):
        state = SemanticGaussianState.initialize(2, 3)
        state.add_evidence(torch.ones((2, 3)))
        state.mark_historical_evidence_stale()
        state.group_disagreement_available = True
        with TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_state.pt"
            state.save(
                path,
                projection_geometry_hash=PROJECTION_GEOMETRY_HASH,
                rgb_model_hash=RGB_MODEL_HASH,
                ontology_hash=ONTOLOGY_HASH,
                palette_hash=PALETTE_HASH,
                recovery_context=RECOVERY_CONTEXT,
            )
            recovery_hash = state.validated_recovery_context_sha256(
                RECOVERY_CONTEXT,
                ontology_hash=ONTOLOGY_HASH,
                prior_total=1.0,
                num_classes=3,
            )

            restored = SemanticGaussianState.load(
                path,
                expected_projection_geometry_hash=PROJECTION_GEOMETRY_HASH,
                expected_ontology_hash=ONTOLOGY_HASH,
                expected_gaussian_count=2,
                expected_recovery_context_sha256=recovery_hash,
            )
            torch.testing.assert_close(restored.alpha_raw, state.alpha_raw)
            self.assertEqual(restored.gaussian_ids.tolist(), [0, 1])
            self.assertTrue(restored.historical_evidence_stale)
            self.assertTrue(restored.group_disagreement_available)
            self.assertEqual(
                restored.recovery_context_sha256(restored.recovery_context),
                recovery_hash,
            )

            with self.assertRaisesRegex(SemanticStateError, "geometry"):
                SemanticGaussianState.load(
                    path,
                    expected_projection_geometry_hash="c" * 64,
                    expected_ontology_hash=ONTOLOGY_HASH,
                    expected_gaussian_count=2,
                    expected_recovery_context_sha256=recovery_hash,
                )

            with self.assertRaisesRegex(SemanticStateError, "count"):
                SemanticGaussianState.load(
                    path,
                    expected_projection_geometry_hash=PROJECTION_GEOMETRY_HASH,
                    expected_ontology_hash=ONTOLOGY_HASH,
                    expected_gaussian_count=3,
                    expected_recovery_context_sha256=recovery_hash,
                )

            with self.assertRaisesRegex(SemanticStateError, "current inputs"):
                SemanticGaussianState.load(
                    path,
                    expected_projection_geometry_hash=PROJECTION_GEOMETRY_HASH,
                    expected_ontology_hash=ONTOLOGY_HASH,
                    expected_gaussian_count=2,
                    expected_recovery_context_sha256="d" * 64,
                )

            payload = torch.load(str(path), map_location="cpu")
            self.assertEqual(payload["schema_version"], 3)
            self.assertEqual(
                payload["semantic_method_version"], SEMANTIC_METHOD_VERSION
            )
            self.assertEqual(payload["ontology_hash"], ONTOLOGY_HASH)
            self.assertEqual(payload["palette_hash"], PALETTE_HASH)
            self.assertEqual(payload["rgb_model_hash"], RGB_MODEL_HASH)

    def test_sidecar_explicitly_rejects_old_schema_and_method(self):
        state = SemanticGaussianState.initialize(1, 3)
        recovery_hash = state.validated_recovery_context_sha256(
            RECOVERY_CONTEXT, ONTOLOGY_HASH, 1.0, 3
        )
        payload = state.state_dict(
            PROJECTION_GEOMETRY_HASH,
            RGB_MODEL_HASH,
            ONTOLOGY_HASH,
            PALETTE_HASH,
            RECOVERY_CONTEXT,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_state.pt"
            payload["schema_version"] = 2
            torch.save(payload, str(path))
            with self.assertRaisesRegex(SemanticStateError, "schema"):
                SemanticGaussianState.load(
                    path,
                    PROJECTION_GEOMETRY_HASH,
                    ONTOLOGY_HASH,
                    1,
                    recovery_hash,
                )

            payload["schema_version"] = 3
            payload["semantic_method_version"] = "g1-alpha-v1"
            torch.save(payload, str(path))
            with self.assertRaisesRegex(SemanticStateError, "method version"):
                SemanticGaussianState.load(
                    path,
                    PROJECTION_GEOMETRY_HASH,
                    ONTOLOGY_HASH,
                    1,
                    recovery_hash,
                )

    def test_recovery_context_rejects_placeholders(self):
        state = SemanticGaussianState.initialize(1, 3)
        with self.assertRaisesRegex(SemanticStateError, "schema"):
            state.state_dict(
                PROJECTION_GEOMETRY_HASH,
                RGB_MODEL_HASH,
                ONTOLOGY_HASH,
                PALETTE_HASH,
                {name: None for name in state.RECOVERY_CONTEXT_KEYS},
            )

    def test_add_evidence_validation_is_atomic(self):
        state = SemanticGaussianState.initialize(1, 2)
        before = state.alpha_raw.clone()
        with self.assertRaisesRegex(SemanticStateError, "non-negative"):
            state.add_evidence(
                torch.ones((1, 2)), support=torch.tensor([-1.0])
            )
        torch.testing.assert_close(state.alpha_raw, before)

    def test_new_evidence_marks_only_changed_rows_dirty(self):
        state = SemanticGaussianState.initialize(2, 2)
        state.mark_clean()
        state.add_evidence(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
        self.assertEqual(state.dirty.tolist(), [True, False])

    def test_active_projection_dirty_and_historical_staleness_are_independent(self):
        state = SemanticGaussianState.initialize(2, 2)
        state.mark_clean()
        state.mark_historical_evidence_stale()

        self.assertFalse(bool(state.active_projection_dirty.any()))
        self.assertTrue(state.historical_evidence_stale)
        state.mark_dirty(torch.tensor([True, False]))
        self.assertFalse(bool(state.active_projection_dirty.any()))
        state.mark_active_projection_dirty(torch.tensor([True, False]))
        self.assertEqual(state.active_projection_dirty.tolist(), [True, False])
        state.mark_active_projection_current()
        state.mark_clean()
        self.assertTrue(state.historical_evidence_stale)
        state.mark_historical_evidence_current()
        self.assertFalse(state.historical_evidence_stale)

    def test_invalidated_state_cannot_be_reused_or_saved(self):
        state = SemanticGaussianState.initialize(1, 3)
        state.invalidate("observer failed")
        with self.assertRaisesRegex(SemanticStateError, "requires rebuild"):
            state.add_evidence(torch.zeros((1, 3)))
        with self.assertRaisesRegex(SemanticStateError, "requires rebuild"):
            state.state_dict(
                PROJECTION_GEOMETRY_HASH,
                RGB_MODEL_HASH,
                ONTOLOGY_HASH,
                PALETTE_HASH,
                RECOVERY_CONTEXT,
            )

    def test_noop_delta_does_not_rebuild_or_increment_topology(self):
        state = SemanticGaussianState.initialize(2, 2)
        alpha_before = state.alpha_raw
        state.apply_topology(
            TopologyDelta.clone(2, torch.tensor([], dtype=torch.long))
        )
        self.assertIs(state.alpha_raw, alpha_before)
        self.assertEqual(state.topology_version, 0)

    def test_zero_weight_lineage_does_not_copy_group_statistics(self):
        state = SemanticGaussianState.initialize(1, 2)
        state.group_count[0] = 3
        state.group_variance[0] = torch.tensor([0.2, 0.1])
        state.apply_topology(
            TopologyDelta.clone(
                1,
                torch.tensor([0]),
                child_fractions=torch.tensor([1.0]),
            )
        )
        self.assertEqual(state.group_count[0].item(), 0)
        torch.testing.assert_close(state.group_variance[0], torch.zeros(2))


class TopologyDeltaTests(unittest.TestCase):
    def test_clone_rejects_duplicate_parents(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            TopologyDelta.clone(3, torch.tensor([1, 1]))

    def test_split_matches_torch_repeat_child_order(self):
        delta = TopologyDelta.split(4, torch.tensor([1, 3]), 2)
        self.assertEqual(delta.source_indices.tolist(), [0, 2, 1, 3, 1, 3])
        self.assertEqual(
            delta.preserve_source_ids.tolist(),
            [True, True, False, False, False, False],
        )

    def test_split_requires_conservation_for_each_parent(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            TopologyDelta.split(
                2,
                torch.tensor([1]),
                children_per_parent=2,
                child_weights=torch.tensor([0.2, 0.2]),
            )

    def test_parent_indices_must_not_be_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            TopologyDelta.clone(2, torch.tensor([0.5]))

    def test_prune_mask_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "bool"):
            TopologyDelta.prune(torch.tensor([1, 0]))

    def test_public_constructor_cannot_bypass_kind_or_conservation(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            TopologyDelta(
                old_count=1,
                source_indices=torch.tensor([0, 0]),
                evidence_weights=torch.tensor([0.5, 0.5]),
                preserve_source_ids=torch.tensor([True, False]),
                kind="identity",
            )
        with self.assertRaisesRegex(ValueError, "conserve"):
            TopologyDelta(
                old_count=1,
                source_indices=torch.tensor([0]),
                evidence_weights=torch.tensor([0.5]),
                preserve_source_ids=torch.tensor([True]),
                kind="clone",
            )

    def test_mutated_delta_is_revalidated_before_application(self):
        delta = TopologyDelta.clone(1, torch.tensor([0]))
        delta.evidence_weights[0] = 1.0
        state = SemanticGaussianState.initialize(1, 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            state.apply_topology(delta)

    def test_clone_and_split_reject_ambiguous_public_layouts(self):
        with self.assertRaisesRegex(ValueError, "clone"):
            TopologyDelta(
                old_count=1,
                source_indices=torch.tensor([0, -1]),
                evidence_weights=torch.tensor([1.0, 0.0]),
                preserve_source_ids=torch.tensor([True, False]),
                kind="clone",
            )
        with self.assertRaisesRegex(ValueError, "either retained"):
            TopologyDelta(
                old_count=1,
                source_indices=torch.tensor([0, 0]),
                evidence_weights=torch.tensor([1.0, 0.0]),
                preserve_source_ids=torch.tensor([True, False]),
                kind="split",
            )


if __name__ == "__main__":
    unittest.main()
