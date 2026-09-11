import unittest

import torch

from semantic_atdom.semantics.provenance import (
    rgb_model_sha256,
    semantic_projection_sha256,
    semantic_state_sha256,
)
from semantic_atdom.semantics.state import SemanticGaussianState


class SemanticProvenanceTests(unittest.TestCase):
    def test_semantic_state_content_hash_is_stable_and_changes_with_evidence(self):
        state = SemanticGaussianState.initialize(1, 2)
        context = {
            "schema_version": 2,
            "semantic_method_version": "g1-evidence-v2",
            "ledger": {"schema_version": 1, "observations": []},
            "label_provenance": None,
            "open_group": None,
            "ontology": {
                "schema_version": 1,
                "ignore_id": 255,
                "classes": [
                    {"id": 0, "name": "ground"},
                    {"id": 1, "name": "building"},
                ],
            },
            "semantic_config": {
                "schema_version": 2,
                "semantic_method_version": "g1-evidence-v2",
            },
        }
        first = semantic_state_sha256(state, context, "a" * 64, "b" * 64)
        second = semantic_state_sha256(state, context, "a" * 64, "b" * 64)
        state.add_evidence(torch.tensor([[1.0, 0.0]]))
        changed = semantic_state_sha256(state, context, "a" * 64, "b" * 64)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_projection_hash_excludes_sh_while_rgb_hash_includes_it(self):
        attributes = {
            "xyz": torch.zeros((1, 3)),
            "features_dc": torch.zeros((1, 1, 3)),
            "features_rest": torch.zeros((1, 0, 3)),
            "opacity": torch.zeros((1, 1)),
            "scaling": torch.zeros((1, 3)),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
        changed_sh = dict(attributes)
        changed_sh["features_dc"] = torch.ones((1, 1, 3))

        self.assertEqual(
            semantic_projection_sha256(attributes),
            semantic_projection_sha256(changed_sh),
        )
        self.assertNotEqual(
            rgb_model_sha256(attributes), rgb_model_sha256(changed_sh)
        )

    def test_projection_hash_changes_for_projection_attribute(self):
        attributes = {
            "xyz": torch.zeros((1, 3)),
            "features_dc": torch.zeros((1, 1, 3)),
            "features_rest": torch.zeros((1, 0, 3)),
            "opacity": torch.zeros((1, 1)),
            "scaling": torch.zeros((1, 3)),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
        changed = dict(attributes)
        changed["xyz"] = torch.ones((1, 3))

        self.assertNotEqual(
            semantic_projection_sha256(attributes),
            semantic_projection_sha256(changed),
        )


if __name__ == "__main__":
    unittest.main()
