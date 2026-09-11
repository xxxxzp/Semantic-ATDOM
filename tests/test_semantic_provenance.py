import unittest

import torch

from semantic_atdom.semantics.provenance import (
    rgb_model_sha256,
    semantic_projection_sha256,
)


class SemanticProvenanceTests(unittest.TestCase):
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
