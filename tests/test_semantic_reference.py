import unittest

import torch

from semantic_atdom.semantics.reference import (
    aggregate_observation_group,
    evidence_conditioned_belief,
    exact_robust_posterior,
    footprint_observation,
    render_semantic_tdom,
)


class SemanticReferenceTests(unittest.TestCase):
    def test_semantics_package_exports_evidence_contract(self):
        from semantic_atdom import semantics

        self.assertIs(
            semantics.evidence_conditioned_belief,
            evidence_conditioned_belief,
        )
        self.assertEqual(
            semantics.EvidenceBelief.__name__, "EvidenceBelief"
        )

    def test_evidence_conditioning_does_not_invent_background_mass(self):
        alpha = torch.tensor([[0.125, 3.125] + [0.125] * 6])

        result = evidence_conditioned_belief(alpha, prior_total=1.0)

        torch.testing.assert_close(
            result.evidence,
            torch.tensor([[0.0, 3.0] + [0.0] * 6]),
        )
        torch.testing.assert_close(
            result.distribution,
            torch.tensor([[0.0, 1.0] + [0.0] * 6]),
        )
        torch.testing.assert_close(result.evidence_ratio, torch.tensor([0.75]))

    def test_full_footprint_observation_ignores_unknown_pixels(self):
        contributions = torch.tensor(
            [
                [0.8, 0.0],
                [0.2, 0.3],
                [0.0, 0.5],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 255])
        boundary = torch.tensor([1.0, 0.5, 1.0])

        result = footprint_observation(
            contributions, labels, num_classes=2, boundary_weights=boundary
        )

        torch.testing.assert_close(
            result.evidence, torch.tensor([[0.8, 0.1], [0.0, 0.15]])
        )
        torch.testing.assert_close(result.all_mass, torch.tensor([1.0, 0.8]))
        torch.testing.assert_close(result.valid_mass, torch.tensor([0.9, 0.15]))
        torch.testing.assert_close(
            result.distribution, torch.tensor([[8.0 / 9.0, 1.0 / 9.0], [0.0, 1.0]])
        )
        self.assertFalse(result.evidence.requires_grad)

    def test_group_aggregation_contributes_at_most_one_sample(self):
        distributions = torch.tensor(
            [
                [[1.0, 0.0], [0.2, 0.8]],
                [[0.0, 1.0], [0.6, 0.4]],
            ]
        )
        supports = torch.tensor([[0.5, 0.25], [0.5, 0.5]])

        group = aggregate_observation_group(distributions, supports)

        torch.testing.assert_close(
            group.distribution,
            torch.tensor([[0.5, 0.5], [7.0 / 15.0, 8.0 / 15.0]]),
        )
        torch.testing.assert_close(group.support, torch.tensor([0.75, 0.625]))
        self.assertTrue(torch.all(group.support <= 1.0))

    def test_exact_robust_posterior_preserves_two_group_disagreement(self):
        group_distributions = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        group_supports = torch.ones((2, 1))

        result = exact_robust_posterior(group_distributions, group_supports)

        torch.testing.assert_close(result.weights, torch.ones((2, 1)))
        torch.testing.assert_close(result.raw_alpha, torch.tensor([[1.5, 1.5]]))
        torch.testing.assert_close(result.robust_alpha, result.raw_alpha)

    def test_robust_threshold_is_per_gaussian_effective_group_count(self):
        distributions = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )
        supports = torch.tensor(
            [
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )

        result = exact_robust_posterior(distributions, supports)

        self.assertEqual(result.weights[:, 0].tolist(), [1.0, 1.0, 1.0])
        torch.testing.assert_close(
            result.robust_alpha[0], result.raw_alpha[0]
        )

    def test_exact_robust_posterior_consensus_excludes_prior(self):
        distributions = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )

        result = exact_robust_posterior(
            distributions, torch.ones((3, 1))
        )

        torch.testing.assert_close(
            result.js_divergence[2], torch.log(torch.tensor([2.0]))
        )

    def test_exact_robust_posterior_requires_leave_out_support(self):
        distributions = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        supports = torch.tensor([[0.1], [0.1], [1.0]])

        result = exact_robust_posterior(
            distributions,
            supports,
            min_consensus_support=1.0,
        )

        self.assertEqual(result.weights[2].item(), 1.0)

    def test_float_labels_are_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            footprint_observation(
                torch.tensor([[0.5]]),
                torch.tensor([0.5]),
                num_classes=2,
            )

    def test_semantic_tdom_mixes_soft_gaussian_probabilities_before_argmax(self):
        contributions = torch.tensor([[0.6, 0.3], [0.1, 0.0]])
        alpha = torch.tensor([[5.5, 4.5], [0.5, 3.5]])

        result = render_semantic_tdom(
            contributions,
            alpha,
            min_coverage=0.2,
            min_absolute_support=0.1,
            min_support_ratio=0.1,
            min_margin=0.9,
        )

        self.assertEqual(result.class_ids.tolist(), [1, 255])
        self.assertGreater(result.probabilities[0, 1], result.probabilities[0, 0])
        self.assertLess(result.margin[0], 0.9)

    def test_semantic_tdom_class_probabilities_exclude_prior_mass(self):
        result = render_semantic_tdom(
            torch.tensor([[1.0]]),
            torch.tensor([[0.5, 3.5]]),
        )

        torch.testing.assert_close(
            result.probabilities, torch.tensor([[0.0, 1.0]])
        )
        torch.testing.assert_close(
            result.pixel_evidence_support, torch.tensor([0.75])
        )
        self.assertEqual(result.class_ids.item(), 1)

    def test_semantic_tdom_reports_independent_uncertainty_diagnostics(self):
        result = render_semantic_tdom(
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[1.5, 0.5], [0.5, 1.5]]),
            group_disagreement=torch.tensor([0.2, 0.6]),
        )

        torch.testing.assert_close(
            result.probabilities, torch.tensor([[0.5, 0.5]])
        )
        torch.testing.assert_close(
            result.top1_probability, torch.tensor([0.5])
        )
        torch.testing.assert_close(result.margin, torch.tensor([0.0]))
        torch.testing.assert_close(
            result.normalized_entropy, torch.tensor([1.0])
        )
        torch.testing.assert_close(
            result.mixture_disagreement, torch.tensor([0.5])
        )
        torch.testing.assert_close(
            result.group_disagreement, torch.tensor([0.4])
        )
        torch.testing.assert_close(
            result.prior_regularized_variance_proxy,
            torch.tensor([0.0625]),
        )

    def test_semantic_tdom_hard_map_uses_only_the_three_support_gates(self):
        result = render_semantic_tdom(
            torch.tensor(
                [
                    [0.4, 0.0],
                    [0.2, 0.0],
                    [0.2, 0.8],
                    [0.05, 0.0],
                ]
            ),
            torch.tensor([[0.5, 1.5], [0.5, 0.5]]),
            min_coverage=0.1,
            min_absolute_support=0.09,
            min_support_ratio=0.2,
            min_margin=1.1,
        )

        self.assertEqual(result.class_ids.tolist(), [1, 1, 255, 255])

    def test_copied_split_state_preserves_rendered_semantic_field(self):
        parent_alpha = torch.tensor([[2.5, 1.5]])
        before = render_semantic_tdom(
            torch.tensor([[0.7]]), parent_alpha
        )
        after = render_semantic_tdom(
            torch.tensor([[0.3, 0.4]]),
            parent_alpha.repeat(2, 1),
        )

        torch.testing.assert_close(after.probabilities, before.probabilities)
        torch.testing.assert_close(after.coverage, before.coverage)
        torch.testing.assert_close(
            after.pixel_evidence_support,
            before.pixel_evidence_support,
        )
        torch.testing.assert_close(after.support_ratio, before.support_ratio)
        torch.testing.assert_close(after.entropy, before.entropy)
        self.assertEqual(after.class_ids.tolist(), before.class_ids.tolist())

    def test_unknown_pixel_entropy_is_nan_not_false_certainty(self):
        result = render_semantic_tdom(
            torch.tensor([[0.8]]),
            torch.tensor([[0.5, 0.5]]),
        )

        self.assertEqual(result.class_ids.item(), 255)
        self.assertTrue(torch.isnan(result.probabilities).all().item())
        self.assertTrue(torch.isnan(result.entropy).item())
        self.assertTrue(torch.isnan(result.margin).item())


if __name__ == "__main__":
    unittest.main()
