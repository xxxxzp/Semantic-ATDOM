---
status: accepted
date: 2026-09-11
supersedes: prior-weighted rendering interpretation in 0002
---

# Render observed semantic evidence, not prior class mass

The symmetric Dirichlet prior exists to keep an empty Gaussian state well-defined. It is not an observed class label and must not create background or any other class mass in Semantic TDOM.

For each Gaussian, G1 therefore stores a Dirichlet-style state but derives rendered class belief from the non-negative observed evidence `alpha - alpha0`. The evidence-conditioned class distribution supplies class probability, entropy, margin and mixture disagreement; the evidence ratio separately supplies absolute semantic support. The prior-regularized Dirichlet variance remains an explicitly uncalibrated uncertainty proxy, not a classification-error probability.

This decision preserves the lightweight closed-form accumulator selected in ADR 0002 while superseding its interpretation as a standard calibrated categorical posterior for product rendering.
