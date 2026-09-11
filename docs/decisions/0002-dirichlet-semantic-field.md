---
status: accepted
amended_by: 0005-evidence-conditioned-semantic-rendering
---

# Represent closed-set Gaussian semantics with a Dirichlet-style evidence state

Each Gaussian uses an eight-class categorical–Dirichlet-style state with a symmetric prior of total concentration one. This preserves fractional visibility-weighted evidence, separates observation scarcity from class conflict, and supports forward-only TDOM rasterization without iterative semantic training; arbitrary learned feature embeddings are not the V1 authority.

ADR 0005 clarifies that the symmetric prior is not observed class mass: product rendering conditions on `alpha - alpha0`, and any variance derived from `alpha` is an explicitly uncalibrated proxy rather than a classification-error probability.
