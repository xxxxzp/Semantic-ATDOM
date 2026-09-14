---
status: accepted
date: 2026-09-14
supersedes: prior-weighted rendering and evidence-conserving topology in V1
---

# V1.1 uses observed evidence without adopting the complete V2 architecture

## Context

Semantic A-TDOM consumes hard, offline SAM3 labels from correlated nadir UAV
images. The labels have no calibrated per-pixel scores and no manual ground
truth. Semantics must remain a detached sidecar: it may observe RGB geometry
and topology events, but it must never change geometry, opacity, camera poses,
densification, pruning, or RGB optimization.

The validated V1 pipeline satisfies the engineering and geometry-isolation
requirements, including full-resolution tiled rendering and the detached
Semantic TDOM scale-outlier safety filter. Its original semantic mathematics,
however, treats a symmetric Dirichlet prior as class mass and divides evidence
among topology descendants. Both operations change product semantics without
new observations.

## Decision

V1.1 keeps the V1 training, projection, grouping, ledger, exact rebuild,
rendering, palette, and geometry-safety pipeline. It adopts only the following
evidence concepts from the V2 research line.

### Evidence-conditioned class belief

For `K` classes and symmetric prior `alpha0`, define:

```text
E = clamp(alpha - alpha0, min=0)
q = E / (sum(E) + eps)
rho = sum(E) / (sum(alpha) + eps)
```

`q` supplies the official class distribution. `rho` independently supplies
evidence support. The prior is numerical state, not a background observation
and not a vote for any class.

### Eligible leave-one-group-out robustness

The consensus for group `h` excludes both group `h` and the prior. JS-based
downweighting is eligible only when the remaining evidence contains at least
two independent acquisition groups and has total support at least
`min_consensus_support` (default `1.0`). Otherwise reliability is exactly one.

Correlated frames remain collapsed into bounded acquisition groups before this
test, so adjacent flight frames cannot manufacture confidence by repetition.

### Topology continuity

Clone and split descendants copy their parent's local class evidence state.
They do not divide the concentration by the number of descendants. This keeps
rendered probability, evidence ratio, entropy, and hard class continuous under
a geometry-only reparameterization. Online descendants are marked dirty and
closed historical evidence is marked stale; final products rebuild all evidence
on frozen final geometry.

### Primary and conservative raster products

The primary `semantic_id.png` uses only structural gates:

- nonzero/configured geometry coverage;
- absolute evidence support;
- evidence support ratio.

Uncalibrated margin, top-1 probability, entropy and group disagreement do not
silently remove pixels from the primary map. They form a separate conservative
product (`semantic_id_conservative.png`) and `ambiguity_mask.png`. This exposes
the policy choice while retaining uncertainty at large mixed Gaussians and real
class boundaries. Group-disagreement rejection is applied only after an exact
rebuild has produced valid multi-group statistics; the manifest records whether
that gate was available instead of claiming it for online maps.

### Version and authority

The method identity is `g1-evidence-v1.1`; semantic state schema is version 3
and product schema is version 3. V1 sidecars cannot resume under V1.1 because
their probabilities, robust weights and topology inheritance have different
meanings. They remain valid only as frozen A/B inputs.

Online products are explicitly `approx_online` or `stale_online`. A successful
final rebuild is `exact_final` only when every label has source-image binding;
otherwise it is `exact_partial`. A failed requested final rebuild may expose an
`online_fallback`, but semantic failure must not block RGB output.
Source binding means that the declared source hash also matches the RGB bytes
actually loaded by the current training snapshot. A self-consistent metadata
path to an unrelated same-stem image is insufficient. Relative source paths are
resolved from the RGB dataset root, including when semantic data uses a separate
root.

## Deliberately not adopted

V1.1 does not merge the V2 fixed-grid publication system, latest-wins map
scheduler, low-priority CUDA stream design, or broad provenance refactor. Those
are independent engineering candidates and require their own profiling and
acceptance tests. V1's detached scale-outlier product filter remains active
until a source-footprint validity mask is validated against Phantom3.

## Validation order

1. Unit and CUDA integration invariants.
2. Frozen V1 final geometry/state rendering A/B.
3. Exact V1.1 rebuild on frozen Phantom3 geometry.
4. Progressive stage 30 through 100 smoke and resource profile.
5. Full 399-image experiment only after the preceding checks pass.
