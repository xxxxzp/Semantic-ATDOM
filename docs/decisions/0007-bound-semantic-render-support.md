---
status: accepted
date: 2026-09-15
---

# Bound Semantic TDOM support from scale-outlier Gaussians

Progressive geometry can retain a small number of extremely large Gaussians.
Their SH appearance may be black or nearly invisible in RGB TDOM while their
opacity still gives them almost complete alpha coverage. If their semantic
evidence is rendered without qualification, black RGB border pixels receive a
saturated semantic class after evidence normalization.

Semantic TDOM therefore restores the V1 trusted-geometry rule before semantic
rasterization:

\[
\max(\mathbf s_g) \le r_s\,\operatorname{median}_h\max(\mathbf s_h),
\qquad r_s=14\ \text{by default}.
\]

The threshold is configurable through
`semantic_tdom_max_gaussian_scale_ratio`; zero disables the filter. The filter
is applied consistently to semantic geometry, evidence features and diagnostic
features. Its input, retained and excluded counts, median scale and effective
limit are recorded in Semantic TDOM render metrics.

This is a read-only product-rendering support-domain filter, not a fourth
pixel-level hard-map gate. It does not delete Gaussians, alter RGB TDOM, feed
semantic information into geometry, or participate in training, densification,
clone, split or prune. The hard semantic map remains controlled only by geometry
coverage, absolute evidence support and support ratio after the trusted support
domain has been selected.

The rule originated in G1 Prior V1 commit `eb57e4d`, where its engineering
default was 20. G1 Evidence V2 was built from an earlier branch point and
accidentally omitted it; the resulting full Phantom3 product classified all
3,618 exactly black RGB pixels. On the frozen 942,438-Gaussian final snapshot,
ratio 15 still classified 379 black pixels, ratio 14 classified none, and
ratios 10 and 5 removed progressively more otherwise supported pixels. V2
therefore uses 14 as the least restrictive tested value that closes this
regression. This is empirical support-domain calibration, not a semantic
accuracy threshold; it must remain configurable and be reevaluated on geometry
with materially different scale statistics.

## Known limitations and successor direction

This decision is an accepted **provisional containment**, not the final
definition of the Semantic TDOM valid footprint:

- A scene-global median scale ratio is data dependent. It can remove a valid
  large Gaussian in a genuinely multi-scale scene, or retain a smaller but
  still erroneous Gaussian. Ratio 14 is only calibrated against one frozen
  Phantom3 snapshot.
- RGB TDOM still renders the complete Gaussian set while Semantic TDOM renders
  the trusted subset. Their grids and frozen model identity remain paired, but
  their effective per-pixel support domains are therefore not mathematically
  identical.
- Exact or near-black RGB is appearance, not a no-data definition. A valid
  shadow, roof or dark surface may also be black, so RGB intensity must not
  become a semantic rejection gate. The zero-black-pixel regression check
  verifies this particular failure mode but does not prove a general valid
  footprint.

The successor design should define and publish an authoritative per-pixel
footprint independently of RGB color and semantic labels, then use that same
footprint when deciding whether the paired RGB and Semantic products contain
data. Candidate sources include the union of valid source-camera footprints or
existing geometry/training validity masks. Its exact source, update rule and
interaction with the locked three semantic gates require a separate design
decision. It must remain downstream-only and must not allow noisy SAM labels to
affect geometry optimization.
