---
status: accepted
date: 2026-09-22
---

# Recover bounded planar semantic holes with direct multi-view evidence

## Problem

The scale-outlier support filter from ADR 0007 prevents a few very large
Gaussians from spreading semantic classes over the unobserved map boundary.
On the final Phantom3 product, the same filter also removes the only Gaussian
support over a real water region. This produces a bounded 311,232-pixel
unknown component inside otherwise continuous water.

Disabling the filter is not acceptable. A controlled render of the same frozen
model showed that doing so classifies all 13,315,020 boundary-connected outer
pixels as well as the water hole. Lowering the three pixel gates cannot recover
pixels whose trusted Gaussian coverage is zero.

## Decision

Keep the Gaussian support filter and add an opt-in, final-publication-only
fallback for explicitly configured planar classes. The fallback never mutates
the Gaussian model or semantic state and never runs in online training-video
frames.

For each connected unknown component, the implementation requires all of the
following:

1. The component does not touch any map edge.
2. Its dilated boundary has at least 98% consensus for the configured class.
3. At least 32 nearby trusted Gaussians of that class define a local plane;
   robust plane-offset MAD is at most one world unit.
4. Every component pixel is back-projected to that plane and sampled from the
   original rectified semantic labels using the final camera inventory.
5. A pixel needs at least five valid source views, at least 95% class
   agreement, and supporting majorities in at least three independent final
   rebuild observation groups.

The generic default remains disabled. It is enabled by setting
`--semantic_tdom_planar_fallback_class water` (the command-line parser uses
underscored parameter names). All thresholds are serialized in the run config
and Semantic TDOM metrics.

## Product semantics

The hard class map and unknown rasters are updated only for accepted pixels.
The product publishes `class_provenance.png` independently from the class ID:

- `0`: unknown;
- `1`: trusted Gaussian evidence;
- `2`: planar multi-view fallback.

Gaussian support and Gaussian uncertainty rasters retain their original
meaning. Direct-vote entropy/probability diagnostics are updated when those
products are enabled, while Gaussian-only diagnostics remain distinguishable
through provenance. This avoids representing a direct image-space decision as
Gaussian support.

## Acceptance experiment

The frozen final run
`phantom3-v2-progress240-fps1_20260921-180740_c886ce6` was validated without
retraining. Results are stored at:

`/data3/xzp/data/Onthefly/phantom3/outputs/experiments/water-planar-fallback-v1`

The main component was restored as water for 311,232/311,232 pixels. It had
58 contributing cameras, 32--46 valid views per pixel, 20--28 independent
observation groups, and 100% water agreement. A second 10,959-pixel bounded
water hole passed the same gates. Total recovered pixels were 322,191;
boundary-connected newly classified pixels were exactly zero, and every new
class pixel had provenance code 2.

The experiment is reproducible with `tools/validate_planar_fallback.py`; its
`acceptance.json` is the machine-readable result.

## Limits

This repair establishes the semantic class of a bounded planar region; it does
not repair the oversized Gaussian geometry or alter the RGB TDOM. Non-planar
classes must not be enabled merely to remove black pixels. A different dataset
must opt in by class name and pass its own frozen-model acceptance experiment.
