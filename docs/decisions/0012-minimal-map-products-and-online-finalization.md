---
status: accepted
date: 2026-10-21
amends: 0008 default map products and 0011 final semantic policy
---

# Minimal map products and reuse of progressive semantics

The user requested RGB and Semantic TDOM as the normal deliverables, with
other rasters optional, and rejected mandatory full semantic reconstruction
after progressive fusion. These preferences supersede the older mandatory
support/entropy exports and exact-final default.

The trainer now defaults to progressive semantic fusion and final reuse of
that state. This remains enabled when no progress video is requested. Before
final publication, pending work and topology events are drained, remaining
accepted observations are ingested, the open group is closed, input/schema
bindings are checked, and the existing state is checkpointed against the final
geometry. Its status is `approx_final`. Closed historical evidence is not
reprojected; `historical_evidence_stale` remains truthful. The semantic state
does not enter the RGB geometry loss. Geometry refinement, pruning and state
transport retain their existing behavior.

`--semantic_final_rebuild` explicitly requests the previous two-pass final
reprojection and robust fusion. A complete authoritative run then has status
`exact_final`. With final-only maps and no video/checkpoint consumers, this
mode can defer semantic computation until the final boundary. The direct hook
API retains its legacy `final_rebuild=True` default for existing callers;
the CLI passes its selected policy explicitly. Planar fallback currently needs
the exact final groups and therefore requires this flag.

Final validation accepts only the requested status, matching final geometry
stage, zero lag, no deferred observations/errors, a complete hash-verified map
pair, and a valid schema. Approximate finalization permits stale historical
evidence; exact finalization does not. Both policies archive
`final_semantic_metadata.json`; only an executed rebuild emits
`final_rebuild_metrics.json`. Approximate finalization does not silently fall
back to a full rebuild after failure.

Semantic manifest schema 4 defaults to `semantic_color.png` plus small class,
palette and alignment/integrity metadata. Paired publication adds `rgb/tdom.png`.
The optional raster flags are:

- `--semantic_tdom_save_class_ids`: integer class IDs.
- `--semantic_tdom_save_masks`: unknown mask/reason and fallback provenance.
- `--semantic_tdom_save_evidence`: evidence-support TIFF.
- `--semantic_tdom_save_entropy`: entropy TIFF.
- `--semantic_tdom_save_diagnostics`: the complete legacy auxiliary raster set.
- `--semantic_tdom_save_probabilities`: probability cube.

Output estimates and overload degradation account for selected products.
Manifests declare actual exports and hashes; no omitted file is referenced as
present. Class sampling, unknown gates, palette colors, atomic pair publication,
source-data read-only rules and the two mandatory images are unchanged by the
export profile. Model/recovery state is independent of optional map rasters.

Performance is measured on the same saved final pixel grid and the same PNG
configuration. Both required PNG hashes must equal the source. Export replay
does not establish final online-vs-rebuilt semantic quality; that is a separate
policy tradeoff, and agreement between two states is not ground-truth accuracy.
