# Phantom3 stage-30 semantic smoke experiment

Date: 2026-09-10

This experiment verifies that the WP2–WP5 semantic path can consume the
delivered SAM3 labels. It is an engineering and performance smoke test, not a
semantic-accuracy evaluation: the geometry is the stage-30 COLMAP point cloud,
before Gaussian optimization.

## Inputs

- RGB/COLMAP stage: `datas/phantom3_test100/30`
- External semantic root: `datas/semantic_labels`
- Images used: `000001` through `000030`
- Label shape: `897 × 1600`, without optional score tensors
- Class count: 8
- Initial Gaussians: 19,647
- GPU: NVIDIA GeForce RTX 4090

The complete external delivery contains 102 labels and 102 metadata files. All
102 label hashes, dimensions, class IDs, class-spec hashes, and metadata
contracts were validated before this run.

## Result

- Online pass completed for all 30 images.
- Exact final rebuild formed 12 acquisition-overlap groups and reprojected all
  30 images twice.
- All 19,647 initial Gaussians received at least one valid observation.
- Exact rebuild time: 8.886 s
- Projection time within the rebuild: 8.263 s
- Label-read time within the rebuild: 0.539 s
- Fusion time within the rebuild: 0.054 s
- Peak CUDA allocation: 630,859,776 bytes (about 602 MiB)
- Total wall time including scene/camera loading and the online pass: 18.913 s
- Online versus exact mean JS divergence: 0.003902
- Online versus exact maximum JS divergence: 0.084955
- Online versus exact top-1 class agreement: 96.926%

The observed-Gaussian top-1 class counts, in class-ID order, were
`[0, 2328, 6276, 4588, 4340, 194, 61, 1860]`. These counts must not be treated
as final map-quality statistics because Gaussian geometry and opacity were not
optimized in this smoke setup.

## Issue discovered by real data

The CUDA-derived support formula produced values up to `1.00000143` because of
float32 rounding. Support is probabilistic by definition, so the projector now
clamps the final value to `[0, 1]` while downstream aggregation remains strict.
The old PyTorch environment also failed to JIT `torch.prod` for the RTX 4090
architecture; the at-most-four-row group product now uses an equivalent small
iterative multiplication. The successful result above was obtained after both
fixes.
