# Semantic training at resolution 2, 4 and 8

`tools/train_progressive.py --enable_semantics --resolution r` supports
`r = 1, 2, 4, 8`. The original width and height are divided by `r`, rounded
separately, exactly as in RGB camera loading. XinCao's 4032×3024 inputs become
2016×1512, 1008×756 and 504×378. Default CLI behavior is unchanged: semantic
training requires an explicit supported factor; the RGB-only automatic-width
and explicit-width modes remain available.

## Coordinate contract

There are three distinct grids:

1. Original RGB, labels and SfM observations, bound by original dimensions and
   SHA256 hashes in preflight. These inputs are only read.
2. A full-size rectified centered pinhole grid, described by the immutable
   `RectifiedPinhole`. RGB uses linear rectification; labels and boolean masks
   use nearest rectification. Out-of-source semantic pixels are ignore 255.
3. The actual rounded training grid, with shared FoV, pose and projection.
   Its effective focal lengths are `fx * W_train / W_source` and
   `fy * H_train / H_source`. Never overwrite the source calibration with
   scaled dimensions: it is needed when reading full-size original labels.

[COLMAP defines the first pixel center as (0.5, 0.5)](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools).
OpenCV and rasterizer array indices start at center (0, 0). Both principal
points passed to OpenCV's image rectification map subtract 0.5. Stored
calibration and SfM `xys` keep COLMAP edge coordinates. `rectify_points` thus
continues to return edge coordinates; it must not receive another half-pixel
shift.

The array-center resize map is
`u_source = (u_train + 0.5) * W_source / W_train - 0.5` (and likewise for y).
For discrete labels and masks, source indices are
`floor((u_train + 0.5) * W_source / W_train)`.
`pixel_grid.nearest_indices` implements this with integer arithmetic. The same
indices sample class IDs, boolean masks and reliability weights. The previous
torch `nearest` mask resize sampled from the left edge; simply using nearest
in two libraries did not guarantee alignment. RGB retains Pillow's continuous
resize, with the same center convention. No class IDs pass through RGB
interpolation or video encoding.

Initialization and newly added incremental cameras call the same `loadCam`.
Retained cameras whose calibration changes reload and rectify original RGB
onto their saved training size. Frozen semantic cameras preserve both actual
training dimensions and original calibration. Final exact reconstruction and
planar fallback use the same shared label-grid adapter as online projection.

SfM filtering uses each camera's original calibration dimensions to map its
full-grid rectified observations to its training `LogMask`. In COLMAP edge
coordinates, the containing training pixel is `floor(x * W_train / W_source)`.
The legacy global dimensions are only a fallback for camera records without
calibration or source dimensions; no SfM file is rewritten.

Boundary bandwidth remains in full-size rectified source pixels. Compute
the reliability ramp there, then sample it at the same centers as the IDs.
The training mask is lifted onto the source grid with nearest sampling solely
to gate reliability and preserve ignore-boundary behavior. Downsampling a
small class can lose it, but cannot create an interpolated class ID.
Source rectification and boundary computation retain their full-size CPU
cost. Gaussian footprint sampling and evidence support may change with the
training grid; numerical equality of support across factors is not promised.

## Recovery and output

`colmap-center-nearest-v1` is recorded in the semantic configuration and camera
checkpoint records. Semantic recovery rejects an older configuration.
Camera restoration rejects records without the mapping version and rejects
dimensions incompatible with the requested factor. Old training checkpoints
must be restarted rather than silently mixing evidence from different grids.

`--skip_final_evaluation` is included from the main worktree's existing
uncommitted change. It skips final PSNR/SSIM/VGG-LPIPS, records status `skipped`,
and retains exact semantic reconstruction, models, maps and video. It cannot
be combined with `--render_all_cameras`, whose exports still run inside final
evaluation.

The existing video remains a paired RGB/semantic TDOM training timeline.
Its map sampling factor is independent of the training factor. Validate class
IDs using source labels and map ID arrays, not compressed video colors.

## Validation setup

Worktree: `/home/xuzipeng/code/Semantic-A-TDOM/worktrees/semantic-resolution-248`

Branch: `feat/semantic-resolution-248`, based on `c75f622`.

Dataset: `/data3/xzp/data/Onthefly/XinCao`; RGB folder must be explicitly passed
as `--Image_Folder_Path .../images`, since automatic shared-folder discovery
currently recognizes `*_CAM` folders. Semantic delivery root is
`.../XinCao/semantic_labels`.

Acceptance runs select stages 14–16: 14 initial views and two real incremental
stages, with 10 initialization iterations, 5 local + 5 global iterations per
increment, and 10 final refinement iterations (40 total). They enable
`--Use_Tri_Mask`, semantics, TDOM rendering, a three-frame training progress
video and `--skip_final_evaluation`. This is a functional smoke test, not a
reconstruction-quality or full 229-image training benchmark.

Validation artifacts and exact commands are retained outside all input and
training directories at
`/home/xuzipeng/code/Semantic-A-TDOM/validation/semantic-resolution-248/`.
`run_smoke.py` invokes each factor serially through bubblewrap with XinCao,
the main checkout and `/home/xuzipeng/anaconda3` mounted read-only. It prepends
the worktree source to `PYTHONPATH`, suppresses bytecode writes and adds the
existing environment's `bin` to the subprocess PATH. No packages are installed
and no CUDA extension is changed or rebuilt.

`CUDA_VISIBLE_DEVICES=GPU-b12f49e8-3856-55cc-a3d0-0b3c2e65f0f2` selects physical
GPU 4; `--gpu 0` means its visible logical index. Physical GPU 1 is excluded.

CPU regression tests cover odd/rounded dimensions, class-ID and mask center
sampling, analytic RADIAL maps and SfM point round trips, source-size rejection,
shared projection/fallback behavior, source-unit boundary weights, and camera
checkpoint preservation/rejection. The existing CUDA integration tests run
against the already installed extension.

Run tests from this worktree with the same UUID and import path:

```bash
export CUDA_VISIBLE_DEVICES=GPU-b12f49e8-3856-55cc-a3d0-0b3c2e65f0f2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/third_party/src:$PWD"
/home/xuzipeng/anaconda3/envs/semantic-atdom/bin/python -m unittest discover -s tests -v
/home/xuzipeng/anaconda3/envs/semantic-atdom/bin/python -m unittest discover -s tests/integration -v
```

## Acceptance results (2026-10-12)

- 245 unit/regression tests passed; all 22 existing CUDA integration tests
  passed with the installed extension. The nine resolution tests also passed
  after expanding off-center RADIAL and multi-factor mask coverage.
- All three real runs succeeded at final stage 16 and optimizer iteration 40.
  Final inventories contain 16 cameras with correctly scaled focal lengths.

| Factor | Training grid | Final status | Lag | Video frames | Gaussians |
| --- | --- | --- | --- | --- | --- |
| 2 | 2016×1512 | exact_final | 0 | 3 | 12,826 |
| 4 | 1008×756 | exact_final | 0 | 3 | 14,344 |
| 8 | 504×378 | exact_final | 0 | 3 | 16,924 |

Each final rebuild processed all 16 bound source observations with no errors
or deferred IDs and cleared historical staleness. All RGB/semantic map pairs
were published, their manifest hashes checked, and their 5120×3584 grids
matched. Every ID map contains only `{1, 2, 3, 4, 5, 7, 255}`, a valid subset of
the eight source classes plus ignore. Intermediate semantic frames can retain
the existing `stale_online` status after geometry changes; the final frame is
`exact_final`.

All three H.264 videos are 3206×1080 at 4 fps, have exactly three decoded frames,
and include initialization, an incremental stage, and final refinement.
`ffmpeg` decoded each entire video without errors. All final evaluation records
are `skipped` and contain no fabricated quality metrics.

The three archived semantic states reload successfully with matching geometry,
ontology and recovery-context hashes, eight classes, finite nonnegative
evidence and no active projection or historical staleness. Their render-cache
`dirty` flags remain set because the state checkpoint is saved before final
map publication marks the live cache clean; this is separate from evidence
staleness. These checks are recorded in `state-verification.json`.

XinCao's **3,316 files, 8,440,095,515 bytes** have identical path lists, file
sizes, modification timestamps and SHA256 hashes before and after validation.
Both full input manifests have SHA256
`874923af1bbe8cd9a8a85922afdd59189588d9d97dd7857aba1f9dadd0f5c504`.
The main checkout's original three-file uncommitted patch also retains its
original SHA256
`eb5898c9dd3cd522767049c2a03bb591b5ffe3cfc1fa78243bf50e3081ff535b`.

The external validation directory contains `artifact-verification.json`,
`source-verification.json`, before/after input manifests, per-factor exact
commands and console logs, test logs, videos and final-frame previews.
`verify_artifacts.py` reproduces camera, map, semantic-state and video checks.
Do not rerun `run_smoke.py` into the existing nonempty outputs; use fresh output
roots if repeating the experiment.
