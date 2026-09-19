---
status: accepted
date: 2026-09-17
supersedes: output naming and progressive-video sections of 0008
---

# Publish explicit renders and one synchronized map-training video

The output schema uses names that describe artifacts rather than the reason a
developer might inspect them.  Final per-camera renders are published under
`renders/cameras/`; they are not hidden behind the ambiguous term `qa`.
Structured per-camera measurements remain under `metrics/per_camera.csv`.

Optional video output is published under `videos/`.  A run emits only one
video, `videos/training_progress.mp4`, when the user explicitly enables
`--render_training_progress`.  The previous camera-tour and single-product
TDOM videos are removed because they did not show model convergence clearly
and duplicated the published interface.

The public run layout is output schema v2:

```text
<run>/
├── manifest.json
├── model/
├── products/maps/
├── renders/cameras/       # only with --render_all_cameras
├── videos/                # only with --render_training_progress
├── metrics/
├── logs/
└── provenance/
```

## Training-progress video contract

The video presents RGB TDOM and Semantic TDOM side by side at the same
training boundary.  It models online mapping: every frame uses only cameras,
RGB state, and semantic state accepted by that stage.  The map extent may grow
in any direction, while the grid identity, origin, axes, and ground sampling
distance remain unchanged.  During final encoding, variable extents are
placed into one world-aligned black canvas; existing map content is never
re-centered independently per frame.

The renderer uses a video-only TDOM camera with a constant eight-times coarser
ground sampling distance than the publication map.  This bounds render and
IPC cost without changing the spatial relationship between frames.  The
published H.264 video is 1080 pixels high, up to 3840 pixels wide, and 4 FPS by
default.  Its width is fitted once from the final global map aspect ratio and
remains fixed for the complete video.  The two maps share one scale and have a
24-pixel gutter instead of being centered independently in two fixed-width
half canvases.  Its left panel contains the RGB TDOM, accumulated flight
trajectory, current position, scale bar, and local `+Y` direction.  The right
panel contains the unobscured hard-class Semantic TDOM.  A separate legend
column beside it spans the map height and distributes the active run-schema
classes vertically.  Unknown and unobserved pixels are black.  A local-axis
label is used instead of claiming geographic north when the grid has no CRS.

Frame selection reserves one frame for final refinement and distributes the
remaining target frame count uniformly from the initialization boundary
through the final progressive stage.  Defaults are:

```text
--training_progress_target_frames 120
--training_progress_fps 4
```

The first initialized state, final progressive state, and post-refinement
state are mandatory.  Before a frame is rendered, the semantic worker must be
idle and its accepted image boundary must reach the RGB geometry stage.  This
guarantees zero boundary lag, but it does not relabel an approximate or
historically stale online semantic state as exact.  Those states are part of
the real streaming workflow and remain visible in the frame overlay and the
timeline metadata.  Failure to reach the same accepted image boundary fails an
explicitly requested video rather than silently mixing stages.

Temporary RGB JPEG, semantic PNG, and frame metadata live below
`.work/video_frames/` and are deleted only after ffmpeg produces a verified
H.264 stream with the requested frame count and dimensions.  The auditable
timeline is retained as `metrics/training_progress.json`; it records the stage,
phase, optimizer iteration, observed-camera count, semantic boundary, semantic
status/lag/stale-evidence flag/errors, and map window for every encoded frame.

## Command-line contract

Video generation is disabled by default and is enabled with:

```text
--render_training_progress
--training_progress_target_frames N
--training_progress_fps FPS
```

The old `--progress_video`, `--progress_video_content`, and
`--progress_video_stride` interfaces are intentionally removed.  Output schema
v2 does not publish compatibility aliases for `qa/` or `visualizations/`.
