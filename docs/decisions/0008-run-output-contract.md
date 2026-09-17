---
status: accepted
date: 2026-09-16
---

> The public naming and progressive-video sections are superseded by ADR 0009
> and output schema v2.  The lifecycle, retention, recovery, and publication
> decisions below remain in force.

# Separate progressive-training work state from published run results

The historical output tree mixed six different lifecycles at one level:
numeric stage snapshots, the final Gaussian model, paired map products,
semantic recovery slots, evaluation renders, and internal caches.  Names such
as `487/point_cloud/iteration_487`, `Intermediate_Results`, and
`OutputImages` exposed implementation details without telling a consumer
which files were final, resumable, optional, or safe to remove.

A run is now a self-describing publication.  Mutable state lives under
`.work/` while training is active.  A successful run publishes only these
top-level interfaces:

```text
<run>/
├── manifest.json
├── model/
├── products/maps/
├── metrics/
├── logs/
├── provenance/
├── qa/                  # only when explicitly requested
└── visualizations/      # only when explicitly requested
```

`manifest.json` is the entry point.  It records the run status, schema
version, dataset binding, Git revision, stage and optimizer iteration, enabled
optional outputs, and relative paths to the final model, products, metrics,
logs, and provenance.  Artifact references within a run are relative; source
dataset paths may remain absolute because they identify external inputs rather
than movable run artifacts.  `outputs/latest` may point only to a successfully
published run.

## Retention rules

- The default prioritizes training throughput and writes no progressive RGB
  recovery checkpoint.  A user may explicitly enable a single atomically
  replaced checkpoint with `--recovery_checkpoint_every_stages N`; zero means
  disabled.  It is retained after interruption and removed after success.
- A successful run always keeps the final Gaussian point cloud under `model/`.
  The completed optimizer/training state is disabled by default because it is
  large and a `run_complete` state is intentionally not resumable.  Users who
  need it for debugging may explicitly request
  `--save_final_training_state`, which publishes `model/training_state.pt`.
  Numeric stage directories are not part of the published interface.
- Semantic recovery continues to use two alternating slots while a run is
  mutable.  After successful final publication, the active verified state is
  archived once under `model/semantic/`; the stale rollback slot is discarded.
  The ledger, schema, recovery context, hashes, and final synchronization
  status remain available for audit.
- The paired RGB/Semantic map publication retains its `latest.json`, release
  manifest, product key, and content-hash contract under `products/maps/`.
  Core outputs include RGB, semantic products, evidence support, and semantic
  entropy.  Additional diagnostic rasters and probability cubes are opt-in.
- Per-camera evaluation images are opt-in through
  `--render_all_cameras`.  When enabled, images are stored in one
  `qa/camera_renders/` directory and per-camera metrics in one CSV rather than
  one directory and one text file per camera.
- Logs, aggregate metrics, configuration, environment capture, and dataset
  preflight evidence are always retained.  Empty placeholder files and
  directories are not created.

## Progressive video

Progress visualization is independent of final per-camera evaluation.
`--progress_video` accepts `none`, `camera`, `tdom`, or `both` and defaults to
`none`.  Camera frames show each newly accepted camera rendered from the model
at that progressive point; TDOM frames show the map footprint expanding on a
fixed grid.  `--progress_video_content` accepts `rgb`, `semantic`, or
`side_by_side`; `--progress_video_stride` controls temporal sampling and
`--progress_video_fps` defaults to the historical 4 fps.

Frames are produced only at data/stage boundaries, never at optimizer
iterations.  They are written by a bounded, lower-priority output worker under
`.work/video_frames/` and encoded after training.  Temporary frames are removed
after a verified video is published.  This produces a faithful progression
video without a live server or an encoder in the optimization loop.  Enabling
the option necessarily incurs explicit extra render and I/O cost.

## Failure and migration behavior

Run status transitions through `running` to `succeeded`, `failed`, or
`interrupted`.  A failed run is never exposed as `latest`; it keeps logs,
failure information, and an explicitly enabled recovery checkpoint.  Cleanup
is deliberate rather than silent.

Legacy results are migrated in place only after recording file sizes and
SHA-256 values.  Moves are atomic where the filesystem permits, map-release
internals are not flattened, relative references and manifests are rewritten
before validation, and source files are removed only after the destination has
been verified.  The migration does not invent products that the legacy run did
not generate.

This is an intentional output-schema break.  It removes the ambiguous legacy
tree instead of indefinitely maintaining two public layouts.

## Online semantic state replacement

The output refactor's first full-run attempt exposed a pre-existing async
ordering bug at stage 21.  An out-of-order semantic replay may atomically
replace the active orchestrator state.  Topology updates queued behind that
replay must therefore resolve the active state when the queued operation
executes; they must not bind the predecessor state's method when submitted.
Otherwise the detached predecessor advances while the published active state
silently misses a topology delta.

The invariant is covered by a deterministic CPU regression test that replaces
the orchestrator while a topology update waits in the single-worker queue.  A
real stage 13--22 replay additionally covers AMtown01's non-strict cumulative
identity sequence: stages 18--21 retain a valid online state, and the final
boundary reaches `exact_final` with zero lag and no errors.

When publication is final-only, progress videos contain RGB, and semantic
recovery checkpoints are disabled, no requested artifact consumes progressive
semantic state.  In that case the semantic sidecar is intentionally initialized
only at the final boundary and performs one authoritative exact rebuild.  This
removes otherwise wasted per-stage projection/topology work and GPU contention.
Online semantics remains enabled when any consumer requires it: intermediate
semantic publication, semantic/side-by-side progress video, or semantic
recovery checkpoints.
