# Remove foreground video waiting and reuse prepared labels

Base: `9a96882`, branch `perf/semantic-pipeline-20261021`, isolated worktree.
The geometry sampler and optimizer are unchanged in this repair.

The foreground video capture path previously called `wait_for_idle()` after
each requested stage. It now queues a mandatory frozen full-model capture with
that stage's semantic synchronization. The single semantic worker renders the
two map panels before consuming any later topology event. Two retained capture
slots bound memory; exhausted capacity produces measured foreground backpressure.
Final refinement drains previous captures, rebuilds exact semantics and renders
the final frame. See [ADR 0011](../decisions/0011-ordered-progress-capture-and-label-reuse.md).

Geometry-independent label preparation now has a 128 MiB CPU LRU and a 512 MiB
compressed disk cache under the experiment output. Identity includes the label
array content, source calibration, camera mask, target grid and boundary settings.
No Gaussian projection results are cached. Label loading, provenance validation,
categorical nearest sampling and both final rebuild passes remain in place.

Acceptance artifacts are retained under
`/data3/xzp/experiments/Onthefly/XinCao/semantic-pipeline-20261021`:

- `preparation-before.log` reproduces duplicate boundary preparation on repeated
  projection; the repaired test requires one preparation and two rasterizations.
- `capture-before.log` runs the new foreground-wait regression against the actual
  previous capture function and fails at its `wait_for_idle()` call.
- `final-all-tests.log` and `remaining-root-tests.log`: 285 tests and 68 subtests
  passed, including installed CUDA integration, cache invalidation/corruption/
  bounds, ordered captures across topology changes and error propagation.
- `validation/prepared-raster-verification.json`: all 229 original XinCao
  training-grid ID/weight arrays match the archived uncached reference hashes
  both on cache creation and after disk reload.
- `validation/smoke-verification.json`: stages 30–34, 2560 optimizer iterations,
  final exact semantics, all product hashes and six decoded video frames pass.

Two 80-iteration pipeline smokes and an identical pre-optimization baseline
smoke failed in final map publication with CUDA illegal access. This reproduces
on the accepted baseline too, and is retained as an unresolved pre-existing
short-training/rendering finding. A normal initialization budget passed on the
same data/grid. No extension or shared environment change was made.

Full-run throughput, quality, capture latency and backpressure require the
matched 229-camera experiment, rather than inference from zero foreground wait.

## Complete-run rejection and correction

The first complete pipeline run is retained at
`/data3/xzp/experiments/Onthefly/XinCao/semantic-pipeline-exclusive-20261021`.
It passed all input, state, map, camera, RGB screening and 120-frame video
checks, but failed the performance screen: 5.034 seconds/new image versus
4.543 in the accepted reference. Final semantics improved from 173.096 to
84.492 seconds. Startup capture capacity waiting was 52.916 seconds; next-image
feedback waiting increased from 0.313 to 1.163 seconds/new image. This version
was not accepted as the completed optimization.

The correction shares the trainer's existing FOV/calibration replacement
predicate with the sparse metadata reader. The reader omits full source-grid
mask construction/rectification only for retained cameras whose existing RGB
and mask the trainer will keep. New/changed cameras and legacy callers retain
their previous behavior. On real stage-180 input, 27 old cameras took 3.413
seconds before and 0.002 seconds with the metadata-only path; poses and
calibrations are identical. The initialization frame is also drained before
opening the incremental stream, preserving the reference startup boundary.

The sampling formula and optimizer are unchanged. Earlier input preparation
can change which eligible training iteration supplies the existing adaptive
`Render_New` preview, so final model identity is not guaranteed; matched full
quality checks remain required. An independent OpenCV thread-budget probe did
not show meaningful standalone benefit, and no speculative parent-thread
policy change was added.

## Final complete acceptance

The corrected `7a92357` run completed 229 cameras and 45,800 optimizer steps
with exact final semantics and all 120 audited video frames. Average online
processing improved from 4.543 to 3.612 seconds/new image; geometry was
3.244 versus 3.255 seconds. Final semantics took 80.949 versus 173.096 seconds,
and manifest wall time 1132.448 versus 1445.210 seconds. Capacity backpressure
totaled 0.001199 seconds, with at most one retained capture observed.

The run omitted 7,745 discarded old-camera mask preparations. Final camera
poses, calibration and retained masks exactly match the reference inventory.
All 229 cached training ID/weight arrays match the uncached reference hashes.
288 tests and 68 subtests passed. Training-view RGB mean PSNR/SSIM/LPIPS
changed by +0.04020 dB / +0.001445 / −0.001770; semantic agreement on jointly
known map pixels is 97.024%, a consistency measure without independent truth.
The 3,406 protected file hashes match, and GPU 4 peer samples found no external
compute task. Reference training used physical GPU 2, so device/shared-resource
differences remain a limit on the single-run speedup estimate.

See the [complete report](../exp/semantic-pipeline-full-2026-10-21.md) and
`/data3/xzp/experiments/Onthefly/XinCao/semantic-pipeline-final-20261021`.
The first pipeline trial failed its performance screen and is retained as
evidence; it was not substituted for the final accepted run.
