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
