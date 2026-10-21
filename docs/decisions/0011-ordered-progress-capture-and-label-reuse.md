---
status: accepted
date: 2026-10-21
amends: 0009 training-progress execution and 0006 resource bounds
---

# Queue frozen video boundaries and reuse prepared label rasters

Stage video capture previously waited for the semantic worker on the geometry
thread. The complete XinCao run at `d6df0c1` measured 0.674 seconds of this
waiting per new image, averaged over 199 stages, with 118 video requests.
Source-resolution label preparation was also repeated for unchanged inputs
when geometry changed and during the two-pass final rebuild.

Requested initialization/progressive video frames now execute on the same
single worker as semantic synchronization. They are ordered, mandatory jobs,
not replaceable latest-wins previews. Each job owns frozen full RGB geometry,
projection camera, background, map window/grid, stage/iteration and trajectory.
It renders immediately after that job's semantic synchronization and before
any later stage or topology event. It does not read live trainer geometry or
copy the complete semantic state. Both map panels therefore retain ADR 0009's
same-stage contract, even while the trainer advances to another stage. The
worker no longer needs to be globally idle before rendering its own ordered
capture. Final refinement drains earlier jobs, performs the authoritative
rebuild, and captures the final frame synchronously.

Initialization is a distinct readiness boundary: its first video job is drained
before opening the incremental stream. This preserves the prior startup
contract and avoids charging the initial camera batch to an arbitrary later
stage's capacity wait. Both startup cost and full wall time remain measured.

At most two complete video jobs may be retained, including the active job.
Admission waits when capacity is exhausted; requested frames are never dropped.
This backpressure is measured separately from semantic wait. Existing map
publication and projection snapshot bounds remain in force. Capture errors
are propagated through the existing semantic worker failure mechanism, and an
explicitly requested video must not silently mix stages or publish missing
frames. Independent RGB optimization remains downstream of no semantic loss.

Prepared labels use a 128 MiB CPU LRU and at most 512 MiB of compressed derived
files in the run's semantic working directory, plus one transient atomic-write
file. These limits exclude the current preparation's working arrays. Each key includes label array
content/dtype/shape, original rectification calibration, training-grid dimensions,
camera mask content, ignore ID, boundary parameters, pixel-grid convention and
cache-format version. Disk arrays have shape/dtype/content checksums. Invalid
entries are recomputed; write failure is a cache miss, not permission to consume
stale data. A full label load and schema/provenance validation still precedes
projection, and final input binding and the two geometry-dependent rebuild
passes remain mandatory. No GPU projection results are cached. Source labels
and categorical nearest sampling are unchanged.

Incremental timings record foreground submission and admission backpressure.
The auditable video timeline records background delay, RGB/semantic rendering
and result-ready latency for each requested boundary. End-to-end training time,
final semantic reconstruction and full run wall time must be compared as well;
zero foreground semantic wait alone does not establish a throughput improvement.

The first complete implementation measured 5.034 seconds per new image versus
4.543 in the accepted reference, despite faster final semantics. It failed the
performance screen and was not accepted as the final optimization. Its explicit
capacity wait was 52.916 seconds, mostly at startup; foreground next-image
feedback waiting also increased from 0.313 to 1.163 seconds per new image.

The progressive input reader was rebuilding and rectifying full source-grid
masks for old cameras, then discarding them because the trainer retained its
existing RGB/mask whenever calibration/FOV matched. The reader now receives the
retained projection mappings and shares that exact replacement predicate with
the trainer. Matching old-camera records contain pose metadata only. New views,
changed calibrations/FOV and legacy callers without retained metadata still
prepare masks normally. No new sparse sample, optimizer or geometry rule is
introduced. A real 27-camera stage-180 replay measured 3.413 seconds of this
discarded work versus 0.002 seconds for metadata only, with identical poses and
calibrations. Full acceptance still requires the complete matched experiment.
