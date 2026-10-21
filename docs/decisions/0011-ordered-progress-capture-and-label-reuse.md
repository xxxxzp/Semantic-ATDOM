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
