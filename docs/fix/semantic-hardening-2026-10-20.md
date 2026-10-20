# Semantic hardening, 2026-10-20

Base: `cc99f13` (`semantic/g1-evidence-v2`). Development branch:
`fix/semantic-first-principles-20261020`, in its own worktree. The main
checkout and existing experiments are retained unchanged.

## Repaired contracts

| Review ID | Failure and repair | Regression boundary |
|---|---|---|
| S01 | A failed output consumer could strand the trainer's Queue feeder. Check both workers at every response wait and throughout shutdown, bound the frame queue to two messages, abort all owned producers on failure, and use bounded joins. Startup failure also cleans already started workers. | Real subprocesses with multi-megabyte frames, consumer failure, normal drain, and an unstarted child. |
| S02 | Screen-space barycentric weights were applied directly to world positions. Use normalized `lambda_i / camera_depth_i`, with depth computed from each camera's world-to-camera pose. CPU and tensor paths agree; degenerate triangles and nonpositive/nonfinite vertex depths do not create samples. | Triangle-plane and reprojection constraints under a nontrivial rotation/translation, on NumPy, Torch CPU, and the installed CUDA stack. |
| S03 | Nonpublication stages behind a pending map retained every full snapshot. Give these projection snapshots a separate latest-wins slot and ordered markers. | A paused worker, one pending map, 200 later stages, and an intervening topology append; only the first stage, pending map, and latest cumulative camera inventory execute. |
| S04 | Open observations qualified and contaminated the group-disagreement diagnostic. Store closed-group variance, qualify with closed-group count, weight diagnostic support by closed evidence, and record `online_estimate` / `exact_final`. | One closed plus one open group is unavailable; two closed groups are available; a subsequent open group does not change their variance. Actual CUDA render checks diagnostic eligibility and source. |
| S05 | Preflight omitted generation/score identity and finalization could copy a changed disk schema. Bind canonical model/checkpoint/prompt/threshold metadata plus optional-score identity, reload disk ontology at final boundaries, revalidate authoritative inputs after both rebuild passes, and archive the validated schema snapshot. | Independent mutations of all four generation fields, scores, and cached ontology; formatting and presentation-only changes; stable fresh/resume preflight identity. |
| S06 | An invalid existing observation aborted an explicitly partial rebuild before later valid views. Validate observations independently before forming final groups, keep rejected IDs/errors, and rebuild all admitted views. Strict authoritative binding failure still prevents final acceptance; pass-to-pass identity checks remain mandatory. | Valid / malformed metadata / valid views produce `exact_partial` with the first and third views in the final ledger; changed labels between passes still reject. |
| S07 | NumPy and Torch medians selected different geometry for planar fallback. Share the exact detached Torch eligibility implementation with TDOM. | End-to-end plane membership for odd/even scale distributions and the disabled filter. |
| S08 | Default scene loading created a PLY in source SfM. Keep derived points in memory unless an explicit external output cache is supplied; fresh Scene exports its input PLY under the model output. | Real synthetic COLMAP binary files, source-tree hashes, and explicit output-cache behavior. |

Categorical label resampling remains nearest-neighbor. RGB, original camera
calibration, labels, masks, and the rectified/training-grid mapping retain the
resolution implementation's coordinate contract. No semantic feedback is
introduced into RGB optimization. S02 intentionally corrects upstream geometry
initialization, so new training is not expected to be bit-identical to the
previous sampler.

The async change bounds retained **full stage snapshots** to one active task,
one pending publication, and one pending projection snapshot (plus a transient
replacement allocation). Lightweight FIFO markers/topology events remain
ordered. It does not claim that every control-queue object has a fixed bound.

## Compatibility

Semantic state schema is now **5**; schema 4 stored variance with a different
meaning and must be explicitly rebuilt. Recovery context schema remains 2.
New authoritative preflight reports include `generation_sha256` and
`score_sha256`; old reports without these bindings cannot authorize a new exact
rebuild. These checks do not hash volatile execution options or JSON formatting.
Ontology and presentation remain separate; render palette overrides do not
change accumulated evidence. The archived schema is the validated specification
actually used for fusion, rather than a later copy of a mutable source file.

`--worker_shutdown_timeout` defaults to 300 seconds per worker and must be
positive and finite. A blocked or failed worker causes an explicit failed run
instead of an unbounded join. The frame queue limit bounds message count, not
individual frame size.

## Evidence

Reproduction, regression logs, cleanup backups, and acceptance artifacts:
`/data3/xzp/validation/semantic-hardening-20261020`.
The eight original minimized findings are retained in
`/data3/xzp/validation/semantic-deep-review-20261020`.

Initial complete regression after repair: **273 tests, 8 subtests passed**,
including installed CUDA integration on physical GPU 0. Test processes mount
original datasets, previous experiments, the main checkout, and the shared
Conda installation read-only; no extension rebuild or package installation is
performed. Physical GPU 1 is excluded by UUID selection.

Four obsolete worktrees were removed after checking active references.
The two dirty worktrees have binary patches, untracked-file backups, and
`refs/backup/worktree-cleanup-20261020/*` recovery references. Their branches
remain available. The main checkout and the new fix worktree are retained.

The first production smoke run on original XinCao (`--resolution 4`, stages
30–32, 50 optimizer iterations) completed initialization, incremental training,
exact final reconstruction and H.264 video publication. All **997** protected
file hashes matched. This is a functional smoke check, not the earlier full-run
quality comparison.

A real-SfM projection oracle additionally exposed cancellation in the old Gram
formula for very thin triangles, especially with float32 screen coordinates.
Both samplers now evaluate oriented 2D areas in float64 and return tensor
geometry in its original dtype. A minimal thin-triangle test failed on NumPy,
Torch CPU, and CUDA before this repair. On 392 sampled triangles from four
original XinCao camera poses, the corrected float32 world-point reprojection
has maximum error **0.000876 source pixels**, and float64 maximum error is
below `7e-10` pixels. The oracle independently projects co-visible SfM vertices;
it verifies the sampling constraint, not bundle-adjustment or semantic accuracy.

## Final acceptance

- Final complete suite: **274 tests and 11 subtests passed**, with the existing
  three dependency warnings. Command/log: `final-all-tests-command.json` and
  `final-all-tests.log` in the validation directory above.
- After the thin-triangle correction, a fresh original-data `--resolution 4`
  run (`verified-original-r4`) again completed stages 30–32 and 50 optimizer
  iterations. It published 21,137 Gaussians, an `exact_final` result over all
  32 cumulative cameras, a 512×512 paired map, and four H.264 frames at
  2366×1080. It completed in 241.20 seconds; protected hashes matched.
- A stage-31 checkpoint was captured into the isolated validation directory.
  `resumed-original-r4` restored it, trained stage 32, rebuilt final semantics,
  and published its final map and three-frame H.264 video. The preflight hash
  matched the fresh run, and protected hashes matched again.
- The resumed and continuous runs have the same Gaussian count, group counts,
  and **100% Gaussian hard-class agreement**. Floating geometry and evidence
  are not bit-identical: maximum absolute XYZ difference is 0.003788 world
  units; maximum raw/robust evidence difference is 0.198377. No bit-identical
  recovery trajectory or attribution of those differences is claimed.
- The production standalone `tools/render.py` rendered all **24 cameras in
  the actual local stage-32 COLMAP snapshot** against read-only originals.
  This inventory differs from the progressive run's 32 cumulative cameras.
  Source hashes matched and no source PLY was created.
- Artifact verification reloads schema-5 semantic state through its public
  loader, validates its geometry/context hashes, checks the archived schema
  against final input binding, verifies the final map pointer and categorical
  IDs, and compares video frame counts/dimensions with `ffprobe`. Both fresh
  and resumed artifact sets passed. See `artifact-verification.json`.
- A final partial-mode regression also checks that an admitted later view is
  removed from the deferred inventory after successful exact partial rebuild.

Canonical experiment bundle:
`/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-20261020`.
These short runs establish lifecycle, coordinate, binding and publication
behavior; they do not replace a complete 229-camera quality experiment.
