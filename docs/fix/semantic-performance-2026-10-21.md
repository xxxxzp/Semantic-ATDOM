# Semantic boundary-weight performance

The complete correctness-repair experiment increased incremental video semantic
waiting from 2.368 to 4.342 seconds per new image, while geometry training stayed
near 3.31 seconds. These averages include all 199 incremental stages, including
stages without a requested video frame. The wait is measured around
`semantic_hook.wait_for_idle()` before training-progress map capture; the final
all-camera semantic rebuild is timed separately. Historical timing establishes
the regression, but controlled replay did not identify one repair as its sole
cause.

CPU profiling identified source-grid `label_boundary_weights` as the dominant
cost in a controlled projection replay. Commit `d6df0c1` accelerates that
public function using OpenCV precise L2 distance, recovers exact integer
squared distances in the bounded reliability band, and retains float64 ramp
arithmetic before the original float32 result. Bandwidths above 128, grids
larger than 32768 on either axis, and special parameters use the previous
SciPy path. Class IDs and center-based nearest sampling remain unchanged.
The trainer sets OpenCV's process thread budget to Torch's budget before
starting its semantic executor; it does not change the global setting during
concurrent projection calls. Earlier correctness fixes remain enabled.

The real 4032×3024-label public-function budget failed at a 2.534-second median
before the change and passed at 0.468 seconds afterward. Independent oracle
tests cover diagonal distances, unknown regions, wide IDs, thin arrays,
noncontiguous/read-only inputs, parameter limits, and the fallback. All 229
original-XinCao source-grid and training-grid weight arrays were bit-identical
between old and new implementations. The complete regression suite passed
277 tests and 68 subtests without rebuilding native extensions or installing
packages.

The [complete experiment report](../exp/semantic-performance-full-2026-10-21.md)
records the full 229-camera, 45,800-iteration comparison, final semantic state,
maps, video, training-view quality and protected-input hashes. The experiment
bundle is `/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021`.
This work does not repair the separately documented map column-axis metadata
mismatch.
