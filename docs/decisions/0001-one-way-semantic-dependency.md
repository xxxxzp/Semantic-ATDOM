---
status: accepted
---

# Keep semantic inference downstream of geometry

Semantic labels may read the current Gaussian geometry, visibility and appearance, but semantic losses or evidence must never update geometry, opacity, SH, densification or RGB optimization. The input labels are noisy SAM predictions, so allowing them to shape reconstruction would couple uncertain semantics to the authoritative RGB geometry and make failures difficult to isolate.
