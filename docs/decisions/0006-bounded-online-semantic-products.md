---
status: accepted
date: 2026-09-11
---

# Treat online semantic maps as bounded approximations

Progressive geometry, opacity and camera estimates change after earlier labels have been projected. Reprojecting every historical observation at every stage would defeat the quasi-real-time objective, while refreshing only an active window cannot make all accumulated evidence exact.

Online products are therefore explicitly approximate and use latest-wins publication. A published RGB/Semantic pair must share one frozen projection-geometry snapshot, but intermediate geometry stages need not publish a pair. Lag, deferred observations and historical-evidence staleness determine whether a preview is `approx_online` or `stale_online`.

Authoritative semantics are rebuilt independently from all valid labels on frozen final geometry. Complete inputs produce `exact_final`; explicitly reported missing or invalid inputs produce `exact_partial`. An online state retained after final-rebuild failure is `online_fallback` and is diagnostic only. Semantic failure never blocks or alters RGB geometry.
