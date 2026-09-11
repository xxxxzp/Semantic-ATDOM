---
status: accepted
amended_by: 0006-bounded-online-semantic-products
---

# Synchronize semantics at progressive stage boundaries

Semantic labels are prepared offline, but semantic state is updated throughout progressive A-TDOM reconstruction at initialization, progressive stage boundaries and final refinement. Labels are not re-counted at every RGB iteration; this prevents repeated optimizer sampling from creating false evidence. ADR 0006 clarifies that online products remain bounded approximations and that a latest-wins publisher need not emit a map pair for every geometry stage.
