---
status: accepted
---

# Synchronize semantics at progressive stage boundaries

Semantic labels are prepared offline, but semantic state is updated throughout progressive A-TDOM reconstruction at the end of initialization, every progressive training batch, and final refinement. Labels are not re-counted at every RGB iteration; this keeps Semantic TDOM current while preventing repeated optimizer sampling from creating false probabilistic confidence.
