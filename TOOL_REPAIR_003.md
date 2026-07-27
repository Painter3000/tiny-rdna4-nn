# Phase 4A3-Q0b Tool Repair 003

Marker: `TCNN_RDNA4_P4A3_Q0B_TOOL_REPAIR_003`

## Failure

Revision 2 built and linked successfully, then failed in the fail-fast hook
smoke before the worker matrix:

```text
object.h:186 check failed: input.n() % BATCH_SIZE_GRANULARITY == 0
```

## Root cause

The direct native adapter padded to `16`, confusing the rocWMMA kernel's
internal tile multiple with tiny-cuda-nn's public C++ API batch granularity.
The frozen release defines `BATCH_SIZE_GRANULARITY = 256`; the public Python
path queries the same value through `_C.batch_size_granularity()`.

## Repair boundary

Only test-harness input adaptation, smoke coverage, and diagnostics changed.
The production kernel, production dispatch, numerical contract, measurement
thresholds, matrix, aggregation, confidence interval, outlier policy, and
decision rules are unchanged. No Revision-2 performance worker executed.
