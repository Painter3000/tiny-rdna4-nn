# Phase 3A3 capability blocked

Target: gfx1201 / ROCm 7.2

Phase 3A3 cannot proceed to production integration because no semantically
correct zero-workspace `DRELU_BGRAD` implementation is available.

ROCm 7.2 does not declare `DRELU` or `DRELU_BGRAD`. Probing their numeric
candidates produced misleading successful API statuses but identity-GEMM
validation proved that no derivative mask was applied and no bias-gradient
output was written. Both tested bias sentinels remained unchanged.

Per the Phase 3A3 instructions:

- no production dispatch was changed;
- no tests or thresholds were weakened;
- no integration commit or PASS tag was created.

See `BACKWARD_EPILOGUE_CAPABILITY.md` and the JSON evidence in this directory.
