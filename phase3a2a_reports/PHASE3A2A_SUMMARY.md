# Phase 3A2a summary

Result: **PASS** for the validated gfx1201 / ROCm 7.2 scope.

Phase 3A2a separates hipBLASLt planning from execution and guarantees that distinct HIP streams receive distinct execution handles. The Planning handle is never passed to `hipblasLtMatmul()`. Execution handles are reused by exact stream identity and are capped at 64 per device. Capacity overflow raises an explicit error before launch and never reuses another stream's handle.

## Evidence

- Static production-source audit: PASS.
- Eight unique streams: exactly eight handle creations; second pass creates none; reuse counter increases.
- 64 rounds across eight streams without intermediate global synchronization: PASS.
- Three-stream event chain: PASS.
- 2000 warm-cache calls: handle count 19→19, creations 19→19, descriptors 5→5, heuristic misses 6→6, allocated growth 0, reserved growth 0.
- 20 fresh processes, including four with `HIP_LAUNCH_BLOCKING=1`: PASS.
- Default stream plus both priority pools: the 65th distinct stream is rejected; count and creations remain 64 and overflow count becomes one.
- Full official Phase-3A2 correctness, training, resume, streams, memory, fresh-process and performance gates: PASS.
- Four accepted performance comparisons against the final Phase-3A2 build: PASS.

The supplied gate bundle contained two harness defects which were corrected without weakening a gate: the static audit pointed at a nonexistent binding filename, and the memory test warmed eight live outputs before measuring batches of 128. The corrected memory test warms the same 128-output depth and retains the original 1 MiB threshold. Bundle SHA-256 entries were updated and verified.

Phase 3A2a proves stream-handle safety only for the validated gfx1201 / ROCm 7.2 scope. No ABI compatibility or general GPU compatibility is claimed.

