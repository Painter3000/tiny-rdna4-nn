# Phase 3A2 summary

Result: **PASS** on gfx1201 / ROCm 7.2.

Phase 3A2 adds zero-workspace `BIAS` and `RELU_BIAS` hipBLASLt epilogues to `HipBLASLtMLP`. The global plan cache contains only immutable static signatures and algorithms. Stable bias pointers are bound in bounded, immutable per-instance descriptors, so the same model can launch concurrently on multiple streams without descriptor mutation or a GEMM execution mutex.

`RELU_AUX_BIAS` was deliberately not integrated: training contexts allocate AUX dynamically, and binding those pointers would violate the descriptor safety contract. Native RowMajor output mapping also remains on hipBLASLt GEMM plus the Phase-3A1 postkernel because D's physical row dimension is the batch axis. Neither path falls back to PortableMLP.

## Validation

- Phase 3A2 A-H, explicit streams D/G/H, same model on three streams, 200 mixed batches, and 20 fresh processes: PASS.
- Embedded complete Phase 3A1 C1-C8, GradientModes, four encodings, 500-step training, and bit-exact fresh-process resume: PASS.
- Phase 2E and Phase 2D against the final clean-build candidate: PASS.
- Legacy nine-argument C++ source-compatibility compile smoke: PASS. No ABI claim is made.
- Clean wheel build with `MAX_JOBS=1`, `pip check`, and neutral site-packages import: PASS.
- Memory after warmup: allocated growth 512 bytes; reserved growth 0 bytes.
- No Phase-3A2 DEBUG or ISOLATION markers remain.

The compact workspace no longer contains the external Phase-2E-addendum and Phase-2C harness scripts used during Phase 3A1. Their covered behaviors are exercised by the embedded Phase-3A1 regression (checkpoint/resume, streams, memory, encodings, and encoded networks); the missing external filenames were not represented as newly executed harnesses.

## Performance

Four consecutive post-integration HIP-event runs passed all starter gates. The final clean-build candidate measured 1.415x large-forward and 1.118x large-forward+backward versus Phase 3A1. The preceding three passing runs measured forward 1.482x/1.491x/1.475x and forward+backward 1.129x/1.111x/1.097x.
