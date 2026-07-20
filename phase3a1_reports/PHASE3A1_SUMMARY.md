# Phase 3A1 summary

Result: **PASS** on gfx1201 / ROCm 7.2.

Phase 3A1 adds the explicit FP32 `HipBLASLtMLP` backend. Forward matrix products, input gradients, and weight gradients use hipBLASLt; bias, activation, activation derivative, and bias-gradient operations use dedicated HIP kernels. There is no PortableMLP fallback. PortableMLP selection and behavior remain unchanged.

The flat parameter layout is identical to PortableMLP (`W,b` per layer), so state dictionaries and optimizer state are interchangeable at source/API level. No ABI-compatibility claim is made.

## Gates

- Phase 3A1 cases A-H, explicit streams D/G/H, GradientMode Overwrite/Accumulate/Ignore, Identity/Frequency/OneBlob/HashGrid composition: PASS.
- Default, same explicit, rotating explicit, and three event-ordered streams: PASS.
- 200 repeated mixed-batch forward/backward operations: PASS; allocated growth 1 KiB, reserved growth 0.
- 500-step training stability and PortableMLP state interchange: PASS.
- 100-step checkpoint plus fresh-process 100-step resume: bit-exact with uninterrupted 200 steps.
- 20 fresh Python processes, including four with `HIP_LAUNCH_BLOCKING=1`: PASS.
- Performance gates: PASS; large forward geometric mean 1.346x and large forward+backward geometric mean 1.888x.
- Phase 2E, Phase 2E addendum, Phase 2D, Phase 2C/Phase 1 encoding, and legacy C++ source-compatibility regressions: PASS.

The implementation is intentionally limited to gfx1201, FP32, widths 16/32/64/128, 1-4 hidden layers, and the contracted activation set. Unsupported configurations are rejected explicitly.
