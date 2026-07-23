# Phase 3B1-F1 – Reproducible FP16 Performance

Decision: `PHASE3B1F_PERFORMANCE_GAIN_CONFIRMED`

## Measurement identity

- Basis commit: `04a238b09c275acffc55996ef722a22f9f62c530`
- Run ID: `20260724T001651_182604`
- Active manifest SHA256: `98bd1a1c2d447bfeaba419b9f8cd705320ff0a92f31ef1dca71adf4a3d272708`
- Matrix: 24 cases, 3 fresh processes per case, 72 primary processes, 5 paired rounds
- Result: 24/24 accepted cases, 72/72 valid primary processes, 0 invalid processes, 0 replacements
- Correctness pre/post: PASS/PASS

The run used the documented F0b command and the F0b2 manifest reader. The
failed historical run `20260724T000728_179565` was neither resumed nor
evaluated.

## Result summary

Geometric means below summarize the per-case medians emitted by the checked-in
F0b finalizer. Time and allocated-memory peaks have equal status in the
qualification.

| Category | Cases | Speedup geometric mean | FP32/FP16 peak-memory geometric mean | Maximum FP16 peak | Maximum FP32 peak |
|---|---:|---:|---:|---:|---:|
| All | 24 | 2.2800x | 1.0418x | 150,216,192 B | 153,362,944 B |
| Latency, batch 1024 | 12 | 0.9968x | 1.0368x | 145,915,392 B | 146,113,024 B |
| Throughput, batch 16384 | 12 | 5.2155x | 1.0468x | 150,216,192 B | 153,362,944 B |
| Network-only | 12 | 2.1658x | 1.0310x | 31,516,160 B | 31,903,744 B |
| NetworkWithInputEncoding | 12 | 2.4003x | 1.0527x | 150,216,192 B | 153,362,944 B |

Batch 1024 is classified as a
`latency-bound / launch-overhead-dominated regime`. Batch 16384 is the
throughput regime. The latency result is essentially neutral in aggregate;
the throughput cases show a clear reproducible aggregate gain. Individual
per-case medians, MADs, paired bootstrap 95% intervals, and memory peaks are
recorded in `phase3b1f1_reproducible_performance.json`.

## Raw evidence

- Run directory:
  `/tmp/phase3b1f1_reproducible_runs/20260724T001651_182604`
- Raw index: 20,100 bytes,
  SHA256 `043a69660d4c4cbff37426380f5357c147c905b7ca99bec297cab9c32ed3c2ea`
- Correctness pre: 2,893 bytes,
  SHA256 `b10ce37e897d6f34a15485ae8e2cd34c1eeb33d647b41e251c162ddd1b48785a`
- Correctness post: 2,893 bytes,
  SHA256 `43c9f7d6e4526b0e18d0fd53d6af6834a107b2e6729ed50a01cb7c8e4ce54ed1`
- Terminal log:
  `/tmp/phase3b1f1_logs/phase3b1f1_full_20260724T001651.log`,
  37 bytes,
  SHA256 `a9dbb3cd889d3932538daeb25a9b8485f2455f7cc1512ed72d4481254a664032`

All large per-process JSON and hipBLASLt logs remain in the run directory.
No production files, manifests, thresholds, or measurement scripts were
changed during F1.
