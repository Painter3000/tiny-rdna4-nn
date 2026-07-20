# Phase 3A1 performance

- Result: **PASS**
- Unfused FP32 backend; 20 warmups, 100 HIP-event iterations.

| Case | Forward speedup | Forward+Backward speedup |
|---|---:|---:|
| small | 0.755× | 0.889× |
| medium | 0.703× | 1.207× |
| target_257_w64 | 0.619× | 1.764× |
| large_1024_w64 | 1.001× | 1.808× |
| large_1024_w128 | 1.948× | 1.927× |
| large_4096_w64 | 1.251× | 1.930× |

## Gates

- `no_large_regression_over_15_percent`: PASS
- `one_large_forward_at_least_1_25x`: PASS
- `large_forward_geomean_over_1`: PASS
- `large_forward_backward_geomean_at_least_0_95`: PASS
