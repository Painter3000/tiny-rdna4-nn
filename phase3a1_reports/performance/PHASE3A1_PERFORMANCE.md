# Phase 3A1 performance

- Result: **PASS**
- Unfused FP32 backend; 20 warmups, 100 HIP-event iterations.

| Case | Forward speedup | Forward+Backward speedup |
|---|---:|---:|
| small | 0.713× | 0.845× |
| medium | 0.712× | 1.195× |
| target_257_w64 | 0.623× | 1.769× |
| large_1024_w64 | 1.000× | 1.813× |
| large_1024_w128 | 1.932× | 1.839× |
| large_4096_w64 | 1.244× | 1.916× |

## Gates

- `no_large_regression_over_15_percent`: PASS
- `one_large_forward_at_least_1_25x`: PASS
- `large_forward_geomean_over_1`: PASS
- `large_forward_backward_geomean_at_least_0_95`: PASS
