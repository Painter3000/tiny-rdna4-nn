# Phase 3A1 performance

- Result: **PASS**
- Warmups: 20
- Iterations: 100

| Case | Forward Portable | Forward HipBLASLt | Speedup | F+B Portable | F+B HipBLASLt | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.0793 ms | 0.1037 ms | 0.765× | 0.3182 ms | 0.3783 ms | 0.841× |
| medium | 0.0798 ms | 0.1153 ms | 0.692× | 0.4625 ms | 0.4005 ms | 1.155× |
| target_257_w64 | 0.0792 ms | 0.1237 ms | 0.640× | 1.5856 ms | 0.8927 ms | 1.776× |
| large_1024_w64 | 0.1419 ms | 0.1423 ms | 0.997× | 3.0612 ms | 1.6850 ms | 1.817× |
| large_1024_w128 | 0.3190 ms | 0.1646 ms | 1.938× | 2.6986 ms | 1.4233 ms | 1.896× |
| large_4096_w64 | 0.1266 ms | 0.1100 ms | 1.152× | 4.5738 ms | 2.3761 ms | 1.925× |

## Gates

- `no_large_regression_over_15_percent`: PASS
- `one_large_forward_at_least_1_25x`: PASS
- `large_forward_geomean_over_1`: PASS
- `large_forward_backward_geomean_at_least_0_95`: PASS
