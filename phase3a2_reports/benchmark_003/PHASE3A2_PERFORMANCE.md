# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.482×**
- Large forward+backward vs Phase 3A1: **1.129×**
- Large forward vs PortableMLP: **1.707×**
- Large forward+backward vs PortableMLP: **1.985×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.014× | 1.004× | 0.785× |
| medium | 1.035× | 1.048× | 0.733× |
| target_257_w64 | 1.047× | 1.021× | 0.642× |
| large_1024_w64 | 1.517× | 1.161× | 1.503× |
| large_1024_w128 | 1.758× | 1.206× | 2.463× |
| large_4096_w64 | 1.220× | 1.027× | 1.345× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
