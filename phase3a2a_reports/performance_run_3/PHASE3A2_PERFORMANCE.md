# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.455×**
- Large forward+backward vs Phase 3A1: **1.128×**
- Large forward vs PortableMLP: **1.694×**
- Large forward+backward vs PortableMLP: **1.946×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.010× | 1.022× | 0.675× |
| medium | 1.019× | 1.042× | 0.733× |
| target_257_w64 | 1.052× | 0.976× | 0.673× |
| large_1024_w64 | 1.487× | 1.160× | 1.504× |
| large_1024_w128 | 1.719× | 1.206× | 2.408× |
| large_4096_w64 | 1.205× | 1.025× | 1.342× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
