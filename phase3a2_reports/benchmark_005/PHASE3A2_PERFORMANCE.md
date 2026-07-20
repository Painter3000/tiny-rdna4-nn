# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.475×**
- Large forward+backward vs Phase 3A1: **1.097×**
- Large forward vs PortableMLP: **1.732×**
- Large forward+backward vs PortableMLP: **1.928×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.029× | 0.947× | 0.731× |
| medium | 1.048× | 0.972× | 0.745× |
| target_257_w64 | 1.058× | 1.021× | 0.646× |
| large_1024_w64 | 1.510× | 1.073× | 1.492× |
| large_1024_w128 | 1.742× | 1.202× | 2.583× |
| large_4096_w64 | 1.221× | 1.023× | 1.348× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
