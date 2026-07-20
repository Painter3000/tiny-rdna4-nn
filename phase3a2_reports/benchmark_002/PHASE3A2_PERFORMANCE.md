# Phase 3A2 performance

- Result: **FAIL**
- Large forward vs Phase 3A1: **1.518×**
- Large forward+backward vs Phase 3A1: **1.012×**
- Large forward vs PortableMLP: **1.867×**
- Large forward+backward vs PortableMLP: **1.788×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.048× | 0.916× | 0.680× |
| medium | 1.050× | 0.941× | 0.895× |
| target_257_w64 | 1.060× | 1.001× | 0.649× |
| large_1024_w64 | 1.538× | 1.007× | 1.518× |
| large_1024_w128 | 1.785× | 1.075× | 2.964× |
| large_4096_w64 | 1.273× | 0.958× | 1.446× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: FAIL
