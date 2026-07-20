# Phase 3A2 performance

- Result: **FAIL**
- Large forward vs Phase 3A1: **1.474×**
- Large forward+backward vs Phase 3A1: **0.983×**
- Large forward vs PortableMLP: **1.879×**
- Large forward+backward vs PortableMLP: **1.749×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.027× | 0.999× | 0.706× |
| medium | 1.045× | 1.035× | 0.743× |
| target_257_w64 | 1.046× | 1.001× | 0.641× |
| large_1024_w64 | 1.488× | 1.007× | 1.470× |
| large_1024_w128 | 1.749× | 0.986× | 3.153× |
| large_4096_w64 | 1.231× | 0.958× | 1.431× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: FAIL
