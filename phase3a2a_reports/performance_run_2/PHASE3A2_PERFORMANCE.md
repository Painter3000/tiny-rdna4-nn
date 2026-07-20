# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.461×**
- Large forward+backward vs Phase 3A1: **1.123×**
- Large forward vs PortableMLP: **1.704×**
- Large forward+backward vs PortableMLP: **1.942×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.003× | 0.940× | 0.666× |
| medium | 1.024× | 0.975× | 0.853× |
| target_257_w64 | 1.047× | 0.978× | 0.661× |
| large_1024_w64 | 1.488× | 1.147× | 1.505× |
| large_1024_w128 | 1.755× | 1.210× | 2.464× |
| large_4096_w64 | 1.195× | 1.021× | 1.334× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
