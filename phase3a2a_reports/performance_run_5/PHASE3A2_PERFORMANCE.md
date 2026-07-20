# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.460×**
- Large forward+backward vs Phase 3A1: **1.104×**
- Large forward vs PortableMLP: **1.708×**
- Large forward+backward vs PortableMLP: **1.935×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.008× | 0.968× | 0.791× |
| medium | 1.036× | 0.946× | 0.739× |
| target_257_w64 | 1.046× | 1.023× | 0.851× |
| large_1024_w64 | 1.484× | 1.103× | 1.470× |
| large_1024_w128 | 1.719× | 1.190× | 2.493× |
| large_4096_w64 | 1.221× | 1.024× | 1.359× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
