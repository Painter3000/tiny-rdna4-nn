# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.485×**
- Large forward+backward vs Phase 3A1: **1.134×**
- Large forward vs PortableMLP: **1.729×**
- Large forward+backward vs PortableMLP: **1.955×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.024× | 0.986× | 0.652× |
| medium | 1.036× | 1.004× | 0.738× |
| target_257_w64 | 1.055× | 0.977× | 0.810× |
| large_1024_w64 | 1.514× | 1.174× | 1.532× |
| large_1024_w128 | 1.763× | 1.208× | 2.466× |
| large_4096_w64 | 1.228× | 1.029× | 1.368× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
