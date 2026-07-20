# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.415×**
- Large forward+backward vs Phase 3A1: **1.118×**
- Large forward vs PortableMLP: **1.658×**
- Large forward+backward vs PortableMLP: **1.933×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.025× | 0.978× | 0.746× |
| medium | 1.041× | 0.957× | 0.870× |
| target_257_w64 | 1.108× | 0.978× | 0.698× |
| large_1024_w64 | 1.482× | 1.134× | 1.498× |
| large_1024_w128 | 1.666× | 1.208× | 2.357× |
| large_4096_w64 | 1.148× | 1.020× | 1.291× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
