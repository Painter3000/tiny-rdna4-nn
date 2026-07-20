# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.448×**
- Large forward+backward vs Phase 3A1: **1.123×**
- Large forward vs PortableMLP: **1.682×**
- Large forward+backward vs PortableMLP: **1.937×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.008× | 1.024× | 0.641× |
| medium | 1.019× | 1.040× | 0.731× |
| target_257_w64 | 1.025× | 0.979× | 0.651× |
| large_1024_w64 | 1.473× | 1.144× | 1.487× |
| large_1024_w128 | 1.714× | 1.211× | 2.404× |
| large_4096_w64 | 1.201× | 1.022× | 1.330× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
