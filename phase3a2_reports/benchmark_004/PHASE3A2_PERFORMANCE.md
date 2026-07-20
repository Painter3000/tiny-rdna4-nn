# Phase 3A2 performance

- Result: **PASS**
- Large forward vs Phase 3A1: **1.491×**
- Large forward+backward vs Phase 3A1: **1.111×**
- Large forward vs PortableMLP: **1.739×**
- Large forward+backward vs PortableMLP: **1.941×**

| Case | Forward vs 3A1 | F+B vs 3A1 | Forward vs Portable |
|---|---:|---:|---:|
| small | 1.029× | 1.009× | 0.677× |
| medium | 1.045× | 1.055× | 0.739× |
| target_257_w64 | 1.056× | 1.023× | 0.647× |
| large_1024_w64 | 1.507× | 1.114× | 1.496× |
| large_1024_w128 | 1.757× | 1.204× | 2.535× |
| large_4096_w64 | 1.252× | 1.023× | 1.387× |

## Gates

- `each_large_forward_at_least_0_95x_phase3a1`: PASS
- `large_forward_geomean_at_least_1_08x_phase3a1`: PASS
- `large_forward_backward_geomean_at_least_0_98x_phase3a1`: PASS
- `one_large_forward_at_least_1_10x_phase3a1`: PASS
- `large_forward_vs_portable_not_below_phase3a1_gate`: PASS
- `large_forward_backward_vs_portable_not_below_phase3a1_gate`: PASS
