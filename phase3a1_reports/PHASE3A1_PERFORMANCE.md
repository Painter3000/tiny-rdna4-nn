# Phase 3A1 performance gate

Result: **PASS**. Measurements use 20 warmups followed by 100 HIP-event iterations on AMD Radeon Graphics, gfx1201, ROCm 7.2.53211, PyTorch 2.13.0+rocm7.2. Values compare HipBLASLtMLP with PortableMLP using identical FP32 parameters and inputs.

| Case | Forward | Forward + backward |
|---|---:|---:|
| small | 0.755x | 0.889x |
| medium | 0.703x | 1.207x |
| target 257 / width 64 | 0.619x | 1.764x |
| large 1024 / width 64 | 1.001x | 1.808x |
| large 1024 / width 128 | 1.948x | 1.927x |
| large 4096 / width 64 | 1.251x | 1.930x |

Large-case forward geometric mean is 1.346x; large-case forward+backward geometric mean is 1.888x. All four prescribed gates pass, including the targeted no-regression bound and the 1.25x large-forward threshold.

The process-long plan cache grew from 0 to 51 distinct plans during the suite and recorded 11,012 hits. Cold-forward cost, median/P95 timings, per-case cache snapshots, and allocated/reserved memory are retained in `performance_final/phase3a1_benchmark.json`.
