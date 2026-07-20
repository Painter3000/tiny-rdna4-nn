# Phase 3A2a performance comparison

Four consecutive accepted HIP-event runs (runs 2-5) were compared with the final official Phase-3A2 build. No performance improvement is claimed or required.

| Run | Large forward geomean | Large F+B geomean | Minimum large forward | Result |
|---:|---:|---:|---:|---|
| 2 | 0.9846x | 1.0069x | 0.9737x | PASS |
| 3 | 0.9803x | 1.0110x | 0.9721x | PASS |
| 4 | 1.0009x | 1.0167x | 0.9970x | PASS |
| 5 | 0.9838x | 0.9895x | 0.9720x | PASS |

Required thresholds are 0.98x for both geometric means and 0.97x for every large forward case. The first exploratory run measured 0.9754x forward geomean and was retained in the logs rather than hidden; it was replaced because it did not meet the gate.

The warm-cache safety gate recorded no additional handle creation, descriptor creation, or heuristic cache miss over 2000 calls. Allocated and reserved GPU-memory growth were both zero.

