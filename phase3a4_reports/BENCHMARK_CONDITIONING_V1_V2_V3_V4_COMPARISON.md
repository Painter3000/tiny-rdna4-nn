# Conditioning Protocol v1/v2/v3/v4 comparison

| Protocol | Submission and timing unit | Outcome |
|---|---|---|
| v1 | Individual Python/Pybind Forward calls | Valid performance FAIL |
| v2 | Individual calls with post-measurement blocks | Valid infrastructure FAIL |
| v3 | 128 Python/Pybind calls inside each HIP window | Valid infrastructure FAIL; Python submission limited 19/20 processes |
| v4 | One Python call invokes a native loop of 128 productive Forward calls | Valid infrastructure FAIL; 3/20 processes nonstationary |

V4 removes Python submission from the 128-operation window while preserving
the production inference path, fresh-process isolation, alternating order,
adaptive warm-up, eight-block stationarity gate, and diagnostic counters. V1,
v2, and v3 files and conclusions remain unchanged.

V4 proves that native submission resolves the v3 harness limitation: all
windows retained at least 50.11% measured queue headroom, versus the required
20%. All 20 warm-ups and invariant sets passed. Remaining instability was not
Phase-3A4-specific: two Phase-3A3 processes and one Phase-3A4 process failed
the unchanged 2% stationarity gate. Seven of ten pairs were valid, so the
aggregate performance gate was not evaluated and no official series began.
