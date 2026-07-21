# Conditioning Protocol v1/v2/v3/v4 comparison

| Protocol | Submission and timing unit | Outcome |
|---|---|---|
| v1 | Individual Python/Pybind Forward calls | Valid performance FAIL |
| v2 | Individual calls with post-measurement blocks | Valid infrastructure FAIL |
| v3 | 128 Python/Pybind calls inside each HIP window | Valid infrastructure FAIL; Python submission limited 19/20 processes |
| v4 | One Python call invokes a native loop of 128 productive Forward calls | Predeclared; exploratory result pending |

V4 removes Python submission from the 128-operation window while preserving
the production inference path, fresh-process isolation, alternating order,
adaptive warm-up, eight-block stationarity gate, and diagnostic counters. V1,
v2, and v3 files and conclusions remain unchanged.
