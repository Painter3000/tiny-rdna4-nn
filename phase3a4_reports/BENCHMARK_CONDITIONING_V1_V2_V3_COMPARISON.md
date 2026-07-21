# Conditioning Protocol v1/v2/v3 comparison

| Protocol | Measurement unit | Stationarity | Historical status |
|---|---|---|---|
| v1 | Individual ~90-us Forward call | None after measurement | Valid performance FAIL in official series 1 |
| v2 | Individual ~90-us Forward call | Six blocks of 25 calls | Valid infrastructure FAIL: 11/20 exploratory processes nonstationary |
| v3 | One HIP window of 128 sequential Forward calls, stored per operation | Eight blocks of five 128-call windows | Predeclared; result pending |

V3 reduces sensitivity to individual short-event timing while retaining fresh
process isolation, alternating phase order, adaptive metric-specific warm-up,
post-convergence stationarity gates, and setup/scratch invariants. V1 and v2
files and conclusions remain unchanged.
