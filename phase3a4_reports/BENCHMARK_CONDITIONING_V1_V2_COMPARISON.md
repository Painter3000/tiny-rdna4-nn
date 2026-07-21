# Conditioning Protocol v1 versus v2

Protocol v1 official series 1 is preserved at commit `3006802`. It conditioned
Forward+Backward before separately measuring Forward, and run 4 exposed a
nonstationary Forward interval in `large_1024_w128_relu`.

Protocol v2 isolates every case, metric and variant in a fresh process, uses
the measured metric itself for conditioning, and adds six fixed measurement
stationarity blocks.

## Results

| Protocol | Scope reached | Result | Decisive evidence |
|---|---|---|---|
| v1 | Official series 1, four runs | FAIL | Run 4 Forward geomean `0.9775363556`; Width-128 Forward contained a transient slow middle third |
| v2 | Ten Width-128 Forward exploratory pairs | Infrastructure FAIL | 11/20 metric processes nonstationary; only one complete pair valid |
| v2 | Official series 2 | Not started | Locked by the failed exploratory stationarity gate |

All 20 v2 warm-ups converged, and every handle, heuristic and scratch
measurement invariant passed. The failure is specifically post-convergence
measurement nonstationarity: only 3/10 Phase-3A3 and 6/10 Phase-3A4 processes
passed both 2% block criteria. Maximum block spreads were `13.93754468%` for
Phase 3A3 and `13.46807702%` for Phase 3A4.

Protocol v1 series 1 remains immutable and is not replaced. Protocol v2's
nonstationary exploratory runs are retained and are not repeated. Because the
v2 aggregate requires every process to be valid, its conditioned ratio is
intentionally null and no official-series comparison or PASS selection is
possible.
