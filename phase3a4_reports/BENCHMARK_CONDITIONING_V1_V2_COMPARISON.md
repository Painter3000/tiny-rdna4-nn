# Conditioning Protocol v1 versus v2

Protocol v1 official series 1 is preserved at commit `3006802`. It conditioned
Forward+Backward before separately measuring Forward, and run 4 exposed a
nonstationary Forward interval in `large_1024_w128_relu`.

Protocol v2 isolates every case, metric and variant in a fresh process, uses
the measured metric itself for conditioning, and adds six fixed measurement
stationarity blocks. Protocol v2 results are pending; this report will be
completed mechanically after the predeclared four-run official series 2.
