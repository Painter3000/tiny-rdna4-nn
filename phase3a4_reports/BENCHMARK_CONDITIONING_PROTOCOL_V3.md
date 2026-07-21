# Phase 3A4 benchmark conditioning protocol v3

Status: predeclared Width-128 Forward exploration. Protocol-v1 and v2 evidence
is immutable. No PASS tag is authorized, and no official v3 series is started
by this protocol.

## Isolation and environment

Each `(case, metric, variant)` measurement uses a fresh process. The ten pairs
alternate Phase 3A3 then Phase 3A4 for odd pairs and the reverse for even
pairs. Before the run and again in every child process, `/proc` is checked for
`amd-smi`, `rocm-smi`, `nvtop`, and `radeontop`. Detection produces
`INVALID_ENVIRONMENT`; no process is terminated.

## Window measurement

The sole exploratory workload is `large_1024_w128_relu / forward`. One HIP
start event is recorded before 128 sequential Forward submissions on the same
stream, and one stop event after the 128th. There is no synchronization between
calls; only the stop event is synchronized. The performance sample is
`elapsed_window_ms / 128`.

CPU wall time around submission of the 128 calls is recorded separately and is
never included in the performance value. A window indicates probable stream
starvation when `host_submission_ms >= elapsed_window_ms`. Any such warm-up or
measurement window makes the process an infrastructure FAIL.

## Warm-up and measurement

Three unmeasured Forward calls materialize the model, handle and plans. Warm-up
then uses windows of exactly 128 calls, at least 10 and at most 100. The last
five per-operation values must have at most 1% spread and no monotonic endpoint
drift above 0.5%.

After convergence, exactly 40 windows are recorded: 5,120 Forward calls per
process. The windows are pre-partitioned into eight consecutive blocks of five.
The eight block medians must have at most 2% spread, and every block median must
be within 2% of the overall 40-window median. Handle creations, heuristic
misses, scratch-peak growth, and nonzero partial-live boundaries invalidate the
process. Failed processes are retained and never repeated.

Exactly ten pairs are recorded. All pairs must be valid before applying the
unchanged gate:

`median(Phase3A3 per-operation time) / median(Phase3A4 per-operation time) >= 0.99`

Only a PASS unlocks planning of a later official v3 series. It does not start
that series and does not authorize a PASS tag.
