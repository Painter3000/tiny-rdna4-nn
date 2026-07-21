# Phase 3A4 benchmark conditioning protocol v4

Status: predeclared native-window Width-128 Forward exploration. Protocols v1,
v2, and v3 remain immutable historical evidence. No official series or PASS tag
is authorized by this exploration.

## Test-only native window

A patch is applied only to isolated Phase-3A3 and Phase-3A4 benchmark
worktrees. Production sources and bindings remain unchanged. The added private
Pybind method receives the normal native Module, input, and parameters. Within
one Python/Pybind invocation it allocates one normal output buffer and calls
the unchanged productive `m_module->inference(...)` path exactly 128 times,
using the same current HIP stream, input, parameters, and output buffer. No
replacement kernel or HIP Graph is used.

The method records a HIP start event immediately before the loop and a stop
event immediately after it. There is no synchronization inside the loop; only
the stop event is synchronized. `steady_clock` separately measures native host
enqueue time around the loop. GPU time per operation is the HIP-event window
time divided by 128; host time is never included in the performance value.

## Queue headroom

Every native window must satisfy:

`native_host_enqueue_ms / hip_event_gpu_ms <= 0.80`

This predeclared 20% queue-headroom requirement means the host must finish
submitting the complete window at least 20% sooner than the GPU finishes it.
Failure indicates a credible risk that submission cannot keep the stream fed
and makes the process an infrastructure FAIL.

## Warm-up, measurement, and gates

Three ordinary unmeasured Forward calls first materialize the model, handle,
and plans. Warm-up then uses at least 10 and at most 100 native windows. The
last five per-operation values must have at most 1% spread and no monotonic
endpoint drift above 0.5%.

Measurement contains exactly 40 native windows, or 5,120 productive Forward
calls. Eight predeclared blocks of five windows retain the v3 2% stationarity
requirements. Handle creations, heuristic misses, scratch-peak growth, and
nonzero partial-live boundaries remain forbidden.

Exactly ten fresh-process Width-128 Forward pairs alternate A3/A4 order. Every
process must converge and pass stationarity, invariants, and queue headroom.
Only then is the unchanged `median(A3 per-op) / median(A4 per-op) >= 0.99`
performance gate evaluated. No official complete series is started.

## Recorded exploratory outcome

Protocol v4 is a valid infrastructure FAIL. Native submission removed the v3
bottleneck: all 20 processes converged, passed every handle/heuristic/scratch
invariant, and passed the queue-headroom gate. The median native enqueue/GPU
ratio was `0.42022037`, its maximum was `0.49886919`, leaving at least 50.11%
headroom in every observed window.

Seventeen of 20 processes and seven of ten pairs were stationary. The three
failures were Phase 3A3 in pairs 2 and 9 (block spreads `3.4587%` and `3.3827%`)
and Phase 3A4 in pair 5 (`3.1855%`). Since all ten pairs were not valid, the
aggregate performance ratio is intentionally null. No process is repeated and
no official series is started.
