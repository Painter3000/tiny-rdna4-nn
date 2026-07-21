# Phase 3A4 benchmark conditioning protocol

Status: diagnostic protocol; no PASS tag is authorized by this document.

## Isolation and ordering

Every benchmark case and every Phase-3A3/Phase-3A4 member is executed by
`benchmark_phase3a4_conditioned_case.py` in its own fresh process. The paired
runner alternates the order: odd pairs run Phase 3A3 then Phase 3A4; even pairs
run Phase 3A4 then Phase 3A3. Results are aggregated without best-run selection.

The earlier sequential ReLU-to-None experiment remains available as
`workload_order_sensitivity` in `none_regression_bisect.json`. It is diagnostic
only and is not a release gate for the unchanged None implementation.

## Three distinct stages

1. Model/plan/cache warm-up constructs the model and executes three untimed
   forward and forward-backward operations. This materializes GEMM plans,
   handles and the maximum case-specific fused scratch requirement.
2. GPU steady-state warm-up uses the complete case-specific forward-backward
   workload in 50-operation HIP-event windows. At least five and at most 100
   windows are collected. The last five window medians must span no more than 1%,
   and a monotonic endpoint drift greater than 0.5% invalidates the run.
3. Only after case convergence, the measurement records 150 individual
   HIP-event times for forward and forward-backward. Handle creations,
   heuristic misses and fused-scratch peak growth must all remain zero during
   this stage; partial scratch must be non-live at both boundaries.

For `large_4096_w64_none`, fused stage-1, ReLU-only and bias-finalize launch
deltas must be zero for the whole process. Partial-scratch live and peak must
remain zero. The Phase-3A4 diagnostic fallback counter deliberately accounts
the two unchanged legacy hidden-layer activation-gradient paths per backward;
each stage must equal that deterministic count and its unexpected delta must be
zero. A failure of any invariant invalidates the run.

## Gates

The exploratory None comparison contains at least ten paired fresh-process
runs. Its unchanged gate is:

`median(Phase3A3 conditioned forward+backward) / median(Phase3A4 conditioned forward+backward) >= 0.99`

Exactly four full official runs may be started only after that gate passes.
The existing functional and performance thresholds are unchanged. The
historical `phase3a3_blocked_baseline.json` is never written by these tools;
the newly conditioned reference is stored separately as
`phase3a3_conditioned_baseline.json`.

## Recorded outcome

The ten-pair conditioned None gate passed at `0.9986946037` (Phase-3A3 median
`2.2650722861 ms`, Phase-3A4 median `2.2680329680 ms`). All ten pairs were
valid, and the unexpected legacy-fallback delta was zero in every pair.

Exactly four official runs were recorded. Runs 1 through 3 passed. Run 4 is a
valid FAIL because its forward geomean was `0.9775363556`, below the unchanged
`0.99` gate. The sole failed component was `large_1024_w128_relu` forward at
`0.9261963105`; all four official None forward-backward ratios passed
(`1.0000263065`, `1.0011469164`, `1.0033289285`, and `1.0025310586`). No
official run may be replaced or selected away. Consequently, no PASS tag is
authorized.
