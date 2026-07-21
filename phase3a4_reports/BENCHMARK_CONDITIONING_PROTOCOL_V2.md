# Phase 3A4 benchmark conditioning protocol v2

Status: predeclared protocol for official series 2. No PASS tag is authorized
until the exploratory gate, all four official v2 runs, and all existing
functional gates pass.

## Process isolation

Every `(case, metric, variant)` combination runs in its own fresh process.
Forward processes construct, warm and measure only Forward. Forward+Backward
processes construct, warm and measure only Forward+Backward. No other case,
metric or activation is executed in that process.

Within a pair, Phase 3A3 and Phase 3A4 ordering alternates: odd pairs use A3
then A4; even pairs use A4 then A3. There is no best-result selection.

## Stages and convergence

1. Model/plan/cache warm-up executes three untimed operations of the selected
   metric.
2. GPU steady-state warm-up uses 50-operation windows, at least five and at
   most 100. The last five medians must have at most 1% spread and must not
   show monotonic endpoint drift above 0.5%.
3. After convergence, exactly 150 HIP-event values are recorded. They are
   pre-partitioned into six consecutive blocks of 25 values.

The measurement is stationary only when the six block medians have at most 2%
spread and every block median lies within 2% of the overall median. During the
measurement, handle creations, heuristic misses and scratch-peak growth must
be zero; partial scratch must be non-live at both boundaries.

A nonconvergent or nonstationary process is a valid infrastructure FAIL. It is
retained and is not repeated or replaced.

## Predeclared execution

First, ten `large_1024_w128_relu / forward` pairs are recorded. Their unchanged
gate is `median(Phase3A3) / median(Phase3A4) >= 0.99`, with every process valid
and stationary. Only if this passes may official series 2 create exactly four
new full runs named `conditioned_performance_v2_run_1.json` through `_4.json`.

Official series 1 and its valid Protocol-v1 FAIL remain immutable historical
evidence.
