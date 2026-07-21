# Phase 3A4 environment qualification protocol

Status: predeclared qualification-only run. It is not a replacement performance
protocol, does not evaluate the Phase-3A4 performance gate, and cannot authorize
a PASS tag.

## Fixed environment

The run must start from a Linux TTY with `DISPLAY` and `WAYLAND_DISPLAY` unset.
The installed display manager is `lightdm.service`; it must be inactive. Browser,
video-player, desktop shell/compositor, display server, and GPU-monitor processes
must be absent. A one-shot AMD-SMI process query must report no foreign GPU
compute process. No periodic AMD-SMI, ROCm-SMI, nvtop, radeontop, or `watch`
query is allowed during the run.

The runner also requires a clean Git worktree, the predeclared v4 harness and
binding-manifest hashes, and exact SHA-256 matches for both test-only bindings.
Failure before measurement produces `INVALID_ENVIRONMENT`.

## Unchanged benchmark workload

The runner invokes the unchanged native Protocol-v4 child. Exactly ten pairs
alternate A3→A4 and A4→A3 order. Every variant/pair combination uses a fresh
process. No invalid process or pair is repeated or replaced.

## Qualification gates

The quiet environment qualifies only if all of these predeclared conditions
hold:

- at least 19 of 20 processes are valid;
- at least 9 of 10 complete pairs are valid;
- each of Phase 3A3 and Phase 3A4 has at least 9 of 10 stationary processes;
- all 20 adaptive warm-ups converge;
- all 20 native queue-headroom checks pass;
- all 20 handle, heuristic, partial-live, and scratch invariants pass.

These gates measure yield and symmetry of a quiet measurement environment. No
performance-ratio threshold is applied.

## Recorded diagnostics

The result retains every process and pair and records overall and per-phase
validity, pair validity, warm-up convergence, stationarity, queue headroom,
invariants, valid paired ratios, log-ratio sample deviation, geometric mean,
two-sided 95% Student-t interval, and descriptive A3→A4 versus A4→A3 groups.

The only possible final statuses are `ENVIRONMENT_QUALIFICATION_PASS`,
`ENVIRONMENT_QUALIFICATION_FAIL`, and `INVALID_ENVIRONMENT`. A qualification
PASS means only that this quiet environment is suitable for later sample-size
work. It does not constitute Phase-3A4 performance evidence or authorize a tag.
