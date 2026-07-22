# Phase 3A4 environment qualification protocol

Status: predeclared qualification-only scatter run. This is not a performance
run, does not create Protocol v5, does not evaluate the `>= 0.99` gate, and
cannot authorize a Phase-3A4 PASS tag.

## Fixed v4 context (not re-evaluated)

- geometric mean: `0.997361`
- Student-t 95% interval: `[0.990639, 1.004129]`
- bootstrap 95% interval: `[0.991771, 1.000526]`
- no Phase-3A4 release because n=7 was selected, leave-one-out was fragile,
  and pair 6 had strong influence

## Fixed environment and preflight

The run must start from a real Linux `/dev/tty` with `DISPLAY` and
`WAYLAND_DISPLAY` unset. The launcher discovers the real service behind
`display-manager.service`; it does not assume LightDM, GDM, or SDDM. Before it
stops that service it verifies the TTY, a clean Git worktree, exact v4 harness
hashes, exact A3/A4 binding identities, and absence of foreign GPU compute
processes. Failure produces `INVALID_ENVIRONMENT` without stopping the display
manager.

After the stop, the runner verifies that the discovered service is inactive and
that browser, video-player, desktop shell/compositor, display-server, and GPU
monitor processes are absent. A one-shot AMD-SMI process query must find no GPU
process. Periodic AMD-SMI, ROCm-SMI, nvtop, radeontop, or `watch` is prohibited.

The launcher restores a display manager that it stopped through an EXIT/signal
trap on success, error, Ctrl-C, SIGHUP, SIGINT, and SIGTERM. SIGKILL cannot be
trapped by any userspace program; the documented manual recovery command is the
fallback for that operating-system limitation.

## Unchanged workload and fixed sample count

The runner invokes the unchanged native Protocol-v4 child whose SHA-256 is
predeclared in the runner. Exactly 28 pairs alternate A3→A4 and A4→A3 order.
Every variant/pair combination uses a fresh process. Invalid processes and pairs
are retained and are never repeated, replaced, or selectively discarded.

The number 28 was fixed in advance: at the v4 pair-validity rate of 0.7225 it
yields 20.2 valid pairs in expectation; the quiet TTY is only a safety margin.

## Qualification gate and labels

At least 18 of the 28 complete pairs must be valid. Otherwise the result is
`ENVIRONMENT_QUALIFICATION_FAIL`. Preflight failure is `INVALID_ENVIRONMENT`.
Passing the sole qualification gate produces `ENVIRONMENT_QUALIFICATION_PASS`.
A PASS means only that the quiet environment is suitable for later v5 sample
size planning and that a scatter estimate is available.

These are the only result labels:

- `ENVIRONMENT_QUALIFICATION_PASS`
- `ENVIRONMENT_QUALIFICATION_FAIL`
- `INVALID_ENVIRONMENT`

## Recorded quantities

The result records total and per-A3/A4 process validity, pair validity, warm-up
convergence, stationarity, queue headroom, handle/heuristic/scratch invariants,
and every valid gate-direction ratio `A3 time / A4 time`.

The target quantity is the sample standard deviation of the valid log pair
ratios. Its two-sided 95% confidence interval is computed with chi-square
quantiles and n−1 degrees of freedom. A standard deviation estimated from about
20 observations remains substantially uncertain: its 95% interval is roughly
0.76 to 1.46 times the point estimate. It is therefore a planning quantity with
an error band, not an exact constant.

The geometric mean and its two-sided Student-t 95% interval are reported only
as side products and are not decision values. A3→A4 versus A4→A3 is reported
descriptively and is not a gate.

Protocol v4 ran with a compositor while this qualification runs without one.
Lower scatter is the desired signal. Unchanged scatter does not prove that the
environment was not causal: an apparent null effect may result from the small
sample and must not be overinterpreted.

## Outputs

- `phase3a4_reports/environment_qualification.json`
- `phase3a4_reports/ENVIRONMENT_QUALIFICATION.md`
- `phase3a4_reports/environment_qualification_run.log`

No PASS tag is created.
