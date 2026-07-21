# Phase 3A4 benchmark conditioning protocol v5

Status: predeclared deterministic-environment Width-128 Forward exploration.
Protocols v1 through v4 remain immutable. No official series or PASS tag is
created by this protocol.

## Unchanged measurement harness

V5 invokes `benchmark_phase3a4_conditioned_metric_v4.py` without modification.
Each variant of each pair remains a fresh process using one native call per
128-Forward window, 10–100 warm-up windows, exactly 40 measurement windows,
the unchanged eight-block 2% stationarity gate, 20% native queue headroom, and
the existing handle, heuristic, and scratch invariants.

## AMD-SMI determinism environment

The controller requires root privileges and GPU 0 to identify as R9700. Before
measurement it queries AMD-SMI ASIC, stock clock capability, overdrive, and
current performance level. Overdrive must be zero. The selected GFXCLK is the
maximum GFX/SCLK/SYS DPM frequency explicitly advertised by the installed
driver. This is conservative with respect to correctness: it never exceeds a
stock capability, applies no frequency offset, and uses no observed momentary
clock as an assumption.

The controller enables AMD-SMI performance determinism at that frequency and
then requires an independent structured metric query to return exactly
`PERF_LEVEL: DETERMINISM`. Unsupported commands, missing capabilities, a
different GPU, nonzero overdrive, or failed verification produce
`INVALID_ENVIRONMENT`.

The original performance level is captured before mutation and restored in a
`finally` path after success or any error. Restoration is independently read
back. Failure to restore produces `INVALID_ENVIRONMENT`. A pre-existing
DETERMINISM state is rejected because its exact prior soft-max cannot be
reconstructed safely.

## Fixed sample-retention rule

Exactly 20 pairs are predeclared with alternating A3/A4 order and no retries,
replacement, or selection. At least 16 complete valid pairs are required. The
16/20 threshold is a fixed 80% retention rule: it permits at most four
infrastructure-invalid pairs while retaining at least 16 genuinely paired
observations for a robust median comparison. Invalid processes and pairs stay
in the raw data. This threshold is stricter than accepting the v4 observation
count (17/20 processes but only 7/10 complete pairs) and prevents a small set of
surviving pairs from determining the result.

Only complete valid pairs enter the two phase medians. If fewer than 16 pairs
are valid, the ratio is null and the result is `INFRASTRUCTURE_FAIL`. Otherwise
the unchanged gate is:

`median(valid paired Phase3A3 per-op times) / median(valid paired Phase3A4 per-op times) >= 0.99`

A passing exploration unlocks only planning of a future official series. It
does not start that series and does not authorize a PASS tag.
