# Phase 3A3 backward-epilogue capability

Result: **BLOCKED** for gfx1201 / ROCm 7.2.

The installed ROCm 7.2 `hipblaslt.h` declares `RELU_AUX_BIAS`, `BGRADA`, and
`BGRADB`, but does not declare `DRELU` or `DRELU_BGRAD`. The isolated native
probe also tested the corresponding numeric candidates 136 and 152 instead of
stopping at the compile-time absence.

## Matrix probe

The probe recorded 1,680 cases covering all five requested epilogue names,
widths 16/32/64/128, batches 1/7/64/257/1024/4096, default and explicit
streams, and the requested network roles. Hidden-layer counts and activation
variants are recorded in every result. Unsupported cases remain in the JSON.

For numeric candidates 136 and 152, descriptor setup, zero-workspace
heuristics, and launches returned success in all 336 cases each. This status is
not sufficient evidence of semantic support.

## Empirical semantics

An identity-GEMM probe supplied AUX values `-3`, `-1`, `-0.0`, `+0.0`, `+1`,
and `+3`, with upstream gradients `-2`, `-1`, `0`, `+1`, and `+2`.

- Numeric candidate 136 did not apply a ReLU derivative. Output was identical
  to the unmasked GEMM result for negative, signed-zero, and positive AUX.
- Numeric candidate 152 behaved identically and did not write a bias gradient.
- Repeating candidate 152 with bias-output sentinels 77 and 31 left every
  element unchanged. It therefore neither overwrote nor accumulated BGRAD.

Consequently, the apparent heuristic/launch success is a semantic no-op, not a
usable undocumented DReLU capability. AUX convention, BGRAD axis, and BGRAD
overwrite semantics cannot be established because the requested operation is
not implemented by this ROCm build.

## Decision

There is no correct zero-workspace `DRELU_BGRAD` path for the large ReLU hidden
cases. Phase 3A3B integration is prohibited by the phase contract. Production
dispatch remains unchanged and no Phase 3A3 PASS tag is created.

Raw evidence:

- `backward_epilogue_probe.json`
- `drelu_semantics.json`
- `phase3a3_backward_epilogue_probe.cpp`
- `phase3a3_drelu_semantics_probe.cpp`

