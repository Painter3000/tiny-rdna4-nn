# Phase 4A3-Q0d long4x freeze

## Identität

- Q0c-Freeze-Commit: `47331167a366223c01d9b27c0ce1d4d2db1ab05b`
- Run-ID: `phase4a3_q0d_long4x_20260728T091915Z`
- Profil: `long4x`
- Schedule: `spin`
- Worker: 16
- Auto-Sensitivität: nicht enthalten

## Ergebnis

| Region | Ergebnis |
|---|---|
| LN, Batch 256 | 1/4 gültig – BLOCKED |
| LP, Batch 1 | 4/4 gültig – PASS |
| LP, Batch 31 | 4/4 gültig – PASS |
| LP, Batch 128 | 4/4 gültig – PASS |

Gesamtstatus:

```text
PHASE4A3_Q0D_LATENCY_PROFILE_BLOCKED
spin_profile_qualified=false
performance_claim_allowed=false
```

Das maximale vorab definierte Profil `long4x` qualifizierte LP vollständig,
LN jedoch nicht. Es erfolgen keine identische Wiederholung, keine
Schwellenlockerung, keine Ausreißerentfernung und keine adaptive Verlängerung.

Das ZIP enthält die ursprünglichen Ergebnis-, Manifest-, Worker-, Log-,
PID- und Coredump-Artefakte sowie eine interne Hashliste.

## Bundle

```text
phase4a3_q0d_long4x_20260728T091915Z_freeze_bundle.zip
SHA256=0133a42aa56c2668c7230e56e19ece419cd2cd44ab400d057db9a6b9508c7225
```
