# Phase 4A3 Q0c — frozen run record

## Identity

- Run: `phase4a3_q0c_20260727T152532Z`
- Evidence directory: `/home/oem/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a3_q0c_20260727T152532Z`
- Apparatus repository commit: `8c06a571a71c292bb032491ffff5a2560620b15a`
- Finalizer-manifest repair commit: `8c06a57`

## Integrity

- Full evidence checksum-index SHA256:
  `9bb46adfb21e1376ff08647465c28e964887ef8714fe2d52f74e459d18f32045`
- Worker-manifest SHA256:
  `a4ffd712e35c4cf9e4ff4540065bd7769a6edfd02fe23a7a614e6d37840a9ba0`
- Apparatus-result SHA256:
  `07be6223c29ec8be994be1993f2272ea2f88d56d69d174e2a7f3449ff853462d`

The complete per-file checksum index remains at:

`/home/oem/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a3_q0c_20260727T152532Z/SHA256SUMS_EVIDENCE`

## Process integrity

- Expected workers: 100
- Manifest entries: 100
- Accepted workers: 100
- Return code zero: 100
- Result JSON present and recorded: 100
- Manifest failures: none

## Subphase decisions

| Subphase | Decision |
|---|---|
| P | Provenance PASS |
| LN | Apparatus BLOCKED |
| LP | Apparatus BLOCKED |
| TP | Apparatus PASS |
| TD | Region NOT QUALIFIED |
| G | Diagnostic COMPLETE |

Overall decision:

`PHASE4A3_Q0C_PROTOCOL_PREREQUISITES_BLOCKED`

`performance_claim_allowed=false`

## Qualified Q0c-TP public-product throughput

The ratios compare the public rocWMMA path with the public
hipBLASLt reference path. They include Python, bindings, wrapper,
padding, allocation, launch and synchronization costs.

| Public batch | Ratio | Student-t 95% CI |
|---:|---:|---:|
| 1 | 1.5172x | 1.4984x–1.5362x |
| 31 | 1.5093x | 1.4981x–1.5207x |
| 128 | 1.5114x | 1.5027x–1.5202x |
| 257 | 1.5028x | 1.4993x–1.5063x |
| 512 | 1.6801x | 1.6508x–1.7100x |
| 1024 | 1.6871x | 1.6557x–1.7192x |
| 4096 | 1.6832x | 1.6715x–1.6949x |
| 16384 | 1.4436x | 1.4278x–1.4595x |

Public inputs are padded to multiples of 256. These throughput
results are not isolated kernel timings and must not be presented
as single-sample latency results.

In particular:

- public batches 1, 31 and 128 execute with padded batch 256;
- public batches 257 and 512 execute with padded batch 512;
- padding alone does not explain the different ratios for public
  batches 257 and 512.

No overall Q0c performance claim is authorized by this run.
