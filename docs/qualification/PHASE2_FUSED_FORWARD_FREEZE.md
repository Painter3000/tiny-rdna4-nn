# Phase 2 Fused Forward Freeze

Freeze marker: `P2-GATE-FREEZE-001`

## Qualified state

- Result: `RDNA4_FUSED_MLP_CORRECTNESS: PASS`
- Annotated tag: `phase2-fused-forward-gfx1201-pass`
- Freeze commit: the commit dereferenced by the annotated tag above
- Basis: `phase1b-cross-layer-bridge-gfx1201-pass`
- Architecture: `gfx1201`
- Toolchain: AMD clang version 22.0.0git, ROCm 7.2.0 (build 26014)
- Evidence run: `phase2_fused_forward/evidence/20260729T173705Z_27521`
- External evidence archive: `phase2_fused_forward_gfx1201_pass_20260729.tar.gz`

The tag is the canonical resolver for the final immutable commit. Freeze
verification must prove that it is an annotated tag object, dereference it to a
commit, and prove that the Phase-1B PASS tag is an ancestor of that commit.

## Evidence anchors

| File | SHA-256 |
| --- | --- |
| `evidence/20260729T173705Z_27521/SHA256SUMS` | `1703cbf985d32a46b1e1f13c03e27a19b1ad33452661948a02f5ca42625b90ea` |
| `evidence/20260729T173705Z_27521/verification.json` | `68f46ce5902843328d161bf7715a01bc67f1ad08cded75417d5a780c7b6b97e4` |
| `evidence/20260729T173705Z_27521/provenance.json` | `6d8e42f1e815f4fa1078fd79e0b8bb69dcbfcf19c4e38fc8c3c216dc265d13fe` |
| `evidence/20260729T173705Z_27521/PHASE2_FUSED_FORWARD_REPORT.md` | `ec0403ef47312be2acb3cb6bef1b7caa3133065a5684609568ae87dbcf670a1b` |
| `impl/native/phase2_fused_forward.hip` | `b259f6430148fdfe8a16635d382ae283e52782049aff8268afe9417a51c1bbc2` |
| `impl/phase2_fused_forward_probe.py` | `611b568aea1a6e24ee2de5cc2418544d642dfb64e767fe252aa81a230adbc637` |
| `tools/verify_phase2_isa.py` | `829332c4fb62eeed1168af7e79067d2e861da3d8736cb1341261661d4af32d3f` |
| `TOOL_FIX_ROCM72_ISA_NAMES.md` | `c28316771a0eb9854bf4ab2bd138053cc624b936a3c4f10167ba34ad3ae364c7` |

The outer archive digest is stored beside the archive in
`phase2_fused_forward_gfx1201_pass_20260729.tar.gz.sha256`. Its contents must
also pass the run's internal `SHA256SUMS` after extraction into a fresh
directory.

## Qualified contract

- One fused `64 -> 64 -> 64 -> 64` forward kernel.
- Two canonical LDS cross-layer bridges.
- Three layers and exactly twelve WMMA/MFMA instructions.
- Four qualified cases, each with three bit-identical hardware replays.
- Both bridges reload all 1024 values and preserve the expected parameter
  hashes.
- ISA evidence contains 16 LDS stores, 10 barriers, and 24 LDS reads.
- Metadata records group-segment size 2048, private-segment size 0, scratch 0,
  and no spills.
- HipBLASLt is an independent numerical cross-check, not the qualification
  oracle.

## Frozen tool correction

The ROCm 7.2 ISA verifier correction is part of this freeze. It recognizes the
ROCm 7.2 `ds_store`/`ds_load` spelling and split barrier encoding. Its regression
tests and rationale are included with the verifier so that later tooling
changes cannot silently redefine this kernel's evidence.

## Known limits

This qualification is deliberately limited to the fixed-shape fused forward
probe. It does not qualify bias handling, dynamic shapes, backend/factory
integration, backward propagation, training, or performance. No claim outside
the Phase-2 scope and gates is implied.
