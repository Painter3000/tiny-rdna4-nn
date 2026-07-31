# Phase 3A Fused Backward Freeze

Freeze marker: `P3A-GATE-FREEZE-001`

## Qualified state

- Result: `RDNA4_FUSED_MLP_BACKWARD_CORRECTNESS: PASS`
- Annotated tag: `phase3a-fused-backward-gfx1201-pass`
- Freeze commit: the commit dereferenced by the annotated tag above
- Basis: `phase2-fused-forward-gfx1201-pass`
- Basis commit: `e3ff86c8d0f7aebb1102c3b91e51974d5bef1427`
- Architecture: `gfx1201`
- Toolchain: AMD clang 22.0.0git, ROCm 7.2.0 (build 26014)
- Evidence run: `phase3a_fused_backward/evidence/20260729T185612Z_41268`
- External evidence archive: `phase3a_fused_backward_gfx1201_pass_20260729.tar.gz`

The annotated tag is the canonical resolver for the immutable final commit.
Freeze verification must separately prove the tag object type, dereference the
tag to its commit, and prove that the Phase-2 PASS tag is an ancestor.

## Evidence anchors

| File | SHA-256 |
| --- | --- |
| `evidence/20260729T185612Z_41268/SHA256SUMS` | `e2f0319a76a99c0d55a6ebe4fca7efe987bbb827a3805a31bc2b049d5fece040` |
| `evidence/20260729T185612Z_41268/verification.json` | `a503288743581dcae173179c09999360c4925ee1482c60d6566c5b45cd16b2ff` |
| `evidence/20260729T185612Z_41268/provenance.json` | `ea69129e53d7ba7dfab9e80ebedc08137e5c40ffe4c0cb6e858db165a7b5109d` |
| `evidence/20260729T185612Z_41268/PHASE3A_FUSED_BACKWARD_REPORT.md` | `7ee0364448a58f6dec1a8e145708253c469714db0dc40518d8d7c25fe42f8d0e` |
| `impl/native/phase3a_fused_backward.hip` | `7ad0cc174c25918448b7936bfdca63bf2fdf5aab441063ca3618aefdee135a85` |
| `impl/phase3a_fused_backward_probe.py` | `9f89e45f4cdac4c3e9912b1b5617f9e512a6bd8d9886f1f3e817c6f0f4836886` |
| `tools/verify_phase3a_isa.py` | `700a36a26e146417a148be90ea423389a2f48878f86f11f2c6f70290f3ae6ae9` |
| `tools/verify_phase3a_source.py` | `62f440f6e522a5b7f92b519b323f6c986bcd6be57403fb7a46c653706adb031f` |

The archive digest is stored in the adjacent
`phase3a_fused_backward_gfx1201_pass_20260729.tar.gz.sha256`. The archive must
also be extracted into a fresh directory and pass its internal `SHA256SUMS`.

## Qualified contract

- Fixed `64 -> 64 -> 64 -> 64` backward with FP16 saved boundaries and FP32
  accumulation.
- One rocWMMA backward-tile kernel computes the reverse three-layer dX path and
  complete dW0/dW1/dW2 partials per 16-row batch tile.
- A separate kernel reduces partials elementwise in strictly ascending tile
  order.
- Four required cases each have three fresh, canonically bit-identical hardware
  replays.
- Every replay contains all four complete Phase-1A backward operand-map
  witnesses.
- The independent HipBLASLt backward path uses separate dZ, dX, and dW buffers.
- ISA evidence records 24 WMMA/MFMA instructions, 74 LDS stores, 14 barriers,
  43 LDS loads, and zero atomic instructions.
- The backward kernel uses 6144 bytes of LDS, no scratch, and no spills. The
  reduction kernel uses no LDS, scratch, or spills.
- Thirteen tool and negative regression tests pass.

## Frozen verifier corrections

The freeze includes verifier handling for ROCm 7.2 `ds_store`/`ds_load`
spellings and split signal/wait barriers. It also records effective rocWMMA
call sites through the shared dW device helper and parses resource fields that
ROCm emits before the mangled kernel name. Regression coverage for the ROCm
7.2 ISA spelling is included.

## Known limits

This qualification covers only the fixed-shape mathematical backward and
deterministic dW reduction. It does not qualify bias gradients, arbitrary
topologies, optimizer or parameter updates, loss scaling, checkpoints,
production/factory integration, HashGrid fusion, JIT, atomic alternatives, or
performance. Resource figures are evidence, not a performance claim.
