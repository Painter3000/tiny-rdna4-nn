# Phase 4A0-P4 — Accumulator-to-matrix-A relay proof

Marker:

```text
TCNN_RDNA4_P4A0_P4_ACC_TO_A_RELAY_001
```

P4 consumes the qualified P3 interpretation and generates a deterministic HIP
probe with the exact accumulator→matrix-A permutation embedded in the source.

The GPU pipeline is:

```text
FP16 identity × FP16 source matrix
→ FP32 accumulator
→ output-column bias in accumulator ownership
→ ReLU in FP32
→ P3-derived cross-lane/cross-slot reindex
→ FP32→FP16 conversion
→ rocwmma::from_register_file<FragA>
→ store_matrix_sync row-major
→ bitwise full-matrix comparison against CPU
```

The relay uses uniform Wave32 shuffle participation. For every target matrix-A
slot, all lanes issue all eight candidate-slot shuffles; the P3 mapping selects
the required source lane and source accumulator slot.

This is a correctness proof, not a performance result.

## Install and run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a0_p4_accumulator_to_matrix_a_relay_bundle.zip

chmod +x \
  scripts/generate_phase4a0_p4_relay_source.py \
  scripts/run_phase4a0_p4_accumulator_to_matrix_a_relay.sh

export P3_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p3_20260725T072430Z"

scripts/run_phase4a0_p4_accumulator_to_matrix_a_relay.sh
```

Expected final markers:

```text
PHASE4A0_P3_PREREQUISITE: PASS
PHASE4A0_P4_SOURCE_GENERATION: PASS
ROCWMMA_ACC_TO_A_REINDEX: PASS
ROCWMMA_ACC_TO_A_FP16_CAST: PASS
ROCWMMA_ACC_TO_A_STORED_MATRIX: PASS
ROCWMMA_ACC_TO_A_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A0_P4_JSON_AUDIT: PASS
ROCWMMA_P4_MAP_CONTEXT: RECORDED
PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY: PASS
```
