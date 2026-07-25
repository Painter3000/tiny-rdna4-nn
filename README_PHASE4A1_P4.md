# Phase 4A1-P4 — Width-64 single-LDS reuse and three-layer fused forward

Marker:

```text
TCNN_RDNA4_P4A1_P4_WIDTH64_THREE_LAYER_FUSED_001
```

This checkpoint executes the complete Width-64 forward topology locked in
Phase 4A1-P0:

```text
16×64 FP16 input
→ layer 1: four K tiles per wave
→ FP32 bias 1 + ReLU + FP16
→ hidden 1 in one 16×64 LDS buffer
→ publication barrier
→ layer 2 reads hidden 1 from LDS only
→ four K tiles per wave
→ read-complete barrier
→ FP32 bias 2 + ReLU + FP16
→ overwrite the same LDS buffer with hidden 2
→ publication barrier
→ layer 3 reads hidden 2 from the reused LDS buffer only
→ four K tiles per wave
→ FP32 bias 3
→ final 16×64 FP32 global output
```

Execution contract:

```text
1 kernel
1 block
4 Wave32 waves
128 threads
12 mma_sync calls per wave
48 mma_sync calls per block
1 LDS buffer / 2048 bytes
3 block barriers
```

The read-complete barrier between layer 2 accumulation and the hidden-2 store
is the key new proof. It prevents any wave from overwriting hidden 1 while a
peer wave can still read it. Hidden 2 then replaces hidden 1 in the same
physical LDS allocation.

The read-only hidden oracles are used only for cross-wave validation. Neither
hidden tensor is stored to or reloaded from global memory as a network operand.
The final FP32 output is compared against the P0 CPU-FP64 oracle.

This remains a correctness checkpoint. It does not claim production
integration, occupancy viability, ISA-level absence of spills, or performance.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a1_p4_width64_three_layer_fused_forward_bundle.zip

chmod +x \
  scripts/prepare_phase4a1_p4.py \
  scripts/finalize_phase4a1_p4.py \
  scripts/run_phase4a1_p4_width64_three_layer_fused_forward.sh

export P0_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p0_20260725T080946Z"
export P1_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p1_20260725T082538Z"
export PHASE4A1_P2_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p2_20260725T142608Z"
export PHASE4A1_P3_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p3_20260725T145437Z"

unset PHASE4A1_P4_EVIDENCE

scripts/run_phase4a1_p4_width64_three_layer_fused_forward.sh
```

Expected final markers:

```text
PHASE4A1_P4_PREPARATION: PASS
WIDTH64_LAYER2_READ_COMPLETE_BARRIER: PASS
WIDTH64_SINGLE_LDS_BUFFER_REUSE: PASS
WIDTH64_LAYER2_HIDDEN_BITWISE_CORRECTNESS: PASS
WIDTH64_LAYER3_INPUT_FROM_LDS_ONLY: PASS
WIDTH64_LAYER3_FOUR_K_TILE_ACCUMULATION: PASS
WIDTH64_FINAL_OUTPUT_VS_CPU_FP64: PASS
WIDTH64_NO_INTERMEDIATE_GLOBAL_STORE_RELOAD: PASS
WIDTH64_THREE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS
RDNA4_WIDTH64_THREE_LAYER_FUSED_FORWARD_CORRECTNESS: PASS
PHASE4A1_P4_JSON_AUDIT: PASS
PHASE4A1_P4_MAP_CONTEXT: RECORDED
PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED: PASS
```

The runner accepts only evidence destinations beginning with `phase4a1_p4_`
and refuses to overwrite a non-empty evidence directory.
