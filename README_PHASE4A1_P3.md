# Phase 4A1-P3 — Width-64 two-layer fused forward

Marker:

```text
TCNN_RDNA4_P4A1_P3_WIDTH64_TWO_LAYER_FUSED_001
```

This checkpoint executes two complete Width-64 hidden layers in one HIP kernel.

```text
16×64 FP16 input
→ layer 1: four K tiles per wave
→ FP32 bias + ReLU
→ accumulator→matrix-A relay
→ FP16 hidden 1
→ shared 16×64 LDS buffer
→ block publication barrier
→ layer 2 reads all four K tiles from LDS only
→ FP32 bias + ReLU
→ accumulator→matrix-A relay
→ FP16 hidden 2 diagnostic output
```

Execution: one block, four Wave32 waves, 128 threads, eight `mma_sync` calls
per wave, 32 per block, and 2048 bytes of LDS.

There is no global hidden-1 store and no global hidden-1 reload. The read-only
hidden-1 oracle is used only for a cross-wave LDS validation and is never used
as a layer-2 operand.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe
unzip -o ~/Downloads/phase4a1_p3_width64_two_layer_fused_forward_bundle.zip
chmod +x \
  scripts/prepare_phase4a1_p3.py \
  scripts/finalize_phase4a1_p3.py \
  scripts/run_phase4a1_p3_width64_two_layer_fused_forward.sh

export P0_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p0_20260725T080946Z"
export P1_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p1_20260725T082538Z"
export PHASE4A1_P2_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p2_20260725T142608Z"
unset PHASE4A1_P3_EVIDENCE
scripts/run_phase4a1_p3_width64_two_layer_fused_forward.sh
```

Expected final marker:

```text
PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED: PASS
```

The runner accepts only evidence destinations beginning with `phase4a1_p3_`
and refuses to overwrite a non-empty evidence directory.
