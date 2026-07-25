# Phase 4A0-P5 — Two-layer fused-forward correctness

Marker:

```text
TCNN_RDNA4_P4A0_P5_TWO_LAYER_FUSED_FORWARD_001
```

This is the first genuine two-layer fused MLP-forward correctness probe in the
RDNA4 rocWMMA branch.

Topology:

```text
batch 16
input width 16
hidden width 16
output width 16
```

Single-kernel pipeline:

```text
FP16 input × FP16 weight 1
→ FP32 accumulator 1
→ FP32 hidden bias
→ ReLU
→ P3-derived accumulator→matrix-A Wave32 relay
→ FP32→FP16 hidden conversion
→ FP16 hidden × FP16 weight 2
→ FP32 accumulator 2
→ FP32 output bias
→ FP32 output
```

The hidden matrix is written only for diagnostics. Layer 2 consumes the
register-resident `matrix_a` fragment directly; it does not reload that
diagnostic matrix.

Validation includes:

- P3/P4 prerequisite-chain validation;
- generated source tied to exact P3/P4 evidence hashes;
- one Wave32 and one GPU kernel launch;
- nontrivial input and weight FP16 quantization;
- both ReLU branches;
- nontrivial hidden FP32→FP16 rounding;
- bitwise hidden-matrix comparison;
- final FP32 output against an independent CPU-FP64 reference built from the
  exact FP16-rounded inputs, hidden values, and weights;
- guard regions;
- two fresh-process runs with identical JSON and CSV outputs.

This is a correctness proof, not a performance benchmark.

## Install and run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a0_p5_two_layer_fused_forward_bundle.zip

chmod +x \
  scripts/generate_phase4a0_p5_two_layer_fused_source.py \
  scripts/run_phase4a0_p5_two_layer_fused_forward.sh

export P3_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p3_20260725T072430Z"
export P4_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p4_20260725T073910Z"

scripts/run_phase4a0_p5_two_layer_fused_forward.sh
```

Expected final markers:

```text
PHASE4A0_P3_PREREQUISITE: PASS
PHASE4A0_P4_PREREQUISITE: PASS
PHASE4A0_P5_PREREQUISITE_CHAIN: PASS
PHASE4A0_P5_SOURCE_GENERATION: PASS
ROCWMMA_P5_HIDDEN_BITWISE_CORRECTNESS: PASS
ROCWMMA_P5_NO_INTERMEDIATE_GLOBAL_RELOAD: PASS
ROCWMMA_P5_OUTPUT_VS_CPU_FP64: PASS
ROCWMMA_P5_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A0_P5_JSON_AUDIT: PASS
ROCWMMA_P5_MAP_CONTEXT: RECORDED
RDNA4_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: PASS
PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD: PASS
```
