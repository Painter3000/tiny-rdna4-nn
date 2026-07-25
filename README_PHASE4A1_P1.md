# Phase 4A1-P1 — Width-64 single-layer four-K-tile accumulation

Marker:

```text
TCNN_RDNA4_P4A1_P1_WIDTH64_FOUR_K_TILE_001
```

This is the first Width-64 rocWMMA GPU checkpoint.

## GPU contract

```text
Input:   16×64 FP16 row-major
Weight:  64×64 FP16 column-major
Output:  16×64 FP32
Block:   4 Wave32 waves / 128 threads
Kernel launches: 1
```

Each wave owns one output-column tile:

```text
wave 0 → columns  0..15
wave 1 → columns 16..31
wave 2 → columns 32..47
wave 3 → columns 48..63
```

Each wave keeps one FP32 accumulator fragment alive while executing four
ordered `16×16×16` `mma_sync` operations.

For diagnostic strength, the kernel stores the accumulator after every K tile:

```text
stage 1 → 16 K terms
stage 2 → 32 K terms
stage 3 → 48 K terms
stage 4 → 64 K terms
```

All four partial matrices are independently checked against a CPU-FP64
reference built from the exact P0 FP16 input and weight binaries.

The final stage validates the complete `16×64 × 64×64` multiplication.

This phase does not include bias, activation, LDS hidden storage, another MLP
layer, or performance claims.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a1_p1_width64_single_layer_bundle.zip

chmod +x \
  scripts/finalize_phase4a1_p1.py \
  scripts/run_phase4a1_p1_width64_single_layer.sh

export P0_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p0_20260725T080946Z"

scripts/run_phase4a1_p1_width64_single_layer.sh
```

Expected final markers:

```text
PHASE4A1_P0_PREREQUISITE: PASS
PHASE4A1_P1_INPUT_HASHES: VERIFIED
WIDTH64_K_TILE_STAGE_1: PASS
WIDTH64_K_TILE_STAGE_2: PASS
WIDTH64_K_TILE_STAGE_3: PASS
WIDTH64_K_TILE_STAGE_4: PASS
WIDTH64_FOUR_K_TILE_ACCUMULATION: PASS
WIDTH64_ALL_FOUR_WAVES_OUTPUT_COVERAGE: PASS
WIDTH64_SINGLE_LAYER_VS_CPU_FP64: PASS
WIDTH64_SINGLE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A1_P1_JSON_AUDIT: PASS
PHASE4A1_P1_MAP_CONTEXT: RECORDED
PHASE4A1_P1_WIDTH64_SINGLE_LAYER: PASS
```
