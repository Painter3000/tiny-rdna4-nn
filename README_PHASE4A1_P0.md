# Phase 4A1-P0 — Width-64 fused-MLP tile plan and CPU oracle

Marker:

```text
TCNN_RDNA4_P4A1_P0_WIDTH64_TILE_PLAN_CPU_ORACLE_001
```

This checkpoint is CPU-only. It locks the first Width-64 fused-forward schedule
and produces a deterministic independent reference dataset.

## Locked topology

```text
Batch rows: 16
Input:      64
Hidden 1:   64 + ReLU
Hidden 2:   64 + ReLU
Output:     64
Layers:     3
```

## Planned GPU decomposition

```text
1 block
4 Wave32 waves
128 threads
16×16×16 rocWMMA tiles
```

Each wave owns one 16-column output tile:

```text
wave 0 → columns  0..15
wave 1 → columns 16..31
wave 2 → columns 32..47
wave 3 → columns 48..63
```

Each output tile accumulates four ordered K tiles. The locked schedule therefore
contains:

```text
4 K tiles × 3 layers = 12 mma_sync calls per wave
4 waves × 12 calls   = 48 mma_sync calls per block
```

Hidden tensors use one row-major `16×64` FP16 LDS buffer:

```text
1024 FP16 elements = 2048 bytes
```

Three block barriers make single-buffer reuse safe:

1. publish hidden 1 before layer 2 reads;
2. confirm all hidden-1 reads are finished before overwrite;
3. publish hidden 2 before layer 3 reads.

No intermediate global store or reload is planned.

## CPU oracle

The primary oracle uses scalar CPU-FP64 accumulation from the exact
FP16-rounded inputs and weights. Biases are exact FP32 values promoted to
FP64. Hidden layers apply ReLU followed by explicit IEEE-FP16 quantization.

A separate tiled CPU implementation follows the locked four-K-tile schedule.
Both implementations must produce:

- bitwise-identical FP16 hidden layer 1;
- bitwise-identical FP16 hidden layer 2;
- bitwise-identical FP64 final output.

The evidence contains binary tensors for direct reuse by subsequent GPU probes.

This phase does not claim GPU correctness, occupancy, performance, or
tiny-cuda-nn integration.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a1_p0_width64_tile_plan_cpu_oracle_bundle.zip

chmod +x \
  scripts/phase4a1_p0_width64_tile_plan_and_cpu_oracle.py \
  scripts/run_phase4a1_p0_width64_tile_plan_and_cpu_oracle.sh

export P5_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p5_20260725T075517Z"

scripts/run_phase4a1_p0_width64_tile_plan_and_cpu_oracle.sh
```

Expected final markers:

```text
PHASE4A0_BASELINE_TAG: PASS
PHASE4A0_P5_PREREQUISITE: PASS
PHASE4A1_P0_ORACLE_SELF_TEST: PASS
WIDTH64_TILE_COVERAGE: PASS
WIDTH64_LDS_SINGLE_BUFFER_PLAN: PASS
WIDTH64_SCALAR_VS_TILED_ORACLE: PASS
PHASE4A1_P0_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A1_P0_JSON_AUDIT: PASS
WIDTH64_FUSED_MLP_TILE_PLAN: LOCKED
WIDTH64_CPU_FP64_ORACLE: RECORDED
PHASE4A1_P0_MAP_CONTEXT: RECORDED
PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE: PASS
```
