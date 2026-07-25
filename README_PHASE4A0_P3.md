# Phase 4A0-P3 — Fragment-map interpretation and role equivalence

Marker:

```text
TCNN_RDNA4_P4A0_P3_FRAGMENT_MAP_INTERPRETATION_001
```

This phase consumes the qualified P2 map and performs no GPU compilation or
kernel launch.

It derives:

- exact `(lane, register-file row) → (matrix row, matrix col)` models for
  `matrix_a`, `matrix_b`, and the FP32 accumulator;
- compact GF(2)-affine bit formulas when the complete 256-element map permits
  them;
- generated C++ coordinate-decoder expressions;
- exact inverse coordinate-to-lane/slot tables;
- pairwise role classifications for A→B, A→accumulator, and B→accumulator;
- coordinate-preserving reindex permutations and cycle statistics;
- an explicit fused-kernel policy stating which same-slot cross-role
  operations are safe and which require reindexing.

The analysis describes `rocwmma::to_register_file` geometry. It does **not**
claim physical VGPR numbers and must not be treated as a stable ABI.

## Prerequisite

Phase 4A0-P2 must have passed. By default, the runner selects the newest
`phase4a0_p2_*` evidence directory containing a valid consolidated JSON file.

The known qualified evidence can be supplied explicitly:

```bash
export P2_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p2_20260725T071730Z"
```

## Install and run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a0_p3_fragment_map_interpretation_bundle.zip

chmod +x \
  scripts/analyze_phase4a0_p3_fragment_maps.py \
  scripts/run_phase4a0_p3_fragment_map_interpretation.sh

scripts/run_phase4a0_p3_fragment_map_interpretation.sh
```

Expected final markers:

```text
PHASE4A0_P2_PREREQUISITE: PASS
PHASE4A0_P3_ANALYZER_SELF_TEST: PASS
ROCWMMA_P3_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A0_P3_JSON_AUDIT: PASS
ROCWMMA_P3_MAP_CONTEXT: RECORDED
PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION: PASS
```

The analyzer also prints whether all role and pairwise transformations have
exact GF(2)-affine formulas or require a complete lookup-table fallback.

Evidence contains:

```text
phase4a0_p3_analysis.json
PHASE4A0_P3_ANALYSIS.md
fragment_role_slot_maps.csv
fragment_role_reindex_tables.csv
SHA256SUMS
```
