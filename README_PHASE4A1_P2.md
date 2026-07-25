# Phase 4A1-P2 — Width-64 hidden epilogue and LDS publication

Marker:

```text
TCNN_RDNA4_P4A1_P2_HIDDEN_EPILOGUE_LDS_001
```

This checkpoint combines the proven Width-64 four-K-tile accumulation with the
proven accumulator→matrix-A relay and publishes the complete hidden tensor
through one shared LDS buffer.

## GPU pipeline

```text
16×64 FP16 input
×
64×64 FP16 weight

→ four Wave32 output tiles
→ four ordered K tiles per wave
→ FP32 accumulator
→ FP32 column bias
→ ReLU
→ P3-derived accumulator→matrix-A relay
→ FP32→FP16
→ one shared 16×64 FP16 LDS buffer
→ block barrier
→ cross-wave rotated readback
→ global diagnostic output
```

The shared buffer is:

```text
16 × 64 × FP16 = 2048 bytes
```

Each wave publishes one 16-column tile:

```text
producer wave 0 → columns  0..15
producer wave 1 → columns 16..31
producer wave 2 → columns 32..47
producer wave 3 → columns 48..63
```

After the block barrier, every wave deliberately reads a tile produced by a
different wave:

```text
consumer wave 0 ← producer wave 1
consumer wave 1 ← producer wave 2
consumer wave 2 ← producer wave 3
consumer wave 3 ← producer wave 0
```

This is stronger than having every wave read back its own tile. A bitwise match
against the P0 hidden-layer oracle therefore proves both publication and
cross-wave LDS visibility.

The generated mapping header is bound to the exact P0, P1, P3, and P4 evidence
hashes. This remains a correctness checkpoint, not a performance benchmark.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a1_p2_hidden_epilogue_lds_publication_bundle.zip

chmod +x \
  scripts/generate_phase4a1_p2_mapping_header.py \
  scripts/finalize_phase4a1_p2.py \
  scripts/run_phase4a1_p2_hidden_epilogue_and_lds_publication.sh

export P0_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p0_20260725T080946Z"
export P1_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p1_20260725T082538Z"
export P3_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p3_20260725T072430Z"
export P4_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p4_20260725T073910Z"

unset P2_EVIDENCE
unset PHASE4A1_P2_EVIDENCE

scripts/run_phase4a1_p2_hidden_epilogue_and_lds_publication.sh
```

Expected final markers:

```text
PHASE4A1_P0_PREREQUISITE: PASS
PHASE4A1_P1_PREREQUISITE: PASS
PHASE4A0_P3_MAP_PREREQUISITE: PASS
PHASE4A0_P4_RELAY_PREREQUISITE: PASS
PHASE4A1_P2_INPUT_HASHES: VERIFIED
PHASE4A1_P2_MAPPING_HEADER_GENERATION: PASS
WIDTH64_HIDDEN_EPILOGUE: PASS
WIDTH64_ACCUMULATOR_TO_A_RELAY: PASS
WIDTH64_LDS_ALL_FOUR_WAVES_PUBLICATION: PASS
WIDTH64_CROSS_WAVE_ROTATED_READBACK: PASS
WIDTH64_LDS_HIDDEN_BITWISE_CORRECTNESS: PASS
WIDTH64_LDS_BARRIER_VISIBILITY: PASS
WIDTH64_HIDDEN_LDS_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A1_P2_JSON_AUDIT: PASS
PHASE4A1_P2_MAP_CONTEXT: RECORDED
PHASE4A1_P2_WIDTH64_HIDDEN_LDS: PASS
```


## Evidence-path safety

The runner uses only `PHASE4A1_P2_EVIDENCE`; it deliberately ignores the older
generic `P2_EVIDENCE` variable. It also refuses:

- evidence paths whose basename does not begin with `phase4a1_p2_`;
- non-empty evidence directories.

These checks prevent a later phase from overwriting Phase 4A0-P2 evidence.
