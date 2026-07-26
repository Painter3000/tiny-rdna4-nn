# Phase 4A2-P0 — Width-64 production integration contract

Marker:

```text
TCNN_RDNA4_P4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_001
```

This checkpoint does not modify a production kernel. It binds the completed
Phase 4A1 tag and P5 evidence, inventories the current tiny-cuda-nn production
integration surfaces, and locks the fail-closed contract for the first
production backend.

## Locked backend identity

```text
JSON otype:        RocWMMAWidth64MLP
C++ class:         RocWMMAWidth64MLP
build option:      TCNN_ENABLE_ROCWMMA_WIDTH64_MLP
compile define:    TCNN_WITH_ROCWMMA_WIDTH64_MLP
default:           OFF
selection:         explicit otype only
silent fallback:   forbidden
initial scope:     inference only
```

Existing `PortableMLP`, `HipBLASLtMLPFP16`, `FullyFusedMLP` and `CutlassMLP`
paths remain unchanged.

## Exact first-production eligibility

```text
HIP AMD
runtime architecture gfx1201
Network<__half>
input width 64
hidden width 64
output width 64
2 hidden layers
hidden activation ReLU
output activation None
bias enabled
batch rows >= 16 and divisible by 16
ColumnMajor GPUMatrix [64][batch]
caller-supplied hipStream_t
```

Every unsupported request fails closed. Automatic backend selection and
training/backward are outside this checkpoint.

## Parameter ABI

Each layer is stored as `[weights, bias]` in the normal network parameter
buffer:

```text
W0 4096 FP16 + b0 64 FP16
W1 4096 FP16 + b1 64 FP16
W2 4096 FP16 + b2 64 FP16
total = 12,480 FP16 elements
```

Weights remain column-major. FP16 biases are promoted to FP32 in the epilogue.

## Critical bridge from P4

P4 is a validated standalone kernel, not yet a drop-in network backend. The
contract explicitly records six required adaptations:

1. fixed 16-row launch → one block per 16 batch rows;
2. P4 row-major `[batch][64]` → byte-equivalent ColumnMajor `[64][batch]`;
3. separate FP32 biases → FP16 network biases promoted to FP32;
4. FP32 validation output → public FP16 `Network<__half>` output;
5. oracle pointers and diagnostic counters → removed;
6. standalone launch → caller stream without implicit host synchronization.

No P4 performance claim is inherited.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a2_p0_width64_production_integration_contract_bundle.zip

chmod +x \
  scripts/prepare_phase4a2_p0.py \
  scripts/finalize_phase4a2_p0.py \
  scripts/run_phase4a2_p0_production_integration_contract.sh

export PHASE4A1_P5_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p5_20260725T154122Z"

unset PHASE4A2_P0_EVIDENCE

scripts/run_phase4a2_p0_production_integration_contract.sh
```

## Expected final markers

```text
PHASE4A1_PASS_TAG_BOUND: PASS
PHASE4A1_P5_EVIDENCE_BOUND: PASS
WIDTH64_PRODUCTION_SURFACES_INVENTORIED: PASS
WIDTH64_PUBLIC_OTYPE_COLLISION_FREE: PASS
WIDTH64_OPT_IN_FAIL_CLOSED_CONTRACT: PASS
WIDTH64_NETWORK_ABI_CONTRACT: PASS
WIDTH64_PARAMETER_LAYOUT_CONTRACT: PASS
WIDTH64_BATCH_LAYOUT_CONTRACT: PASS
WIDTH64_INFERENCE_ONLY_SCOPE_LOCKED: PASS
WIDTH64_EXISTING_BACKENDS_UNCHANGED_CONTRACT: PASS
PHASE4A2_P0_FRESH_PROCESS_REPRODUCIBILITY: PASS
PHASE4A2_P0_JSON_AUDIT: PASS
PHASE4A2_P0_MAP_CONTEXT: RECORDED
PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT: PASS
```

The runner accepts only evidence directories beginning with `phase4a2_p0_`
and refuses to overwrite a non-empty destination.
