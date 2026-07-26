# Phase 4A2-P3 — Runtime integration and lifecycle closure

Marker:

```text
TCNN_RDNA4_P4A2_P3_RUNTIME_INTEGRATION_CLOSURE_001
```

P3 changes no production source and performs no rebuild. It reuses the exact
successful P2 extension and closes runtime integration, lifecycle, stream, and
model-isolation risks before the final production code-object audit.

## Bound P2 baseline

```text
Commit:
204a807

Production source SHA-256:
7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4

Mapping-header SHA-256:
f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517
```

The extension under `phase4a2_p2_20260725T174455Z/build/lib` is reused
byte-for-byte.

## Runtime matrix

Public batch sizes:

```text
1, 15, 16, 17,
31, 32, 33,
63, 64, 65,
127, 128, 129,
255, 256, 257,
511, 512, 513,
1023
```

This exercises internal padding to:

```text
256, 512, 768, 1024
```

Each batch is checked against the CPU FP32 reference and run twice for bitwise
repeatability.

## Lifecycle closure

```text
64 repeated launches at batch 257
prefix invariance across six tile boundaries
parameter hot swap A -> B -> A
two independent model instances
two concurrent non-default streams
post-concurrency model replay
PortableMLP factory construction
HipBLASLtMLPFP16 factory construction
training-forward fail-closed behavior
two byte-identical fresh process results
```

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a2_p3_width64_runtime_integration_lifecycle_closure_bundle.zip

chmod +x \
  probes/phase4a2_p3_runtime_integration_probe.py \
  scripts/finalize_phase4a2_p3.py \
  scripts/run_phase4a2_p3_runtime_integration_closure.sh

export PHASE4A2_P2_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a2_p2_20260725T174455Z"

unset PHASE4A2_P3_EVIDENCE

scripts/run_phase4a2_p3_runtime_integration_closure.sh
```

## Expected final markers

```text
WIDTH64_P2_EXTENSION_REUSED: PASS
WIDTH64_RUNTIME_MATRIX_20_BATCHES: PASS
WIDTH64_RUNTIME_PADDING_256_1024: PASS
WIDTH64_RUNTIME_64_LAUNCH_BITWISE: PASS
WIDTH64_RUNTIME_PARAMETER_HOT_SWAP: PASS
WIDTH64_RUNTIME_DUAL_STREAM_MODEL_ISOLATION: PASS
WIDTH64_RUNTIME_EXISTING_FACTORIES: PASS
WIDTH64_RUNTIME_FRESH_PROCESS_REPRODUCIBILITY: PASS
WIDTH64_P3_NO_PRODUCTION_CODE_CHANGE: PASS
PHASE4A2_P3_JSON_AUDIT: PASS
PHASE4A2_P3_MAP_CONTEXT: RECORDED
PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE: PASS
```

## Deliberate boundary

P3 makes no performance, occupancy, spill, resource, or production-ISA claim.
The exact P2 production source remains unchanged. Phase 4A2-P4 will perform the
fresh production rebuild, code-object extraction, ISA/resource audit, and
replay.
