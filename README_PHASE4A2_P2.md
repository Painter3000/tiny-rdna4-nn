# Phase 4A2-P2 — Width-64 production inference and parameter ABI

Marker:

```text
TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001
```

P2 installs the first qualified production inference kernel behind the
explicit `RocWMMAWidth64MLP` opt-in path.

## Bound baseline

```text
Phase 4A2-P1 commit:
c1b95f6

Phase 4A1-P4 source SHA-256:
54e03ee731046bb007d0c554c6d1e6ec2dea99d4f4fe150bb60563e36c4b3382

Validated mapping-header SHA-256:
f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517
```

The validated P4 mapping header is copied byte-for-byte into:

```text
include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h
```

## Production adaptations

```text
P4 fixed [16][64] input
→ one block per 16 samples

P4 separate weights and FP32 biases
→ normal FP16 parameter buffer:
   W0,b0,W1,b1,W2,b2

P4 final FP32 validation output
→ public FP16 output

P4 oracle pointers and diagnostics
→ removed

P4 standalone launch
→ caller-supplied HIP stream
→ no host synchronization
```

The kernel retains:

```text
4 Wave32 waves
128 threads per block
4 K tiles per layer
12 rocWMMA operations per wave
one reused 2,048-byte LDS buffer
three source barriers
ReLU after hidden layers
no output activation
```

## Parameter ABI

```text
W0: 4096 FP16, column-major
b0:   64 FP16
W1: 4096 FP16, column-major
b1:   64 FP16
W2: 4096 FP16, column-major
b2:   64 FP16

Total: 12,480 FP16 elements
```

Bias values are promoted to FP32 inside each epilogue.

## Public correctness probe

The Python entry point is tested at batch sizes:

```text
1, 16, 17, 255, 256, 257
```

The existing Python padding contract therefore exercises internal 256- and
512-sample launches. Each result is compared with a CPU FP32 reference that
retains FP16 input, parameter, hidden-boundary, and final-output casts.

Additional gates cover:

```text
bitwise repeatability
tile-prefix invariance across batch boundaries
non-default HIP stream execution
FP16 public output
positive and negative output values
training-forward fail-closed behavior
two fresh-process result identity
```

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a2_p2_width64_production_inference_parameter_abi_bundle.zip

chmod +x \
  scripts/apply_phase4a2_p2.py \
  scripts/finalize_phase4a2_p2.py \
  scripts/run_phase4a2_p2_production_inference.sh \
  probes/phase4a2_p2_inference_probe.py

export PHASE4A2_P1_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a2_p1_20260725T170127Z"

export PHASE4A1_P4_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p4_20260725T150928Z"

unset PHASE4A2_P2_EVIDENCE

scripts/run_phase4a2_p2_production_inference.sh
```

Optional build parallelism:

```bash
export MAX_JOBS=4
```

## Expected final markers

```text
WIDTH64_PRODUCTION_KERNEL_INSTALLED: PASS
WIDTH64_P4_SOURCE_AND_MAPPING_BOUND: PASS
WIDTH64_FP16_PARAMETER_ABI_12480: PASS
WIDTH64_COLUMN_MAJOR_BATCH_BRIDGE: PASS
WIDTH64_MULTI_BLOCK_BATCH_CORRECTNESS: PASS
WIDTH64_INFERENCE_VS_CPU_REFERENCE: PASS
WIDTH64_FRESH_PROCESS_REPRODUCIBILITY: PASS
WIDTH64_NONDEFAULT_STREAM_CORRECTNESS: PASS
WIDTH64_TRAINING_BACKWARD_FAIL_CLOSED: PASS
WIDTH64_NO_ORACLE_DIAGNOSTIC_PATH: PASS
PHASE4A2_P2_JSON_AUDIT: PASS
PHASE4A2_P2_MAP_CONTEXT: RECORDED
PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI: PASS
```

## Deliberate boundary

Training forward and backward remain fail-closed. P2 makes no automatic
backend-selection, performance, occupancy, spill, resource, or production-ISA
claim. The production code-object audit remains reserved for Phase 4A2-P4.


## v2 source-gate correction

The first apply attempt failed only in a static source-string gate. The C++
training-forward diagnostic is intentionally split across adjacent string
literals:

```cpp
"training forward is not "
"qualified; use inference/no-grad. No fallback was executed."
```

C++ concatenates these literals at compile time, but the Python source scanner
searched for one contiguous source-text substring. The corrected gate verifies
the exact split fragments plus the explicit no-fallback clause.

The transactional apply restored the P1 source and bridge automatically, and
the payload was not deleted because apply did not complete. A normal rerun is
therefore safe; no manual cleanup or resume mode is required.


## v3 register-file size correction

The first production build reached the new source and exposed two compile-time
assertions that used the wrong rocWMMA fragment property:

```cpp
RegA::num_elements
RegAcc::num_elements
```

For a register-file transform, `num_elements` describes the packed
compatibility/storage representation. The epilogue and generated mapping work
with the unpacked per-lane element view, whose public constexpr API is
`fragment::size()`.

The corrected assertions are:

```cpp
static_assert(RegA::size() == SLOTS);
static_assert(RegAcc::size() == SLOTS);
```

Corrected production-source SHA-256:

```text
030635f88913f57f94a893cc462a0d5b05a1817ec9eb94545948e0c693ec517e
```

The included resume script patches the already-applied source and reuses the
failed P2 evidence directory without repeating the apply step.


## v4 HIP host-pass correction

The v3 compiler output resolved the remaining ambiguity:

```text
expression evaluates to '4 == 8'
2 errors generated when compiling for host
```

The assertions describe the `gfx1201` Wave32 device register mapping. HIP
performs separate host and device compilation passes. The host pass has no
target-specific Wave32 geometry and sees the rocWMMA register-file fallback
with four elements. The device pass for `gfx1201` retains the validated eight
elements per lane.

The assertions are therefore preserved and scoped correctly:

```cpp
#if defined(__HIP_DEVICE_COMPILE__)
static_assert(RegA::size() == SLOTS);
static_assert(RegAcc::size() == SLOTS);
#endif
```

Corrected production-source SHA-256:

```text
e51159d05ecfcb16e5053b4efafd9f8724df773d2c75183ea1c86510d2f6f853
```

Run the same resume script after installing this bundle.


## v5 robust resume correction

The v4 resume script matched an exact comment block. The current worktree had
the same two assertions but a different surrounding comment state after the
earlier resume attempts, so the exact-text patch refused to proceed.

The v5 resume patch is state-tolerant. It locates the unique `RegA` and
`RegAcc` `SLOTS` assertions, accepts either `num_elements` or `size()`, accepts
guarded or unguarded forms, normalizes the block, and then updates the contract
and apply manifest with the actual resulting source SHA-256.

No kernel math, mapping table, parameter ABI, or launch geometry changes.
