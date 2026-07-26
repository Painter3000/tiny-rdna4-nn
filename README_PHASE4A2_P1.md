# Phase 4A2-P1 — Opt-in class, build and factory skeleton

Marker:

```text
TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001
```

P1 introduces the production-facing backend identity and verifies both build
states. It deliberately installs **no production rocWMMA kernel**.

## Added production identity

```text
otype:              RocWMMAWidth64MLP
C++ class:          RocWMMAWidth64MLP final : Network<__half>
build switch:       TCNN_ENABLE_ROCWMMA_WIDTH64_MLP
compile definition: TCNN_WITH_ROCWMMA_WIDTH64_MLP
default:            OFF
```

## Build and factory proof

The runner performs two isolated extension builds.

### Default OFF

- the new source is absent from the compile log;
- the new compile definition is absent;
- an explicit factory request fails with `was not compiled`;
- `PortableMLP` and `HipBLASLtMLPFP16` still construct.

### Explicit ON

- the new source is compiled;
- the dedicated compile definition is present;
- the explicit factory constructs on `gfx1201`;
- the parameter ABI is exactly `12,480` FP16 elements;
- wrong shape, activation, precision, and bias requests fail closed;
- inference and forward fail before any production kernel or fallback;
- two fresh processes produce identical evidence.

## Surface addendum

The validated PyTorch entry point requires two surfaces that P0 did not list
in its planned-surface ledger:

```text
bindings/torch/setup.py
src/cpp_api.cu
```

The addendum records them without changing any P0 semantic gate.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a2_p1_opt_in_class_build_factory_skeleton_bundle.zip

chmod +x \
  scripts/apply_phase4a2_p1.py \
  scripts/finalize_phase4a2_p1.py \
  scripts/run_phase4a2_p1_opt_in_class_build_factory_skeleton.sh \
  probes/phase4a2_p1_factory_probe.py

export PHASE4A2_P0_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a2_p0_20260725T160447Z"

unset PHASE4A2_P1_EVIDENCE

scripts/run_phase4a2_p1_opt_in_class_build_factory_skeleton.sh
```

Optional build parallelism:

```bash
export MAX_JOBS=4
```

## Expected final markers

```text
WIDTH64_DEFAULT_OFF_BUILD: PASS
WIDTH64_DEFAULT_OFF_FACTORY_FAIL_CLOSED: PASS
WIDTH64_EXISTING_FACTORY_REGRESSION: PASS
WIDTH64_EXPLICIT_ON_BUILD: PASS
WIDTH64_EXPLICIT_FACTORY_CONSTRUCTION: PASS
WIDTH64_PARAMETER_ABI_12480: PASS
WIDTH64_INVALID_CONFIG_FAIL_CLOSED: PASS
WIDTH64_INFERENCE_FORWARD_PREKERNEL_FAIL_CLOSED: PASS
WIDTH64_ENABLED_FRESH_PROCESS_REPRODUCIBILITY: PASS
WIDTH64_NO_PRODUCTION_KERNEL_INSTALLED: PASS
PHASE4A2_P1_SURFACE_ADDENDUM: RECORDED
PHASE4A2_P1_JSON_AUDIT: PASS
PHASE4A2_P1_MAP_CONTEXT: RECORDED
PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON: PASS
```

## Deliberate boundary

P1 makes no inference-correctness, training, backward, occupancy, or
performance claim. The class is usable only for construction and ABI
inspection; every execution path fails closed.


## v2 probe correction

The first complete ON build exposed a probe-only mistake. A Python
`tcnn.Network` is represented natively by `NetworkWithInputEncoding`, so the
top-level `hyperparams()` object describes the wrapper. The backend
hyperparameters are nested inside it.

The corrected probe recursively locates the nested object whose `otype` is
`RocWMMAWidth64MLP`. No C++, build switch, factory, ABI, or compiled extension
changed.

The included resume script reuses the already successful OFF and ON builds:

```bash
export PHASE4A2_P1_RESUME_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a2_p1_20260725T170127Z"

scripts/resume_phase4a2_p1_after_probe_fix.sh
```
