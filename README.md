# tiny-rdna4-nn

A community port of [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for **AMD RDNA4 / `gfx1201`**, developed and validated on the **AMD Radeon AI PRO R9700** with **ROCm 7.2**.

This repository is not a CUDA build and does not require an NVIDIA GPU. The validated path is the PyTorch extension compiled with HIP for AMD RDNA4.

> **Project status**
>
> - **Phase 3B1:** qualified FP16 forward, backward, training, `NetworkWithInputEncoding`, and reproducible performance validation.
> - **Phase 4A2:** qualified opt-in `RocWMMAWidth64MLP` production **inference** path for AMD RDNA4 / `gfx1201` (forward only, no performance claim yet).
>
> **Validated tags:**
> - [`phase3b1-fp16-gfx1201-rocm72-pass`](https://github.com/Painter3000/tiny-rdna4-nn/releases/tag/phase3b1-fp16-gfx1201-rocm72-pass)
> - [`phase4a2-width64-production-inference-gfx1201-pass`](https://github.com/Painter3000/tiny-rdna4-nn/releases/tag/phase4a2-width64-production-inference-gfx1201-pass)

## Scope

The current port focuses on:

- AMD RDNA4 architecture `gfx1201`
- Radeon AI PRO R9700
- ROCm 7.2
- Linux
- PyTorch ROCm extension
- FP32 portable network backend
- hipBLASLt FP32 and FP16 MLP backends
- an opt-in rocWMMA Width-64 fused inference backend (`RocWMMAWidth64MLP`)
- forward, backward, and Adam training (portable and hipBLASLt backends)
- standalone encodings and `NetworkWithInputEncoding`
- deterministic correctness validation across the qualified backends
- reproducible performance validation for the Phase 3B1 hipBLASLt FP16 path

Other AMD architectures may require additional work and are not claimed as validated by this repository.

## Validated environment

The qualification work was performed with:

- Ubuntu 24.04
- AMD Radeon AI PRO R9700
- GPU architecture: `gfx1201`
- ROCm 7.2
- Python 3.12
- PyTorch 2.13.0 with ROCm 7.2

## Performance on Radeon AI PRO R9700

Phase 3B1-F1 measured 24 cases using 72 fresh processes and five paired FP16/FP32 rounds per process. All 24 cases and all 72 primary processes were valid, with correctness passing before and after measurement.

| Category | FP16 speedup vs. FP32 | FP32/FP16 peak-memory factor |
|---|---:|---:|
| Overall | **2.2800x** | **1.0418x** |
| Latency regime, batch 1024 | 0.9968x | 1.0368x |
| Throughput regime, batch 16384 | **5.2155x** | 1.0468x |
| Network only | **2.1658x** | 1.0310x |
| Network with input encoding | **2.4003x** | 1.0527x |

Small batch cases are largely launch-overhead dominated and are neutral in aggregate. Throughput-oriented cases show the main FP16 benefit. Results vary by topology; the full report includes every case and process.

> These figures describe the Phase 3B1 hipBLASLt FP16 path. The Phase 4A2 `RocWMMAWidth64MLP` backend makes **no performance claim** yet; only its functional inference correctness and code-object resources are published.

See:

- [Phase 3B1-F1 reproducible performance report](phase3b1_reports/PHASE3B1F1_REPRODUCIBLE_PERFORMANCE.md)
- [Machine-readable F1 results](phase3b1_reports/phase3b1f1_reproducible_performance.json)
- [Raw-data index](phase3b1_reports/phase3b1f1_raw_index.jsonl)

## Installation

### Requirements

- Linux
- ROCm 7.2 installed, normally under `/opt/rocm`
- ROCm-enabled PyTorch
- Python 3.12
- C++17-capable host toolchain
- Git and CMake

Clone the repository with its submodules:

```bash
git clone --recursive https://github.com/Painter3000/tiny-rdna4-nn.git
cd tiny-rdna4-nn
```

Activate the Python environment containing the ROCm-enabled PyTorch build, then compile the extension:

```bash
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1201

cd bindings/torch
python -m pip install --no-build-isolation .
```

The build intentionally requires `PYTORCH_ROCM_ARCH=gfx1201` for the validated RDNA4 path.

### Optional: opt-in rocWMMA Width-64 inference backend

The `RocWMMAWidth64MLP` backend is **disabled by default**. It is only compiled into the extension when explicitly requested at build time. With the switch off, the backend source is not compiled and carries no build overhead.

```bash
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1201
export TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1
export TCNN_HALF_PRECISION=1

cd bindings/torch
python -m pip install --no-build-isolation .
```

## Quick verification

PyTorch continues to expose ROCm devices through its `torch.cuda` compatibility namespace:

```bash
python - <<'PY'
import torch
import tinycudann as tcnn

assert torch.version.hip is not None, "This PyTorch build has no ROCm support"
assert torch.cuda.is_available(), "No ROCm GPU is visible to PyTorch"

print("PyTorch:", torch.__version__)
print("ROCm/HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("Architecture:", torch.cuda.get_device_properties(0).gcnArchName)
print("tinycudann:", tcnn.__file__)
print("tinycudann import: PASS")
PY
```

### Expected GPU memory arena warning

On the validated Radeon AI PRO R9700 / ROCm 7.2 setup, model creation may print:

```text
tiny-cuda-nn warning: GPUMemoryArena: GPU 0 does not support virtual memory.
Falling back to regular allocations, which will be larger and can cause
occasional stutter.
```

This warning is **expected on the currently qualified setup** and does not indicate a failed installation. The tiny-cuda-nn GPU memory arena does not obtain a usable virtual memory path on this platform.

Treat the warning as non-fatal when the verification test completes successfully and all functional PASS markers appear.

## Reproducible fresh-clone smoke

The repository includes a versioned end-to-end smoke test for the qualified ROCm / `gfx1201` path:

- `scripts/fresh_clone_user_smoke.sh`
- `scripts/fresh_clone_user_smoke.py`

The full smoke requires:

- a clean recursive checkout,
- an activated Python virtual environment,
- ROCm-enabled PyTorch already installed,
- no existing `tinycudann` installation in that environment.

Run the complete backend validation with:

```bash
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1201
export MAX_JOBS=1

scripts/fresh_clone_user_smoke.sh --all-backends
```

To store the complete evidence outside the repository:

```bash
EVIDENCE="$HOME/tiny-rdna4-nn-evidence/$(date -u +%Y%m%dT%H%M%SZ)"

scripts/fresh_clone_user_smoke.sh \
  --all-backends \
  --evidence-dir "$EVIDENCE"
```

The test performs:

- clean-package precondition checks,
- wheel build and installation,
- Python and native-module provenance checks,
- `HashGrid + PortableMLP` forward, backward, and Adam training,
- checkpoint reload in a second Python process,
- explicit validation of `PortableMLP`, `HipBLASLtMLP`, and `HipBLASLtMLPFP16`,
- rejection of `FullyFusedMLP` rather than silently aliasing it,
- native ROCm/PyTorch library resolution,
- repository and recursive-submodule cleanliness checks,
- machine-readable JSON evidence generation.

The expected `GPUMemoryArena` warning is retained in the log and classified as non-fatal when all functional checks pass.

For a runtime-only check of an already installed build:

```bash
scripts/fresh_clone_user_smoke.sh --runtime-only --all-backends
```

The smoke is a correctness and reproducibility test, not a performance benchmark.

The standard smoke validates the portable and hipBLASLt backends. It does **not** cover the default-off `RocWMMAWidth64MLP` backend, which requires an explicit build switch and supports inference only.

## Phase 4A2: rocWMMA Width-64 opt-in inference backend

`RocWMMAWidth64MLP` is a fused three-linear-layer rocWMMA inference kernel for `gfx1201`, validated as an explicit opt-in production path. It is deliberately narrow and honest about what it does and does not do.

### What is qualified

- Explicit **opt-in only**; there is no automatic selection and no silent fallback.
- Architecture `gfx1201`, Wave32.
- FP16 parameters and FP16 output; consumes the normal tiny-cuda-nn parameter buffer. The qualified topology uses exactly 12,480 FP16 parameter values, corresponding to 24,960 bytes.
- Exact topology `64 -> 64 -> 64 -> 64` with two ReLU hidden layers (three linear layers).
- **Inference (forward) only.** Backward and training are intentionally fail-closed.
- Deterministic correctness validated against a **CPU FP32 reference**, plus 64 bitwise-identical launch repeats, prefix invariance, parameter hot-swap, and dual-stream model isolation.
- Public batch sizes are internally padded to the required tile boundaries. Production inference (Phase 4A2-P2) was validated across the batch sizes `1, 16, 17, 255, 256, 257` against the CPU reference (FP32 matmul with FP16 layer boundaries); the separate runtime-lifecycle stage (Phase 4A2-P3) exercised a 20-case batch matrix and padding boundaries from 256 to 1024.
- The linked production kernel was bound by an ISA/code-object audit at the release commit, and the full runtime matrix was replayed twice, byte-identically, from that commit.

### Minimal usage

```python
import torch
import tinycudann as tcnn

model = tcnn.Network(
    n_input_dims=64,
    n_output_dims=64,
    network_config={
        "otype": "RocWMMAWidth64MLP",
        "precision": "Fp16",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    },
).to("cuda").eval()

# FP16 input; non-multiple-of-16 batch is padded internally.
x = torch.randn(257, 64, device="cuda", dtype=torch.float16)

with torch.inference_mode():
    y = model(x)

assert y.shape == (257, 64)
assert y.dtype == torch.float16
assert torch.isfinite(y).all()
print("RocWMMAWidth64MLP inference: PASS")
```

Requires the extension to have been built with `TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1` (see the opt-in build section). Without the switch, the name is rejected rather than aliased. The network config must also set `"precision": "Fp16"`; the backend does not select a precision implicitly and rejects the request otherwise (fail-closed).

### Audited code-object resources

The Phase 4A2-P4 audit recorded the following facts for the linked production kernel on `gfx1201`:

```text
Group segment (LDS):   2048 bytes
Private segment:          0 bytes
Scratch instructions:     0
MFMA/WMMA instructions:  12
ds_load_b128:             8
ds_store_b128:            2
ds_bpermute_b32:        192
Block barriers:           6
VGPR field:              92
SGPR field:              74
```

> **Claim boundary.** These are recorded resource and instruction facts, not a performance result. Register values alone do not justify an occupancy or throughput claim, and the ISA audit does not contain a performance hypothesis. The measured fragment/register mapping used during development is a version-bound diagnostic for ROCm 7.2 / rocWMMA on `gfx1201` and is treated as `LAYOUT_STABILITY: NOT_GUARANTEED_BY_ROCWMMA_API`; the kernel relies on official rocWMMA accesses rather than hard-wiring that mapping as a constant.

## Python API

The Python package and import name remain `tinycudann`. The high-level model classes are kept where practical, but network backend names are selected explicitly on the ROCm path. NVIDIA backend names are not aliased to AMD backends.

The conservative FP32 reference path uses `PortableMLP`:

```python
import torch
import tinycudann as tcnn

model = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=4,
    encoding_config={
        "otype": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 2,
        "log2_hashmap_size": 12,
        "base_resolution": 4,
        "per_level_scale": 2.0,
    },
    network_config={
        "otype": "PortableMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 16,
        "n_hidden_layers": 1,
    },
).to("cuda")

x = torch.rand(256, 3, device="cuda", requires_grad=True)
y = model(x)
loss = y.float().square().mean()
loss.backward()
torch.cuda.synchronize()

assert torch.isfinite(y).all()
assert x.grad is not None and torch.isfinite(x.grad).all()
print("PortableMLP forward/backward: PASS")
```

### ROCm network backend selection

| `otype` | Precision | Training | Purpose |
|---|---|---|---|
| `PortableMLP` | FP32 | yes | Portable correctness-first reference backend |
| `HipBLASLtMLP` | FP32 | yes | Explicit accelerated AMD hipBLASLt backend |
| `HipBLASLtMLPFP16` | FP16 | yes | Explicit qualified FP16 backend; requires `"precision": "Fp16"` |
| `RocWMMAWidth64MLP` | FP16 | no (inference only) | Opt-in fused rocWMMA backend; `gfx1201`/Wave32, exact `64->64->64->64`, disabled by default. Requires `"precision": "Fp16"` |
| `FullyFusedMLP` | — | — | Not implemented on the qualified ROCm path and intentionally rejected rather than aliased |

Example FP16 network configuration:

```python
network_config = {
    "otype": "HipBLASLtMLPFP16",
    "precision": "Fp16",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2,
}
```

`MLP`, `CutlassMLP`, `FullyFusedMLP`, and `MegakernelMLP` are NVIDIA-oriented backend names in upstream tiny-cuda-nn. They are deliberately not treated as aliases for the AMD backends because the implementations differ substantially and silent aliasing invites confusion.

## Important differences from upstream tiny-cuda-nn

- The validated build target is HIP/ROCm, not CUDA.
- `gfx1201` is the currently qualified architecture.
- NVIDIA CUTLASS, FullyFusedMLP, CUDA RTC, and CUDA JIT fusion are not part of the qualified ROCm path.
- A fused AMD path does exist as the opt-in, inference-only `RocWMMAWidth64MLP` backend, but it is disabled by default and is not a replacement for `FullyFusedMLP`.
- AMD network backends must be selected explicitly as `PortableMLP`, `HipBLASLtMLP`, `HipBLASLtMLPFP16`, or (opt-in) `RocWMMAWidth64MLP`.
- The root native CMake project still contains substantial upstream CUDA-oriented infrastructure. The supported and validated RDNA4 build path is currently `bindings/torch`.
- File paths, C++ namespaces, header paths such as `tiny-cuda-nn/...`, and the Python import `tinycudann` remain unchanged where practical to preserve source compatibility.

## Validation status

The following functional blocks have completed their audited candidate qualification:

- FP16 forward
- FP16 backward
- FP16 training and loss scaling
- checkpoint/resume behavior
- `NetworkWithInputEncoding`
- deterministic numerical comparison
- reproducible FP16 performance measurement

The Phase 3B1 PASS tag points to commit:

```text
2d7087c03442c66f8c4b6491c111e32cae2b40de
```

The versioned fresh-clone smoke was validated successfully against commit:

```text
49c08d0cd5ec4b5078c09d36a6016f0cbf659538
```

These are two distinct records: the PASS tag documents the frozen performance/correctness state, while the smoke commit documents the publicly available reproducibility tooling.

### Phase 4A2 rocWMMA Width-64 inference

The opt-in `RocWMMAWidth64MLP` inference path has completed its audited qualification:

- opt-in, default-off build switch with fail-closed factory behavior
- exact `64 -> 64 -> 64 -> 64` topology, two ReLU hidden layers, FP16
- production inference correctness against a CPU FP32 reference
- runtime lifecycle: multi-batch, padding boundaries, 64 bitwise repeats, prefix invariance, parameter hot-swap, dual-stream model isolation
- training-mode execution is unsupported and fails closed; the qualified runtime matrix explicitly validates training-forward rejection, and backward/training is not implemented
- ISA/code-object audit of the linked production kernel with a byte-identical runtime replay

The Phase 4A2 PASS tag points to commit:

```text
cd1330a21452f7e2edab9e676567b7a040f922bc
```

## Current limitations

- Only `gfx1201` on the Radeon AI PRO R9700 has completed the full qualification described here.
- Native C++ example and benchmark workflows from the upstream CUDA README are not yet the recommended RDNA4 entry point.
- `FullyFusedMLP`, CUDA RTC, and CUDA JIT fusion are unavailable on the ROCm path.
- The `RocWMMAWidth64MLP` backend is inference-only (no backward/training), fixed to the exact `64 -> 64 -> 64 -> 64` topology, opt-in and disabled by default, and carries no performance claim yet.
- This port does not claim support for NVIDIA GPUs or for all ROCm-capable AMD architectures.
- Performance depends strongly on batch size and topology; latency-bound workloads may see little or no FP16 speedup.
- The validated ROCm 7.2 setup currently uses the regular-allocation fallback because the tiny-cuda-nn GPU memory arena does not obtain a usable virtual memory path. This results in higher memory usage than the virtual-allocation path would achieve.

## Upstream project and attribution

This project is derived from [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) by Thomas Müller and contributors.

`tiny-rdna4-nn` is an independent community port. It is not affiliated with or endorsed by NVIDIA or AMD.

The original copyright notices and BSD 3-Clause license are retained. See [LICENSE.txt](LICENSE.txt).
