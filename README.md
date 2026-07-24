# tiny-rdna4-nn

A community port of [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for **AMD RDNA4 / `gfx1201`**, developed and validated on the **AMD Radeon AI PRO R9700** with **ROCm 7.2**.

This repository is not a CUDA build and does not require an NVIDIA GPU. The validated path is the PyTorch extension compiled with HIP for AMD RDNA4.

> **Project status:** Phase 3B1 FP16 correctness, training, `NetworkWithInputEncoding`, and reproducible performance qualification passed.
>
> **Validated tag:** [`phase3b1-fp16-gfx1201-rocm72-pass`](https://github.com/Painter3000/tiny-rdna4-nn/releases/tag/phase3b1-fp16-gfx1201-rocm72-pass)

## Scope

The current port focuses on:

- AMD RDNA4 architecture `gfx1201`
- Radeon AI PRO R9700
- ROCm 7.2
- Linux
- PyTorch ROCm extension
- FP32 portable network backend
- hipBLASLt FP32 and FP16 MLP backends
- forward, backward, and Adam training
- standalone encodings and `NetworkWithInputEncoding`
- deterministic correctness and reproducible performance validation

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

Phase 3B1-F1 measured 24 cases using 72 fresh processes and five paired FP16/FP32 rounds per process. All 24 cases and all 72 primary processes were valid, with correctness passing before and after me[...]

| Category | FP16 speedup vs. FP32 | FP32/FP16 peak-memory factor |
|---|---:|---:|
| Overall | **2.2800x** | **1.0418x** |
| Latency regime, batch 1024 | 0.9968x | 1.0368x |
| Throughput regime, batch 16384 | **5.2155x** | 1.0468x |
| Network only | **2.1658x** | 1.0310x |
| Network with input encoding | **2.4003x** | 1.0527x |

Small batch cases are largely launch-overhead dominated and are neutral in aggregate. Throughput-oriented cases show the main FP16 benefit. Results vary by topology; the full report includes every cas[...]

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

This warning is **expected on the currently qualified setup** and does not indicate a failed installation. The tiny-cuda-nn GPU memory arena does not obtain a usable virtual memory path on this platform, so the runtime falls back to regular GPU allocations. Forward, backward, training, and checkpoint operations remain fully functional, but memory usage may be higher and allocation-related pauses may occasionally occur.

Treat the warning as non-fatal when the verification test completes successfully and all functional PASS markers appear.

## Python API

The Python package and import name remain `tinycudann`. The high-level model classes are kept where practical, but network backend names are selected explicitly on the ROCm path. NVIDIA backend names [...]

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

| `otype` | Precision | Purpose |
|---|---|---|
| `PortableMLP` | FP32 | Portable correctness-first reference backend |
| `HipBLASLtMLP` | FP32 | Explicit accelerated AMD hipBLASLt backend |
| `HipBLASLtMLPFP16` | FP16 | Explicit qualified FP16 backend; requires `"precision": "Fp16"` |
| `FullyFusedMLP` | — | Not implemented on the qualified ROCm path and intentionally rejected rather than aliased |

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

`MLP`, `CutlassMLP`, `FullyFusedMLP`, and `MegakernelMLP` are NVIDIA-oriented backend names in upstream tiny-cuda-nn. They are deliberately not treated as aliases for the AMD backends because the impl[...]

## Important differences from upstream tiny-cuda-nn

- The validated build target is HIP/ROCm, not CUDA.
- `gfx1201` is the currently qualified architecture.
- NVIDIA CUTLASS, FullyFusedMLP, CUDA RTC, and CUDA JIT fusion are not part of the qualified ROCm path.
- AMD network backends must be selected explicitly as `PortableMLP`, `HipBLASLtMLP`, or `HipBLASLtMLPFP16`.
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

The public PASS tag points to commit:

```text
2d7087c03442c66f8c4b6491c111e32cae2b40de
```

A separate fresh-clone user validation of `main` confirmed recursive cloning, wheel build and installation in an independent Python environment, package and native-module provenance, ROCm library reso[...]

## Current limitations

- Only `gfx1201` on the Radeon AI PRO R9700 has completed the full qualification described here.
- Native C++ example and benchmark workflows from the upstream CUDA README are not yet the recommended RDNA4 entry point.
- `FullyFusedMLP`, CUDA RTC, and CUDA JIT fusion are unavailable on the ROCm path.
- This port does not claim support for NVIDIA GPUs or for all ROCm-capable AMD architectures.
- Performance depends strongly on batch size and topology; latency-bound workloads may see little or no FP16 speedup.
- The validated ROCm 7.2 setup currently uses the regular-allocation fallback because the tiny-cuda-nn GPU memory arena does not obtain a usable virtual memory path. This results in higher memory usage and occasional allocation-related stutter, but does not affect correctness or training functionality.

## Upstream project and attribution

This project is derived from [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) by Thomas Müller and contributors.

`tiny-rdna4-nn` is an independent community port. It is not affiliated with or endorsed by NVIDIA or AMD.

The original copyright notices and BSD 3-Clause license are retained. See [LICENSE.txt](LICENSE.txt).
