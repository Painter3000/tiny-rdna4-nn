# tiny-rdna4-nn

<!-- PAINTER3000_STATUS_BLOCK_START -->
## Repository snapshot

- **Repository type:** Community ROCm port / RDNA4 fused-MLP research branch based on `NVlabs/tiny-cuda-nn`.
- **Target GPU:** AMD Radeon AI PRO R9700.
- **Target architecture:** RDNA4 / `gfx1201`.
- **Target stack:** Ubuntu 24.04, ROCm 7.2, Python 3.12, PyTorch `2.13.0+rocm7.2`, HIP `7.2.53211`, AMD Clang 22.0.0.
- **Validation status:** Phase 4A2 Model-B Public **PASS**; `RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA: PASS`.
- **Qualified release:** `phase4a2-model-b-public-gfx1201-pass`.
- **Upstream base:** `NVlabs/tiny-cuda-nn`, with recursive source/submodule provenance frozen by the qualified release tag.
<!-- PAINTER3000_STATUS_BLOCK_END -->


A community ROCm port of [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for **AMD RDNA4 / `gfx1201`**, developed and validated on the **AMD Radeon AI PRO R9700** with **ROCm 7.2**.

The current public milestone integrates a dedicated **rocWMMA Width-64 fused MLP** with the **AMD `gsplat` image path** and validates real forward, backward, optimization, replay, resume, runtime-attestation, and 100-step execution on RDNA4.

> **Current status:** Phase 4A2 Model-B Public — **PASS**
>
> **Validated release:** [`phase4a2-model-b-public-gfx1201-pass`](https://github.com/Painter3000/tiny-rdna4-nn/releases/tag/phase4a2-model-b-public-gfx1201-pass)
>
> **Overall gate:** `RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA: PASS`

## Qualified source anchor

The Phase-4A2 release freezes the following exact source state:

| Item | Value |
|---|---|
| Annotated tag | `phase4a2-model-b-public-gfx1201-pass` |
| Tag object | `636dfc01673c97c43815c0205f53f5ed937b3249` |
| Commit | `b98bdcc6b2878f6cb6c10a2141e50867cec6d96a` |
| Git tree | `a8ffaaa3f509400c40f6de58e8a74fb047f8e16e` |
| Target architecture | `gfx1201` |

`main` may contain later documentation commits. Use the annotated tag or the release source archive when reproducing the exact qualified source state.

## What Phase 4A2 validates

The public qualification covers:

- recursive source and submodule provenance;
- direct ROCm Clang compilation for `gfx1201`;
- the dedicated rocWMMA Width-64 fused MLP path;
- staged `tinycudann` and AMD `gsplat` runtime imports;
- decoder and rasterizer forward passes;
- decoder and rasterizer backward passes;
- real optimizer updates;
- rejection of CPU fallback;
- 100-step finite-value, loss, signal, render-output, and VRAM-stability gates;
- local replay determinism;
- local resume equivalence;
- fused-backend runtime attestation;
- negative tests for backend activation;
- result-interpretation attestation.

An independent recursive clone from GitHub was also built and executed successfully:

```text
GITHUB_FRESHCLONE_FUNCTIONAL_VALIDATION: PASS
```

## Validated environment

| Component | Qualified value |
|---|---|
| Operating system | Ubuntu 24.04 |
| GPU | AMD Radeon AI PRO R9700 |
| GPU architecture | `gfx1201` |
| ROCm | 7.2 |
| Python | 3.12 |
| PyTorch | `2.13.0+rocm7.2` |
| HIP reported by PyTorch | `7.2.53211` |
| Compiler | AMD Clang 22.0.0 from ROCm 7.2 |

Other AMD GPUs, architectures, ROCm releases, PyTorch releases, and operating systems are not covered by this qualification.

## Quick start: Phase 4A2 image path

### Requirements

- Linux with ROCm 7.2 installed, normally at `/opt/rocm`;
- Python 3.12 in an activated virtual environment;
- PyTorch `2.13.0+rocm7.2`;
- Git, CMake, and a C++17-capable host toolchain;
- an initialized recursive clone;
- an AMD Radeon AI PRO R9700 / `gfx1201` for the GPU demo.

The native build itself does not require a visible GPU. The public demo does.

### 1. Clone recursively

```bash
git clone --recurse-submodules \
  https://github.com/Painter3000/tiny-rdna4-nn.git
cd tiny-rdna4-nn
```

To reproduce the frozen Phase-4A2 source exactly:

```bash
git checkout phase4a2-model-b-public-gfx1201-pass
git submodule update --init --recursive
```

### 2. Verify the ROCm Python environment

Activate the virtual environment containing the qualified ROCm-enabled PyTorch build, then check it:

```bash
python - <<'PY'
import torch

assert torch.__version__ == "2.13.0+rocm7.2"
assert torch.version.hip == "7.2.53211"
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1201"

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("Architecture:", torch.cuda.get_device_properties(0).gcnArchName)
print("RDNA4_ENVIRONMENT: PASS")
PY
```

### 3. Build the native modules

```bash
PHASE4A2_PYTHON="$VIRTUAL_ENV/bin/python" \
PHASE4A2_ROCM_REAL=/opt/rocm \
./scripts/build_phase4a2_image_path_modules.sh
```

Expected final marker:

```text
PHASE4A2_IMAGE_PATH_BUILD: PASS
```

The script creates an ignored local build tree at:

```text
.phase4a2_image_build/
```

It builds and stages both native runtimes without installing them globally into the virtual environment.

### 4. Run the complete public GPU demo

```bash
PHASE4A2_PYTHON="$VIRTUAL_ENV/bin/python" \
./scripts/run_phase4a2_public_demo.sh
```

Expected final marker:

```text
RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA: PASS
```

The demo writes its evidence beneath:

```text
phase4a2_public_demo_result/
```

## Public command surface

The repository exposes exactly three fail-closed public validation commands:

| Command | Purpose |
|---|---|
| `./scripts/build_phase4a2_image_path_modules.sh` | Build and stage the Phase-4A2 `tiny-rdna4-nn` and AMD `gsplat` native modules |
| `./scripts/run_phase4a2_public_demo.sh` | Run the qualified fused-MLP + rasterizer GPU path and all Phase-4A2 gates |
| `./scripts/fresh_clone_user_smoke.sh` | Run the earlier portable backend smoke for the explicit PyTorch network backends |

See [the public validation contract](docs/VALIDATION.md) for the exact claim levels and exclusions.

## Validation model

The public validation contract uses three levels:

1. **Tier 1 — Portable self-contained validation:** the public contract passes on one concrete host.
2. **Tier 2 — Reference comparison:** the local result agrees with frozen, environment-bound anchors; this is informative rather than universally required.
3. **Tier 3 — External field validation:** an independent foreign host passes the public contract with complete environment metadata.

The completed GitHub fresh-clone run is an independent checkout on the qualified R9700 host. It proves that the public repository can be cloned, built, and executed from scratch in that environment. It is not a claim of universal ROCm or RDNA4 compatibility.

## Reproducibility boundary

The following claims are supported:

- exact public commit, tree, and recursive submodule provenance;
- successful native build from an independent GitHub fresh clone;
- self-consistent SHA256 fingerprints for each concrete build instance;
- successful complete public GPU validation from the fresh clone.

**Checkout-path-independent binary reproducibility is not qualified.**

Absolute source and build paths are embedded in generated host and device binaries. Building the same source at a different absolute checkout path therefore produced different raw ELF/HIP SHA256 values. Section-level analysis also found differing `.text` and `.rodata` bytes in selected AMD `gsplat` code objects.

The binary hashes below identify concrete build instances; they are not cross-checkout reproducible-build identifiers:

| Build instance | `tiny-rdna4-nn` | AMD `gsplat` |
|---|---|---|
| Qualified checkout | `883f89efdad7bb909a4a3899ab79b2defe9713fdb5c7cf22cf4882c626b3efc4` | `b5eab3c002cd622aa08b47094b03e642babd86b15a42318c5a465b5932882360` |
| Independent GitHub fresh clone | `aeeb40781cb16bfb2d31d4a5d95b2f550b54e50d30615323b6214599b78fc08d` | `ad931a31572cde8d4b794683387cc11b5fa2cbc9bd7ca081a94db4f39367fb7c` |

Native binaries retained in the release evidence are provenance artifacts, not a universal binary distribution. Rebuild the modules locally from the tagged source.

## Release assets

The validated release contains seven uploaded assets. The three primary archives are:

| Asset | SHA256 |
|---|---|
| Deterministic source archive with recursive submodules | `b95e325af79131c7e049fea2a3bcff8950ec1f19f668843331ea9dbe7e4450e7` |
| Sanitized public evidence archive | `b082a33ca687f802b03540ec47dad67b582b9591040763d49c1dc5ac83f201e5` |
| Git bundle with the qualified commit and annotated tag | `04624aca638bdd6eb709b17b4cd64acffc4e44a4e0509d10a9e5184a25e91e2f` |

The release also includes:

- `RELEASE_NOTES.md`;
- `ASSET_SCOPE.md`;
- `PACKAGE_COMPLETE.txt`;
- `RELEASE_SHA256SUMS`.

All uploaded assets were downloaded again from the GitHub draft release and verified byte-for-byte before publication.

## Pinned recursive dependencies

| Dependency | Qualified commit |
|---|---|
| AMD `gsplat` | `2c62b22552c0ad4ed120aae304ce66ae27bc5d08` |
| nested GLM | `33b4a621a697a305bc3a7610d290677b96beb181` |
| CUTLASS source dependency | `82f5075946e2569589439d500733b700a3141374` |
| fmt | `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa` |
| cmrc | `952ffddba731fc110bd50409e8d2b8a06abbd237` |

These are Git-pinned source dependencies. Their presence does not imply that NVIDIA CUDA or NVIDIA hardware is required for the qualified ROCm path.

## Network backends

The Python package and import name remain `tinycudann` for source compatibility.

| Backend or path | Precision | Status and purpose |
|---|---|---|
| `PortableMLP` | FP32 | Portable correctness-first reference backend |
| `HipBLASLtMLP` | FP32 | Explicit AMD hipBLASLt backend |
| `HipBLASLtMLPFP16` | FP16 | Qualified explicit FP16 hipBLASLt backend; requires `"precision": "Fp16"` |
| rocWMMA Width-64 fused MLP | FP16-oriented fused path | Dedicated Phase-4A2 path qualified with the AMD `gsplat` image pipeline |
| upstream generic `FullyFusedMLP` name | — | Not claimed as a universal drop-in CUDA-compatible backend on ROCm |

The dedicated rocWMMA Width-64 path should not be interpreted as universal support for every upstream `FullyFusedMLP` topology or configuration.

### Portable backend smoke

The earlier backend-focused fresh-clone smoke remains available:

```bash
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1201
export MAX_JOBS=1

./scripts/fresh_clone_user_smoke.sh --all-backends
```

This smoke covers the explicit `PortableMLP`, `HipBLASLtMLP`, and `HipBLASLtMLPFP16` backends. It is separate from the Phase-4A2 fused image-path build and demo.

## Earlier FP16 backend performance qualification

Phase 3B1-F1 measured explicit FP16 hipBLASLt backend performance against the FP32 reference on the R9700:

| Category | FP16 speedup vs. FP32 | FP32/FP16 peak-memory factor |
|---|---:|---:|
| Overall | **2.2800x** | **1.0418x** |
| Latency regime, batch 1024 | 0.9968x | 1.0368x |
| Throughput regime, batch 16384 | **5.2155x** | 1.0468x |
| Network only | **2.1658x** | 1.0310x |
| Network with input encoding | **2.4003x** | 1.0527x |

These are earlier backend measurements, not an end-to-end Phase-4A2 AMD `gsplat` pipeline speedup claim.

Reports:

- [Phase 3B1-F1 reproducible performance report](phase3b1_reports/PHASE3B1F1_REPRODUCIBLE_PERFORMANCE.md)
- [Machine-readable F1 results](phase3b1_reports/phase3b1f1_reproducible_performance.json)
- [Raw-data index](phase3b1_reports/phase3b1f1_raw_index.jsonl)

## Expected GPU memory arena warning

On the qualified R9700 / ROCm 7.2 setup, model creation may print:

```text
tiny-cuda-nn warning: GPUMemoryArena: GPU 0 does not support virtual memory.
Falling back to regular allocations, which will be larger and can cause
occasional stutter.
```

This warning is expected on the qualified setup and is non-fatal when the functional PASS markers complete. The runtime uses regular GPU allocations instead. Memory usage may be higher and allocation-related pauses may occur, but the validated forward, backward, optimizer, replay, resume, and 100-step paths remain functional.

## Current limitations

- The complete Phase-4A2 qualification currently applies only to the AMD Radeon AI PRO R9700 / `gfx1201` environment documented above.
- The qualified horizon is 100 steps; longer training or convergence quality is not claimed by Phase 4A2.
- Phase 4A2 does not claim SuGaR integration or AMD `nvdiffrast` integration.
- Phase 4A2 does not claim general performance superiority over other backends or GPUs.
- Checkout-path-independent bit-reproducible binaries are not qualified.
- CUDA RTC, CUDA JIT fusion, and universal drop-in compatibility with NVIDIA-specific upstream backends are not part of the qualified path.
- Native release binaries are evidence artifacts, not portable prebuilt packages for arbitrary systems.
- Other AMD architectures may require additional porting and qualification work.

## Upstream project and attribution

This project is derived from [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) by Thomas Müller and contributors.

`tiny-rdna4-nn` is an independent community port. It is not affiliated with or endorsed by NVIDIA or AMD.

The original copyright notices and BSD 3-Clause license are retained. See [LICENSE.txt](LICENSE.txt).
