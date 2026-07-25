# Phase 4A0-P1 — Minimal rocWMMA GEMM vs CPU-FP64

Marker:

```text
TCNN_RDNA4_P4A0_P1_MINIMAL_ROCWMMA_GEMM_001
```

This checkpoint performs exactly one `16×16×16` rocWMMA GEMM:

- A: FP16, row-major
- B: FP16, column-major
- accumulation: FP32
- output: FP32, row-major
- primary oracle: CPU FP64, calculated from the exact FP16-rounded inputs
- launch: one `gfx1201` Wave32 (`grid=1`, `block=32`)

It does not dump fragment registers, use hipBLASLt, modify the productive MLP,
or make a performance claim.

## Install into the prepared worktree

Run from the repository root:

```bash
unzip -o ~/Downloads/phase4a0_p1_minimal_rocwmma_gemm_bundle.zip
chmod +x scripts/run_phase4a0_p1_minimal_gemm.sh
```

## Execute

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe
scripts/run_phase4a0_p1_minimal_gemm.sh
```

Expected final markers:

```text
ROCWMMA_P1_WAVE32_CONTEXT: PASS
ROCWMMA_P1_INPUT_QUANTIZATION: PASS
ROCWMMA_NUMERICAL_RESULT_VS_CPU: PASS
PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM: PASS
PHASE4A0_P1_JSON_AUDIT: PASS
```

Evidence is written outside the repository under:

```text
~/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p1_<UTC timestamp>/
```

A successful P1 authorizes P2, the raw lane/register fragment-map probe. It
does not yet establish the fragment layout.
