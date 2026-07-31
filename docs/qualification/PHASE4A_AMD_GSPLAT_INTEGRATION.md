# Phase 4A: HipBLASLt amd-gsplat integration baseline

This integration trains Gaussian colors through the real amd-gsplat
differentiable rasterizer. Two selectable MLP backends receive the same
features, scene, cameras, target render, loss, seed, and step count:

- `reference`: a PyTorch FP32 MLP;
- `tiny-rdna4-nn`: the native FP32 `HipBLASLtMLP`.

## Backend scope

Phase 4A qualifies real amd-gsplat integration with the native
`HipBLASLtMLP` backend, including rasterizer backward, optimizer updates, loss
decrease, and render output.

`TCNN_ENABLE_ROCWMMA_WIDTH64_MLP` is not enabled. The fused rocWMMA production
path is `NOT_TESTED_IN_AMD_GSPLAT`, and Phase 4A does not extend the qualified
100-step horizon of that fused kernel.

The result table is a `functional_integration_comparison`, not a
`backend_numerical_equivalence` result. Different initial losses demonstrate
that these runs are not a controlled same-model/same-parameter comparison.
Their strong loss decrease proves real gradient flow and trainability, but
does not by itself prove element-wise decoder correctness.

The default run executes both paths for 200 steps and writes a compact result
directory containing the target, before/after renders, loss CSV, hashes,
environment metadata, metrics, and gates.

```bash
scripts/run_phase4a_fresh_integration.sh
```

The command verifies the bundled sources, builds both native extensions inside
`.phase4a_build`, and runs the integration from those fresh modules. Activate a
ROCm PyTorch environment first or set `PHASE4A_PYTHON` to its interpreter.
Nothing is downloaded or installed into that environment.
