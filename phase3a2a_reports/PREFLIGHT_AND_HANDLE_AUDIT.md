# Phase 3A2a preflight and static handle audit

Validated base: `1d53d05256858009b0b0159e38669a4881eed940`, tagged `phase3a2-hipblaslt-epilogue-tuning-gfx1201-rocm72-pass`. Development branch: `rdna4-phase3a2a-stream-handle-safety`.

The Phase-3A2 source had one device-global `hipblasLtHandle_t`. It was used by both `hipblasLtMatmulAlgoGetHeuristic()` and every `hipblasLtMatmul()` call, including calls submitted to different HIP streams. There is one production `hipblasLtMatmul()` callsite in `src/hipblaslt_mlp.cu`; all forward, input-gradient, and weight-gradient paths converge there.

Phase 3A2a replaces that ownership with:

- exactly one planning handle per initialized device, used only under the DeviceContext plan mutex by `hipblasLtMatmulAlgoGetHeuristic()`;
- a bounded map of exact `hipStream_t` identities to distinct execution handles, capacity 64 per device;
- a per-execution-handle submit mutex held only across the host `hipblasLtMatmul()` call;
- RAII registry and DeviceContext teardown, with device activation, plan destruction, execution-handle destruction, and finally planning-handle destruction;
- no production `hipDeviceSynchronize()` or `hipStreamSynchronize()`, no global GEMM lock, no eviction, and no PortableMLP fallback.

The epilogue descriptor cache now ignores batch-dependent matrix dimensions because a launch descriptor contains only transpose flags, epilogue kind, AUX presence, and a stable bias pointer. With at most five layers, two stable parameter sets, and two accepted transpose variants, the maximum expected descriptor count is 20 per model. The validated model uses five.

The official static audit reports PASS in `static_audit_001.json` and in the complete official gate output.

