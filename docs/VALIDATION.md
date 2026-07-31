# Public validation contract

**Tier 1 — Portable Self-contained Validation:** PASS means the portable
contract passed on this concrete host. It does not prove universal ROCm/GPU
compatibility and is not external field validation.

**Tier 2 — Reference Comparison:** MATCH means the local run agrees with
frozen, environment-bound reference anchors. A mismatch does not automatically
invalidate Tier 1.

**Tier 3 — External Field Validation:** PASS means an independent foreign host
passed the public contract and supplied complete environment metadata.
Incomplete submissions are `INSUFFICIENT_ENVIRONMENT_METADATA`.

The qualified scope is gfx1201 / Radeon AI PRO R9700, ROCm 7.2, PyTorch
2.13.0+rocm7.2, Width64 fused rocWMMA MLP, amd-gsplat forward/backward, and a
100-step horizon. Replays and resume are locally deterministic only within the
frozen build, backend, hardware, runtime, and scene configuration.

No claim is made for every RDNA4 GPU or ROCm version, general performance or
convergence superiority, qualification beyond 100 steps, SuGaR integration, or
amd-nvdiffrast integration in Phase 4A2.
