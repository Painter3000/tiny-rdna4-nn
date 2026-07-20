# Phase 2E robustness summary

- Native GradientMode: Overwrite, Accumulate, double Accumulate, and Ignore PASS for 1x16, 4x64/Sigmoid, and HashGrid composition.
- Repeated backward: 50 changing-batch contexts PASS; retained graph supported and accumulates to 2G.
- Main training: 1000/1000 finite steps, six cyclic batch sizes; mean loss `0.191506` to `0.0000720051` (ratio `0.000376`).
- HashGrid training: 300/300 finite steps; network change `3.93867`, encoding change `3.28193`.
- File checkpoint: fresh-process output bitwise exact; SHA-256 recorded in JSON.
- Resume: uninterrupted 200 versus 100+fresh-process+100 is bitwise exact for parameters/output/loss.
- Streams: default, fixed explicit, rotating, and three independent streams PASS.
- Fresh processes: 20/20 PASS, including 4 with `HIP_LAUNCH_BLOCKING=1`.
- Edge contracts: invalid configuration rejected early; empty batches and public non-contiguous inputs supported.
- Memory: zero live-allocation growth after warm-up in 200 small and 100 large repetitions.
- Phase 2D regression: 8 required + 2 stream + 8 fresh PASS.
- Phase 2C regression: 12 stream + 12 fresh + 8 encoding regression PASS.

The only production semantic defect found was silent acceptance of explicit FP16 network precision; it is now rejected as unsupported. The native mode hook is confined below the public `torch.nn.Module` API.
