# Phase 3A2 marker inventory

- `TCNN_RDNA4_P3A2_EPILOGUE_001` in `src/hipblaslt_mlp.cu`: immutable epilogue plan and launch-descriptor architecture.
- `TCNN_RDNA4_P3A2_EPILOGUE_001` in `bindings/torch/tinycudann/modules.py`: `torch.no_grad()` selects the native inference entry point.

No `TCNN_RDNA4_P3A2_DEBUG_*` or `TCNN_RDNA4_P3A2_ISOLATION_*` marker remains.
