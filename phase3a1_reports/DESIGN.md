# Phase 3A1 HipBLASLtMLP design

## Scope and identity

`HipBLASLtMLP` is an explicit, unfused FP32-only backend for ROCm 7.2 on `gfx1201`. `PortableMLP` remains unchanged and is never selected as a silent fallback. Both backends share the canonical flat `W0,b0,...,Wout,bout` parameter layout and therefore support direct parameter and `state_dict()` interchange.

## Files

- `include/tiny-cuda-nn/networks/hipblaslt_mlp.h`: network class and independent forward-context contract.
- `src/hipblaslt_mlp.cu`: hipBLASLt plan cache, GEMM launches, pointwise/reduction kernels and network implementation.
- `src/portable_network.cu`: explicit factory selection for the two AMD FP32 backends.
- `bindings/torch/setup.py`: compile the new source and link `libhipblaslt`.
- `scripts/validate_phase3a1.py` and `scripts/benchmark_phase3a1.py`: correctness and HIP-event performance gates.

## GEMM and layout contract

All tiny-cuda-nn matrices used by the PyTorch path are contiguous column-major views of row-major PyTorch buffers. Phase 3A0's validated swapped-column-major mapping is used directly:

- Forward: `Y_col(out,batch) = transpose(W_col(in,out)) * X_col(in,batch)`.
- dX: `dX_col(in,batch) = W_col(in,out) * dY_col(out,batch)`.
- dW storage: `dW_col(in,out) = X_col(in,batch) * transpose(dY_col(out,batch))`.

No parameter, input, output, or gradient transpose/copy is introduced.

## Plan cache

There is one process-long device context per HIP device. It owns one hipBLASLt handle, a construction mutex, and immutable plans keyed by device, physical A/B/C dimensions, transpose flags, and operation role. Lookup first takes a shared cache mutex briefly; only cache misses perform descriptor/heuristic construction. Each plan selects the first successful zero-workspace algorithm. Plans and handles intentionally live until process shutdown so descriptors cannot be destroyed while queued GPU work still references them.

## Forward and backward

Forward performs one hipBLASLt GEMM per layer followed by one kernel that adds bias and applies `None`, `ReLU`, or `Sigmoid`. Independent contexts retain hidden preactivations and activations.

Backward applies the output derivative, then walks layers in reverse: hipBLASLt dW, separate bias reduction, hipBLASLt dX, and the preceding hidden derivative. `Overwrite` uses GEMM beta 0, `Accumulate` beta 1, and `Ignore` skips dW/db while still computing dX. All operations use the caller stream without device-wide or unconditional stream synchronization.

## Rejections

Construction rejects non-FP32 precision, non-`gfx1201`, widths outside 16/32/64/128, hidden counts outside 1/2/4, hidden activations outside None/ReLU, output activations outside None/Sigmoid, and missing zero-workspace algorithms. The public wrapper retains its existing input-shape and contiguity normalization contracts.

## Performance expectation

Large GEMMs should outperform PortableMLP's scalar kernels. Small batches may be slower due to unfused launches. No automatic dispatch threshold is introduced. A PASS tag is permitted only if all correctness gates and the separately documented performance gates pass.
