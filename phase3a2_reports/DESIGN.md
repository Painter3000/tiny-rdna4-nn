# Phase 3A2 epilogue design

Phase 3A2 keeps matrix layouts and the selected zero-workspace algorithm in the global static-signature cache. The cache key now additionally contains epilogue kind and AUX presence. Dynamic pointers are never written into that shared descriptor.

Each `HipBLASLtMLP` instance owns a bounded set of launch descriptors keyed by the complete matrix signature and the stable parameter-bias pointer. Descriptors are constructed once under a per-instance lookup mutex and remain immutable during all launches. The GEMM itself is never protected by a global or per-instance mutex, so the same model may enqueue work concurrently on multiple streams.

`BIAS` is used for activation `None`. `RELU_BIAS` is used for ReLU. ReLU backward uses the post-activation value for its sign predicate, which is mathematically identical to testing the preactivation and avoids a dynamic AUX binding. Sigmoid retains BIAS plus an activation-only kernel in inference and the Phase-3A1 postkernel in training because its derivative requires the activation state.

The transposed physical mapping used for a native RowMajor output gives hipBLASLt the batch dimension as D's row dimension, so its bias broadcast axis is wrong for tiny-cuda-nn's logical output. That path deliberately remains hipBLASLt GEMM plus the Phase-3A1 postkernel and increments the diagnostic fallback counter. It never switches to PortableMLP.

No ABI-compatibility claim is made.
