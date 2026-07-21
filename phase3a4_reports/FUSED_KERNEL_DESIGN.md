# Phase 3A4 fused ReLU backward and bias-gradient design

## Existing path

`HipBLASLtMLP::backward_impl` first computes the hidden upstream tensor with
hipBLASLt. It then launches `activation_gradient` to write `dZ`, followed by
`bias_gradient` when parameter gradients are enabled. The latter assigns for
`Overwrite`, adds for `Accumulate`, and is skipped for `Ignore`.

The preferred mask source is the stored post-ReLU hidden activation. The
predicate is strictly `activation > 0.0f`; consequently negative values,
`-0.0f`, and `+0.0f` all have derivative zero. No pre-activation is materialized
or reconstructed for the fused path.

## Released layout

The initial fused dispatch is restricted to contiguous ColumnMajor tiny-cuda-nn
matrices with logical shape `(width, batch)`. Its physical indexing is
`sample * width + feature`, i.e. coalesced `[batch,width]`. Contiguous RowMajor
matrices use the existing Phase-3A2a HIP kernels and increment the fallback
counter.

## Deterministic two-stage reduction

Stage 1 uses `blockDim=(WIDTH, 256/WIDTH)`. One thread owns one feature of one
sample row. It writes that `dZ` element exactly once and stores it in shared
memory. Threads with `y==0` sum the tile rows in increasing order and write one
partial per feature. There are no atomics and no wave-shuffle operations.

Stage 2 assigns one thread to each feature and consumes partials in increasing
tile order. `Overwrite` assigns the sum and `Accumulate` adds it. In `Ignore`,
Stage 1 is instantiated without bias computation, no partial storage is
allocated, and Stage 2 is not launched. A second Accumulate therefore produces
`2G` without final bias-gradient atomics.

For `ROWS_PER_TILE = 256 / WIDTH`, required scratch is

```text
ceil(batch / ROWS_PER_TILE) * width * sizeof(float)
```

plus the allocator's existing alignment. The maximum contracted case is width
128, batch 4096: 1,048,576 bytes before alignment. Scratch ownership belongs to
the exclusive forward/backward context; it is never cached globally or shared
between concurrent contexts.

## Dispatch and fallback

The fused path requires `HipBLASLtMLP`, FP32, hidden ReLU, width
16/32/64/128, and contiguous ColumnMajor layout. Every other case retains the
legacy activation-gradient plus bias-gradient HIP path. There is no CPU or
PortableMLP fallback and no host/device synchronization in production.

