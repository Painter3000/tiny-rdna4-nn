# PortableMLP Phase 2D design

`TCNN_RDNA4_P2D_FIX_001` defines a stable layer-major parameter layout. For every
linear layer, including the output layer, parameters are `[weights, bias]`; weights
are row-major `[output_width, input_width]`. Layers are concatenated from input to
output. Consequently, the Phase 2C one-hidden-layer layout remains byte-for-byte
`W0,b0,W1,b1` compatible.

Offsets are accumulated with `size_t`: `weight_offset` is the previous layer end,
`bias_offset = weight_offset + output_width * input_width`, and the next layer
starts after `output_width` bias values. All linear layers have bias.

Forward retains one preactivation and one activation `GPUMatrixDynamic<float>` per
hidden layer. These matrices use the input/predecessor layout and are owned by the
forward context. The output activation is computed directly into the caller output;
Sigmoid backward uses that saved public output value.

Backward first forms the output delta, then walks layer metadata in reverse. Each
step computes full weight and bias gradients. A transposed matrix-vector product
forms the preceding upstream gradient, followed by the hidden None/ReLU derivative.
The last step optionally writes the input gradient. The context therefore contains
all and only the hidden preactivations/activations required for first backward.

Every allocation and kernel receives the public `hipStream_t`; no auxiliary stream
or synchronization is introduced. Kernels bounds-check each flattened element, so
batch 1 and batches not divisible by launch block sizes (7 and 257 in validation)
follow the same path. Matrices are accessed through their declared row/column-major
layout. `tcnn.Network` internally composes Identity encoding; for public input widths
2 and 3, its padded channels are the Identity-defined constant 1 and are included in
the numerical reference.

Supported scope is deliberately FP32, 1 or more hidden layers, widths 16/32/64/128,
hidden None/ReLU, output None/Sigmoid. Performance backends, mixed precision, and
second backward remain excluded.
