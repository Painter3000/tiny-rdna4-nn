/* TCNN_RDNA4_P2_FIX_002 / TCNN_RDNA4_P2D_FIX_002: generalized FP32 PortableMLP. */

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/networks/portable_mlp.h>

#include <cmath>
#include <type_traits>

namespace tcnn {
namespace {

TCNN_DEVICE inline size_t matrix_index(MatrixLayout layout, uint32_t row, uint32_t column,
	uint32_t rows, uint32_t columns) {
	return layout == MatrixLayout::ColumnMajor ? static_cast<size_t>(row) + static_cast<size_t>(column) * rows
		: static_cast<size_t>(column) + static_cast<size_t>(row) * columns;
}

TCNN_DEVICE inline float activate(float value, Activation activation) {
	if (activation == Activation::ReLU) return value > 0.0f ? value : 0.0f;
	if (activation == Activation::Sigmoid) return 1.0f / (1.0f + expf(-value));
	return value;
}

__global__ void linear_forward_kernel(uint32_t n_elements, const float* input, MatrixLayout input_layout,
	const float* weights, const float* biases, float* preactivation, float* output,
	MatrixLayout output_layout, uint32_t batch, uint32_t input_width, uint32_t output_width,
	Activation activation) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / output_width;
	const uint32_t row = index - sample * output_width;
	float value = biases[row];
	for (uint32_t column = 0; column < input_width; ++column) {
		value = fmaf(input[matrix_index(input_layout, column, sample, input_width, batch)],
			weights[static_cast<size_t>(row) * input_width + column], value);
	}
	const size_t dst = matrix_index(output_layout, row, sample, output_width, batch);
	if (preactivation) preactivation[dst] = value;
	output[dst] = activate(value, activation);
}

__global__ void activation_gradient_kernel(uint32_t n_elements, const float* preactivation,
	const float* activation_value, MatrixLayout layout, const float* upstream, MatrixLayout upstream_layout,
	float* result, uint32_t batch, uint32_t width, Activation activation) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / width;
	const uint32_t row = index - sample * width;
	const size_t idx = matrix_index(layout, row, sample, width, batch);
	float derivative = 1.0f;
	if (activation == Activation::ReLU) derivative = preactivation[idx] > 0.0f ? 1.0f : 0.0f;
	else if (activation == Activation::Sigmoid) {
		const float y = activation_value[idx];
		derivative = y * (1.0f - y);
	}
	result[idx] = upstream[matrix_index(upstream_layout, row, sample, width, batch)] * derivative;
}

__global__ void transpose_matvec_kernel(uint32_t n_elements, const float* delta, MatrixLayout delta_layout,
	const float* weights, float* result, MatrixLayout result_layout, uint32_t batch,
	uint32_t input_width, uint32_t output_width) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / input_width;
	const uint32_t column = index - sample * input_width;
	float value = 0.0f;
	for (uint32_t row = 0; row < output_width; ++row) {
		value = fmaf(delta[matrix_index(delta_layout, row, sample, output_width, batch)],
			weights[static_cast<size_t>(row) * input_width + column], value);
	}
	result[matrix_index(result_layout, column, sample, input_width, batch)] = value;
}

TCNN_DEVICE inline void write_gradient(float* dst, size_t idx, float value, GradientMode mode) {
	if (mode == GradientMode::Overwrite) dst[idx] = value;
	else if (mode == GradientMode::Accumulate) dst[idx] += value;
}

__global__ void weight_gradient_kernel(uint32_t n_elements, const float* input, MatrixLayout input_layout,
	const float* delta, MatrixLayout delta_layout, float* gradients, size_t offset, GradientMode mode,
	uint32_t batch, uint32_t input_width, uint32_t output_width) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t row = index / input_width;
	const uint32_t column = index - row * input_width;
	float value = 0.0f;
	for (uint32_t sample = 0; sample < batch; ++sample) {
		value = fmaf(delta[matrix_index(delta_layout, row, sample, output_width, batch)],
			input[matrix_index(input_layout, column, sample, input_width, batch)], value);
	}
	write_gradient(gradients, offset + index, value, mode);
}

__global__ void bias_gradient_kernel(uint32_t n_elements, const float* delta, MatrixLayout layout,
	float* gradients, size_t offset, GradientMode mode, uint32_t batch, uint32_t width) {
	const uint32_t row = threadIdx.x + blockIdx.x * blockDim.x;
	if (row >= n_elements) return;
	float value = 0.0f;
	for (uint32_t sample = 0; sample < batch; ++sample) value += delta[matrix_index(layout, row, sample, width, batch)];
	write_gradient(gradients, offset + row, value, mode);
}

inline void launch_linear(hipStream_t stream, const GPUMatrixDynamic<float>& input, const float* params,
	GPUMatrixDynamic<float>* preactivation, GPUMatrixDynamic<float>& output, uint32_t input_width,
	uint32_t output_width, size_t weight_offset, size_t bias_offset, Activation activation) {
	linear_kernel(linear_forward_kernel, 0, stream, input.n() * output_width, input.data(), input.layout(),
		params + weight_offset, params + bias_offset, preactivation ? preactivation->data() : nullptr,
		output.data(), output.layout(), input.n(), input_width, output_width, activation);
	CUDA_CHECK_THROW(hipGetLastError());
}

} // namespace

template <typename T>
PortableMLP<T>::PortableMLP(uint32_t input_width, uint32_t hidden_width, uint32_t output_width,
	uint32_t n_hidden_layers, Activation activation, Activation output_activation)
: m_input_width{input_width}, m_hidden_width{hidden_width}, m_output_width{output_width},
	m_n_hidden_layers{n_hidden_layers}, m_activation{activation}, m_output_activation{output_activation} {
	static_assert(std::is_same<T, float>::value, "PortableMLP is FP32-only.");
	if (!input_width || !output_width || !n_hidden_layers) throw std::runtime_error{"PortableMLP dimensions and n_hidden_layers must be positive."};
	if (hidden_width != 16 && hidden_width != 32 && hidden_width != 64 && hidden_width != 128)
		throw std::runtime_error{"PortableMLP supports n_neurons 16, 32, 64, or 128."};
	if (activation != Activation::None && activation != Activation::ReLU)
		throw std::runtime_error{"PortableMLP hidden activation must be None or ReLU."};
	if (output_activation != Activation::None && output_activation != Activation::Sigmoid)
		throw std::runtime_error{"PortableMLP output activation must be None or Sigmoid."};
	for (uint32_t layer = 0; layer <= n_hidden_layers; ++layer) {
		const uint32_t in = layer == 0 ? input_width : hidden_width;
		const uint32_t out = layer == n_hidden_layers ? output_width : hidden_width;
		const size_t weight_offset = m_total_n_params;
		const size_t bias_offset = weight_offset + static_cast<size_t>(out) * in;
		m_layers.push_back({in, out, weight_offset, bias_offset});
		m_total_n_params = bias_offset + out;
	}
}

template <typename T>
void PortableMLP<T>::inference_mixed_precision_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output, bool use_inference_params) {
	std::vector<GPUMatrixDynamic<T>> buffers;
	buffers.reserve(m_n_hidden_layers);
	const GPUMatrixDynamic<T>* current = &input;
	for (uint32_t layer = 0; layer < m_n_hidden_layers; ++layer) {
		buffers.emplace_back(m_hidden_width, input.n(), stream, input.layout());
		launch_linear(stream, *current, selected_params(use_inference_params), nullptr, buffers.back(),
			m_layers[layer].input_width, m_layers[layer].output_width, m_layers[layer].weight_offset,
			m_layers[layer].bias_offset, m_activation);
		current = &buffers.back();
	}
	const auto& last = m_layers.back();
	launch_linear(stream, *current, selected_params(use_inference_params), nullptr, output,
		last.input_width, last.output_width, last.weight_offset, last.bias_offset, m_output_activation);
}

template <typename T>
std::unique_ptr<Context> PortableMLP<T>::forward_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>* output, bool use_inference_params, bool) {
	auto forward = std::make_unique<ForwardContext>();
	forward->preactivations.reserve(m_n_hidden_layers);
	forward->activations.reserve(m_n_hidden_layers);
	const GPUMatrixDynamic<T>* current = &input;
	for (uint32_t layer = 0; layer < m_n_hidden_layers; ++layer) {
		forward->preactivations.emplace_back(m_hidden_width, input.n(), stream, input.layout());
		forward->activations.emplace_back(m_hidden_width, input.n(), stream, input.layout());
		launch_linear(stream, *current, selected_params(use_inference_params), &forward->preactivations.back(),
			forward->activations.back(), m_layers[layer].input_width, m_layers[layer].output_width,
			m_layers[layer].weight_offset, m_layers[layer].bias_offset, m_activation);
		current = &forward->activations.back();
	}
	GPUMatrixDynamic<T>* actual_output = output;
	if (!actual_output) { forward->owned_output = {m_output_width, input.n(), stream, input.layout()}; actual_output = &forward->owned_output; }
	const auto& last = m_layers.back();
	launch_linear(stream, *current, selected_params(use_inference_params), nullptr, *actual_output,
		last.input_width, last.output_width, last.weight_offset, last.bias_offset, m_output_activation);
	return forward;
}

template <typename T>
void PortableMLP<T>::backward_impl(hipStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& dL_doutput, GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params, GradientMode param_gradients_mode) {
	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
	const uint32_t batch = input.n();
	const float* params = selected_params(use_inference_params);
	std::vector<GPUMatrixDynamic<T>> deltas;
	deltas.reserve(m_layers.size());
	for (uint32_t layer = 0; layer < m_layers.size(); ++layer)
		deltas.emplace_back(m_layers[layer].output_width, batch, stream,
			layer + 1 == m_layers.size() ? output.layout() : forward.activations[layer].layout());
	linear_kernel(activation_gradient_kernel, 0, stream, batch * m_output_width, output.data(), output.data(),
		output.layout(), dL_doutput.data(), dL_doutput.layout(), deltas.back().data(), batch, m_output_width, m_output_activation);
	CUDA_CHECK_THROW(hipGetLastError());

	float* gradients = nullptr;
	if (param_gradients_mode != GradientMode::Ignore) {
		gradients = this->gradients();
		if (!gradients) throw std::runtime_error{"PortableMLP gradient memory was not provided."};
	}
	for (int32_t layer = static_cast<int32_t>(m_layers.size()) - 1; layer >= 0; --layer) {
		const auto& meta = m_layers[layer];
		const GPUMatrixDynamic<T>& layer_input = layer == 0 ? input : forward.activations[layer - 1];
		if (gradients) {
			linear_kernel(weight_gradient_kernel, 0, stream, meta.output_width * meta.input_width,
				layer_input.data(), layer_input.layout(), deltas[layer].data(), deltas[layer].layout(), gradients,
				meta.weight_offset, param_gradients_mode, batch, meta.input_width, meta.output_width);
			linear_kernel(bias_gradient_kernel, 0, stream, meta.output_width, deltas[layer].data(), deltas[layer].layout(),
				gradients, meta.bias_offset, param_gradients_mode, batch, meta.output_width);
			CUDA_CHECK_THROW(hipGetLastError());
		}
		if (layer == 0) {
			if (dL_dinput) {
				linear_kernel(transpose_matvec_kernel, 0, stream, batch * meta.input_width, deltas[layer].data(),
					deltas[layer].layout(), params + meta.weight_offset, dL_dinput->data(), dL_dinput->layout(),
					batch, meta.input_width, meta.output_width);
				CUDA_CHECK_THROW(hipGetLastError());
			}
		} else {
			GPUMatrixDynamic<T> upstream{meta.input_width, batch, stream, forward.activations[layer - 1].layout()};
			linear_kernel(transpose_matvec_kernel, 0, stream, batch * meta.input_width, deltas[layer].data(),
				deltas[layer].layout(), params + meta.weight_offset, upstream.data(), upstream.layout(), batch,
				meta.input_width, meta.output_width);
			linear_kernel(activation_gradient_kernel, 0, stream, batch * meta.input_width,
				forward.preactivations[layer - 1].data(), forward.activations[layer - 1].data(),
				forward.activations[layer - 1].layout(), upstream.data(), upstream.layout(), deltas[layer - 1].data(),
				batch, meta.input_width, m_activation);
			CUDA_CHECK_THROW(hipGetLastError());
		}
	}
}

template <typename T>
void PortableMLP<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
	for (const auto& layer : m_layers) {
		const float bound = scale * std::sqrt(6.0f / (layer.input_width + layer.output_width));
		generate_random_uniform<float>(rnd, static_cast<size_t>(layer.output_width) * layer.input_width,
			params_full_precision + layer.weight_offset, -bound, bound);
		CUDA_CHECK_THROW(hipMemset(params_full_precision + layer.bias_offset, 0, layer.output_width * sizeof(float)));
	}
}

template <typename T>
std::vector<std::pair<uint32_t, uint32_t>> PortableMLP<T>::layer_sizes() const {
	std::vector<std::pair<uint32_t, uint32_t>> result;
	for (const auto& layer : m_layers) result.emplace_back(layer.input_width, layer.output_width);
	return result;
}

template <typename T> uint32_t PortableMLP<T>::width(uint32_t layer) const {
	if (layer >= m_n_hidden_layers) throw std::runtime_error{"PortableMLP hidden activation index out of range."};
	return m_hidden_width;
}

template <typename T>
std::pair<const T*, MatrixLayout> PortableMLP<T>::forward_activations(const Context& ctx, uint32_t layer) const {
	if (layer >= m_n_hidden_layers) throw std::runtime_error{"PortableMLP hidden activation index out of range."};
	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
	return {forward.activations[layer].data(), forward.activations[layer].layout()};
}

template <typename T> json PortableMLP<T>::hyperparams() const {
	return {{"otype", "PortableMLP"}, {"activation", to_string(m_activation)},
		{"output_activation", to_string(m_output_activation)}, {"n_neurons", m_hidden_width},
		{"n_hidden_layers", m_n_hidden_layers}, {"bias", true}, {"precision", "Fp32"}};
}

template class PortableMLP<float>;
} // namespace tcnn
