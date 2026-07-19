/* TCNN_RDNA4_P2_FIX_002: PortableMLP FP32 HIP implementation. */

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/networks/portable_mlp.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace tcnn {
namespace {

TCNN_DEVICE inline size_t matrix_index(
	MatrixLayout layout,
	uint32_t row,
	uint32_t column,
	uint32_t rows,
	uint32_t columns
) {
	return layout == MatrixLayout::ColumnMajor
		? static_cast<size_t>(row) + static_cast<size_t>(column) * rows
		: static_cast<size_t>(column) + static_cast<size_t>(row) * columns;
}

__global__ void hidden_forward_kernel(
	uint32_t n_elements,
	const float* input,
	MatrixLayout input_layout,
	const float* weight1,
	const float* bias1,
	float* preactivation,
	float* activation,
	MatrixLayout hidden_layout,
	uint32_t batch,
	uint32_t input_width,
	uint32_t hidden_width
) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / hidden_width;
	const uint32_t hidden = index - sample * hidden_width;

	float value = bias1[hidden];
	for (uint32_t input_idx = 0; input_idx < input_width; ++input_idx) {
		value = fmaf(
			input[matrix_index(input_layout, input_idx, sample, input_width, batch)],
			weight1[static_cast<size_t>(hidden) * input_width + input_idx],
			value
		);
	}
	const size_t dst = matrix_index(hidden_layout, hidden, sample, hidden_width, batch);
	preactivation[dst] = value;
	activation[dst] = value > 0.0f ? value : 0.0f;
}

__global__ void output_forward_kernel(
	uint32_t n_elements,
	const float* activation,
	MatrixLayout hidden_layout,
	const float* weight2,
	const float* bias2,
	float* output,
	MatrixLayout output_layout,
	uint32_t batch,
	uint32_t hidden_width,
	uint32_t output_width
) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / output_width;
	const uint32_t output_idx = index - sample * output_width;

	float value = bias2[output_idx];
	for (uint32_t hidden = 0; hidden < hidden_width; ++hidden) {
		value = fmaf(
			activation[matrix_index(hidden_layout, hidden, sample, hidden_width, batch)],
			weight2[static_cast<size_t>(output_idx) * hidden_width + hidden],
			value
		);
	}
	output[matrix_index(output_layout, output_idx, sample, output_width, batch)] = value;
}

__global__ void hidden_gradient_kernel(
	uint32_t n_elements,
	const float* preactivation,
	MatrixLayout hidden_layout,
	const float* weight2,
	const float* dL_doutput,
	MatrixLayout output_layout,
	float* dL_dhidden,
	uint32_t batch,
	uint32_t hidden_width,
	uint32_t output_width
) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / hidden_width;
	const uint32_t hidden = index - sample * hidden_width;

	float value = 0.0f;
	for (uint32_t output_idx = 0; output_idx < output_width; ++output_idx) {
		value = fmaf(
			dL_doutput[matrix_index(output_layout, output_idx, sample, output_width, batch)],
			weight2[static_cast<size_t>(output_idx) * hidden_width + hidden],
			value
		);
	}
	const size_t idx = matrix_index(hidden_layout, hidden, sample, hidden_width, batch);
	dL_dhidden[idx] = preactivation[idx] > 0.0f ? value : 0.0f;
}

__global__ void input_gradient_kernel(
	uint32_t n_elements,
	const float* dL_dhidden,
	MatrixLayout hidden_layout,
	const float* weight1,
	float* dL_dinput,
	MatrixLayout input_layout,
	uint32_t batch,
	uint32_t input_width,
	uint32_t hidden_width
) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t sample = index / input_width;
	const uint32_t input_idx = index - sample * input_width;

	float value = 0.0f;
	for (uint32_t hidden = 0; hidden < hidden_width; ++hidden) {
		value = fmaf(
			dL_dhidden[matrix_index(hidden_layout, hidden, sample, hidden_width, batch)],
			weight1[static_cast<size_t>(hidden) * input_width + input_idx],
			value
		);
	}
	dL_dinput[matrix_index(input_layout, input_idx, sample, input_width, batch)] = value;
}

TCNN_DEVICE inline void write_gradient(float* dst, size_t idx, float value, GradientMode mode) {
	if (mode == GradientMode::Overwrite) dst[idx] = value;
	else if (mode == GradientMode::Accumulate) dst[idx] += value;
}

__global__ void weight1_gradient_kernel(
	uint32_t n_elements,
	const float* input,
	MatrixLayout input_layout,
	const float* dL_dhidden,
	MatrixLayout hidden_layout,
	float* gradients,
	size_t offset,
	GradientMode mode,
	uint32_t batch,
	uint32_t input_width,
	uint32_t hidden_width
) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t hidden = index / input_width;
	const uint32_t input_idx = index - hidden * input_width;
	float value = 0.0f;
	for (uint32_t sample = 0; sample < batch; ++sample) {
		value = fmaf(
			dL_dhidden[matrix_index(hidden_layout, hidden, sample, hidden_width, batch)],
			input[matrix_index(input_layout, input_idx, sample, input_width, batch)],
			value
		);
	}
	write_gradient(gradients, offset + index, value, mode);
}

__global__ void bias1_gradient_kernel(
	uint32_t n_elements,
	const float* dL_dhidden,
	MatrixLayout hidden_layout,
	float* gradients,
	size_t offset,
	GradientMode mode,
	uint32_t batch,
	uint32_t hidden_width
) {
	const uint32_t hidden = threadIdx.x + blockIdx.x * blockDim.x;
	if (hidden >= n_elements) return;
	float value = 0.0f;
	for (uint32_t sample = 0; sample < batch; ++sample) {
		value += dL_dhidden[matrix_index(hidden_layout, hidden, sample, hidden_width, batch)];
	}
	write_gradient(gradients, offset + hidden, value, mode);
}

__global__ void weight2_gradient_kernel(
	uint32_t n_elements,
	const float* activation,
	MatrixLayout hidden_layout,
	const float* dL_doutput,
	MatrixLayout output_layout,
	float* gradients,
	size_t offset,
	GradientMode mode,
	uint32_t batch,
	uint32_t hidden_width,
	uint32_t output_width
) {
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n_elements) return;
	const uint32_t output_idx = index / hidden_width;
	const uint32_t hidden = index - output_idx * hidden_width;
	float value = 0.0f;
	for (uint32_t sample = 0; sample < batch; ++sample) {
		value = fmaf(
			dL_doutput[matrix_index(output_layout, output_idx, sample, output_width, batch)],
			activation[matrix_index(hidden_layout, hidden, sample, hidden_width, batch)],
			value
		);
	}
	write_gradient(gradients, offset + index, value, mode);
}

__global__ void bias2_gradient_kernel(
	uint32_t n_elements,
	const float* dL_doutput,
	MatrixLayout output_layout,
	float* gradients,
	size_t offset,
	GradientMode mode,
	uint32_t batch,
	uint32_t output_width
) {
	const uint32_t output_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_idx >= n_elements) return;
	float value = 0.0f;
	for (uint32_t sample = 0; sample < batch; ++sample) {
		value += dL_doutput[matrix_index(output_layout, output_idx, sample, output_width, batch)];
	}
	write_gradient(gradients, offset + output_idx, value, mode);
}

inline void launch_forward(
	hipStream_t stream,
	const GPUMatrixDynamic<float>& input,
	const float* params,
	GPUMatrixDynamic<float>& preactivation,
	GPUMatrixDynamic<float>& activation,
	GPUMatrixDynamic<float>& output,
	uint32_t input_width,
	uint32_t hidden_width,
	uint32_t output_width,
	size_t bias1_offset,
	size_t weight2_offset,
	size_t bias2_offset
) {
	const uint32_t batch = input.n();
	linear_kernel(hidden_forward_kernel, 0, stream, batch * hidden_width,
		input.data(), input.layout(), params, params + bias1_offset,
		preactivation.data(), activation.data(), activation.layout(),
		batch, input_width, hidden_width);
	CUDA_CHECK_THROW(hipGetLastError());

	linear_kernel(output_forward_kernel, 0, stream, batch * output_width,
		activation.data(), activation.layout(), params + weight2_offset,
		params + bias2_offset, output.data(), output.layout(),
		batch, hidden_width, output_width);
	CUDA_CHECK_THROW(hipGetLastError());
}

} // namespace

template <typename T>
PortableMLP<T>::PortableMLP(
	uint32_t input_width,
	uint32_t hidden_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
	m_input_width{input_width},
	m_hidden_width{hidden_width},
	m_output_width{output_width},
	m_total_n_params{
		static_cast<size_t>(hidden_width) * input_width + hidden_width +
		static_cast<size_t>(output_width) * hidden_width + output_width
	}
{
	static_assert(std::is_same<T, float>::value, "PortableMLP Phase 2C is FP32-only.");
	if (input_width == 0 || output_width == 0) throw std::runtime_error{"PortableMLP requires positive input/output widths."};
	if (hidden_width != 16 && hidden_width != 32) throw std::runtime_error{"PortableMLP supports n_neurons 16 or 32."};
	if (n_hidden_layers != 1) throw std::runtime_error{"PortableMLP supports exactly one hidden layer."};
	if (activation != Activation::ReLU) throw std::runtime_error{"PortableMLP supports ReLU only."};
	if (output_activation != Activation::None) throw std::runtime_error{"PortableMLP supports output_activation=None only."};
}

template <typename T>
void PortableMLP<T>::inference_mixed_precision_impl(
	hipStream_t stream,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output,
	bool use_inference_params
) {
	GPUMatrixDynamic<T> preactivation{m_hidden_width, input.n(), stream, input.layout()};
	GPUMatrixDynamic<T> activation{m_hidden_width, input.n(), stream, input.layout()};
	launch_forward(stream, input, selected_params(use_inference_params), preactivation,
		activation, output, m_input_width, m_hidden_width, m_output_width,
		bias1_offset(), weight2_offset(), bias2_offset());
}

template <typename T>
std::unique_ptr<Context> PortableMLP<T>::forward_impl(
	hipStream_t stream,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>* output,
	bool use_inference_params,
	bool
) {
	auto forward = std::make_unique<ForwardContext>();
	forward->preactivation = {m_hidden_width, input.n(), stream, input.layout()};
	forward->activation = {m_hidden_width, input.n(), stream, input.layout()};
	GPUMatrixDynamic<T>* actual_output = output;
	if (!actual_output) {
		forward->owned_output = {m_output_width, input.n(), stream, input.layout()};
		actual_output = &forward->owned_output;
	}
	launch_forward(stream, input, selected_params(use_inference_params),
		forward->preactivation, forward->activation, *actual_output,
		m_input_width, m_hidden_width, m_output_width,
		bias1_offset(), weight2_offset(), bias2_offset());
	return forward;
}

template <typename T>
void PortableMLP<T>::backward_impl(
	hipStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>&,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	GradientMode param_gradients_mode
) {
	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
	const uint32_t batch = input.n();
	const float* params = selected_params(use_inference_params);
	GPUMatrixDynamic<T> dL_dhidden{m_hidden_width, batch, stream, forward.activation.layout()};

	linear_kernel(hidden_gradient_kernel, 0, stream, batch * m_hidden_width,
		forward.preactivation.data(), forward.preactivation.layout(),
		params + weight2_offset(), dL_doutput.data(), dL_doutput.layout(),
		dL_dhidden.data(), batch, m_hidden_width, m_output_width);
	CUDA_CHECK_THROW(hipGetLastError());

	if (dL_dinput) {
		linear_kernel(input_gradient_kernel, 0, stream, batch * m_input_width,
			dL_dhidden.data(), dL_dhidden.layout(), params + weight1_offset(),
			dL_dinput->data(), dL_dinput->layout(), batch, m_input_width, m_hidden_width);
		CUDA_CHECK_THROW(hipGetLastError());
	}

	if (param_gradients_mode == GradientMode::Ignore) return;
	float* gradients = this->gradients();
	if (!gradients) throw std::runtime_error{"PortableMLP gradient memory was not provided."};

	linear_kernel(weight1_gradient_kernel, 0, stream, m_hidden_width * m_input_width,
		input.data(), input.layout(), dL_dhidden.data(), dL_dhidden.layout(),
		gradients, weight1_offset(), param_gradients_mode, batch, m_input_width, m_hidden_width);
	CUDA_CHECK_THROW(hipGetLastError());

	linear_kernel(bias1_gradient_kernel, 0, stream, m_hidden_width,
		dL_dhidden.data(), dL_dhidden.layout(), gradients, bias1_offset(),
		param_gradients_mode, batch, m_hidden_width);
	CUDA_CHECK_THROW(hipGetLastError());

	linear_kernel(weight2_gradient_kernel, 0, stream, m_output_width * m_hidden_width,
		forward.activation.data(), forward.activation.layout(), dL_doutput.data(),
		dL_doutput.layout(), gradients, weight2_offset(), param_gradients_mode,
		batch, m_hidden_width, m_output_width);
	CUDA_CHECK_THROW(hipGetLastError());

	linear_kernel(bias2_gradient_kernel, 0, stream, m_output_width,
		dL_doutput.data(), dL_doutput.layout(), gradients, bias2_offset(),
		param_gradients_mode, batch, m_output_width);
	CUDA_CHECK_THROW(hipGetLastError());
}

template <typename T>
void PortableMLP<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
	const size_t weight1_count = static_cast<size_t>(m_hidden_width) * m_input_width;
	const size_t weight2_count = static_cast<size_t>(m_output_width) * m_hidden_width;
	const float bound1 = scale * std::sqrt(6.0f / (m_input_width + m_hidden_width));
	const float bound2 = scale * std::sqrt(6.0f / (m_hidden_width + m_output_width));
	generate_random_uniform<float>(rnd, weight1_count, params_full_precision + weight1_offset(), -bound1, bound1);
	// TCNN_RDNA4_P2_FIX_008: params_full_precision is GPU memory in the
	// PyTorch binding. Never use host std::fill_n on this pointer.
	CUDA_CHECK_THROW(hipMemset(
		params_full_precision + bias1_offset(),
		0,
		m_hidden_width * sizeof(float)
	));
	generate_random_uniform<float>(
		rnd,
		weight2_count,
		params_full_precision + weight2_offset(),
		-bound2,
		bound2
	);
	CUDA_CHECK_THROW(hipMemset(
		params_full_precision + bias2_offset(),
		0,
		m_output_width * sizeof(float)
	));
}

template <typename T>
std::pair<const T*, MatrixLayout> PortableMLP<T>::forward_activations(const Context& ctx, uint32_t layer) const {
	if (layer != 0) throw std::runtime_error{"PortableMLP only has one hidden activation."};
	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
	return {forward.activation.data(), forward.activation.layout()};
}

template class PortableMLP<float>;

} // namespace tcnn
