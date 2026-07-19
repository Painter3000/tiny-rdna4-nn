/*
 * Portable correctness-first MLP for the AMD RDNA4 tiny-cuda-nn port.
 * TCNN_RDNA4_P2_FIX_001
 * TCNN_RDNA4_P2_FIX_012: explicit HIP stream interface.
 */
#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>

namespace tcnn {

template <typename T>
class PortableMLP : public Network<T> {
public:
	PortableMLP(
		uint32_t input_width,
		uint32_t hidden_width,
		uint32_t output_width,
		uint32_t n_hidden_layers,
		Activation activation,
		Activation output_activation
	);

	void inference_mixed_precision_impl(
		hipStream_t stream,
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override;

	std::unique_ptr<Context> forward_impl(
		hipStream_t stream,
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override;

	void backward_impl(
		hipStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override;

	void set_params_impl(T*, T*, T*) override {}
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;

	size_t n_params() const override { return m_total_n_params; }
	uint32_t input_width() const override { return m_input_width; }
	uint32_t padded_output_width() const override { return m_output_width; }
	uint32_t output_width() const override { return m_output_width; }
	static uint32_t REQUIRED_ALIGNMENT() { return 1; }
	uint32_t required_input_alignment() const override { return REQUIRED_ALIGNMENT(); }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return {{m_input_width, m_hidden_width}, {m_hidden_width, m_output_width}};
	}

	uint32_t width(uint32_t layer) const override {
		if (layer != 0) {
			throw std::runtime_error{"PortableMLP only has one hidden activation."};
		}
		return m_hidden_width;
	}

	uint32_t num_forward_activations() const override { return 1; }
	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override;

	json hyperparams() const override {
		return {
			{"otype", "PortableMLP"},
			{"activation", "ReLU"},
			{"output_activation", "None"},
			{"n_neurons", m_hidden_width},
			{"n_hidden_layers", 1},
			{"bias", true},
			{"precision", "Fp32"},
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrixDynamic<T> preactivation;
		GPUMatrixDynamic<T> activation;
		GPUMatrixDynamic<T> owned_output;
	};

	const T* selected_params(bool use_inference_params) const {
		return use_inference_params ? this->inference_params() : this->params();
	}

	size_t weight1_offset() const { return 0; }
	size_t bias1_offset() const { return static_cast<size_t>(m_hidden_width) * m_input_width; }
	size_t weight2_offset() const { return bias1_offset() + m_hidden_width; }
	size_t bias2_offset() const { return weight2_offset() + static_cast<size_t>(m_output_width) * m_hidden_width; }

	uint32_t m_input_width;
	uint32_t m_hidden_width;
	uint32_t m_output_width;
	size_t m_total_n_params;
};

} // namespace tcnn
