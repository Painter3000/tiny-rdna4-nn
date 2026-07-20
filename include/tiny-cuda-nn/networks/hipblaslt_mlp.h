/* TCNN_RDNA4_P3A1_HIPBLASLT_001: unfused FP32 hipBLASLt MLP. */
#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>

namespace tcnn {

uint64_t hipblaslt_mlp_cache_hits();
uint64_t hipblaslt_mlp_cache_misses();
uint64_t hipblaslt_mlp_cache_size();

template <typename T>
class HipBLASLtMLP : public Network<T> {
public:
	HipBLASLtMLP(uint32_t input_width, uint32_t hidden_width, uint32_t output_width,
		uint32_t n_hidden_layers, Activation activation, Activation output_activation);

	void inference_mixed_precision_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<T>& output, bool use_inference_params = true) override;
	std::unique_ptr<Context> forward_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false,
		bool prepare_input_gradients = false) override;
	void backward_impl(hipStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite) override;

	void set_params_impl(T*, T*, T*) override {}
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;
	size_t n_params() const override { return m_total_n_params; }
	uint32_t input_width() const override { return m_input_width; }
	uint32_t padded_output_width() const override { return m_output_width; }
	uint32_t output_width() const override { return m_output_width; }
	static uint32_t REQUIRED_ALIGNMENT() { return 1; }
	uint32_t required_input_alignment() const override { return REQUIRED_ALIGNMENT(); }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
	uint32_t width(uint32_t layer) const override;
	uint32_t num_forward_activations() const override { return m_n_hidden_layers; }
	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override;
	json hyperparams() const override;

private:
	struct Layer { uint32_t input_width; uint32_t output_width; size_t weight_offset; size_t bias_offset; };
	struct ForwardContext : public Context {
		std::vector<GPUMatrixDynamic<T>> preactivations;
		std::vector<GPUMatrixDynamic<T>> activations;
		GPUMatrixDynamic<T> output_preactivation;
		GPUMatrixDynamic<T> owned_output;
	};
	const T* selected_params(bool inference) const { return inference ? this->inference_params() : this->params(); }

	uint32_t m_input_width, m_hidden_width, m_output_width, m_n_hidden_layers;
	Activation m_activation, m_output_activation;
	std::vector<Layer> m_layers;
	size_t m_total_n_params = 0;
};

} // namespace tcnn
