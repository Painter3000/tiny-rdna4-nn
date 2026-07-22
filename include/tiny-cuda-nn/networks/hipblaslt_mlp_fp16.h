/* TCNN_RDNA4_P3B1B_FP16_FORWARD_001: opt-in FP16 forward-only hipBLASLt MLP. */
#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>

namespace tcnn {

uint64_t hipblaslt_fp16_cache_hits();
uint64_t hipblaslt_fp16_cache_misses();
uint64_t hipblaslt_fp16_cache_size();
uint64_t hipblaslt_fp16_heuristic_queries();
uint64_t hipblaslt_fp16_execution_handle_count();
uint64_t hipblaslt_fp16_execution_handle_creations();
uint64_t hipblaslt_fp16_execution_handle_reuses();
uint64_t hipblaslt_fp16_descriptor_count();
uint64_t hipblaslt_fp16_bias_launches();
uint64_t hipblaslt_fp16_relu_bias_launches();
uint64_t hipblaslt_fp16_scratch_bytes_live();
uint64_t hipblaslt_fp16_scratch_bytes_peak();

class HipBLASLtMLPFP16 final : public Network<__half> {
public:
	HipBLASLtMLPFP16(uint32_t input_width, uint32_t hidden_width, uint32_t output_width,
		uint32_t n_hidden_layers, Activation activation, Activation output_activation);
	~HipBLASLtMLPFP16() override;
	void inference_mixed_precision_impl(hipStream_t, const GPUMatrixDynamic<__half>&,
		GPUMatrixDynamic<__half>&, bool = true) override;
	std::unique_ptr<Context> forward_impl(hipStream_t, const GPUMatrixDynamic<__half>&,
		GPUMatrixDynamic<__half>* = nullptr, bool = false, bool = false) override;
	void backward_impl(hipStream_t, const Context&, const GPUMatrixDynamic<__half>&,
		const GPUMatrixDynamic<__half>&, const GPUMatrixDynamic<__half>&,
		GPUMatrixDynamic<__half>* = nullptr, bool = false,
		GradientMode = GradientMode::Overwrite) override;
	void set_params_impl(__half*, __half*, __half*) override {}
	void initialize_params(pcg32&, float*, float = 1) override;
	size_t n_params() const override { return m_total_n_params; }
	uint32_t input_width() const override { return m_input_width; }
	uint32_t padded_output_width() const override { return m_output_width; }
	uint32_t output_width() const override { return m_output_width; }
	static uint32_t REQUIRED_ALIGNMENT() { return 1; }
	uint32_t required_input_alignment() const override { return REQUIRED_ALIGNMENT(); }
	std::vector<std::pair<uint32_t,uint32_t>> layer_sizes() const override;
	uint32_t width(uint32_t) const override;
	uint32_t num_forward_activations() const override { return m_n_hidden_layers; }
	std::pair<const __half*,MatrixLayout> forward_activations(const Context&, uint32_t) const override;
	json hyperparams() const override;

private:
	struct DescriptorState;
	struct Layer { uint32_t in, out; size_t weights, bias; };
	struct ForwardContext final : Context {
		std::vector<GPUMatrixDynamic<__half>> activations;
		GPUMatrixDynamic<__half> owned_output;
	};
	const __half* selected_params(bool inference) const { return inference ? this->inference_params() : this->params(); }
	void linear(hipStream_t, const GPUMatrixDynamic<__half>&, const __half*, const __half*,
		GPUMatrixDynamic<__half>&, uint32_t, uint32_t, Activation);
	uint32_t m_input_width, m_hidden_width, m_output_width, m_n_hidden_layers;
	Activation m_activation, m_output_activation;
	std::vector<Layer> m_layers;
	std::unique_ptr<DescriptorState> m_descriptors;
	size_t m_total_n_params = 0;
};
} // namespace tcnn
