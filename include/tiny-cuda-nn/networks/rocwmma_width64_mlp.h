/*
 * TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001
 * Explicit, fail-closed Width-64 rocWMMA production-backend skeleton.
 */
#pragma once

#include <tiny-cuda-nn/network.h>

namespace tcnn {

class RocWMMAWidth64MLP final : public Network<__half> {
public:
	static constexpr uint32_t WIDTH = 64;
	static constexpr uint32_t N_HIDDEN_LAYERS = 2;
	static constexpr uint32_t N_LINEAR_LAYERS = 3;
	static constexpr uint32_t TILE_ROWS = 16;
	static constexpr uint32_t REQUIRED_ALIGNMENT_VALUE = 16;

	static constexpr size_t WEIGHT_ELEMENTS = WIDTH * WIDTH;
	static constexpr size_t BIAS_ELEMENTS = WIDTH;

	static constexpr size_t WEIGHT_0_OFFSET = 0;
	static constexpr size_t BIAS_0_OFFSET = WEIGHT_0_OFFSET + WEIGHT_ELEMENTS;
	static constexpr size_t WEIGHT_1_OFFSET = BIAS_0_OFFSET + BIAS_ELEMENTS;
	static constexpr size_t BIAS_1_OFFSET = WEIGHT_1_OFFSET + WEIGHT_ELEMENTS;
	static constexpr size_t WEIGHT_2_OFFSET = BIAS_1_OFFSET + BIAS_ELEMENTS;
	static constexpr size_t BIAS_2_OFFSET = WEIGHT_2_OFFSET + WEIGHT_ELEMENTS;
	static constexpr size_t TOTAL_PARAMETER_ELEMENTS = BIAS_2_OFFSET + BIAS_ELEMENTS;

	RocWMMAWidth64MLP(
		uint32_t input_width,
		uint32_t hidden_width,
		uint32_t output_width,
		uint32_t n_hidden_layers,
		Activation activation,
		Activation output_activation
	);

	void inference_mixed_precision_impl(
		hipStream_t stream,
		const GPUMatrixDynamic<__half>& input,
		GPUMatrixDynamic<__half>& output,
		bool use_inference_params = true
	) override;

	std::unique_ptr<Context> forward_impl(
		hipStream_t stream,
		const GPUMatrixDynamic<__half>& input,
		GPUMatrixDynamic<__half>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override;

	void backward_impl(
		hipStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<__half>& input,
		const GPUMatrixDynamic<__half>& output,
		const GPUMatrixDynamic<__half>& dL_doutput,
		GPUMatrixDynamic<__half>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override;

	void set_params_impl(__half*, __half*, __half*) override {}

	void initialize_params(
		pcg32& rnd,
		float* params_full_precision,
		float scale = 1
	) override;

	size_t n_params() const override {
		return TOTAL_PARAMETER_ELEMENTS;
	}

	uint32_t input_width() const override {
		return WIDTH;
	}

	uint32_t padded_output_width() const override {
		return WIDTH;
	}

	uint32_t output_width() const override {
		return WIDTH;
	}

	static uint32_t REQUIRED_ALIGNMENT() {
		return REQUIRED_ALIGNMENT_VALUE;
	}

	uint32_t required_input_alignment() const override {
		return REQUIRED_ALIGNMENT();
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;

	uint32_t width(uint32_t layer) const override;

	uint32_t num_forward_activations() const override {
		return N_HIDDEN_LAYERS;
	}

	std::pair<const __half*, MatrixLayout> forward_activations(
		const Context& ctx,
		uint32_t layer
	) const override;

	json hyperparams() const override;
};

static_assert(
	RocWMMAWidth64MLP::TOTAL_PARAMETER_ELEMENTS == 12480,
	"Phase 4A2 Width-64 parameter ABI drifted."
);

} // namespace tcnn
