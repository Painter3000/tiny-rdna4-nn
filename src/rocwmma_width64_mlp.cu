/*
 * TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001
 * No production rocWMMA kernel is installed in this checkpoint.
 */
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/networks/rocwmma_width64_mlp.h>
#include <tiny-cuda-nn/random.h>

#include <cmath>
#include <string>

namespace tcnn {
namespace {

constexpr const char* SKELETON_FAILURE =
	"RocWMMAWidth64MLP Phase 4A2-P1 skeleton: the production inference "
	"kernel is not qualified; no fallback was executed.";

[[noreturn]] void fail_closed() {
	throw std::runtime_error{SKELETON_FAILURE};
}

bool is_exact_gfx1201(const char* raw_arch) {
	if (!raw_arch) {
		return false;
	}

	const std::string arch{raw_arch};
	return arch == "gfx1201" || arch.rfind("gfx1201:", 0) == 0;
}

} // namespace

RocWMMAWidth64MLP::RocWMMAWidth64MLP(
	uint32_t input_width,
	uint32_t hidden_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) {
	int device = 0;
	CUDA_CHECK_THROW(hipGetDevice(&device));

	hipDeviceProp_t properties{};
	CUDA_CHECK_THROW(hipGetDeviceProperties(&properties, device));

	if (!is_exact_gfx1201(properties.gcnArchName)) {
		throw std::runtime_error{fmt::format(
			"RocWMMAWidth64MLP supports gfx1201 only, but the active device "
			"reports {}.",
			properties.gcnArchName
		)};
	}

	if (input_width != WIDTH || hidden_width != WIDTH || output_width != WIDTH) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires input, hidden, and output widths of 64."
		};
	}

	if (n_hidden_layers != N_HIDDEN_LAYERS) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires exactly two hidden layers."
		};
	}

	if (activation != Activation::ReLU) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires hidden activation ReLU."
		};
	}

	if (output_activation != Activation::None) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires output activation None."
		};
	}

	this->set_jit_fusion(false);
}

void RocWMMAWidth64MLP::inference_mixed_precision_impl(
	hipStream_t,
	const GPUMatrixDynamic<__half>&,
	GPUMatrixDynamic<__half>&,
	bool
) {
	fail_closed();
}

std::unique_ptr<Context> RocWMMAWidth64MLP::forward_impl(
	hipStream_t,
	const GPUMatrixDynamic<__half>&,
	GPUMatrixDynamic<__half>*,
	bool,
	bool
) {
	fail_closed();
}

void RocWMMAWidth64MLP::backward_impl(
	hipStream_t,
	const Context&,
	const GPUMatrixDynamic<__half>&,
	const GPUMatrixDynamic<__half>&,
	const GPUMatrixDynamic<__half>&,
	GPUMatrixDynamic<__half>*,
	bool,
	GradientMode
) {
	throw std::runtime_error{
		"RocWMMAWidth64MLP Phase 4A2-P1 skeleton: backward is not "
		"qualified; no fallback was executed."
	};
}

void RocWMMAWidth64MLP::initialize_params(
	pcg32& rnd,
	float* params_full_precision,
	float scale
) {
	if (!params_full_precision) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP parameter initialization received a null pointer."
		};
	}

	const float bound = scale * std::sqrt(
		6.0f / static_cast<float>(WIDTH + WIDTH)
	);

	const size_t weight_offsets[N_LINEAR_LAYERS] = {
		WEIGHT_0_OFFSET,
		WEIGHT_1_OFFSET,
		WEIGHT_2_OFFSET,
	};

	const size_t bias_offsets[N_LINEAR_LAYERS] = {
		BIAS_0_OFFSET,
		BIAS_1_OFFSET,
		BIAS_2_OFFSET,
	};

	for (uint32_t layer = 0; layer < N_LINEAR_LAYERS; ++layer) {
		generate_random_uniform<float>(
			rnd,
			WEIGHT_ELEMENTS,
			params_full_precision + weight_offsets[layer],
			-bound,
			bound
		);

		CUDA_CHECK_THROW(hipMemset(
			params_full_precision + bias_offsets[layer],
			0,
			BIAS_ELEMENTS * sizeof(float)
		));
	}
}

std::vector<std::pair<uint32_t, uint32_t>>
RocWMMAWidth64MLP::layer_sizes() const {
	return {
		{WIDTH, WIDTH},
		{WIDTH, WIDTH},
		{WIDTH, WIDTH},
	};
}

uint32_t RocWMMAWidth64MLP::width(uint32_t layer) const {
	if (layer >= N_HIDDEN_LAYERS) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP hidden layer index is out of range."
		};
	}

	return WIDTH;
}

std::pair<const __half*, MatrixLayout>
RocWMMAWidth64MLP::forward_activations(
	const Context&,
	uint32_t
) const {
	throw std::runtime_error{
		"RocWMMAWidth64MLP Phase 4A2-P1 skeleton has no qualified "
		"forward-activation context."
	};
}

json RocWMMAWidth64MLP::hyperparams() const {
	return {
		{"otype", "RocWMMAWidth64MLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_input_dims", WIDTH},
		{"n_neurons", WIDTH},
		{"n_output_dims", WIDTH},
		{"n_hidden_layers", N_HIDDEN_LAYERS},
		{"bias", true},
		{"operand_precision", "Fp16"},
		{"accumulation_precision", "Fp32"},
		{"hidden_output_precision", "Fp16"},
		{"final_output_precision", "Fp16"},
		{"parameter_elements", TOTAL_PARAMETER_ELEMENTS},
		{"runtime_architecture", "gfx1201"},
		{"selection", "explicit_otype_only"},
		{"silent_fallback", false},
		{"inference_qualified", false},
		{"forward_qualified", false},
		{"backward_qualified", false},
		{"phase", "4A2-P1-skeleton"},
	};
}

} // namespace tcnn
