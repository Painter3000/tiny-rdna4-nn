/*
 * TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001
 *
 * Width-64, three-linear-layer rocWMMA inference kernel for gfx1201.
 * The Phase 4A1 dataflow is retained while production-only adaptations
 * remove diagnostic arguments, consume the normal FP16 parameter buffer,
 * launch one block per 16 samples, and store the public FP16 output.
 */
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h>
#include <tiny-cuda-nn/networks/rocwmma_width64_mlp.h>
#include <tiny-cuda-nn/random.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

namespace tcnn {
namespace {

constexpr uint32_t WIDTH = RocWMMAWidth64MLP::WIDTH;
constexpr uint32_t TILE = RocWMMAWidth64MLP::TILE_ROWS;
constexpr uint32_t K_TILES = WIDTH / TILE;
constexpr uint32_t WAVES = 4;
constexpr uint32_t WAVE_SIZE = 32;
constexpr uint32_t THREADS = WAVES * WAVE_SIZE;
constexpr uint32_t SLOTS = 8;
constexpr uint32_t ELEMENTS = TILE * WIDTH;

constexpr size_t WEIGHT_0_OFFSET =
	RocWMMAWidth64MLP::WEIGHT_0_OFFSET;
constexpr size_t BIAS_0_OFFSET =
	RocWMMAWidth64MLP::BIAS_0_OFFSET;
constexpr size_t WEIGHT_1_OFFSET =
	RocWMMAWidth64MLP::WEIGHT_1_OFFSET;
constexpr size_t BIAS_1_OFFSET =
	RocWMMAWidth64MLP::BIAS_1_OFFSET;
constexpr size_t WEIGHT_2_OFFSET =
	RocWMMAWidth64MLP::WEIGHT_2_OFFSET;
constexpr size_t BIAS_2_OFFSET =
	RocWMMAWidth64MLP::BIAS_2_OFFSET;

constexpr const char* PRODUCTION_MARKER =
	"TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001";
constexpr const char* P4_SOURCE_SHA256 =
	"54e03ee731046bb007d0c554c6d1e6ec2dea99d4f4fe150bb60563e36c4b3382";
constexpr const char* P4_MAPPING_HEADER_SHA256 =
	"f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517";

using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;

using FragA = rocwmma::fragment<
	rocwmma::matrix_a,
	TILE,
	TILE,
	TILE,
	Half,
	rocwmma::row_major
>;

using FragB = rocwmma::fragment<
	rocwmma::matrix_b,
	TILE,
	TILE,
	TILE,
	Half,
	rocwmma::col_major
>;

using FragAcc = rocwmma::fragment<
	rocwmma::accumulator,
	TILE,
	TILE,
	TILE,
	F32
>;

using RegA = rocwmma::apply_register_file_t<FragA>;
using RegAcc = rocwmma::apply_register_file_t<FragAcc>;

static_assert(sizeof(Half) == sizeof(__half));
static_assert(WIDTH == 64);
static_assert(TILE == 16);
static_assert(K_TILES == 4);
static_assert(THREADS == 128);
static_assert(ELEMENTS * sizeof(Half) == 2048);
// TCNN_RDNA4_P4A2_P2_HIP_HOST_PASS_HOTFIX_005:
// The HIP host pass has no target-specific Wave32 geometry.
// Validate the gfx1201/Wave32 register view only in the device pass.
#if defined(__HIP_DEVICE_COMPILE__)
static_assert(RegA::size() == SLOTS);
static_assert(RegAcc::size() == SLOTS);
#endif
static_assert(RocWMMAWidth64MLP::TOTAL_PARAMETER_ELEMENTS == 12480);

template <bool APPLY_RELU>
__device__ inline void accumulator_bias_to_matrix_a(
	FragAcc const& accumulator,
	const __half* bias,
	uint32_t output_col_begin,
	RegA& output_reg_a
) {
	const uint32_t lane = threadIdx.x % WAVE_SIZE;
	auto const& reg_acc = rocwmma::to_register_file(accumulator);

	F32 epilogue[SLOTS];

	for (uint32_t slot = 0; slot < SLOTS; ++slot) {
		const uint32_t acc_index = lane * SLOTS + slot;
		const uint32_t local_col =
			phase4a1_p2_generated::kAccumulatorColumn[acc_index];
		const uint32_t global_col = output_col_begin + local_col;
		const F32 biased =
			reg_acc[slot] + static_cast<F32>(bias[global_col]);
		epilogue[slot] =
			APPLY_RELU && biased < 0.0f ? 0.0f : biased;
	}

	for (
		uint32_t target_slot = 0;
		target_slot < SLOTS;
		++target_slot
	) {
		const uint32_t target_index =
			lane * SLOTS + target_slot;
		const uint32_t desired_lane =
			phase4a1_p2_generated::kAccLaneForATargetA[
				target_index
			];
		const uint32_t desired_slot =
			phase4a1_p2_generated::kAccSlotForATargetA[
				target_index
			];

		F32 selected = 0.0f;

		for (
			uint32_t candidate_slot = 0;
			candidate_slot < SLOTS;
			++candidate_slot
		) {
			const F32 shuffled = __shfl(
				epilogue[candidate_slot],
				desired_lane,
				WAVE_SIZE
			);

			if (candidate_slot == desired_slot) {
				selected = shuffled;
			}
		}

		output_reg_a[target_slot] = static_cast<Half>(selected);
	}
}

__global__ void rocwmma_width64_inference_kernel(
	const Half* input,
	const __half* params,
	Half* output
) {
	__shared__ __align__(16) Half hidden_lds[ELEMENTS];

	const uint32_t wave = threadIdx.x / WAVE_SIZE;
	const uint32_t output_col_begin = wave * TILE;

	if (
		warpSize != WAVE_SIZE ||
		blockDim.x != THREADS ||
		wave >= WAVES
	) {
		return;
	}

	const Half* input_tile =
		input + static_cast<size_t>(blockIdx.x) * ELEMENTS;
	Half* output_tile =
		output + static_cast<size_t>(blockIdx.x) * ELEMENTS;

	const Half* weight_0 = reinterpret_cast<const Half*>(
		params + WEIGHT_0_OFFSET
	);
	const __half* bias_0 = params + BIAS_0_OFFSET;
	const Half* weight_1 = reinterpret_cast<const Half*>(
		params + WEIGHT_1_OFFSET
	);
	const __half* bias_1 = params + BIAS_1_OFFSET;
	const Half* weight_2 = reinterpret_cast<const Half*>(
		params + WEIGHT_2_OFFSET
	);
	const __half* bias_2 = params + BIAS_2_OFFSET;

	// Layer 0: global input -> hidden 1 in the single LDS buffer.
	FragAcc layer_0_accumulator;
	rocwmma::fill_fragment(layer_0_accumulator, 0.0f);

	for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
		FragA frag_a;
		FragB frag_b;
		const Half* a_tile = input_tile + k_tile * TILE;
		const Half* b_tile =
			weight_0 + output_col_begin * WIDTH + k_tile * TILE;
		rocwmma::load_matrix_sync(frag_a, a_tile, WIDTH);
		rocwmma::load_matrix_sync(frag_b, b_tile, WIDTH);
		rocwmma::mma_sync(
			layer_0_accumulator,
			frag_a,
			frag_b,
			layer_0_accumulator
		);
	}

	RegA hidden_1_reg_a;
	accumulator_bias_to_matrix_a<true>(
		layer_0_accumulator,
		bias_0,
		output_col_begin,
		hidden_1_reg_a
	);

	auto const& hidden_1_frag_a =
		rocwmma::from_register_file<FragA>(hidden_1_reg_a);

	rocwmma::store_matrix_sync(
		hidden_lds + output_col_begin,
		hidden_1_frag_a,
		WIDTH,
		rocwmma::mem_row_major
	);

	// Barrier 1: publish hidden 1 to all four Wave32 waves.
	__syncthreads();

	// Layer 1: hidden 1 from LDS -> FP32 accumulation.
	FragAcc layer_1_accumulator;
	rocwmma::fill_fragment(layer_1_accumulator, 0.0f);

	for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
		FragA frag_a;
		FragB frag_b;
		const Half* a_tile = hidden_lds + k_tile * TILE;
		const Half* b_tile =
			weight_1 + output_col_begin * WIDTH + k_tile * TILE;
		rocwmma::load_matrix_sync(frag_a, a_tile, WIDTH);
		rocwmma::load_matrix_sync(frag_b, b_tile, WIDTH);
		rocwmma::mma_sync(
			layer_1_accumulator,
			frag_a,
			frag_b,
			layer_1_accumulator
		);
	}

	// Barrier 2: every wave completed all hidden-1 LDS reads before
	// the same physical buffer is reused for hidden 2.
	__syncthreads();

	RegA hidden_2_reg_a;
	accumulator_bias_to_matrix_a<true>(
		layer_1_accumulator,
		bias_1,
		output_col_begin,
		hidden_2_reg_a
	);

	auto const& hidden_2_frag_a =
		rocwmma::from_register_file<FragA>(hidden_2_reg_a);

	rocwmma::store_matrix_sync(
		hidden_lds + output_col_begin,
		hidden_2_frag_a,
		WIDTH,
		rocwmma::mem_row_major
	);

	// Barrier 3: publish hidden 2 after safe single-buffer reuse.
	__syncthreads();

	// Layer 2: hidden 2 from LDS -> final public FP16 output.
	FragAcc layer_2_accumulator;
	rocwmma::fill_fragment(layer_2_accumulator, 0.0f);

	for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
		FragA frag_a;
		FragB frag_b;
		const Half* a_tile = hidden_lds + k_tile * TILE;
		const Half* b_tile =
			weight_2 + output_col_begin * WIDTH + k_tile * TILE;
		rocwmma::load_matrix_sync(frag_a, a_tile, WIDTH);
		rocwmma::load_matrix_sync(frag_b, b_tile, WIDTH);
		rocwmma::mma_sync(
			layer_2_accumulator,
			frag_a,
			frag_b,
			layer_2_accumulator
		);
	}

	RegA output_reg_a;
	accumulator_bias_to_matrix_a<false>(
		layer_2_accumulator,
		bias_2,
		output_col_begin,
		output_reg_a
	);

	auto const& output_frag_a =
		rocwmma::from_register_file<FragA>(output_reg_a);

	rocwmma::store_matrix_sync(
		output_tile + output_col_begin,
		output_frag_a,
		WIDTH,
		rocwmma::mem_row_major
	);
}

bool is_exact_gfx1201(const char* raw_arch) {
	if (!raw_arch) {
		return false;
	}

	const std::string arch{raw_arch};
	return arch == "gfx1201" || arch.rfind("gfx1201:", 0) == 0;
}

[[noreturn]] void fail_training_forward() {
	throw std::runtime_error{
		"RocWMMAWidth64MLP Phase 4A2-P2: training forward is not "
		"qualified; use inference/no-grad. No fallback was executed."
	};
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
			"RocWMMAWidth64MLP supports gfx1201 only, but the active "
			"device reports {}.",
			properties.gcnArchName
		)};
	}

	if (input_width != WIDTH || hidden_width != WIDTH || output_width != WIDTH) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires input, hidden, and output "
			"widths of 64."
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
	hipStream_t stream,
	const GPUMatrixDynamic<__half>& input,
	GPUMatrixDynamic<__half>& output,
	bool use_inference_params
) {
	if (
		input.layout() != MatrixLayout::ColumnMajor ||
		output.layout() != MatrixLayout::ColumnMajor
	) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires ColumnMajor input and output."
		};
	}

	if (!input.is_contiguous() || !output.is_contiguous()) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires contiguous input and output."
		};
	}

	if (input.n() == 0 || input.n() % TILE_ROWS != 0) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP requires a nonzero batch divisible by 16."
		};
	}

	const __half* selected_params =
		use_inference_params ? this->inference_params() : this->params();

	if (!selected_params) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP parameters were not provided."
		};
	}

	const dim3 grid{input.n() / TILE_ROWS};
	const dim3 block{THREADS};

	hipLaunchKernelGGL(
		rocwmma_width64_inference_kernel,
		grid,
		block,
		0,
		stream,
		reinterpret_cast<const Half*>(input.data()),
		selected_params,
		reinterpret_cast<Half*>(output.data())
	);

	CUDA_CHECK_THROW(hipGetLastError());
}

std::unique_ptr<Context> RocWMMAWidth64MLP::forward_impl(
	hipStream_t,
	const GPUMatrixDynamic<__half>&,
	GPUMatrixDynamic<__half>*,
	bool,
	bool
) {
	fail_training_forward();
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
		"RocWMMAWidth64MLP Phase 4A2-P2: backward is not qualified; "
		"no fallback was executed."
	};
}

void RocWMMAWidth64MLP::initialize_params(
	pcg32& rnd,
	float* params_full_precision,
	float scale
) {
	if (!params_full_precision) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP parameter initialization received a "
			"null pointer."
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
		"RocWMMAWidth64MLP Phase 4A2-P2 inference path does not "
		"materialize public hidden activations."
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
		{"bias_storage_precision", "Fp16"},
		{"hidden_output_precision", "Fp16"},
		{"final_output_precision", "Fp16"},
		{"parameter_elements", TOTAL_PARAMETER_ELEMENTS},
		{"parameter_layout", "W0,b0,W1,b1,W2,b2"},
		{"weight_layout", "ColumnMajor64x64"},
		{"network_input_layout", "ColumnMajor64xBatch"},
		{"batch_tile_rows", TILE_ROWS},
		{"threads_per_block", THREADS},
		{"wave_size", WAVE_SIZE},
		{"waves_per_block", WAVES},
		{"lds_bytes", ELEMENTS * sizeof(Half)},
		{"source_barriers", 3},
		{"runtime_architecture", "gfx1201"},
		{"selection", "explicit_otype_only"},
		{"silent_fallback", false},
		{"inference_qualified", true},
		{"forward_qualified", false},
		{"backward_qualified", false},
		{"caller_stream", true},
		{"host_synchronization", false},
		{"diagnostic_oracle_arguments", false},
		{"diagnostic_counters", false},
		{"phase4a1_p4_source_sha256", P4_SOURCE_SHA256},
		{"mapping_header_sha256", P4_MAPPING_HEADER_SHA256},
		{"marker", PRODUCTION_MARKER},
		{"phase", "4A2-P2-production-inference"},
	};
}

} // namespace tcnn
