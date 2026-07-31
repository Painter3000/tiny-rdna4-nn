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

#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

namespace tcnn {
namespace {

std::atomic<uint64_t> g_phase4a2_forward_view_refresh_count{0};
std::atomic<uint64_t> g_phase4a2_fused_forward_launch_count{0};
std::atomic<uint64_t> g_phase4a2_logical_weight_version{0};
std::atomic<uint64_t> g_phase4a2_packed_view_version{0};

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
	Half* output,
	float* saved_pre_activation_0,
	Half* saved_hidden_0,
	float* saved_pre_activation_1,
	Half* saved_hidden_1
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
	if (saved_pre_activation_0) {
		auto const& reg_acc = rocwmma::to_register_file(layer_0_accumulator);
		const uint32_t lane = threadIdx.x % WAVE_SIZE;
		for (uint32_t slot = 0; slot < SLOTS; ++slot) {
			const uint32_t index = lane * SLOTS + slot;
			const uint32_t row = slot + (lane >= 16 ? 8 : 0);
			const uint32_t col =
				phase4a1_p2_generated::kAccumulatorColumn[index];
			saved_pre_activation_0[
				static_cast<size_t>(blockIdx.x) * ELEMENTS +
				row * WIDTH + output_col_begin + col
			] = reg_acc[slot] + static_cast<F32>(
				bias_0[output_col_begin + col]
			);
		}
	}
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
	if (saved_hidden_0) {
		Half* saved_tile =
			saved_hidden_0 + static_cast<size_t>(blockIdx.x) * ELEMENTS;
		for (uint32_t i = threadIdx.x; i < ELEMENTS; i += THREADS) {
			saved_tile[i] = hidden_lds[i];
		}
	}
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
	if (saved_pre_activation_1) {
		auto const& reg_acc = rocwmma::to_register_file(layer_1_accumulator);
		const uint32_t lane = threadIdx.x % WAVE_SIZE;
		for (uint32_t slot = 0; slot < SLOTS; ++slot) {
			const uint32_t index = lane * SLOTS + slot;
			const uint32_t row = slot + (lane >= 16 ? 8 : 0);
			const uint32_t col =
				phase4a1_p2_generated::kAccumulatorColumn[index];
			saved_pre_activation_1[
				static_cast<size_t>(blockIdx.x) * ELEMENTS +
				row * WIDTH + output_col_begin + col
			] = reg_acc[slot] + static_cast<F32>(
				bias_1[output_col_begin + col]
			);
		}
	}
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
	if (saved_hidden_1) {
		Half* saved_tile =
			saved_hidden_1 + static_cast<size_t>(blockIdx.x) * ELEMENTS;
		for (uint32_t i = threadIdx.x; i < ELEMENTS; i += THREADS) {
			saved_tile[i] = hidden_lds[i];
		}
	}
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

__global__ void phase4a2_refresh_forward_weight_view_kernel(
	const __half* logical,
	__half* physical
) {
	const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if (element >= RocWMMAWidth64MLP::TOTAL_PARAMETER_ELEMENTS) {
		return;
	}
	const size_t weight_offsets[3] = {
		WEIGHT_0_OFFSET,
		WEIGHT_1_OFFSET,
		WEIGHT_2_OFFSET,
	};
	const size_t bias_offsets[3] = {
		BIAS_0_OFFSET,
		BIAS_1_OFFSET,
		BIAS_2_OFFSET,
	};
	for (uint32_t layer = 0; layer < 3; ++layer) {
		const size_t weight_begin = weight_offsets[layer];
		const size_t weight_end = weight_begin + WIDTH * WIDTH;
		if (element >= weight_begin && element < weight_end) {
			const uint32_t local = element - weight_begin;
			const uint32_t physical_row = local / WIDTH;
			const uint32_t physical_col = local % WIDTH;
			physical[element] = logical[
				weight_begin + physical_col * WIDTH + physical_row
			];
			return;
		}
		const size_t bias_begin = bias_offsets[layer];
		if (element >= bias_begin && element < bias_begin + WIDTH) {
			physical[element] = logical[element];
			return;
		}
	}
}

void refresh_forward_weight_view(
	hipStream_t stream,
	const __half* logical,
	__half* physical
) {
	if (!logical || !physical) {
		throw std::runtime_error{
			"Phase 4A2 forward weight view received a null parameter buffer."
		};
	}
	hipLaunchKernelGGL(
		phase4a2_refresh_forward_weight_view_kernel,
		dim3(div_round_up(
			static_cast<uint32_t>(
				RocWMMAWidth64MLP::TOTAL_PARAMETER_ELEMENTS
			),
			256u
		)),
		dim3(256),
		0,
		stream,
		logical,
		physical
	);
	CUDA_CHECK_THROW(hipGetLastError());
}

void launch_forward_with_current_view(
	hipStream_t stream,
	uint32_t rows,
	const Half* input,
	const __half* logical_params,
	__half* physical_params,
	Half* output,
	float* saved_pre_activation_0,
	Half* saved_hidden_0,
	float* saved_pre_activation_1,
	Half* saved_hidden_1,
	uint64_t logical_weight_version = 0,
	uint64_t* packed_view_version = nullptr,
	uint64_t* refresh_count = nullptr,
	uint64_t* launch_count = nullptr,
	bool suppress_refresh = false
) {
	if (!suppress_refresh) {
		refresh_forward_weight_view(
			stream,
			logical_params,
			physical_params
		);
		const uint64_t runtime_version =
			g_phase4a2_logical_weight_version.fetch_add(1) + 1;
		g_phase4a2_packed_view_version.store(runtime_version);
		g_phase4a2_forward_view_refresh_count.fetch_add(1);
		if (packed_view_version) {
			*packed_view_version = logical_weight_version;
		}
		if (refresh_count) {
			++*refresh_count;
		}
	}
	if (
		packed_view_version &&
		*packed_view_version != logical_weight_version
	) {
		throw std::runtime_error{
			"Phase 4A2 stale forward view rejected before fused kernel launch."
		};
	}
	hipLaunchKernelGGL(
		rocwmma_width64_inference_kernel,
		dim3(rows / RocWMMAWidth64MLP::TILE_ROWS),
		dim3(THREADS),
		0,
		stream,
		input,
		physical_params,
		output,
		saved_pre_activation_0,
		saved_hidden_0,
		saved_pre_activation_1,
		saved_hidden_1
	);
	CUDA_CHECK_THROW(hipGetLastError());
	g_phase4a2_fused_forward_launch_count.fetch_add(1);
	if (launch_count) {
		++*launch_count;
	}
}

__device__ inline void store_half_gradient(
	FragAcc const& accumulator,
	Half* output,
	uint32_t output_col_begin,
	const Half* relu_gate
) {
	const uint32_t lane = threadIdx.x % WAVE_SIZE;
	auto const& reg = rocwmma::to_register_file(accumulator);
	for (uint32_t slot = 0; slot < SLOTS; ++slot) {
		const uint32_t row = slot + (lane >= 16 ? 8 : 0);
		const uint32_t col = lane & 15;
		F32 value = reg[slot];
		if (
			relu_gate &&
			static_cast<float>(
				relu_gate[row * WIDTH + output_col_begin + col]
			) <= 0.0f
		) {
			value = 0.0f;
		}
		output[row * WIDTH + output_col_begin + col] =
			static_cast<Half>(value);
	}
}

__device__ inline void prepare_dw_operands(
	const Half* left,
	const Half* right,
	uint32_t rows,
	uint32_t row_begin,
	Half* left_transposed,
	Half* right_transposed
) {
	for (uint32_t i = threadIdx.x; i < WIDTH * TILE; i += THREADS) {
		const uint32_t feature = i / TILE;
		const uint32_t local_row = i % TILE;
		const uint32_t global_row = row_begin + local_row;
		left_transposed[i] = global_row < rows
			? left[global_row * WIDTH + feature]
			: static_cast<Half>(0.0f);
		right_transposed[i] = global_row < rows
			? right[global_row * WIDTH + feature]
			: static_cast<Half>(0.0f);
	}
	__syncthreads();
}

__device__ inline void compute_dw_layer(
	const Half* left,
	const Half* right,
	uint32_t rows,
	uint32_t row_begin,
	Half* left_transposed,
	Half* right_transposed,
	float* partial
) {
	const uint32_t wave = threadIdx.x / WAVE_SIZE;
	const uint32_t output_col_begin = wave * TILE;
	prepare_dw_operands(
		left,
		right,
		rows,
		row_begin,
		left_transposed,
		right_transposed
	);
	for (uint32_t input_tile = 0; input_tile < K_TILES; ++input_tile) {
		FragA a;
		FragB b;
		FragAcc accumulator;
		rocwmma::fill_fragment(accumulator, 0.0f);
		rocwmma::load_matrix_sync(
			a,
			left_transposed + input_tile * TILE * TILE,
			TILE
		);
		rocwmma::load_matrix_sync(
			b,
			right_transposed + output_col_begin * TILE,
			TILE
		);
		rocwmma::mma_sync(accumulator, a, b, accumulator);
		rocwmma::store_matrix_sync(
			partial + input_tile * TILE * WIDTH + output_col_begin,
			accumulator,
			WIDTH,
			rocwmma::mem_row_major
		);
	}
	__syncthreads();
}

__global__ void rocwmma_width64_backward_tile_kernel(
	const Half* input,
	const __half* params,
	const Half* hidden_0,
	const Half* hidden_1,
	const Half* doutput,
	uint32_t rows,
	Half* dz1_all,
	Half* dz0_all,
	Half* dinput_all,
	float* partials
) {
	__shared__ __align__(16) Half gradient[TILE * WIDTH];
	__shared__ __align__(16) Half left_transposed[WIDTH * TILE];
	__shared__ __align__(16) Half right_transposed[WIDTH * TILE];

	const uint32_t row_begin = blockIdx.x * TILE;
	const uint32_t wave = threadIdx.x / WAVE_SIZE;
	const uint32_t output_col_begin = wave * TILE;
	const uint32_t valid =
		row_begin < rows ? min(TILE, rows - row_begin) : 0;
	const Half* dy = doutput + row_begin * WIDTH;
	Half* dz1 = dz1_all + row_begin * WIDTH;
	Half* dz0 = dz0_all + row_begin * WIDTH;
	Half* dx = dinput_all ? dinput_all + row_begin * WIDTH : nullptr;
	const Half* gate_1 = hidden_1 + row_begin * WIDTH;
	const Half* gate_0 = hidden_0 + row_begin * WIDTH;

	const Half* weight_0 = reinterpret_cast<const Half*>(
		params + WEIGHT_0_OFFSET
	);
	const Half* weight_1 = reinterpret_cast<const Half*>(
		params + WEIGHT_1_OFFSET
	);
	const Half* weight_2 = reinterpret_cast<const Half*>(
		params + WEIGHT_2_OFFSET
	);

	for (uint32_t i = threadIdx.x; i < TILE * WIDTH; i += THREADS) {
		gradient[i] =
			i / WIDTH < valid ? dy[i] : static_cast<Half>(0.0f);
	}
	__syncthreads();

	FragAcc dx2;
	rocwmma::fill_fragment(dx2, 0.0f);
	for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
		FragA a;
		FragB b;
		rocwmma::load_matrix_sync(a, gradient + k_tile * TILE, WIDTH);
		rocwmma::load_matrix_sync(
			b,
			weight_2 + output_col_begin * WIDTH + k_tile * TILE,
			WIDTH
		);
		rocwmma::mma_sync(dx2, a, b, dx2);
	}
	store_half_gradient(dx2, dz1, output_col_begin, gate_1);
	__syncthreads();
	compute_dw_layer(
		hidden_1,
		doutput,
		rows,
		row_begin,
		left_transposed,
		right_transposed,
		partials + (2 * gridDim.x + blockIdx.x) * WIDTH * WIDTH
	);

	for (uint32_t i = threadIdx.x; i < TILE * WIDTH; i += THREADS) {
		gradient[i] =
			i / WIDTH < valid ? dz1[i] : static_cast<Half>(0.0f);
	}
	__syncthreads();

	FragAcc dx1;
	rocwmma::fill_fragment(dx1, 0.0f);
	for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
		FragA a;
		FragB b;
		rocwmma::load_matrix_sync(a, gradient + k_tile * TILE, WIDTH);
		rocwmma::load_matrix_sync(
			b,
			weight_1 + output_col_begin * WIDTH + k_tile * TILE,
			WIDTH
		);
		rocwmma::mma_sync(dx1, a, b, dx1);
	}
	store_half_gradient(dx1, dz0, output_col_begin, gate_0);
	__syncthreads();
	compute_dw_layer(
		hidden_0,
		dz1_all,
		rows,
		row_begin,
		left_transposed,
		right_transposed,
		partials + (1 * gridDim.x + blockIdx.x) * WIDTH * WIDTH
	);

	for (uint32_t i = threadIdx.x; i < TILE * WIDTH; i += THREADS) {
		gradient[i] =
			i / WIDTH < valid ? dz0[i] : static_cast<Half>(0.0f);
	}
	__syncthreads();

	FragAcc dx0;
	rocwmma::fill_fragment(dx0, 0.0f);
	for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
		FragA a;
		FragB b;
		rocwmma::load_matrix_sync(a, gradient + k_tile * TILE, WIDTH);
		rocwmma::load_matrix_sync(
			b,
			weight_0 + output_col_begin * WIDTH + k_tile * TILE,
			WIDTH
		);
		rocwmma::mma_sync(dx0, a, b, dx0);
	}
	if (dx) {
		store_half_gradient(dx0, dx, output_col_begin, nullptr);
	}
	__syncthreads();
	compute_dw_layer(
		input,
		dz0_all,
		rows,
		row_begin,
		left_transposed,
		right_transposed,
		partials + blockIdx.x * WIDTH * WIDTH
	);
}

__global__ void rocwmma_width64_publish_gradient_kernel(
	const float* partials,
	uint32_t tile_count,
	const Half* dz0,
	const Half* dz1,
	const Half* doutput,
	uint32_t rows,
	__half* gradients,
	uint32_t mode
) {
	const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if (element < 3 * WIDTH * WIDTH) {
		const uint32_t layer = element / (WIDTH * WIDTH);
		const uint32_t offset = element % (WIDTH * WIDTH);
		float value = 0.0f;
		for (uint32_t tile = 0; tile < tile_count; ++tile) {
			value += partials[
				(layer * tile_count + tile) * WIDTH * WIDTH + offset
			];
		}
		const size_t weight_offsets[3] = {
			WEIGHT_0_OFFSET,
			WEIGHT_1_OFFSET,
			WEIGHT_2_OFFSET,
		};
		const size_t destination = weight_offsets[layer] + offset;
		gradients[destination] = mode == 1
			? static_cast<__half>(
				static_cast<float>(gradients[destination]) + value
			)
			: static_cast<__half>(value);
	}
	if (element < 3 * WIDTH) {
		const uint32_t layer = element / WIDTH;
		const uint32_t column = element % WIDTH;
		const Half* source =
			layer == 0 ? dz0 : layer == 1 ? dz1 : doutput;
		float value = 0.0f;
		for (uint32_t row = 0; row < rows; ++row) {
			value += static_cast<float>(source[row * WIDTH + column]);
		}
		const size_t bias_offsets[3] = {
			BIAS_0_OFFSET,
			BIAS_1_OFFSET,
			BIAS_2_OFFSET,
		};
		const size_t destination = bias_offsets[layer] + column;
		gradients[destination] = mode == 1
			? static_cast<__half>(
				static_cast<float>(gradients[destination]) + value
			)
			: static_cast<__half>(value);
	}
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

void phase4a2_reset_runtime_attestation() {
	g_phase4a2_forward_view_refresh_count.store(0);
	g_phase4a2_fused_forward_launch_count.store(0);
	g_phase4a2_logical_weight_version.store(0);
	g_phase4a2_packed_view_version.store(0);
}

Phase4A2RuntimeAttestation phase4a2_runtime_attestation() {
	const uint64_t refresh =
		g_phase4a2_forward_view_refresh_count.load();
	const uint64_t launch =
		g_phase4a2_fused_forward_launch_count.load();
	const uint64_t logical =
		g_phase4a2_logical_weight_version.load();
	const uint64_t packed =
		g_phase4a2_packed_view_version.load();
	return {
		refresh,
		launch,
		logical,
		packed,
		refresh == launch && logical == packed,
	};
}

void phase4a2_test_forward_states(
	hipStream_t stream,
	uint32_t rows,
	const __half* input,
	const __half* logical_params,
	float* pre_activation_0,
	__half* hidden_0,
	float* pre_activation_1,
	__half* hidden_1,
	__half* output
) {
	if (rows == 0 || rows % RocWMMAWidth64MLP::TILE_ROWS != 0) {
		throw std::runtime_error{
			"Phase 4A2 state probe requires a nonzero batch divisible by 16."
		};
	}
	GPUMemory<__half> forward_params{
		RocWMMAWidth64MLP::TOTAL_PARAMETER_ELEMENTS
	};
	launch_forward_with_current_view(
		stream,
		rows,
		reinterpret_cast<const Half*>(input),
		logical_params,
		forward_params.data(),
		reinterpret_cast<Half*>(output),
		pre_activation_0,
		reinterpret_cast<Half*>(hidden_0),
		pre_activation_1,
		reinterpret_cast<Half*>(hidden_1)
	);
}

Phase4A2StalenessAttestation phase4a2_test_staleness_guard(
	hipStream_t stream,
	uint32_t rows,
	const __half* input,
	const __half* logical_params_before,
	const __half* logical_params_after,
	__half* output
) {
	GPUMemory<__half> physical{
		RocWMMAWidth64MLP::TOTAL_PARAMETER_ELEMENTS
	};
	Phase4A2StalenessAttestation result{};
	auto launch = [&](const __half* logical, bool suppress) {
		launch_forward_with_current_view(
			stream,
			rows,
			reinterpret_cast<const Half*>(input),
			logical,
			physical.data(),
			reinterpret_cast<Half*>(output),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			result.logical_weight_version,
			&result.packed_view_version,
			&result.forward_view_refresh_count,
			&result.fused_forward_launch_count,
			suppress
		);
	};
	++result.logical_weight_mutation_count;
	++result.logical_weight_version;
	launch(logical_params_before, false);
	++result.logical_weight_mutation_count;
	++result.logical_weight_version;
	launch(logical_params_after, false);
	result.forward_view_is_current =
		result.packed_view_version == result.logical_weight_version;
	try {
		const uint64_t stale_logical_version =
			result.logical_weight_version + 1;
		launch_forward_with_current_view(
			stream,
			rows,
			reinterpret_cast<const Half*>(input),
			logical_params_after,
			physical.data(),
			reinterpret_cast<Half*>(output),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			stale_logical_version,
			&result.packed_view_version,
			&result.forward_view_refresh_count,
			&result.fused_forward_launch_count,
			true
		);
	} catch (const std::runtime_error&) {
		result.stale_launch_rejected_before_kernel = true;
	}
	return result;
}

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

	const __half* logical_params =
		use_inference_params ? this->inference_params() : this->params();

	if (!logical_params) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP parameters were not provided."
		};
	}

	__half* selected_params = use_inference_params
		? m_forward_inference_params.data()
		: m_forward_params.data();
	launch_forward_with_current_view(
		stream,
		input.n(),
		reinterpret_cast<const Half*>(input.data()),
		logical_params,
		selected_params,
		reinterpret_cast<Half*>(output.data()),
		nullptr,
		nullptr,
		nullptr,
		nullptr
	);
}

std::unique_ptr<Context> RocWMMAWidth64MLP::forward_impl(
	hipStream_t stream,
	const GPUMatrixDynamic<__half>& input,
	GPUMatrixDynamic<__half>* output,
	bool use_inference_params,
	bool
) {
	if (
		input.layout() != MatrixLayout::ColumnMajor ||
		!input.is_contiguous() ||
		input.n() == 0 ||
		input.n() % TILE_ROWS != 0
	) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP training requires contiguous ColumnMajor "
			"input and a nonzero batch divisible by 16."
		};
	}
	const __half* logical_params =
		use_inference_params ? this->inference_params() : this->params();
	if (!logical_params) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP parameters were not provided."
		};
	}
	auto context = std::make_unique<ForwardContext>();
	context->pre_activation_0 = {
		WIDTH,
		input.n(),
		stream,
		MatrixLayout::ColumnMajor
	};
	context->hidden_0 = {
		WIDTH,
		input.n(),
		stream,
		MatrixLayout::ColumnMajor
	};
	context->pre_activation_1 = {
		WIDTH,
		input.n(),
		stream,
		MatrixLayout::ColumnMajor
	};
	context->hidden_1 = {
		WIDTH,
		input.n(),
		stream,
		MatrixLayout::ColumnMajor
	};
	if (!output) {
		context->owned_output = {
			WIDTH,
			input.n(),
			stream,
			MatrixLayout::ColumnMajor
		};
		output = &context->owned_output;
	}
	__half* selected_params = use_inference_params
		? m_forward_inference_params.data()
		: m_forward_params.data();
	launch_forward_with_current_view(
		stream,
		input.n(),
		reinterpret_cast<const Half*>(input.data()),
		logical_params,
		selected_params,
		reinterpret_cast<Half*>(output->data()),
		context->pre_activation_0.data(),
		reinterpret_cast<Half*>(context->hidden_0.data()),
		context->pre_activation_1.data(),
		reinterpret_cast<Half*>(context->hidden_1.data())
	);
	return context;
}

void RocWMMAWidth64MLP::set_params_impl(
	__half*,
	__half*,
	__half*
) {
	m_forward_params.resize(TOTAL_PARAMETER_ELEMENTS);
	m_forward_inference_params.resize(TOTAL_PARAMETER_ELEMENTS);
}

void RocWMMAWidth64MLP::backward_impl(
	hipStream_t stream,
	const Context& raw_context,
	const GPUMatrixDynamic<__half>& input,
	const GPUMatrixDynamic<__half>&,
	const GPUMatrixDynamic<__half>& doutput,
	GPUMatrixDynamic<__half>* dinput,
	bool use_inference_params,
	GradientMode mode
) {
	const auto& context = dynamic_cast<const ForwardContext&>(raw_context);
	const __half* selected_params =
		use_inference_params ? this->inference_params() : this->params();
	if (!selected_params) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP parameters were not provided."
		};
	}
	if (mode != GradientMode::Ignore && !this->gradients()) {
		throw std::runtime_error{
			"RocWMMAWidth64MLP gradient memory was not provided."
		};
	}
	const uint32_t rows = input.n();
	const uint32_t tile_count = rows / TILE_ROWS;
	GPUMatrixDynamic<__half> dz1{
		WIDTH,
		rows,
		stream,
		MatrixLayout::ColumnMajor
	};
	GPUMatrixDynamic<__half> dz0{
		WIDTH,
		rows,
		stream,
		MatrixLayout::ColumnMajor
	};
	GPUMatrixDynamic<float> partials{
		WIDTH * WIDTH,
		3 * tile_count,
		stream,
		MatrixLayout::ColumnMajor
	};
	hipLaunchKernelGGL(
		rocwmma_width64_backward_tile_kernel,
		dim3(tile_count),
		dim3(THREADS),
		0,
		stream,
		reinterpret_cast<const Half*>(input.data()),
		selected_params,
		reinterpret_cast<const Half*>(context.hidden_0.data()),
		reinterpret_cast<const Half*>(context.hidden_1.data()),
		reinterpret_cast<const Half*>(doutput.data()),
		rows,
		reinterpret_cast<Half*>(dz1.data()),
		reinterpret_cast<Half*>(dz0.data()),
		dinput ? reinterpret_cast<Half*>(dinput->data()) : nullptr,
		partials.data()
	);
	CUDA_CHECK_THROW(hipGetLastError());
	if (mode != GradientMode::Ignore) {
		const uint32_t count = 3 * WIDTH * WIDTH;
		hipLaunchKernelGGL(
			rocwmma_width64_publish_gradient_kernel,
			dim3(div_round_up(count, 256u)),
			dim3(256),
			0,
			stream,
			partials.data(),
			tile_count,
			reinterpret_cast<const Half*>(dz0.data()),
			reinterpret_cast<const Half*>(dz1.data()),
			reinterpret_cast<const Half*>(doutput.data()),
			rows,
			this->gradients(),
			mode == GradientMode::Accumulate ? 1u : 0u
		);
		CUDA_CHECK_THROW(hipGetLastError());
	}
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
		{"forward_qualified", true},
		{"backward_qualified", true},
		{"training_integration_phase", "4A2"},
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
