#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>

#include <stdexcept>

namespace tcnn {

template <uint32_t WIDTH, bool COMPUTE_BIAS>
__global__ void fused_relu_backward_biasgrad_stage1(const float* upstream, const float* relu_mask_source,
	float* dz, float* partial_db, uint32_t batch) {
	constexpr uint32_t rows_per_tile = 256 / WIDTH;
	__shared__ float tile[rows_per_tile][WIDTH];
	const uint32_t feature = threadIdx.x;
	const uint32_t sample = blockIdx.x * rows_per_tile + threadIdx.y;
	float value = 0.0f;
	if (sample < batch) {
		const uint32_t index = sample * WIDTH + feature;
		value = relu_mask_source[index] > 0.0f ? upstream[index] : 0.0f;
		dz[index] = value;
	}
	if constexpr (COMPUTE_BIAS) {
		tile[threadIdx.y][feature] = value;
		__syncthreads();
		if (threadIdx.y == 0) {
			float sum = 0.0f;
#pragma unroll
			for (uint32_t row = 0; row < rows_per_tile; ++row) sum += tile[row][feature];
			partial_db[blockIdx.x * WIDTH + feature] = sum;
		}
	}
}

template <uint32_t WIDTH, bool ACCUMULATE>
__global__ void finalize_biasgrad(const float* partial_db, float* db, uint32_t n_partials) {
	const uint32_t feature = threadIdx.x;
	if (feature >= WIDTH) return;
	float sum = 0.0f;
	for (uint32_t partial = 0; partial < n_partials; ++partial) sum += partial_db[partial * WIDTH + feature];
	if constexpr (ACCUMULATE) db[feature] += sum; else db[feature] = sum;
}

template <uint32_t WIDTH>
void launch_width(hipStream_t stream, const float* upstream, const float* mask, float* dz,
	float* partial, float* db, uint32_t batch, GradientMode mode) {
	constexpr uint32_t rows_per_tile = 256 / WIDTH;
	const uint32_t n_partials = div_round_up(batch, rows_per_tile);
	const dim3 block{WIDTH, rows_per_tile};
	if (mode == GradientMode::Ignore) {
		fused_relu_backward_biasgrad_stage1<WIDTH, false><<<n_partials, block, 0, stream>>>(upstream, mask, dz, nullptr, batch);
	} else {
		fused_relu_backward_biasgrad_stage1<WIDTH, true><<<n_partials, block, 0, stream>>>(upstream, mask, dz, partial, batch);
		if (mode == GradientMode::Accumulate) finalize_biasgrad<WIDTH, true><<<1, WIDTH, 0, stream>>>(partial, db, n_partials);
		else finalize_biasgrad<WIDTH, false><<<1, WIDTH, 0, stream>>>(partial, db, n_partials);
	}
}

void launch_fused_relu_backward(uint32_t width, hipStream_t stream, const float* upstream, const float* mask,
	float* dz, float* partial, float* db, uint32_t batch, GradientMode mode) {
	switch (width) {
		case 16: launch_width<16>(stream,upstream,mask,dz,partial,db,batch,mode); break;
		case 32: launch_width<32>(stream,upstream,mask,dz,partial,db,batch,mode); break;
		case 64: launch_width<64>(stream,upstream,mask,dz,partial,db,batch,mode); break;
		case 128: launch_width<128>(stream,upstream,mask,dz,partial,db,batch,mode); break;
		default: throw std::runtime_error{"Unsupported Phase-3A4 fused width."};
	}
}

} // namespace tcnn
