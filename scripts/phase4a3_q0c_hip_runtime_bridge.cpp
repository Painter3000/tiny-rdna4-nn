// TCNN_RDNA4_P4A3_Q0C_APPARATUS_REDESIGN_001
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <ctime>

namespace {
uint64_t raw_ns() {
	timespec ts{};
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
		static_cast<uint64_t>(ts.tv_nsec);
}
}

extern "C" unsigned q0c_schedule_auto() {
	return static_cast<unsigned>(hipDeviceScheduleAuto);
}
extern "C" unsigned q0c_schedule_spin() {
	return static_cast<unsigned>(hipDeviceScheduleSpin);
}
extern "C" unsigned q0c_schedule_mask() {
	return static_cast<unsigned>(hipDeviceScheduleMask);
}
extern "C" int q0c_set_schedule(unsigned value) {
	return static_cast<int>(hipSetDeviceFlags(value));
}
extern "C" int q0c_get_flags(unsigned* value) {
	return static_cast<int>(hipGetDeviceFlags(value));
}
extern "C" const char* q0c_error_name(int code) {
	return hipGetErrorName(static_cast<hipError_t>(code));
}
extern "C" const char* q0c_error_string(int code) {
	return hipGetErrorString(static_cast<hipError_t>(code));
}

extern "C" int q0c_measure_floors(
	std::size_t count,
	uint64_t* timer_ns,
	uint64_t* empty_sync_ns,
	uint64_t* minimal_gpu_ns
) {
	if (!count || !timer_ns || !empty_sync_ns || !minimal_gpu_ns) {
		return static_cast<int>(hipErrorInvalidValue);
	}
	hipStream_t stream{};
	void* storage = nullptr;
	hipError_t status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
	if (status != hipSuccess) return static_cast<int>(status);
	status = hipMalloc(&storage, sizeof(uint32_t));
	if (status != hipSuccess) {
		const hipError_t ignored = hipStreamDestroy(stream);
		(void)ignored;
		return static_cast<int>(status);
	}
	status = hipMemsetAsync(storage, 0, sizeof(uint32_t), stream);
	if (status == hipSuccess) status = hipStreamSynchronize(stream);
	if (status != hipSuccess) {
		const hipError_t ignored_free = hipFree(storage);
		const hipError_t ignored_stream = hipStreamDestroy(stream);
		(void)ignored_free; (void)ignored_stream;
		return static_cast<int>(status);
	}
	for (std::size_t i = 0; i < count; ++i) {
		const uint64_t a = raw_ns();
		const uint64_t b = raw_ns();
		timer_ns[i] = b - a;

		const uint64_t c = raw_ns();
		status = hipStreamSynchronize(stream);
		const uint64_t d = raw_ns();
		if (status != hipSuccess) break;
		empty_sync_ns[i] = d - c;

		const uint64_t e = raw_ns();
		status = hipMemsetAsync(
			storage, static_cast<int>(i & 0xffu), sizeof(uint32_t), stream
		);
		if (status == hipSuccess) status = hipStreamSynchronize(stream);
		const uint64_t f = raw_ns();
		if (status != hipSuccess) break;
		minimal_gpu_ns[i] = f - e;
	}
	const hipError_t free_status = hipFree(storage);
	const hipError_t stream_status = hipStreamDestroy(stream);
	if (status != hipSuccess) return static_cast<int>(status);
	if (free_status != hipSuccess) return static_cast<int>(free_status);
	return static_cast<int>(stream_status);
}
