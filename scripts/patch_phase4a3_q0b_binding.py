#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pathlib
import shutil

MARKER = "TCNN_RDNA4_P4A3_Q0B_TEST_ONLY_BINDING_HOOK_001"
INCLUDE_ANCHOR = "#include <pybind11/functional.h>\n"
INCLUDES = """#include <pybind11/functional.h>
#include <chrono>
#include <thread>
#include <vector>
"""
METHOD_ANCHOR = "\ttorch::Tensor initial_params(size_t seed) {\n"
METHOD = r'''
	// TCNN_RDNA4_P4A3_Q0B_TEST_ONLY_BINDING_HOOK_001
	std::tuple<nlohmann::json, torch::Tensor>
	phase4a3_q0b_benchmark_inference(
		torch::Tensor input,
		torch::Tensor params,
		uint32_t iterations,
		bool synchronize_each,
		uint64_t gap_ns
	) {
		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(input.dim() == 2);
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(iterations > 0);
		CHECK_THROW(input.device() == params.device());

		const at::Device device = input.device();
#if defined(__HIP_PLATFORM_AMD__)
		const c10::hip::HIPGuardMasqueradingAsCUDA device_guard{device};
		hipStream_t stream =
			c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();
#else
		const at::cuda::CUDAGuard device_guard{device};
		hipStream_t stream = at::cuda::getCurrentCUDAStream();
#endif
		const uint32_t batch_size = input.size(0);
		const uint32_t batch_size_granularity =
			tcnn::cpp::batch_size_granularity();
		CHECK_THROW(batch_size_granularity > 0);
		CHECK_THROW(batch_size % batch_size_granularity == 0);
		torch::Tensor output = torch::empty(
			{batch_size, n_output_dims()},
			torch::TensorOptions()
				.dtype(c10_output_precision())
				.device(device)
		);

		auto launch = [&]() {
			m_module->inference(
				stream,
				batch_size,
				input.data_ptr<float>(),
				void_data_ptr(output),
				void_data_ptr(params)
			);
		};

		nlohmann::json record;
		record["marker"] =
			"TCNN_RDNA4_P4A3_Q0B_TEST_ONLY_BINDING_HOOK_001";
		record["iterations"] = iterations;
		record["synchronize_each"] = synchronize_each;
		record["gap_ns"] = gap_ns;
		record["batch_size"] = batch_size;
		record["batch_size_granularity"] = batch_size_granularity;

		if (synchronize_each) {
			std::vector<uint64_t> host_ns;
			host_ns.reserve(iterations);
			for (uint32_t i = 0; i < iterations; ++i) {
				if (gap_ns) {
					std::this_thread::sleep_for(
						std::chrono::nanoseconds{static_cast<std::chrono::nanoseconds::rep>(gap_ns)}
					);
				}
				const auto begin = std::chrono::steady_clock::now();
				launch();
				CHECK_THROW(hipStreamSynchronize(stream) == hipSuccess);
				const auto end = std::chrono::steady_clock::now();
				host_ns.push_back(static_cast<uint64_t>(
					std::chrono::duration_cast<std::chrono::nanoseconds>(
						end - begin
					).count()
				));
			}
			record["host_ns"] = host_ns;
			return {record, output};
		}

		hipEvent_t start{};
		hipEvent_t stop{};
		CHECK_THROW(hipEventCreate(&start) == hipSuccess);
		CHECK_THROW(hipEventCreate(&stop) == hipSuccess);
		CHECK_THROW(hipEventRecord(start, stream) == hipSuccess);
		const auto host_begin = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < iterations; ++i) {
			launch();
		}
		const auto host_submitted = std::chrono::steady_clock::now();
		CHECK_THROW(hipEventRecord(stop, stream) == hipSuccess);
		CHECK_THROW(hipEventSynchronize(stop) == hipSuccess);
		const auto host_complete = std::chrono::steady_clock::now();
		float event_ms = 0.0f;
		CHECK_THROW(hipEventElapsedTime(&event_ms, start, stop) == hipSuccess);
		CHECK_THROW(hipEventDestroy(start) == hipSuccess);
		CHECK_THROW(hipEventDestroy(stop) == hipSuccess);

		record["event_ms"] = event_ms;
		record["host_submission_ns"] = static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				host_submitted - host_begin
			).count()
		);
		record["host_total_ns"] = static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				host_complete - host_begin
			).count()
		);
		return {record, output};
	}

'''
PYBIND_ANCHOR = '\t\t.def("fwd", &Module::fwd)\n'
PYBIND = (
    '\t\t.def("fwd", &Module::fwd)\n'
    '\t\t.def("phase4a3_q0b_benchmark_inference", '
    '&Module::phase4a3_q0b_benchmark_inference)\n'
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path, required=True)
    parser.add_argument("--backup", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("apply", "restore"), required=True)
    args = parser.parse_args()

    if args.mode == "restore":
        if not args.backup.is_file():
            raise RuntimeError("Backup is missing")
        shutil.copy2(args.backup, args.file)
        print("PHASE4A3_Q0B_BINDING_RESTORED: PASS")
        return 0

    text = args.file.read_text()
    if MARKER in text:
        raise RuntimeError("Q0b marker already present")
    for anchor in (INCLUDE_ANCHOR, METHOD_ANCHOR, PYBIND_ANCHOR):
        if text.count(anchor) != 1:
            raise RuntimeError(f"Expected exactly one anchor: {anchor!r}")
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.file, args.backup)
    text = text.replace(INCLUDE_ANCHOR, INCLUDES, 1)
    text = text.replace(METHOD_ANCHOR, METHOD + METHOD_ANCHOR, 1)
    text = text.replace(PYBIND_ANCHOR, PYBIND, 1)
    args.file.write_text(text)
    if MARKER not in text:
        raise RuntimeError("Q0b marker insertion failed")
    print("PHASE4A3_Q0B_TEST_ONLY_BINDING_PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
