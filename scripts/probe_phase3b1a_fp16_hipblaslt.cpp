// TCNN_RDNA4_P3B1A_FP16_CAPABILITY_001: standalone FP16/FP32-accumulation probe.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <json/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

constexpr const char* kMarker = "TCNN_RDNA4_P3B1A_FP16_CAPABILITY_001";
constexpr int64_t kM = 16;
constexpr int64_t kN = 16;
constexpr size_t kGuardBytes = 256;
constexpr uint8_t kOutputSentinel = 0xa5;
constexpr uint8_t kOutputGuard = 0x5a;
constexpr uint8_t kWorkspaceSentinel = 0xc3;
constexpr uint8_t kWorkspaceGuard = 0x7e;
constexpr size_t kMaxWorkspace = 64ull << 20;

void hip_check(hipError_t status, const char* operation) {
	if (status != hipSuccess) {
		throw std::runtime_error{std::string{operation} + ": " + hipGetErrorString(status)};
	}
}

void lt_check(hipblasStatus_t status, const char* operation) {
	if (status != HIPBLAS_STATUS_SUCCESS) {
		throw std::runtime_error{std::string{operation} + ": hipBLASLt status " + std::to_string((int)status)};
	}
}

std::string bytes_hex(const void* pointer, size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(pointer);
	std::ostringstream stream;
	stream << std::hex << std::setfill('0');
	for (size_t i = 0; i < size; ++i) stream << std::setw(2) << (unsigned)bytes[i];
	return stream.str();
}

struct GuardedDeviceBuffer {
	void* base = nullptr;
	size_t payload_bytes = 0;
	uint8_t sentinel = 0;
	uint8_t guard = 0;

	GuardedDeviceBuffer(size_t bytes, uint8_t sentinel_, uint8_t guard_)
	: payload_bytes{bytes}, sentinel{sentinel_}, guard{guard_} {
		hip_check(hipMalloc(&base, kGuardBytes + payload_bytes + kGuardBytes), "hipMalloc guarded buffer");
		reset();
	}
	~GuardedDeviceBuffer() { if (base) hipFree(base); }
	GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
	GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
	void* data() const { return static_cast<uint8_t*>(base) + kGuardBytes; }
	void reset() {
		std::vector<uint8_t> host(kGuardBytes + payload_bytes + kGuardBytes, guard);
		std::fill(host.begin() + kGuardBytes, host.begin() + kGuardBytes + payload_bytes, sentinel);
		hip_check(hipMemcpy(base, host.data(), host.size(), hipMemcpyHostToDevice), "reset guarded buffer");
	}
	json inspect() const {
		std::vector<uint8_t> host(kGuardBytes + payload_bytes + kGuardBytes);
		hip_check(hipMemcpy(host.data(), base, host.size(), hipMemcpyDeviceToHost), "inspect guarded buffer");
		const bool prefix = std::all_of(host.begin(), host.begin() + kGuardBytes, [&](uint8_t v) { return v == guard; });
		const bool suffix = std::all_of(host.end() - kGuardBytes, host.end(), [&](uint8_t v) { return v == guard; });
		const bool unchanged = std::all_of(host.begin() + kGuardBytes, host.begin() + kGuardBytes + payload_bytes,
			[&](uint8_t v) { return v == sentinel; });
		return {{"prefix_guard_intact", prefix}, {"suffix_guard_intact", suffix},
			{"payload_unchanged", unchanged}, {"payload_bytes", payload_bytes}};
	}
};

float half_round(float value) { return __half2float(__float2half(value)); }

float pattern_a(int64_t k, int64_t row) {
	static constexpr float values[16] = {
		255.0f, 255.0f, 64.0f, 64.0f, 1.0f, 1.0f, 0.5f, 0.5f,
		32.0f, 32.0f, 0.03125f, 0.03125f, 3.1415927f, 3.1415927f, 0.33325f, 0.33325f,
	};
	const float row_adjust = 1.0f + (float)(row % 3) * 0.0009765625f;
	return half_round(values[k % 16] * row_adjust);
}

float pattern_b(int64_t k, int64_t col) {
	static constexpr float values[16] = {
		255.0f, -255.0f, 64.0f, -63.96875f, 1.0f, -0.9995117f, 0.03125f, -0.0307617f,
		2.0f, -1.9990234f, 0.03125f, -0.0307617f, 1.0f, -0.9995117f, 1.0f, -0.9995117f,
	};
	const float col_adjust = 1.0f - (float)(col % 3) * 0.00048828125f;
	return half_round(values[k % 16] * col_adjust);
}

size_t col_major(int64_t row, int64_t col, int64_t ld) { return (size_t)row + (size_t)col * ld; }

struct HostProblem {
	std::vector<__half> a;
	std::vector<__half> b;
	std::vector<double> r64;
	std::vector<float> r32;
	std::vector<float> r16;
};

HostProblem make_problem(int64_t k, bool trans_a, bool trans_b) {
	const int64_t a_rows = trans_a ? k : kM, a_cols = trans_a ? kM : k;
	const int64_t b_rows = trans_b ? kN : k, b_cols = trans_b ? k : kN;
	HostProblem result;
	result.a.resize((size_t)a_rows * a_cols);
	result.b.resize((size_t)b_rows * b_cols);
	for (int64_t row = 0; row < kM; ++row) for (int64_t inner = 0; inner < k; ++inner) {
		const __half value = __float2half(pattern_a(inner, row));
		if (trans_a) result.a[col_major(inner, row, a_rows)] = value;
		else result.a[col_major(row, inner, a_rows)] = value;
	}
	for (int64_t inner = 0; inner < k; ++inner) for (int64_t col = 0; col < kN; ++col) {
		const __half value = __float2half(pattern_b(inner, col));
		if (trans_b) result.b[col_major(col, inner, b_rows)] = value;
		else result.b[col_major(inner, col, b_rows)] = value;
	}
	result.r64.resize(kM * kN);
	result.r32.resize(kM * kN);
	result.r16.resize(kM * kN);
	for (int64_t col = 0; col < kN; ++col) for (int64_t row = 0; row < kM; ++row) {
		double acc64 = 0.0;
		float acc32 = 0.0f;
		__half acc16 = __float2half(0.0f);
		for (int64_t inner = 0; inner < k; ++inner) {
			const float av = pattern_a(inner, row), bv = pattern_b(inner, col);
			acc64 += (double)av * (double)bv;
			acc32 += av * bv;
			const __half product = __float2half(av * bv);
			acc16 = __float2half(__half2float(acc16) + __half2float(product));
		}
		const size_t index = col_major(row, col, kM);
		result.r64[index] = acc64;
		result.r32[index] = acc32;
		result.r16[index] = __half2float(acc16);
	}
	return result;
}

template <typename T> std::vector<float> to_float(const std::vector<T>& values) {
	std::vector<float> result(values.size());
	for (size_t i = 0; i < values.size(); ++i) result[i] = (float)values[i];
	return result;
}
template <> std::vector<float> to_float(const std::vector<__half>& values) {
	std::vector<float> result(values.size());
	for (size_t i = 0; i < values.size(); ++i) result[i] = __half2float(values[i]);
	return result;
}

json run_probe(const std::string& direction, int64_t k, const std::string& output_type, int fresh_repeat) {
	const bool trans_a = direction[0] == 'T';
	const bool trans_b = direction[1] == 'T';
	const hipblasOperation_t op_a = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	const hipblasOperation_t op_b = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	const int64_t a_rows = trans_a ? k : kM, a_cols = trans_a ? kM : k, lda = a_rows;
	const int64_t b_rows = trans_b ? kN : k, b_cols = trans_b ? k : kN, ldb = b_rows;
	const int64_t ldc = kM, ldd = kM;
	const hipDataType out_type = output_type == "f16" ? HIP_R_16F : HIP_R_32F;
	const size_t out_element = output_type == "f16" ? sizeof(__half) : sizeof(float);
	const size_t out_bytes = (size_t)kM * kN * out_element;
	HostProblem host = make_problem(k, trans_a, trans_b);

	void *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
	hip_check(hipMalloc(&dev_a, host.a.size() * sizeof(__half)), "hipMalloc A");
	hip_check(hipMalloc(&dev_b, host.b.size() * sizeof(__half)), "hipMalloc B");
	hip_check(hipMalloc(&dev_c, out_bytes), "hipMalloc C");
	hip_check(hipMemcpy(dev_a, host.a.data(), host.a.size() * sizeof(__half), hipMemcpyHostToDevice), "copy A");
	hip_check(hipMemcpy(dev_b, host.b.data(), host.b.size() * sizeof(__half), hipMemcpyHostToDevice), "copy B");
	hip_check(hipMemset(dev_c, 0, out_bytes), "clear C");
	GuardedDeviceBuffer output(out_bytes, kOutputSentinel, kOutputGuard);

	hipblasLtHandle_t planning = nullptr, handles[2] = {nullptr, nullptr};
	hipStream_t streams[2] = {nullptr, nullptr};
	hipblasLtMatmulDesc_t operation = nullptr;
	hipblasLtMatrixLayout_t layout_a = nullptr, layout_b = nullptr, layout_c = nullptr, layout_d = nullptr;
	hipblasLtMatmulPreference_t preference = nullptr;
	lt_check(hipblasLtCreate(&planning), "create planning handle");
	int device = 0, lt_version = 0;
	hipDeviceProp_t properties{};
	hip_check(hipGetDevice(&device), "get device");
	hip_check(hipGetDeviceProperties(&properties, device), "get device properties");
	lt_check(hipblasLtGetVersion(planning, &lt_version), "get hipBLASLt version");
	std::array<char, 128> lt_revision{};
	lt_check(hipblasLtGetGitRevision(planning, lt_revision.data()), "get hipBLASLt revision");
	for (int i = 0; i < 2; ++i) {
		hip_check(hipStreamCreate(&streams[i]), "create stream");
		lt_check(hipblasLtCreate(&handles[i]), "create execution handle");
	}
	lt_check(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F), "create operation");
	lt_check(hipblasLtMatmulDescSetAttribute(operation, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)), "set trans A");
	lt_check(hipblasLtMatmulDescSetAttribute(operation, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)), "set trans B");
	lt_check(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16F, a_rows, a_cols, lda), "layout A");
	lt_check(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16F, b_rows, b_cols, ldb), "layout B");
	lt_check(hipblasLtMatrixLayoutCreate(&layout_c, out_type, kM, kN, ldc), "layout C");
	lt_check(hipblasLtMatrixLayoutCreate(&layout_d, out_type, kM, kN, ldd), "layout D");
	lt_check(hipblasLtMatmulPreferenceCreate(&preference), "create preference");
	uint64_t maximum_workspace = kMaxWorkspace;
	lt_check(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
		&maximum_workspace, sizeof(maximum_workspace)), "set maximum workspace");
	std::array<hipblasLtMatmulHeuristicResult_t, 32> candidates{};
	int heuristic_count = 0;
	const hipblasStatus_t heuristic_status = hipblasLtMatmulAlgoGetHeuristic(planning, operation,
		layout_a, layout_b, layout_c, layout_d, preference, candidates.size(), candidates.data(), &heuristic_count);
	int selected = -1;
	for (int i = 0; i < heuristic_count; ++i) {
		if (candidates[i].state == HIPBLAS_STATUS_SUCCESS && candidates[i].workspaceSize <= kMaxWorkspace) {
			selected = i;
			break;
		}
	}
	json result = {
		{"marker", kMarker}, {"direction", direction}, {"k", k}, {"m", kM}, {"n", kN},
		{"output_type", output_type}, {"fresh_repeat", fresh_repeat},
		{"descriptor", {{"compute_type", "HIPBLAS_COMPUTE_32F"}, {"scale_type", "HIP_R_32F"},
			{"a_type", "HIP_R_16F"}, {"b_type", "HIP_R_16F"}, {"c_type", output_type == "f16" ? "HIP_R_16F" : "HIP_R_32F"},
			{"d_type", output_type == "f16" ? "HIP_R_16F" : "HIP_R_32F"},
			{"trans_a", direction.substr(0, 1)}, {"trans_b", direction.substr(1, 1)},
			{"a_rows", a_rows}, {"a_cols", a_cols}, {"b_rows", b_rows}, {"b_cols", b_cols},
			{"lda", lda}, {"ldb", ldb}, {"ldc", ldc}, {"ldd", ldd},
			{"batch_count", 1}, {"stride_a", 0}, {"stride_b", 0}, {"stride_c", 0}, {"stride_d", 0}}},
		{"heuristic_status", (int)heuristic_status}, {"heuristic_count", heuristic_count},
		{"selected_candidate", selected}, {"stream_count", 2}, {"execution_handle_count", 2},
		{"planning_handle_count", 1}, {"heuristic_queries_before_steady_state", 1},
		{"environment", {{"device", properties.name}, {"gcn_arch", properties.gcnArchName},
			{"hip_runtime_version", HIP_VERSION}, {"hipblaslt_version", lt_version},
			{"hipblaslt_git_revision", lt_revision.data()}}},
	};
	if (heuristic_status != HIPBLAS_STATUS_SUCCESS || selected < 0) {
		result["probe_state"] = "UNSUPPORTED";
	} else {
		auto& chosen = candidates[selected];
		const size_t workspace_bytes = chosen.workspaceSize;
		GuardedDeviceBuffer workspace(std::max<size_t>(workspace_bytes, 1), kWorkspaceSentinel, kWorkspaceGuard);
		void* workspace_pointer = workspace_bytes ? workspace.data() : nullptr;
		result["algorithm"] = {
			{"index", hipblaslt_ext::getIndexFromAlgo(chosen.algo)},
			{"solution_name", hipblaslt_ext::getSolutionNameFromAlgo(planning, chosen.algo)},
			{"kernel_name", hipblaslt_ext::getKernelNameFromAlgo(planning, chosen.algo)},
			{"serialized", bytes_hex(&chosen.algo, sizeof(chosen.algo))},
			{"workspace_required_bytes", workspace_bytes}, {"workspace_actual_bytes", workspace_bytes},
			{"waves_count", chosen.wavesCount},
		};
		const float alpha = 1.0f, beta = 0.0f;
		json launches = json::array();
		std::vector<float> final_output;
		for (int launch = 0; launch < 6; ++launch) {
			const int stream_index = launch % 2;
			output.reset();
			const hipblasStatus_t matmul_status = hipblasLtMatmul(handles[stream_index], operation, &alpha,
				dev_a, layout_a, dev_b, layout_b, &beta, dev_c, layout_c, output.data(), layout_d,
				&chosen.algo, workspace_pointer, workspace_bytes, streams[stream_index]);
			const hipError_t sync_status = hipStreamSynchronize(streams[stream_index]);
			json inspection = output.inspect();
			launches.push_back({{"launch", launch + 1}, {"stream_index", stream_index},
				{"stream", (uint64_t)(uintptr_t)streams[stream_index]}, {"matmul_status", (int)matmul_status},
				{"synchronize_status", (int)sync_status}, {"output_memory", inspection}});
			if (matmul_status == HIPBLAS_STATUS_SUCCESS && sync_status == hipSuccess) {
				if (output_type == "f16") {
					std::vector<__half> values(kM * kN);
					hip_check(hipMemcpy(values.data(), output.data(), out_bytes, hipMemcpyDeviceToHost), "copy D f16");
					final_output = to_float(values);
				} else {
					std::vector<float> values(kM * kN);
					hip_check(hipMemcpy(values.data(), output.data(), out_bytes, hipMemcpyDeviceToHost), "copy D f32");
					final_output = values;
				}
			}
		}
		result["launches"] = launches;
		result["workspace_memory"] = workspace.inspect();
		result["steady_state"] = {{"launches", 4}, {"new_handles", 0}, {"new_heuristic_queries", 0},
			{"descriptor_growth", 0}, {"workspace_growth_bytes", 0}};
		result["gpu_output"] = final_output;
		std::vector<float> projected_r32(kM * kN), projected_r16(kM * kN);
		for (size_t i = 0; i < projected_r32.size(); ++i) {
			projected_r32[i] = output_type == "f16" ? half_round(host.r32[i]) : host.r32[i];
			projected_r16[i] = output_type == "f16" ? half_round(host.r16[i]) : host.r16[i];
		}
		result["reference_r64"] = host.r64;
		result["reference_r32_projected_to_d"] = projected_r32;
		result["reference_r16_projected_to_d"] = projected_r16;
		result["probe_state"] = "EXECUTED";
	}

	if (preference) hipblasLtMatmulPreferenceDestroy(preference);
	if (layout_d) hipblasLtMatrixLayoutDestroy(layout_d);
	if (layout_c) hipblasLtMatrixLayoutDestroy(layout_c);
	if (layout_b) hipblasLtMatrixLayoutDestroy(layout_b);
	if (layout_a) hipblasLtMatrixLayoutDestroy(layout_a);
	if (operation) hipblasLtMatmulDescDestroy(operation);
	for (int i = 0; i < 2; ++i) {
		if (handles[i]) hipblasLtDestroy(handles[i]);
		if (streams[i]) hipStreamDestroy(streams[i]);
	}
	if (planning) hipblasLtDestroy(planning);
	if (dev_c) hipFree(dev_c);
	if (dev_b) hipFree(dev_b);
	if (dev_a) hipFree(dev_a);
	return result;
}

} // namespace

int main(int argc, char** argv) {
	json result = {{"marker", kMarker}};
	try {
		if (argc != 5) throw std::runtime_error{"usage: probe DIRECTION K OUTPUT_TYPE FRESH_REPEAT"};
		const std::string direction = argv[1], output_type = argv[3];
		const int64_t k = std::stoll(argv[2]);
		const int repeat = std::stoi(argv[4]);
		if (direction != "NN" && direction != "NT" && direction != "TN") throw std::runtime_error{"invalid direction"};
		if (output_type != "f16" && output_type != "f32") throw std::runtime_error{"invalid output type"};
		result = run_probe(direction, k, output_type, repeat);
		result["process_status"] = "PASS";
	} catch (const std::exception& error) {
		result["process_status"] = "ERROR";
		result["error"] = error.what();
	}
	std::cout << result.dump() << std::endl;
	return result.value("process_status", "ERROR") == "PASS" ? 0 : 1;
}
