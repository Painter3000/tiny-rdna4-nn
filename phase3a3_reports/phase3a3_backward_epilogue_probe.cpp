#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void hip_ok(hipError_t s, const char* what) {
	if (s != hipSuccess) { std::cerr << what << ": " << hipGetErrorString(s) << "\n"; std::exit(2); }
}

struct Buffer {
	void* p = nullptr;
	explicit Buffer(size_t n) { hip_ok(hipMalloc(&p, n), "hipMalloc"); }
	~Buffer() { if (p) hipFree(p); }
};

const char* status_name(hipblasStatus_t s) {
	switch (s) {
		case HIPBLAS_STATUS_SUCCESS: return "success";
		case HIPBLAS_STATUS_INVALID_VALUE: return "invalid_value";
		case HIPBLAS_STATUS_NOT_SUPPORTED: return "not_supported";
		default: return "other";
	}
}

struct Candidate { const char* name; uint32_t value; bool declared; };

} // namespace

int main(int argc, char** argv) {
	const std::string output = argc > 1 ? argv[1] : "backward_epilogue_probe.json";
	hipDeviceProp_t prop{};
	hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");

	// ROCm 7.2 declares no DRELU values. 136/152 are probed deliberately as
	// the bit-pattern counterparts to RELU_AUX and DGELU_BGRAD, not assumed
	// to be supported API constants.
	const std::vector<Candidate> candidates = {
		{"RELU_AUX_BIAS", static_cast<uint32_t>(HIPBLASLT_EPILOGUE_RELU_AUX_BIAS), true},
		{"DRELU", 136u, false},
		{"DRELU_BGRAD", 152u, false},
		{"BGRADA", static_cast<uint32_t>(HIPBLASLT_EPILOGUE_BGRADA), true},
		{"BGRADB", static_cast<uint32_t>(HIPBLASLT_EPILOGUE_BGRADB), true},
	};
	const std::vector<int64_t> widths = {16, 32, 64, 128};
	const std::vector<int64_t> batches = {1, 7, 64, 257, 1024, 4096};
	const std::vector<const char*> roles = {"forward_hidden", "hidden_dx", "dw", "output_dx", "output_dw", "column_major_transpose", "row_major_fallback"};

	hipblasLtHandle_t handle = nullptr;
	if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) return 3;
	Buffer storage(128ull * 4096ull * sizeof(float) * 4ull);
	Buffer bias(4096ull * sizeof(float));
	Buffer aux(128ull * 4096ull * sizeof(float));
	std::ofstream out(output);
	hip_ok(hipMemset(storage.p, 0, storage.p ? 128ull * 4096ull * sizeof(float) * 4ull : 0), "hipMemset storage");
	hip_ok(hipMemset(aux.p, 0, 128ull * 4096ull * sizeof(float)), "hipMemset aux");
	hip_ok(hipMemset(bias.p, 0, 4096ull * sizeof(float)), "hipMemset bias");
	out << "{\n  \"schema\":1,\n  \"device\":\"" << prop.name << "\",\n  \"arch\":\"" << prop.gcnArchName
	    << "\",\n  \"rocm_header_has_drelu\":false,\n  \"results\":[\n";
	bool first = true;
	for (const auto& candidate : candidates) for (const auto* role : roles) for (int64_t width : widths) for (int64_t batch : batches) for (int stream_kind = 0; stream_kind < 2; ++stream_kind) {
		hipStream_t stream = nullptr;
		if (stream_kind) hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate");
		hipblasLtMatmulDesc_t desc = nullptr;
		hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
		hipblasLtMatmulPreference_t pref = nullptr;
		hipblasStatus_t create = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
		hipblasStatus_t set = create;
		hipblasStatus_t heuristic = create;
		hipblasStatus_t launch = HIPBLAS_STATUS_NOT_SUPPORTED;
		int returned = 0;
		hipblasLtMatmulHeuristicResult_t selected{};
		if (create == HIPBLAS_STATUS_SUCCESS) {
			const auto epilogue = static_cast<hipblasLtEpilogue_t>(candidate.value);
			set = hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
			void* bp = bias.p; void* ap = aux.p; int64_t aux_ld = width; hipDataType type = HIP_R_32F;
			if (set == HIPBLAS_STATUS_SUCCESS) set = hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp));
			if (set == HIPBLAS_STATUS_SUCCESS) set = hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &type, sizeof(type));
			if (set == HIPBLAS_STATUS_SUCCESS) set = hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &ap, sizeof(ap));
			if (set == HIPBLAS_STATUS_SUCCESS) set = hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
			if (set == HIPBLAS_STATUS_SUCCESS) {
				hipblasLtMatrixLayoutCreate(&a, HIP_R_32F, width, width, width);
				hipblasLtMatrixLayoutCreate(&b, HIP_R_32F, width, batch, width);
				hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, width, batch, width);
				hipblasLtMatrixLayoutCreate(&d, HIP_R_32F, width, batch, width);
				hipblasLtMatmulPreferenceCreate(&pref);
				uint64_t zero = 0;
				hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &zero, sizeof(zero));
				hipblasLtMatmulHeuristicResult_t results[8]{};
				heuristic = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 8, results, &returned);
				if (heuristic == HIPBLAS_STATUS_SUCCESS && returned > 0) {
					selected = results[0];
					const float alpha = 1.0f, beta = 0.0f;
					char* base = static_cast<char*>(storage.p);
					void* pa = base;
					void* pb = base + 128ull * 128ull * sizeof(float);
					void* pc = base + (128ull * 128ull + 128ull * 4096ull) * sizeof(float);
					launch = hipblasLtMatmul(handle, desc, &alpha, pa, a, pb, b, &beta, pc, c, pc, d, &selected.algo, nullptr, 0, stream);
					if (launch == HIPBLAS_STATUS_SUCCESS) {
						const hipError_t sync = hipStreamSynchronize(stream);
						if (sync != hipSuccess) launch = HIPBLAS_STATUS_EXECUTION_FAILED;
					}
				}
			}
		}
		if (!first) out << ",\n"; first = false;
		out << "    {\"epilogue\":\"" << candidate.name << "\",\"numeric_value\":" << candidate.value
		    << ",\"declared_by_rocm72_header\":" << (candidate.declared ? "true" : "false")
		    << ",\"role\":\"" << role << "\",\"width\":" << width << ",\"batch\":" << batch
		    << ",\"hidden_layers\":[1,2,4],\"hidden_activations\":[\"ReLU\",\"None\"]"
		    << ",\"output_activations\":[\"None\",\"Sigmoid\"],\"stream\":\"" << (stream_kind ? "explicit" : "default") << "\""
		    << ",\"descriptor_status\":\"" << status_name(set) << "\",\"descriptor_status_code\":" << static_cast<int>(set)
		    << ",\"heuristic_status\":\"" << status_name(heuristic) << "\",\"heuristic_status_code\":" << static_cast<int>(heuristic)
		    << ",\"zero_workspace_algorithm\":" << ((heuristic == HIPBLAS_STATUS_SUCCESS && returned > 0) ? "true" : "false")
		    << ",\"heuristic_count\":" << returned << ",\"launch_status\":\"" << status_name(launch) << "\",\"launch_status_code\":" << static_cast<int>(launch)
		    << ",\"aux_semantics_verified\":false,\"bgrad_axis_verified\":false,\"selected_for_integration\":false}";
		if (pref) hipblasLtMatmulPreferenceDestroy(pref);
		if (d) hipblasLtMatrixLayoutDestroy(d); if (c) hipblasLtMatrixLayoutDestroy(c);
		if (b) hipblasLtMatrixLayoutDestroy(b); if (a) hipblasLtMatrixLayoutDestroy(a);
		if (desc) hipblasLtMatmulDescDestroy(desc);
		if (stream) hipStreamDestroy(stream);
	}
	out << "\n  ]\n}\n";
	hipblasLtDestroy(handle);
	return 0;
}
