// TCNN_RDNA4_P4A0_P1_MINIMAL_ROCWMMA_GEMM_001
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t M = 16;
constexpr uint32_t N = 16;
constexpr uint32_t K = 16;
constexpr double MAX_ABS_TOLERANCE = 5.0e-5;
constexpr double NORMALIZED_L2_TOLERANCE = 2.0e-5;
constexpr const char* MARKER = "TCNN_RDNA4_P4A0_P1_MINIMAL_ROCWMMA_GEMM_001";

using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;

#define HIP_CHECK(call)                                                                    \
    do {                                                                                   \
        const hipError_t error_ = (call);                                                  \
        if (error_ != hipSuccess) {                                                        \
            std::cerr << "HIP_ERROR: " << #call << ": " << hipGetErrorString(error_)       \
                      << " (" << static_cast<int>(error_) << ")" << std::endl;              \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (false)

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

double half_to_double(Half value) {
    return static_cast<double>(static_cast<float>(value));
}

__global__ void minimal_rocwmma_gemm(const Half* a, const Half* b, F32* c, uint32_t* diagnostics) {
    using FragA = rocwmma::fragment<
        rocwmma::matrix_a, M, N, K, Half, rocwmma::row_major>;
    using FragB = rocwmma::fragment<
        rocwmma::matrix_b, M, N, K, Half, rocwmma::col_major>;
    using FragAcc = rocwmma::fragment<
        rocwmma::accumulator, M, N, K, F32>;

    FragA frag_a;
    FragB frag_b;
    FragAcc frag_acc;

    rocwmma::fill_fragment(frag_acc, 0.0f);
    rocwmma::load_matrix_sync(frag_a, a, K);
    rocwmma::load_matrix_sync(frag_b, b, K);
    rocwmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    rocwmma::store_matrix_sync(c, frag_acc, N, rocwmma::mem_row_major);

    if (threadIdx.x == 0) {
        diagnostics[0] = static_cast<uint32_t>(warpSize);
        diagnostics[1] = static_cast<uint32_t>(blockDim.x);
        diagnostics[2] = static_cast<uint32_t>(gridDim.x);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string json_path = argc >= 2 ? argv[1] : "phase4a0_p1_minimal_rocwmma_gemm.json";

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    const std::string arch = props.gcnArchName;
    const bool context_ok = props.warpSize == 32 && arch.rfind("gfx1201", 0) == 0;

    std::vector<Half> host_a(M * K);
    std::vector<Half> host_b(K * N);
    std::vector<F32> host_c(M * N, std::numeric_limits<F32>::quiet_NaN());
    std::vector<double> reference(M * N, 0.0);

    uint32_t quantization_changed_a = 0;
    uint32_t quantization_changed_b = 0;

    // Deterministic, small, deliberately non-binary-decimal source values.
    // The CPU oracle is built only from the values after FP16 quantization.
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < K; ++col) {
            const int code = static_cast<int>((row * 17u + col * 13u + 3u) % 31u) - 15;
            const float source = static_cast<float>(code) * 0.0173f;
            const Half quantized = static_cast<Half>(source);
            host_a[row * K + col] = quantized;
            if (static_cast<float>(quantized) != source) {
                ++quantization_changed_a;
            }
        }
    }

    // B is physically column-major: element (row, col) is at col*K + row.
    for (uint32_t row = 0; row < K; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const int code = static_cast<int>((row * 11u + col * 19u + 5u) % 29u) - 14;
            const float source = static_cast<float>(code) * 0.0137f;
            const Half quantized = static_cast<Half>(source);
            host_b[col * K + row] = quantized;
            if (static_cast<float>(quantized) != source) {
                ++quantization_changed_b;
            }
        }
    }

    const bool quantization_ok = quantization_changed_a > 0 && quantization_changed_b > 0;

    // Independent primary oracle: FP64 accumulation from the exact FP16-rounded inputs.
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            double accum = 0.0;
            for (uint32_t inner = 0; inner < K; ++inner) {
                const double av = half_to_double(host_a[row * K + inner]);
                const double bv = half_to_double(host_b[col * K + inner]);
                accum += av * bv;
            }
            reference[row * N + col] = accum;
        }
    }

    Half* device_a = nullptr;
    Half* device_b = nullptr;
    F32* device_c = nullptr;
    uint32_t* device_diagnostics = nullptr;

    HIP_CHECK(hipMalloc(&device_a, host_a.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&device_b, host_b.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&device_c, host_c.size() * sizeof(F32)));
    HIP_CHECK(hipMalloc(&device_diagnostics, 3 * sizeof(uint32_t)));

    HIP_CHECK(hipMemcpy(
        device_a, host_a.data(), host_a.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        device_b, host_b.data(), host_b.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        device_c, host_c.data(), host_c.size() * sizeof(F32), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_diagnostics, 0, 3 * sizeof(uint32_t)));

    hipLaunchKernelGGL(
        minimal_rocwmma_gemm,
        dim3(1),
        dim3(32),
        0,
        0,
        device_a,
        device_b,
        device_c,
        device_diagnostics);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> diagnostics(3, 0);
    HIP_CHECK(hipMemcpy(
        host_c.data(), device_c, host_c.size() * sizeof(F32), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        diagnostics.data(),
        device_diagnostics,
        diagnostics.size() * sizeof(uint32_t),
        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_c));
    HIP_CHECK(hipFree(device_diagnostics));

    uint32_t nonfinite_count = 0;
    double max_abs = 0.0;
    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;
    uint32_t max_index = 0;

    for (uint32_t index = 0; index < M * N; ++index) {
        const double gpu = static_cast<double>(host_c[index]);
        const double ref = reference[index];

        if (!std::isfinite(gpu)) {
            ++nonfinite_count;
            continue;
        }

        const double diff = std::abs(gpu - ref);
        if (diff > max_abs) {
            max_abs = diff;
            max_index = index;
        }
        sum_diff_sq += diff * diff;
        sum_ref_sq += ref * ref;
    }

    const double normalized_l2 =
        std::sqrt(sum_diff_sq / std::max(sum_ref_sq, std::numeric_limits<double>::min()));

    const bool wave_context_ok =
        context_ok && diagnostics[0] == 32 && diagnostics[1] == 32 && diagnostics[2] == 1;
    const bool numerical_ok =
        nonfinite_count == 0 &&
        max_abs <= MAX_ABS_TOLERANCE &&
        normalized_l2 <= NORMALIZED_L2_TOLERANCE;
    const bool passed = wave_context_ok && quantization_ok && numerical_ok;

    std::cout << std::setprecision(17);
    std::cout << "marker: " << MARKER << '\n';
    std::cout << "device: " << props.name << '\n';
    std::cout << "arch: " << arch << '\n';
    std::cout << "host_warp_size: " << props.warpSize << '\n';
    std::cout << "kernel_warp_size: " << diagnostics[0] << '\n';
    std::cout << "block_x: " << diagnostics[1] << '\n';
    std::cout << "grid_x: " << diagnostics[2] << '\n';
    std::cout << "quantization_changed_a: " << quantization_changed_a << '\n';
    std::cout << "quantization_changed_b: " << quantization_changed_b << '\n';
    std::cout << "max_abs: " << max_abs << '\n';
    std::cout << "normalized_l2: " << normalized_l2 << '\n';
    std::cout << "nonfinite_count: " << nonfinite_count << '\n';
    std::cout << "max_error_element: row=" << (max_index / N)
              << " col=" << (max_index % N)
              << " gpu=" << static_cast<double>(host_c[max_index])
              << " cpu_fp64=" << reference[max_index] << '\n';

    std::cout << "ROCWMMA_P1_WAVE32_CONTEXT: "
              << (wave_context_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P1_INPUT_QUANTIZATION: "
              << (quantization_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_NUMERICAL_RESULT_VS_CPU: "
              << (numerical_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM: "
              << (passed ? "PASS" : "FAIL") << '\n';

    std::ofstream json(json_path);
    if (!json) {
        std::cerr << "JSON_ERROR: cannot open " << json_path << std::endl;
        return 3;
    }

    json << std::setprecision(17);
    json << "{\n";
    json << "  \"marker\": \"" << MARKER << "\",\n";
    json << "  \"decision\": \""
         << (passed ? "PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM_PASS"
                    : "PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM_FAIL")
         << "\",\n";
    json << "  \"device\": {\n";
    json << "    \"name\": \"" << json_escape(props.name) << "\",\n";
    json << "    \"arch\": \"" << json_escape(arch) << "\",\n";
    json << "    \"host_warp_size\": " << props.warpSize << ",\n";
    json << "    \"kernel_warp_size\": " << diagnostics[0] << ",\n";
    json << "    \"block_x\": " << diagnostics[1] << ",\n";
    json << "    \"grid_x\": " << diagnostics[2] << "\n";
    json << "  },\n";
    json << "  \"matrix\": {\"m\": 16, \"n\": 16, \"k\": 16},\n";
    json << "  \"contract\": {\n";
    json << "    \"a_memory_layout\": \"row_major\",\n";
    json << "    \"b_memory_layout\": \"col_major\",\n";
    json << "    \"output_memory_layout\": \"row_major\",\n";
    json << "    \"input\": \"FP16 after explicit host quantization\",\n";
    json << "    \"gpu_accumulation\": \"FP32 rocWMMA accumulator\",\n";
    json << "    \"gpu_output\": \"FP32\",\n";
    json << "    \"primary_oracle\": \"CPU FP64 accumulation from exact FP16-rounded inputs\"\n";
    json << "  },\n";
    json << "  \"quantization\": {\n";
    json << "    \"changed_a\": " << quantization_changed_a << ",\n";
    json << "    \"changed_b\": " << quantization_changed_b << ",\n";
    json << "    \"passed\": " << (quantization_ok ? "true" : "false") << "\n";
    json << "  },\n";
    json << "  \"metrics\": {\n";
    json << "    \"max_abs\": " << max_abs << ",\n";
    json << "    \"normalized_l2\": " << normalized_l2 << ",\n";
    json << "    \"nonfinite_count\": " << nonfinite_count << ",\n";
    json << "    \"max_error_row\": " << (max_index / N) << ",\n";
    json << "    \"max_error_col\": " << (max_index % N) << ",\n";
    json << "    \"max_error_gpu\": " << static_cast<double>(host_c[max_index]) << ",\n";
    json << "    \"max_error_cpu_fp64\": " << reference[max_index] << "\n";
    json << "  },\n";
    json << "  \"tolerances\": {\n";
    json << "    \"max_abs\": " << MAX_ABS_TOLERANCE << ",\n";
    json << "    \"normalized_l2\": " << NORMALIZED_L2_TOLERANCE << "\n";
    json << "  },\n";
    json << "  \"gates\": {\n";
    json << "    \"wave32_context\": " << (wave_context_ok ? "true" : "false") << ",\n";
    json << "    \"input_quantization\": " << (quantization_ok ? "true" : "false") << ",\n";
    json << "    \"numerical_vs_cpu_fp64\": " << (numerical_ok ? "true" : "false") << "\n";
    json << "  }\n";
    json << "}\n";

    return passed ? 0 : 1;
}
