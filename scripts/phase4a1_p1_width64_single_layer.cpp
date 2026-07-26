// TCNN_RDNA4_P4A1_P1_WIDTH64_FOUR_K_TILE_001
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t BATCH = 16;
constexpr uint32_t WIDTH = 64;
constexpr uint32_t TILE = 16;
constexpr uint32_t N_TILES = 4;
constexpr uint32_t K_TILES = 4;
constexpr uint32_t WAVE_SIZE = 32;
constexpr uint32_t WAVES = 4;
constexpr uint32_t THREADS = WAVE_SIZE * WAVES;
constexpr uint32_t ELEMENTS = BATCH * WIDTH;
constexpr uint32_t GUARD = 16;
constexpr uint32_t STAGE_STRIDE = ELEMENTS + 2 * GUARD;

constexpr double MAX_ABS_TOLERANCE = 5.0e-5;
constexpr double NORMALIZED_L2_TOLERANCE = 2.0e-5;

constexpr const char* MARKER =
    "TCNN_RDNA4_P4A1_P1_WIDTH64_FOUR_K_TILE_001";

using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;

using FragA = rocwmma::fragment<
    rocwmma::matrix_a, TILE, TILE, TILE, Half, rocwmma::row_major>;
using FragB = rocwmma::fragment<
    rocwmma::matrix_b, TILE, TILE, TILE, Half, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<
    rocwmma::accumulator, TILE, TILE, TILE, F32>;

#define HIP_CHECK(call)                                                                    \
    do {                                                                                   \
        const hipError_t error_ = (call);                                                  \
        if (error_ != hipSuccess) {                                                        \
            std::cerr << "HIP_ERROR: " << #call << ": " << hipGetErrorString(error_)       \
                      << " (" << static_cast<int>(error_) << ")" << std::endl;              \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (false)

union FloatBits {
    float value;
    uint32_t bits;
};

__host__ __device__ uint32_t float_bits(float value) {
    FloatBits converter{};
    converter.value = value;
    return converter.bits;
}

__global__ void width64_four_k_tile_kernel(
    const Half* input,
    const Half* weight,
    float* stage_outputs,
    uint32_t* diagnostics) {

    const uint32_t wave = threadIdx.x / WAVE_SIZE;
    const uint32_t lane = threadIdx.x % WAVE_SIZE;

    if (threadIdx.x == 0) {
        diagnostics[0] = static_cast<uint32_t>(warpSize);
        diagnostics[1] = static_cast<uint32_t>(blockDim.x);
        diagnostics[2] = static_cast<uint32_t>(gridDim.x);
        diagnostics[3] = WAVES;
        diagnostics[4] = K_TILES;
    }

    if (warpSize != WAVE_SIZE ||
        blockDim.x != THREADS ||
        gridDim.x != 1 ||
        wave >= WAVES) {
        if (threadIdx.x == 0) {
            diagnostics[5] = 1u;
        }
        return;
    }

    if (lane == 0) {
        atomicAdd(&diagnostics[8 + wave], 1u);
    }

    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);

    const uint32_t n_tile = wave;
    const uint32_t output_col_begin = n_tile * TILE;

    for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
        FragA frag_a;
        FragB frag_b;

        const Half* a_tile =
            input + k_tile * TILE;

        const Half* b_tile =
            weight
            + output_col_begin * WIDTH
            + k_tile * TILE;

        rocwmma::load_matrix_sync(
            frag_a,
            a_tile,
            WIDTH);

        rocwmma::load_matrix_sync(
            frag_b,
            b_tile,
            WIDTH);

        rocwmma::mma_sync(
            accumulator,
            frag_a,
            frag_b,
            accumulator);

        float* stage_base =
            stage_outputs
            + k_tile * STAGE_STRIDE
            + GUARD
            + output_col_begin;

        rocwmma::store_matrix_sync(
            stage_base,
            accumulator,
            WIDTH,
            rocwmma::mem_row_major);
    }

    if (lane == 0) {
        atomicAdd(&diagnostics[16 + wave], 1u);
    }
}

std::vector<char> read_binary(
    const std::filesystem::path& path,
    size_t expected_bytes) {

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot open binary file: " + path.string());
    }

    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (size != static_cast<std::streamsize>(expected_bytes)) {
        std::ostringstream message;
        message
            << "unexpected file size for " << path
            << ": got " << size
            << ", expected " << expected_bytes;
        throw std::runtime_error(message.str());
    }

    std::vector<char> bytes(expected_bytes);
    if (!input.read(bytes.data(), size)) {
        throw std::runtime_error(
            "failed to read binary file: " + path.string());
    }

    return bytes;
}

std::vector<Half> read_fp16(
    const std::filesystem::path& path,
    size_t elements) {

    static_assert(sizeof(Half) == 2, "rocWMMA FP16 must be 2 bytes");

    const std::vector<char> bytes =
        read_binary(path, elements * sizeof(Half));

    std::vector<Half> values(elements);
    std::memcpy(
        values.data(),
        bytes.data(),
        bytes.size());
    return values;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u"
                        << std::hex
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<int>(ch)
                        << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

struct StageMetrics {
    uint32_t stage = 0;
    uint32_t k_terms = 0;
    uint32_t nonfinite_count = 0;
    uint32_t max_error_index = 0;
    double max_abs = 0.0;
    double normalized_l2 = 0.0;
    bool guards_ok = false;
    bool passed = false;
    std::vector<double> reference;
};

struct WaveMetrics {
    uint32_t stage = 0;
    uint32_t wave = 0;
    uint32_t col_begin = 0;
    uint32_t col_end = 0;
    uint32_t nonfinite_count = 0;
    uint32_t max_error_index = 0;
    double max_abs = 0.0;
    double normalized_l2 = 0.0;
    bool passed = false;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " <input_fp16_row_major.bin>"
            << " <weight_fp16_col_major.bin>"
            << " <result.json>"
            << " <stage_output.csv>"
            << std::endl;
        return 64;
    }

    const std::filesystem::path input_path = argv[1];
    const std::filesystem::path weight_path = argv[2];
    const std::filesystem::path json_path = argv[3];
    const std::filesystem::path csv_path = argv[4];

    std::filesystem::create_directories(json_path.parent_path());
    std::filesystem::create_directories(csv_path.parent_path());

    std::vector<Half> input;
    std::vector<Half> weight;

    try {
        input = read_fp16(input_path, ELEMENTS);
        weight = read_fp16(weight_path, WIDTH * WIDTH);
    } catch (const std::exception& error) {
        std::cerr << "INPUT_ERROR: " << error.what() << std::endl;
        return 3;
    }

    int device = 0;
    int runtime_version = 0;
    int driver_version = 0;

    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));

    const float guard_value = -12345.25f;
    const uint32_t guard_bits = float_bits(guard_value);

    std::vector<float> stage_outputs(
        K_TILES * STAGE_STRIDE,
        guard_value);

    for (uint32_t stage = 0; stage < K_TILES; ++stage) {
        std::fill(
            stage_outputs.begin() + stage * STAGE_STRIDE + GUARD,
            stage_outputs.begin() + stage * STAGE_STRIDE + GUARD + ELEMENTS,
            std::numeric_limits<float>::quiet_NaN());
    }

    Half* d_input = nullptr;
    Half* d_weight = nullptr;
    float* d_stage_outputs = nullptr;
    uint32_t* d_diagnostics = nullptr;

    HIP_CHECK(hipMalloc(
        &d_input,
        input.size() * sizeof(Half)));

    HIP_CHECK(hipMalloc(
        &d_weight,
        weight.size() * sizeof(Half)));

    HIP_CHECK(hipMalloc(
        &d_stage_outputs,
        stage_outputs.size() * sizeof(float)));

    HIP_CHECK(hipMalloc(
        &d_diagnostics,
        32 * sizeof(uint32_t)));

    HIP_CHECK(hipMemcpy(
        d_input,
        input.data(),
        input.size() * sizeof(Half),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        d_weight,
        weight.data(),
        weight.size() * sizeof(Half),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        d_stage_outputs,
        stage_outputs.data(),
        stage_outputs.size() * sizeof(float),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(
        d_diagnostics,
        0,
        32 * sizeof(uint32_t)));

    hipLaunchKernelGGL(
        width64_four_k_tile_kernel,
        dim3(1),
        dim3(THREADS),
        0,
        0,
        d_input,
        d_weight,
        d_stage_outputs,
        d_diagnostics);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> diagnostics(32, 0u);

    HIP_CHECK(hipMemcpy(
        stage_outputs.data(),
        d_stage_outputs,
        stage_outputs.size() * sizeof(float),
        hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemcpy(
        diagnostics.data(),
        d_diagnostics,
        diagnostics.size() * sizeof(uint32_t),
        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_stage_outputs));
    HIP_CHECK(hipFree(d_diagnostics));

    const bool context_ok =
        std::string(props.gcnArchName).rfind("gfx1201", 0) == 0 &&
        props.warpSize == static_cast<int>(WAVE_SIZE) &&
        diagnostics[0] == WAVE_SIZE &&
        diagnostics[1] == THREADS &&
        diagnostics[2] == 1 &&
        diagnostics[3] == WAVES &&
        diagnostics[4] == K_TILES &&
        diagnostics[5] == 0;

    bool wave_entry_ok = true;
    bool wave_exit_ok = true;
    for (uint32_t wave = 0; wave < WAVES; ++wave) {
        wave_entry_ok &= diagnostics[8 + wave] == 1u;
        wave_exit_ok &= diagnostics[16 + wave] == 1u;
    }

    std::vector<StageMetrics> stage_metrics;
    std::vector<WaveMetrics> wave_metrics;

    for (uint32_t stage = 0; stage < K_TILES; ++stage) {
        StageMetrics metrics;
        metrics.stage = stage + 1;
        metrics.k_terms = (stage + 1) * TILE;
        metrics.reference.resize(ELEMENTS);

        for (uint32_t row = 0; row < BATCH; ++row) {
            for (uint32_t col = 0; col < WIDTH; ++col) {
                double accumulator = 0.0;

                for (uint32_t inner = 0;
                     inner < metrics.k_terms;
                     ++inner) {

                    const double a =
                        static_cast<double>(
                            static_cast<float>(
                                input[row * WIDTH + inner]));

                    const double b =
                        static_cast<double>(
                            static_cast<float>(
                                weight[col * WIDTH + inner]));

                    accumulator += a * b;
                }

                metrics.reference[row * WIDTH + col] =
                    accumulator;
            }
        }

        metrics.guards_ok = true;
        const uint32_t stage_base = stage * STAGE_STRIDE;

        for (uint32_t index = 0; index < GUARD; ++index) {
            metrics.guards_ok &=
                float_bits(stage_outputs[stage_base + index])
                    == guard_bits;

            metrics.guards_ok &=
                float_bits(
                    stage_outputs[
                        stage_base + GUARD + ELEMENTS + index])
                    == guard_bits;
        }

        double sum_diff_sq = 0.0;
        double sum_ref_sq = 0.0;

        for (uint32_t index = 0; index < ELEMENTS; ++index) {
            const double gpu =
                static_cast<double>(
                    stage_outputs[stage_base + GUARD + index]);

            const double cpu = metrics.reference[index];

            if (!std::isfinite(gpu)) {
                ++metrics.nonfinite_count;
                continue;
            }

            const double diff = std::abs(gpu - cpu);
            if (diff > metrics.max_abs) {
                metrics.max_abs = diff;
                metrics.max_error_index = index;
            }

            sum_diff_sq += diff * diff;
            sum_ref_sq += cpu * cpu;
        }

        metrics.normalized_l2 =
            std::sqrt(
                sum_diff_sq /
                std::max(
                    sum_ref_sq,
                    std::numeric_limits<double>::min()));

        metrics.passed =
            metrics.guards_ok &&
            metrics.nonfinite_count == 0 &&
            metrics.max_abs <= MAX_ABS_TOLERANCE &&
            metrics.normalized_l2 <= NORMALIZED_L2_TOLERANCE;

        stage_metrics.push_back(metrics);

        for (uint32_t wave = 0; wave < WAVES; ++wave) {
            WaveMetrics wave_result;
            wave_result.stage = stage + 1;
            wave_result.wave = wave;
            wave_result.col_begin = wave * TILE;
            wave_result.col_end = wave_result.col_begin + TILE - 1;

            double wave_sum_diff_sq = 0.0;
            double wave_sum_ref_sq = 0.0;

            for (uint32_t row = 0; row < BATCH; ++row) {
                for (uint32_t col = wave_result.col_begin;
                     col <= wave_result.col_end;
                     ++col) {

                    const uint32_t index = row * WIDTH + col;
                    const double gpu =
                        static_cast<double>(
                            stage_outputs[
                                stage_base + GUARD + index]);

                    const double cpu = metrics.reference[index];

                    if (!std::isfinite(gpu)) {
                        ++wave_result.nonfinite_count;
                        continue;
                    }

                    const double diff = std::abs(gpu - cpu);
                    if (diff > wave_result.max_abs) {
                        wave_result.max_abs = diff;
                        wave_result.max_error_index = index;
                    }

                    wave_sum_diff_sq += diff * diff;
                    wave_sum_ref_sq += cpu * cpu;
                }
            }

            wave_result.normalized_l2 =
                std::sqrt(
                    wave_sum_diff_sq /
                    std::max(
                        wave_sum_ref_sq,
                        std::numeric_limits<double>::min()));

            wave_result.passed =
                wave_result.nonfinite_count == 0 &&
                wave_result.max_abs <= MAX_ABS_TOLERANCE &&
                wave_result.normalized_l2 <= NORMALIZED_L2_TOLERANCE;

            wave_metrics.push_back(wave_result);
        }
    }

    const bool all_stages_ok =
        std::all_of(
            stage_metrics.begin(),
            stage_metrics.end(),
            [](const StageMetrics& metrics) {
                return metrics.passed;
            });

    const bool all_waves_ok =
        std::all_of(
            wave_metrics.begin(),
            wave_metrics.end(),
            [](const WaveMetrics& metrics) {
                return metrics.passed;
            });

    const bool final_stage_ok =
        stage_metrics.back().passed;

    const bool passed =
        context_ok &&
        wave_entry_ok &&
        wave_exit_ok &&
        all_stages_ok &&
        all_waves_ok &&
        final_stage_ok;

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr
            << "OUTPUT_ERROR: cannot write "
            << csv_path
            << std::endl;
        return 4;
    }

    csv
        << "stage,k_terms,wave,row,col,gpu,cpu_fp64,abs_diff\n";

    for (uint32_t stage = 0; stage < K_TILES; ++stage) {
        const uint32_t stage_base = stage * STAGE_STRIDE;
        const StageMetrics& metrics = stage_metrics[stage];

        for (uint32_t row = 0; row < BATCH; ++row) {
            for (uint32_t col = 0; col < WIDTH; ++col) {
                const uint32_t index = row * WIDTH + col;
                const double gpu =
                    static_cast<double>(
                        stage_outputs[
                            stage_base + GUARD + index]);
                const double cpu = metrics.reference[index];

                csv
                    << (stage + 1) << ','
                    << metrics.k_terms << ','
                    << (col / TILE) << ','
                    << row << ','
                    << col << ','
                    << std::setprecision(17) << gpu << ','
                    << cpu << ','
                    << std::abs(gpu - cpu) << '\n';
            }
        }
    }

    std::ofstream json(json_path);
    if (!json) {
        std::cerr
            << "OUTPUT_ERROR: cannot write "
            << json_path
            << std::endl;
        return 4;
    }

    json << std::setprecision(17);
    json << "{\n";
    json << "  \"marker\": \"" << MARKER << "\",\n";
    json << "  \"decision\": \""
         << (
                passed
                    ? "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_PASS"
                    : "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_FAIL"
            )
         << "\",\n";
    json << "  \"context\": {\n";
    json << "    \"device\": \""
         << json_escape(props.name)
         << "\",\n";
    json << "    \"arch\": \""
         << json_escape(props.gcnArchName)
         << "\",\n";
    json << "    \"hip_runtime_version\": "
         << runtime_version
         << ",\n";
    json << "    \"hip_driver_version\": "
         << driver_version
         << ",\n";
#ifdef __clang_version__
    json << "    \"compiler_version\": \""
         << json_escape(__clang_version__)
         << "\",\n";
#endif
    json << "    \"warp_size\": "
         << diagnostics[0]
         << ",\n";
    json << "    \"threads_per_block\": "
         << diagnostics[1]
         << ",\n";
    json << "    \"waves_per_block\": "
         << diagnostics[3]
         << ",\n";
    json << "    \"k_tiles\": "
         << diagnostics[4]
         << "\n";
    json << "  },\n";
    json << "  \"topology\": {\n";
    json << "    \"batch_rows\": 16,\n";
    json << "    \"input_width\": 64,\n";
    json << "    \"output_width\": 64,\n";
    json << "    \"tile\": \"16x16x16\",\n";
    json << "    \"output_tiles\": 4,\n";
    json << "    \"k_tiles_per_output_tile\": 4,\n";
    json << "    \"mma_sync_calls_per_wave\": 4,\n";
    json << "    \"mma_sync_calls_per_block\": 16,\n";
    json << "    \"kernel_launches\": 1,\n";
    json << "    \"input_type\": \"FP16\",\n";
    json << "    \"weight_type\": \"FP16\",\n";
    json << "    \"accumulator_type\": \"FP32\",\n";
    json << "    \"output_type\": \"FP32\"\n";
    json << "  },\n";
    json << "  \"tolerances\": {\n";
    json << "    \"max_abs\": "
         << MAX_ABS_TOLERANCE
         << ",\n";
    json << "    \"normalized_l2\": "
         << NORMALIZED_L2_TOLERANCE
         << "\n";
    json << "  },\n";
    json << "  \"stages\": [\n";

    for (size_t index = 0; index < stage_metrics.size(); ++index) {
        const StageMetrics& metrics = stage_metrics[index];

        json << "    {\n";
        json << "      \"stage\": "
             << metrics.stage
             << ",\n";
        json << "      \"k_terms\": "
             << metrics.k_terms
             << ",\n";
        json << "      \"guards_ok\": "
             << (metrics.guards_ok ? "true" : "false")
             << ",\n";
        json << "      \"nonfinite_count\": "
             << metrics.nonfinite_count
             << ",\n";
        json << "      \"max_abs\": "
             << metrics.max_abs
             << ",\n";
        json << "      \"normalized_l2\": "
             << metrics.normalized_l2
             << ",\n";
        json << "      \"max_error_row\": "
             << (metrics.max_error_index / WIDTH)
             << ",\n";
        json << "      \"max_error_col\": "
             << (metrics.max_error_index % WIDTH)
             << ",\n";
        json << "      \"passed\": "
             << (metrics.passed ? "true" : "false")
             << "\n";
        json << "    }"
             << (index + 1 == stage_metrics.size() ? "\n" : ",\n");
    }

    json << "  ],\n";
    json << "  \"wave_tiles\": [\n";

    for (size_t index = 0; index < wave_metrics.size(); ++index) {
        const WaveMetrics& metrics = wave_metrics[index];

        json << "    {\n";
        json << "      \"stage\": "
             << metrics.stage
             << ",\n";
        json << "      \"wave\": "
             << metrics.wave
             << ",\n";
        json << "      \"col_begin\": "
             << metrics.col_begin
             << ",\n";
        json << "      \"col_end\": "
             << metrics.col_end
             << ",\n";
        json << "      \"nonfinite_count\": "
             << metrics.nonfinite_count
             << ",\n";
        json << "      \"max_abs\": "
             << metrics.max_abs
             << ",\n";
        json << "      \"normalized_l2\": "
             << metrics.normalized_l2
             << ",\n";
        json << "      \"passed\": "
             << (metrics.passed ? "true" : "false")
             << "\n";
        json << "    }"
             << (index + 1 == wave_metrics.size() ? "\n" : ",\n");
    }

    json << "  ],\n";
    json << "  \"diagnostics\": {\n";
    json << "    \"wave_entry_counts\": ["
         << diagnostics[8] << ", "
         << diagnostics[9] << ", "
         << diagnostics[10] << ", "
         << diagnostics[11] << "],\n";
    json << "    \"wave_exit_counts\": ["
         << diagnostics[16] << ", "
         << diagnostics[17] << ", "
         << diagnostics[18] << ", "
         << diagnostics[19] << "]\n";
    json << "  },\n";
    json << "  \"gates\": {\n";
    json << "    \"wave32_context\": "
         << (context_ok ? "true" : "false")
         << ",\n";
    json << "    \"all_four_waves_entered_once\": "
         << (wave_entry_ok ? "true" : "false")
         << ",\n";
    json << "    \"all_four_waves_exited_once\": "
         << (wave_exit_ok ? "true" : "false")
         << ",\n";
    json << "    \"all_four_partial_stages_vs_cpu_fp64\": "
         << (all_stages_ok ? "true" : "false")
         << ",\n";
    json << "    \"all_wave_tiles_vs_cpu_fp64\": "
         << (all_waves_ok ? "true" : "false")
         << ",\n";
    json << "    \"final_width64_output_vs_cpu_fp64\": "
         << (final_stage_ok ? "true" : "false")
         << "\n";
    json << "  }\n";
    json << "}\n";

    std::cout << "marker: " << MARKER << '\n';
    std::cout << "wave_entry_counts: "
              << diagnostics[8] << ","
              << diagnostics[9] << ","
              << diagnostics[10] << ","
              << diagnostics[11] << '\n';
    std::cout << "wave_exit_counts: "
              << diagnostics[16] << ","
              << diagnostics[17] << ","
              << diagnostics[18] << ","
              << diagnostics[19] << '\n';

    for (const StageMetrics& metrics : stage_metrics) {
        std::cout
            << "stage_" << metrics.stage
            << "_k_terms: " << metrics.k_terms
            << '\n';

        std::cout
            << "stage_" << metrics.stage
            << "_max_abs: " << metrics.max_abs
            << '\n';

        std::cout
            << "stage_" << metrics.stage
            << "_normalized_l2: "
            << metrics.normalized_l2
            << '\n';

        std::cout
            << "WIDTH64_K_TILE_STAGE_"
            << metrics.stage
            << ": "
            << (metrics.passed ? "PASS" : "FAIL")
            << '\n';
    }

    std::cout << "WIDTH64_FOUR_K_TILE_ACCUMULATION: "
              << (all_stages_ok ? "PASS" : "FAIL")
              << '\n';

    std::cout << "WIDTH64_ALL_FOUR_WAVES_OUTPUT_COVERAGE: "
              << (
                    wave_entry_ok &&
                    wave_exit_ok &&
                    all_waves_ok
                        ? "PASS"
                        : "FAIL"
                 )
              << '\n';

    std::cout << "WIDTH64_SINGLE_LAYER_VS_CPU_FP64: "
              << (final_stage_ok ? "PASS" : "FAIL")
              << '\n';

    std::cout << "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_PROCESS: "
              << (passed ? "PASS" : "FAIL")
              << '\n';

    return passed ? 0 : 1;
}
