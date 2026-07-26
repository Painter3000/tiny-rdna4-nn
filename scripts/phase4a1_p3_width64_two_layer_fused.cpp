// TCNN_RDNA4_P4A1_P3_WIDTH64_TWO_LAYER_FUSED_001
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "phase4a1_p3_mapping_generated.hpp"

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
constexpr uint32_t K_TILES = 4;
constexpr uint32_t WAVES = 4;
constexpr uint32_t WAVE_SIZE = 32;
constexpr uint32_t THREADS = WAVES * WAVE_SIZE;
constexpr uint32_t SLOTS = 8;
constexpr uint32_t ELEMENTS = BATCH * WIDTH;
constexpr uint32_t GUARD = 16;

constexpr const char* MARKER =
    "TCNN_RDNA4_P4A1_P3_WIDTH64_TWO_LAYER_FUSED_001";

using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;

using FragA = rocwmma::fragment<
    rocwmma::matrix_a, TILE, TILE, TILE, Half, rocwmma::row_major>;
using FragB = rocwmma::fragment<
    rocwmma::matrix_b, TILE, TILE, TILE, Half, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<
    rocwmma::accumulator, TILE, TILE, TILE, F32>;

using RegA = rocwmma::apply_register_file_t<FragA>;
using RegAcc = rocwmma::apply_register_file_t<FragAcc>;

#define HIP_CHECK(call)                                                                    \
    do {                                                                                   \
        const hipError_t error_ = (call);                                                  \
        if (error_ != hipSuccess) {                                                        \
            std::cerr << "HIP_ERROR: " << #call << ": " << hipGetErrorString(error_)       \
                      << " (" << static_cast<int>(error_) << ")" << std::endl;              \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (false)

union HalfBits {
    Half value;
    uint16_t bits;
};

__host__ __device__ uint16_t half_bits(Half value) {
    HalfBits converter{};
    converter.value = value;
    return converter.bits;
}

__device__ inline void accumulator_epilogue_to_matrix_a(
    FragAcc const& accumulator,
    const float* bias,
    uint32_t output_col_begin,
    RegA& output_reg_a,
    uint32_t* diagnostics) {

    const uint32_t lane = threadIdx.x % WAVE_SIZE;
    auto const& reg_acc = rocwmma::to_register_file(accumulator);

    F32 epilogue[SLOTS];

    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        const uint32_t acc_index = lane * SLOTS + slot;
        const uint32_t local_col =
            phase4a1_p2_generated::kAccumulatorColumn[acc_index];
        const uint32_t global_col = output_col_begin + local_col;

        const F32 biased = reg_acc[slot] + bias[global_col];
        epilogue[slot] = biased > 0.0f ? biased : 0.0f;
    }

    for (uint32_t target_slot = 0;
         target_slot < SLOTS;
         ++target_slot) {

        const uint32_t target_index = lane * SLOTS + target_slot;
        const uint32_t desired_lane =
            phase4a1_p2_generated::kAccLaneForATargetA[target_index];
        const uint32_t desired_slot =
            phase4a1_p2_generated::kAccSlotForATargetA[target_index];

        if (desired_lane >= WAVE_SIZE || desired_slot >= SLOTS) {
            atomicAdd(&diagnostics[24], 1u);
        }

        F32 selected = 0.0f;

        for (uint32_t candidate_slot = 0;
             candidate_slot < SLOTS;
             ++candidate_slot) {

            const F32 shuffled = __shfl(
                epilogue[candidate_slot],
                desired_lane,
                WAVE_SIZE);

            if (candidate_slot == desired_slot) {
                selected = shuffled;
            }
        }

        output_reg_a[target_slot] = static_cast<Half>(selected);
    }
}

__global__ void width64_two_layer_fused_kernel(
    const Half* input,
    const Half* weight_1,
    const Half* weight_2,
    const float* bias_1,
    const float* bias_2,
    const Half* expected_hidden_1,
    Half* hidden_2_output,
    uint32_t* diagnostics) {

    __shared__ __align__(16) Half hidden_1_lds[ELEMENTS];

    const uint32_t wave = threadIdx.x / WAVE_SIZE;
    const uint32_t lane = threadIdx.x % WAVE_SIZE;

    if (threadIdx.x == 0) {
        diagnostics[0] = static_cast<uint32_t>(warpSize);
        diagnostics[1] = static_cast<uint32_t>(blockDim.x);
        diagnostics[2] = static_cast<uint32_t>(gridDim.x);
        diagnostics[3] = WAVES;
        diagnostics[4] = K_TILES;
        diagnostics[5] = RegAcc::num_elements;
        diagnostics[6] = RegA::num_elements;
        diagnostics[7] = 2u;
    }

    if (warpSize != WAVE_SIZE
        || blockDim.x != THREADS
        || gridDim.x != 1
        || RegAcc::num_elements != SLOTS
        || RegA::num_elements != SLOTS
        || wave >= WAVES) {

        if (threadIdx.x == 0) {
            diagnostics[26] = 1u;
        }
        return;
    }

    if (lane == 0) {
        atomicAdd(&diagnostics[8 + wave], 1u);
    }

    const uint32_t output_col_begin = wave * TILE;

    FragAcc layer_1_accumulator;
    rocwmma::fill_fragment(layer_1_accumulator, 0.0f);

    for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
        FragA frag_a;
        FragB frag_b;
        const Half* a_tile = input + k_tile * TILE;
        const Half* b_tile =
            weight_1 + output_col_begin * WIDTH + k_tile * TILE;
        rocwmma::load_matrix_sync(frag_a, a_tile, WIDTH);
        rocwmma::load_matrix_sync(frag_b, b_tile, WIDTH);
        rocwmma::mma_sync(
            layer_1_accumulator,
            frag_a,
            frag_b,
            layer_1_accumulator);
    }

    RegA hidden_1_reg_a;
    accumulator_epilogue_to_matrix_a(
        layer_1_accumulator,
        bias_1,
        output_col_begin,
        hidden_1_reg_a,
        diagnostics);

    auto const& hidden_1_frag_a =
        rocwmma::from_register_file<FragA>(hidden_1_reg_a);

    rocwmma::store_matrix_sync(
        hidden_1_lds + output_col_begin,
        hidden_1_frag_a,
        WIDTH,
        rocwmma::mem_row_major);

    if (lane == 0) {
        atomicAdd(&diagnostics[12 + wave], 1u);
    }

    __syncthreads();

    const uint32_t producer_wave = (wave + 1u) % WAVES;
    const uint32_t producer_col_begin = producer_wave * TILE;
    uint32_t local_hidden_1_mismatches = 0;

    for (uint32_t linear = lane;
         linear < TILE * TILE;
         linear += WAVE_SIZE) {

        const uint32_t row = linear / TILE;
        const uint32_t local_col = linear % TILE;
        const uint32_t global_col = producer_col_begin + local_col;
        const uint32_t index = row * WIDTH + global_col;

        if (half_bits(hidden_1_lds[index])
            != half_bits(expected_hidden_1[index])) {
            ++local_hidden_1_mismatches;
        }
    }

    if (local_hidden_1_mismatches != 0) {
        atomicAdd(&diagnostics[25], local_hidden_1_mismatches);
    }

    if (lane == 0) {
        atomicAdd(&diagnostics[16 + wave], 1u);
        diagnostics[28 + wave] = producer_wave;
    }

    FragAcc layer_2_accumulator;
    rocwmma::fill_fragment(layer_2_accumulator, 0.0f);

    for (uint32_t k_tile = 0; k_tile < K_TILES; ++k_tile) {
        FragA frag_a;
        FragB frag_b;
        const Half* a_tile = hidden_1_lds + k_tile * TILE;
        const Half* b_tile =
            weight_2 + output_col_begin * WIDTH + k_tile * TILE;
        rocwmma::load_matrix_sync(frag_a, a_tile, WIDTH);
        rocwmma::load_matrix_sync(frag_b, b_tile, WIDTH);
        rocwmma::mma_sync(
            layer_2_accumulator,
            frag_a,
            frag_b,
            layer_2_accumulator);
    }

    if (lane == 0) {
        diagnostics[32 + wave] = K_TILES;
    }

    RegA hidden_2_reg_a;
    accumulator_epilogue_to_matrix_a(
        layer_2_accumulator,
        bias_2,
        output_col_begin,
        hidden_2_reg_a,
        diagnostics);

    auto const& hidden_2_frag_a =
        rocwmma::from_register_file<FragA>(hidden_2_reg_a);

    rocwmma::store_matrix_sync(
        hidden_2_output + output_col_begin,
        hidden_2_frag_a,
        WIDTH,
        rocwmma::mem_row_major);

    if (lane == 0) {
        atomicAdd(&diagnostics[20 + wave], 1u);
    }
}

std::vector<char> read_binary(
    const std::filesystem::path& path,
    size_t expected_bytes) {

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open binary file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(expected_bytes)) {
        std::ostringstream message;
        message << "unexpected file size for " << path
                << ": got " << size << ", expected " << expected_bytes;
        throw std::runtime_error(message.str());
    }
    std::vector<char> bytes(expected_bytes);
    if (!input.read(bytes.data(), size)) {
        throw std::runtime_error("failed to read binary file: " + path.string());
    }
    return bytes;
}

std::vector<Half> read_fp16(
    const std::filesystem::path& path,
    size_t elements) {
    static_assert(sizeof(Half) == 2);
    const auto bytes = read_binary(path, elements * sizeof(Half));
    std::vector<Half> values(elements);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

std::vector<float> read_fp32(
    const std::filesystem::path& path,
    size_t elements) {
    static_assert(sizeof(float) == 4);
    const auto bytes = read_binary(path, elements * sizeof(float));
    std::vector<float> values(elements);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(ch)
                        << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <input> <weight1> <weight2> <bias1> <bias2>"
                  << " <expected_hidden1> <expected_hidden2> <result.json> <output.csv>"
                  << std::endl;
        return 64;
    }

    const std::filesystem::path input_path = argv[1];
    const std::filesystem::path weight_1_path = argv[2];
    const std::filesystem::path weight_2_path = argv[3];
    const std::filesystem::path bias_1_path = argv[4];
    const std::filesystem::path bias_2_path = argv[5];
    const std::filesystem::path expected_hidden_1_path = argv[6];
    const std::filesystem::path expected_hidden_2_path = argv[7];
    const std::filesystem::path json_path = argv[8];
    const std::filesystem::path csv_path = argv[9];

    std::filesystem::create_directories(json_path.parent_path());
    std::filesystem::create_directories(csv_path.parent_path());

    std::vector<Half> input;
    std::vector<Half> weight_1;
    std::vector<Half> weight_2;
    std::vector<float> bias_1;
    std::vector<float> bias_2;
    std::vector<Half> expected_hidden_1;
    std::vector<Half> expected_hidden_2;

    try {
        input = read_fp16(input_path, ELEMENTS);
        weight_1 = read_fp16(weight_1_path, WIDTH * WIDTH);
        weight_2 = read_fp16(weight_2_path, WIDTH * WIDTH);
        bias_1 = read_fp32(bias_1_path, WIDTH);
        bias_2 = read_fp32(bias_2_path, WIDTH);
        expected_hidden_1 = read_fp16(expected_hidden_1_path, ELEMENTS);
        expected_hidden_2 = read_fp16(expected_hidden_2_path, ELEMENTS);
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

    const Half guard_value = static_cast<Half>(-123.5f);
    const uint16_t guard_bits = half_bits(guard_value);
    const Half interior_nan = static_cast<Half>(
        std::numeric_limits<float>::quiet_NaN());

    std::vector<Half> output(ELEMENTS + 2 * GUARD, guard_value);
    std::fill(
        output.begin() + GUARD,
        output.begin() + GUARD + ELEMENTS,
        interior_nan);

    Half* d_input = nullptr;
    Half* d_weight_1 = nullptr;
    Half* d_weight_2 = nullptr;
    float* d_bias_1 = nullptr;
    float* d_bias_2 = nullptr;
    Half* d_expected_hidden_1 = nullptr;
    Half* d_output = nullptr;
    uint32_t* d_diagnostics = nullptr;

    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_weight_1, weight_1.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_weight_2, weight_2.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_bias_1, bias_1.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias_2, bias_2.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &d_expected_hidden_1,
        expected_hidden_1.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_diagnostics, 64 * sizeof(uint32_t)));

    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_1, weight_1.data(), weight_1.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight_2, weight_2.data(), weight_2.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias_1, bias_1.data(), bias_1.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias_2, bias_2.data(), bias_2.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expected_hidden_1, expected_hidden_1.data(), expected_hidden_1.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output, output.data(), output.size() * sizeof(Half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_diagnostics, 0, 64 * sizeof(uint32_t)));

    hipLaunchKernelGGL(
        width64_two_layer_fused_kernel,
        dim3(1),
        dim3(THREADS),
        0,
        0,
        d_input,
        d_weight_1,
        d_weight_2,
        d_bias_1,
        d_bias_2,
        d_expected_hidden_1,
        d_output + GUARD,
        d_diagnostics);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> diagnostics(64, 0u);
    HIP_CHECK(hipMemcpy(output.data(), d_output, output.size() * sizeof(Half), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(diagnostics.data(), d_diagnostics, diagnostics.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_weight_1));
    HIP_CHECK(hipFree(d_weight_2));
    HIP_CHECK(hipFree(d_bias_1));
    HIP_CHECK(hipFree(d_bias_2));
    HIP_CHECK(hipFree(d_expected_hidden_1));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_diagnostics));

    bool guards_ok = true;
    for (uint32_t index = 0; index < GUARD; ++index) {
        guards_ok &= half_bits(output[index]) == guard_bits;
        guards_ok &= half_bits(output[GUARD + ELEMENTS + index]) == guard_bits;
    }

    uint32_t hidden_2_mismatch_count = 0;
    uint32_t first_mismatch_index = 0;
    uint32_t nonfinite_count = 0;
    uint32_t zero_count = 0;
    uint32_t positive_count = 0;
    double max_abs = 0.0;
    uint32_t max_error_index = 0;

    for (uint32_t index = 0; index < ELEMENTS; ++index) {
        const Half gpu = output[GUARD + index];
        const Half cpu = expected_hidden_2[index];
        const float gpu_float = static_cast<float>(gpu);
        const float cpu_float = static_cast<float>(cpu);
        if (!std::isfinite(gpu_float)) ++nonfinite_count;
        if ((half_bits(gpu) & 0x7FFFu) == 0u) ++zero_count;
        if (gpu_float > 0.0f) ++positive_count;
        if (half_bits(gpu) != half_bits(cpu)) {
            if (hidden_2_mismatch_count == 0) first_mismatch_index = index;
            ++hidden_2_mismatch_count;
            const double difference = std::abs(
                static_cast<double>(gpu_float) - static_cast<double>(cpu_float));
            if (difference > max_abs) {
                max_abs = difference;
                max_error_index = index;
            }
        }
    }

    const bool context_ok =
        std::string(props.gcnArchName).rfind("gfx1201", 0) == 0
        && props.warpSize == static_cast<int>(WAVE_SIZE)
        && diagnostics[0] == WAVE_SIZE
        && diagnostics[1] == THREADS
        && diagnostics[2] == 1
        && diagnostics[3] == WAVES
        && diagnostics[4] == K_TILES
        && diagnostics[5] == SLOTS
        && diagnostics[6] == SLOTS
        && diagnostics[7] == 2
        && diagnostics[26] == 0;

    bool wave_entry_ok = true;
    bool layer_1_publication_ok = true;
    bool layer_1_cross_wave_check_ok = true;
    bool layer_2_output_ok = true;
    bool layer_2_k_tiles_ok = true;
    for (uint32_t wave = 0; wave < WAVES; ++wave) {
        wave_entry_ok &= diagnostics[8 + wave] == 1u;
        layer_1_publication_ok &= diagnostics[12 + wave] == 1u;
        layer_1_cross_wave_check_ok &= diagnostics[16 + wave] == 1u;
        layer_2_output_ok &= diagnostics[20 + wave] == 1u;
        layer_2_k_tiles_ok &= diagnostics[32 + wave] == K_TILES;
    }

    const bool rotated_sources_ok =
        diagnostics[28] == 1u && diagnostics[29] == 2u
        && diagnostics[30] == 3u && diagnostics[31] == 0u;
    const bool mapping_ok =
        diagnostics[24] == 0u
        && phase4a1_p2_generated::kRelayMovedEntries == 240u;
    const bool hidden_1_bitwise_ok = diagnostics[25] == 0u;
    const bool hidden_2_bitwise_ok =
        hidden_2_mismatch_count == 0u && nonfinite_count == 0u;
    const bool hidden_2_epilogue_exercised =
        zero_count > 0u && positive_count > 0u;
    const bool passed =
        context_ok && wave_entry_ok && layer_1_publication_ok
        && layer_1_cross_wave_check_ok && rotated_sources_ok
        && mapping_ok && hidden_1_bitwise_ok && layer_2_k_tiles_ok
        && layer_2_output_ok && hidden_2_bitwise_ok
        && hidden_2_epilogue_exercised && guards_ok;

    std::ofstream csv(csv_path);
    if (!csv) return 4;
    csv << "row,col,wave,gpu,gpu_bits,cpu,cpu_bits,abs_diff\n";
    for (uint32_t row = 0; row < BATCH; ++row) {
        for (uint32_t col = 0; col < WIDTH; ++col) {
            const uint32_t index = row * WIDTH + col;
            const Half gpu = output[GUARD + index];
            const Half cpu = expected_hidden_2[index];
            csv << row << ',' << col << ',' << (col / TILE) << ','
                << static_cast<float>(gpu) << ',' << half_bits(gpu) << ','
                << static_cast<float>(cpu) << ',' << half_bits(cpu) << ','
                << std::abs(static_cast<double>(static_cast<float>(gpu)) - static_cast<double>(static_cast<float>(cpu)))
                << '\n';
        }
    }

    std::ofstream json(json_path);
    if (!json) return 4;
    json << std::setprecision(17);
    json << "{\n";
    json << "  \"marker\": \"" << MARKER << "\",\n";
    json << "  \"decision\": \"" << (passed ? "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_PASS" : "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_FAIL") << "\",\n";
    json << "  \"context\": {\n";
    json << "    \"device\": \"" << json_escape(props.name) << "\",\n";
    json << "    \"arch\": \"" << json_escape(props.gcnArchName) << "\",\n";
    json << "    \"hip_runtime_version\": " << runtime_version << ",\n";
    json << "    \"hip_driver_version\": " << driver_version << ",\n";
#ifdef __clang_version__
    json << "    \"compiler_version\": \"" << json_escape(__clang_version__) << "\",\n";
#endif
    json << "    \"warp_size\": " << diagnostics[0] << ",\n";
    json << "    \"threads_per_block\": " << diagnostics[1] << ",\n";
    json << "    \"waves_per_block\": " << diagnostics[3] << ",\n";
    json << "    \"k_tiles_per_layer\": " << diagnostics[4] << ",\n";
    json << "    \"accumulator_slots_per_lane\": " << diagnostics[5] << ",\n";
    json << "    \"matrix_a_slots_per_lane\": " << diagnostics[6] << "\n";
    json << "  },\n";
    json << "  \"topology\": {\n";
    json << "    \"batch_rows\": 16,\n";
    json << "    \"input_width\": 64,\n";
    json << "    \"hidden_1_width\": 64,\n";
    json << "    \"hidden_2_width\": 64,\n";
    json << "    \"layers\": 2,\n";
    json << "    \"tile\": \"16x16x16\",\n";
    json << "    \"k_tiles_per_layer\": 4,\n";
    json << "    \"mma_sync_calls_per_wave\": 8,\n";
    json << "    \"mma_sync_calls_per_block\": 32,\n";
    json << "    \"kernel_launches\": 1,\n";
    json << "    \"lds_bytes\": 2048,\n";
    json << "    \"publication_barriers\": 1,\n";
    json << "    \"hidden_1_transport\": \"LDS_only\",\n";
    json << "    \"hidden_1_global_store\": false,\n";
    json << "    \"hidden_1_global_reload\": false,\n";
    json << "    \"final_output\": \"hidden_2_fp16_diagnostic_store\"\n";
    json << "  },\n";
    json << "  \"diagnostics\": {\n";
    json << "    \"wave_entry_counts\": [" << diagnostics[8] << ", " << diagnostics[9] << ", " << diagnostics[10] << ", " << diagnostics[11] << "],\n";
    json << "    \"layer_1_publication_counts\": [" << diagnostics[12] << ", " << diagnostics[13] << ", " << diagnostics[14] << ", " << diagnostics[15] << "],\n";
    json << "    \"layer_1_cross_wave_check_counts\": [" << diagnostics[16] << ", " << diagnostics[17] << ", " << diagnostics[18] << ", " << diagnostics[19] << "],\n";
    json << "    \"layer_2_output_counts\": [" << diagnostics[20] << ", " << diagnostics[21] << ", " << diagnostics[22] << ", " << diagnostics[23] << "],\n";
    json << "    \"mapping_error_count\": " << diagnostics[24] << ",\n";
    json << "    \"hidden_1_lds_mismatch_count\": " << diagnostics[25] << ",\n";
    json << "    \"cross_wave_sources\": [" << diagnostics[28] << ", " << diagnostics[29] << ", " << diagnostics[30] << ", " << diagnostics[31] << "],\n";
    json << "    \"layer_2_k_tile_counts\": [" << diagnostics[32] << ", " << diagnostics[33] << ", " << diagnostics[34] << ", " << diagnostics[35] << "]\n";
    json << "  },\n";
    json << "  \"metrics\": {\n";
    json << "    \"hidden_2_mismatch_count\": " << hidden_2_mismatch_count << ",\n";
    json << "    \"first_mismatch_row\": " << (first_mismatch_index / WIDTH) << ",\n";
    json << "    \"first_mismatch_col\": " << (first_mismatch_index % WIDTH) << ",\n";
    json << "    \"nonfinite_count\": " << nonfinite_count << ",\n";
    json << "    \"zero_count\": " << zero_count << ",\n";
    json << "    \"positive_count\": " << positive_count << ",\n";
    json << "    \"max_abs\": " << max_abs << ",\n";
    json << "    \"max_error_row\": " << (max_error_index / WIDTH) << ",\n";
    json << "    \"max_error_col\": " << (max_error_index % WIDTH) << ",\n";
    json << "    \"relay_moved_entries_per_tile\": " << phase4a1_p2_generated::kRelayMovedEntries << "\n";
    json << "  },\n";
    json << "  \"gates\": {\n";
    json << "    \"wave32_context\": " << (context_ok ? "true" : "false") << ",\n";
    json << "    \"all_four_waves_entered_once\": " << (wave_entry_ok ? "true" : "false") << ",\n";
    json << "    \"layer_1_all_four_waves_published_once\": " << (layer_1_publication_ok ? "true" : "false") << ",\n";
    json << "    \"layer_1_cross_wave_visibility\": " << (layer_1_cross_wave_check_ok && rotated_sources_ok ? "true" : "false") << ",\n";
    json << "    \"layer_1_hidden_bitwise_equal\": " << (hidden_1_bitwise_ok ? "true" : "false") << ",\n";
    json << "    \"accumulator_to_matrix_a_mapping\": " << (mapping_ok ? "true" : "false") << ",\n";
    json << "    \"layer_2_four_k_tiles_per_wave\": " << (layer_2_k_tiles_ok ? "true" : "false") << ",\n";
    json << "    \"layer_2_all_four_output_tiles\": " << (layer_2_output_ok ? "true" : "false") << ",\n";
    json << "    \"hidden_2_epilogue_exercised\": " << (hidden_2_epilogue_exercised ? "true" : "false") << ",\n";
    json << "    \"hidden_2_bitwise_equal\": " << (hidden_2_bitwise_ok ? "true" : "false") << ",\n";
    json << "    \"guard_regions\": " << (guards_ok ? "true" : "false") << ",\n";
    json << "    \"layer_2_input_from_lds_only\": true,\n";
    json << "    \"no_intermediate_global_store_or_reload\": true\n";
    json << "  }\n";
    json << "}\n";

    std::cout << "marker: " << MARKER << '\n';
    std::cout << "hidden_1_lds_mismatch_count: " << diagnostics[25] << '\n';
    std::cout << "hidden_2_mismatch_count: " << hidden_2_mismatch_count << '\n';
    std::cout << "nonfinite_count: " << nonfinite_count << '\n';
    std::cout << "zero_count: " << zero_count << '\n';
    std::cout << "positive_count: " << positive_count << '\n';
    std::cout << "max_abs: " << max_abs << '\n';
    std::cout << "wave_entry_counts: " << diagnostics[8] << "," << diagnostics[9] << "," << diagnostics[10] << "," << diagnostics[11] << '\n';
    std::cout << "layer_1_publication_counts: " << diagnostics[12] << "," << diagnostics[13] << "," << diagnostics[14] << "," << diagnostics[15] << '\n';
    std::cout << "cross_wave_sources: " << diagnostics[28] << "," << diagnostics[29] << "," << diagnostics[30] << "," << diagnostics[31] << '\n';
    std::cout << "layer_2_k_tile_counts: " << diagnostics[32] << "," << diagnostics[33] << "," << diagnostics[34] << "," << diagnostics[35] << '\n';

    std::cout << "WIDTH64_LAYER1_HIDDEN_BITWISE_CORRECTNESS: " << (hidden_1_bitwise_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "WIDTH64_LAYER1_LDS_PUBLICATION: " << (layer_1_publication_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "WIDTH64_LAYER1_CROSS_WAVE_VISIBILITY: " << (layer_1_cross_wave_check_ok && rotated_sources_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "WIDTH64_LAYER2_INPUT_FROM_LDS_ONLY: PASS\n";
    std::cout << "WIDTH64_LAYER2_FOUR_K_TILE_ACCUMULATION: " << (layer_2_k_tiles_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "WIDTH64_LAYER2_HIDDEN_BITWISE_CORRECTNESS: " << (hidden_2_bitwise_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "WIDTH64_NO_INTERMEDIATE_GLOBAL_STORE_RELOAD: PASS\n";
    std::cout << "RDNA4_WIDTH64_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: " << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_PROCESS: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
