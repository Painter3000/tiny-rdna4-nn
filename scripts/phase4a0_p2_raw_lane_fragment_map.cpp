// TCNN_RDNA4_P4A0_P2_RAW_LANE_FRAGMENT_MAP_003
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
constexpr uint32_t WAVE = 32;
constexpr uint32_t MAX_SLOTS = 16;
constexpr uint32_t GUARD_WORDS = 16;
constexpr uint32_t TOTAL_CAPACITY = WAVE * MAX_SLOTS;

constexpr uint32_t DATA_SENTINEL = 0xDEADBEEFu;
constexpr uint32_t RAW_SENTINEL = 0xBAD0C0DEu;
constexpr uint32_t DATA_GUARD = 0xA5A5A5A5u;
constexpr uint32_t COUNT_GUARD = 0xC0FFEE11u;

constexpr uint32_t A_BASE = 1u;
constexpr uint32_t B_BASE = 1u;
constexpr uint32_t ACC_BASE = 1001u;

constexpr const char* MARKER =
    "TCNN_RDNA4_P4A0_P2_RAW_LANE_FRAGMENT_MAP_003";

using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;

using FragA = rocwmma::fragment<
    rocwmma::matrix_a, M, N, K, Half, rocwmma::row_major>;
using FragB = rocwmma::fragment<
    rocwmma::matrix_b, M, N, K, Half, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<
    rocwmma::accumulator, M, N, K, F32>;

using RegA = rocwmma::apply_register_file_t<FragA>;
using RegB = rocwmma::apply_register_file_t<FragB>;
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

union FloatBits {
    float value;
    uint32_t bits;
};

__host__ __device__ uint16_t half_bits(Half value) {
    HalfBits converter{};
    converter.value = value;
    return converter.bits;
}

__host__ __device__ uint32_t float_bits(float value) {
    FloatBits converter{};
    converter.value = value;
    return converter.bits;
}

struct DeviceView {
    uint32_t* markers;
    uint32_t* raw_bits;
    uint32_t* counts;
};

struct CaptureBuffer {
    std::vector<uint32_t> markers;
    std::vector<uint32_t> raw_bits;
    std::vector<uint32_t> counts;

    uint32_t* d_markers = nullptr;
    uint32_t* d_raw_bits = nullptr;
    uint32_t* d_counts = nullptr;

    CaptureBuffer()
        : markers(TOTAL_CAPACITY + 2 * GUARD_WORDS, DATA_GUARD),
          raw_bits(TOTAL_CAPACITY + 2 * GUARD_WORDS, DATA_GUARD),
          counts(TOTAL_CAPACITY + 2 * GUARD_WORDS, COUNT_GUARD) {
        std::fill(
            markers.begin() + GUARD_WORDS,
            markers.begin() + GUARD_WORDS + TOTAL_CAPACITY,
            DATA_SENTINEL);
        std::fill(
            raw_bits.begin() + GUARD_WORDS,
            raw_bits.begin() + GUARD_WORDS + TOTAL_CAPACITY,
            RAW_SENTINEL);
        std::fill(
            counts.begin() + GUARD_WORDS,
            counts.begin() + GUARD_WORDS + TOTAL_CAPACITY,
            0u);
    }

    void allocate_and_upload() {
        HIP_CHECK(hipMalloc(&d_markers, markers.size() * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&d_raw_bits, raw_bits.size() * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&d_counts, counts.size() * sizeof(uint32_t)));

        HIP_CHECK(hipMemcpy(
            d_markers,
            markers.data(),
            markers.size() * sizeof(uint32_t),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(
            d_raw_bits,
            raw_bits.data(),
            raw_bits.size() * sizeof(uint32_t),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(
            d_counts,
            counts.data(),
            counts.size() * sizeof(uint32_t),
            hipMemcpyHostToDevice));
    }

    DeviceView device_view() const {
        return {
            d_markers + GUARD_WORDS,
            d_raw_bits + GUARD_WORDS,
            d_counts + GUARD_WORDS,
        };
    }

    void download() {
        HIP_CHECK(hipMemcpy(
            markers.data(),
            d_markers,
            markers.size() * sizeof(uint32_t),
            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(
            raw_bits.data(),
            d_raw_bits,
            raw_bits.size() * sizeof(uint32_t),
            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(
            counts.data(),
            d_counts,
            counts.size() * sizeof(uint32_t),
            hipMemcpyDeviceToHost));
    }

    void release() {
        if (d_markers) HIP_CHECK(hipFree(d_markers));
        if (d_raw_bits) HIP_CHECK(hipFree(d_raw_bits));
        if (d_counts) HIP_CHECK(hipFree(d_counts));
        d_markers = nullptr;
        d_raw_bits = nullptr;
        d_counts = nullptr;
    }

    uint32_t marker(uint32_t lane, uint32_t slot) const {
        return markers[GUARD_WORDS + lane * MAX_SLOTS + slot];
    }

    uint32_t bits(uint32_t lane, uint32_t slot) const {
        return raw_bits[GUARD_WORDS + lane * MAX_SLOTS + slot];
    }

    uint32_t count(uint32_t lane, uint32_t slot) const {
        return counts[GUARD_WORDS + lane * MAX_SLOTS + slot];
    }
};

__global__ void capture_kernel(
    const Half* marker_a,
    const Half* marker_b,
    const Half* identity_a,
    const Half* output_b,
    DeviceView capture_a,
    DeviceView capture_b,
    DeviceView capture_acc,
    float* stored_output,
    uint32_t* diagnostics) {

    if (threadIdx.x == 0) {
        diagnostics[0] = static_cast<uint32_t>(warpSize);
        diagnostics[1] = static_cast<uint32_t>(blockDim.x);
        diagnostics[2] = static_cast<uint32_t>(gridDim.x);

        // These values are deliberately recorded in device code. Host-side
        // rocWMMA traits use the library's host fallback wave size and are
        // not authoritative for gfx1201 Wave32 geometry.
        diagnostics[3] = RegA::num_elements;
        diagnostics[4] = RegB::num_elements;
        diagnostics[5] = RegAcc::num_elements;
    }

    if (warpSize != WAVE || blockDim.x != WAVE || gridDim.x != 1) {
        return;
    }

    if (RegA::num_elements > MAX_SLOTS ||
        RegB::num_elements > MAX_SLOTS ||
        RegAcc::num_elements > MAX_SLOTS) {
        return;
    }

    const uint32_t lane = threadIdx.x;

    FragA frag_a;
    FragB frag_b;
    FragAcc frag_acc;

    rocwmma::load_matrix_sync(frag_a, marker_a, K);
    rocwmma::load_matrix_sync(frag_b, marker_b, K);

    auto const& reg_a = rocwmma::to_register_file(frag_a);
    auto const& reg_b = rocwmma::to_register_file(frag_b);

    for (uint32_t slot = 0; slot < RegA::num_elements; ++slot) {
        const uint32_t offset = lane * MAX_SLOTS + slot;
        const Half value = reg_a[slot];
        capture_a.markers[offset] =
            static_cast<uint32_t>(static_cast<float>(value));
        capture_a.raw_bits[offset] =
            static_cast<uint32_t>(half_bits(value));
        atomicAdd(&capture_a.counts[offset], 1u);
    }

    for (uint32_t slot = 0; slot < RegB::num_elements; ++slot) {
        const uint32_t offset = lane * MAX_SLOTS + slot;
        const Half value = reg_b[slot];
        capture_b.markers[offset] =
            static_cast<uint32_t>(static_cast<float>(value));
        capture_b.raw_bits[offset] =
            static_cast<uint32_t>(half_bits(value));
        atomicAdd(&capture_b.counts[offset], 1u);
    }

    // Independent accumulator map: I * B = B, where B contains unique exact
    // integer markers 1001..1256.
    rocwmma::load_matrix_sync(frag_a, identity_a, K);
    rocwmma::load_matrix_sync(frag_b, output_b, K);
    rocwmma::fill_fragment(frag_acc, 0.0f);
    rocwmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

    auto const& reg_acc = rocwmma::to_register_file(frag_acc);

    for (uint32_t slot = 0; slot < RegAcc::num_elements; ++slot) {
        const uint32_t offset = lane * MAX_SLOTS + slot;
        const float value = reg_acc[slot];
        capture_acc.markers[offset] = static_cast<uint32_t>(value);
        capture_acc.raw_bits[offset] = float_bits(value);
        atomicAdd(&capture_acc.counts[offset], 1u);
    }

    rocwmma::store_matrix_sync(
        stored_output,
        frag_acc,
        N,
        rocwmma::mem_row_major);
}

bool guards_intact(const std::vector<uint32_t>& values, uint32_t guard) {
    for (uint32_t index = 0; index < GUARD_WORDS; ++index) {
        if (values[index] != guard) return false;
        if (values[GUARD_WORDS + TOTAL_CAPACITY + index] != guard) return false;
    }
    return true;
}

bool buffer_guards_intact(const CaptureBuffer& capture) {
    return
        guards_intact(capture.markers, DATA_GUARD) &&
        guards_intact(capture.raw_bits, DATA_GUARD) &&
        guards_intact(capture.counts, COUNT_GUARD);
}

bool slot_ownership_valid(
    const CaptureBuffer& capture,
    uint32_t active_slots) {

    if (active_slots == 0 || active_slots > MAX_SLOTS) return false;

    for (uint32_t lane = 0; lane < WAVE; ++lane) {
        for (uint32_t slot = 0; slot < MAX_SLOTS; ++slot) {
            const bool active = slot < active_slots;
            if (active) {
                if (capture.count(lane, slot) != 1u) return false;
            } else {
                if (capture.count(lane, slot) != 0u) return false;
                if (capture.marker(lane, slot) != DATA_SENTINEL) return false;
                if (capture.bits(lane, slot) != RAW_SENTINEL) return false;
            }
        }
    }

    return true;
}

bool validate_half_map(
    const CaptureBuffer& capture,
    uint32_t slots,
    uint32_t marker_base) {

    if (slots * WAVE != M * K) return false;

    std::vector<uint32_t> occurrences(M * K, 0u);
    for (uint32_t lane = 0; lane < WAVE; ++lane) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            const uint32_t marker = capture.marker(lane, slot);
            if (marker < marker_base || marker >= marker_base + M * K) {
                return false;
            }

            ++occurrences[marker - marker_base];

            const uint32_t expected_bits =
                static_cast<uint32_t>(half_bits(static_cast<Half>(marker)));
            if (capture.bits(lane, slot) != expected_bits) {
                return false;
            }
        }
    }

    return std::all_of(
        occurrences.begin(),
        occurrences.end(),
        [](uint32_t count) { return count == 1u; });
}

bool validate_float_map(
    const CaptureBuffer& capture,
    uint32_t slots,
    uint32_t marker_base) {

    if (slots * WAVE != M * N) return false;

    std::vector<uint32_t> occurrences(M * N, 0u);
    for (uint32_t lane = 0; lane < WAVE; ++lane) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            const uint32_t marker = capture.marker(lane, slot);
            if (marker < marker_base || marker >= marker_base + M * N) {
                return false;
            }

            ++occurrences[marker - marker_base];

            const uint32_t expected_bits =
                float_bits(static_cast<float>(marker));
            if (capture.bits(lane, slot) != expected_bits) {
                return false;
            }
        }
    }

    return std::all_of(
        occurrences.begin(),
        occurrences.end(),
        [](uint32_t count) { return count == 1u; });
}

bool stored_output_valid(const std::vector<float>& output) {
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const float expected =
                static_cast<float>(ACC_BASE + row * N + col);
            if (output[row * N + col] != expected) return false;
        }
    }
    return true;
}

std::string hex_bits(uint32_t value, bool half) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << std::setfill('0')
        << std::setw(half ? 4 : 8)
        << (half ? (value & 0xFFFFu) : value);
    return out.str();
}

void write_map_rows(
    std::ofstream& out,
    const char* record_type,
    const char* role,
    const CaptureBuffer& capture,
    uint32_t slots,
    uint32_t marker_base,
    bool half) {

    for (uint32_t slot = 0; slot < slots; ++slot) {
        for (uint32_t lane = 0; lane < WAVE; ++lane) {
            const uint32_t marker = capture.marker(lane, slot);
            const uint32_t linear = marker - marker_base;

            out << record_type << '\t'
                << role << '\t'
                << slot << '\t'
                << lane << '\t'
                << marker << '\t'
                << (linear / N) << '\t'
                << (linear % N) << '\t'
                << hex_bits(capture.bits(lane, slot), half) << '\t'
                << capture.count(lane, slot) << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <capture.tsv>" << std::endl;
        return 64;
    }

    const std::filesystem::path output_path = argv[1];
    std::filesystem::create_directories(output_path.parent_path());

    int device = 0;
    int runtime_version = 0;
    int driver_version = 0;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));

    std::vector<Half> marker_a(M * K);
    std::vector<Half> marker_b(K * N);
    std::vector<Half> identity_a(M * K);
    std::vector<Half> output_b(K * N);

    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < K; ++col) {
            marker_a[row * K + col] =
                static_cast<Half>(A_BASE + row * K + col);
            identity_a[row * K + col] =
                static_cast<Half>(row == col ? 1.0f : 0.0f);
        }
    }

    for (uint32_t row = 0; row < K; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            marker_b[col * K + row] =
                static_cast<Half>(B_BASE + row * N + col);
            output_b[col * K + row] =
                static_cast<Half>(ACC_BASE + row * N + col);
        }
    }

    Half* d_marker_a = nullptr;
    Half* d_marker_b = nullptr;
    Half* d_identity_a = nullptr;
    Half* d_output_b = nullptr;
    float* d_stored_output = nullptr;
    uint32_t* d_diagnostics = nullptr;

    HIP_CHECK(hipMalloc(&d_marker_a, marker_a.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_marker_b, marker_b.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_identity_a, identity_a.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_output_b, output_b.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_stored_output, M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_diagnostics, 6 * sizeof(uint32_t)));

    HIP_CHECK(hipMemcpy(
        d_marker_a,
        marker_a.data(),
        marker_a.size() * sizeof(Half),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_marker_b,
        marker_b.data(),
        marker_b.size() * sizeof(Half),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_identity_a,
        identity_a.data(),
        identity_a.size() * sizeof(Half),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_output_b,
        output_b.data(),
        output_b.size() * sizeof(Half),
        hipMemcpyHostToDevice));

    std::vector<float> stored_output(
        M * N,
        std::numeric_limits<float>::quiet_NaN());
    HIP_CHECK(hipMemcpy(
        d_stored_output,
        stored_output.data(),
        stored_output.size() * sizeof(float),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_diagnostics, 0, 6 * sizeof(uint32_t)));

    CaptureBuffer capture_a;
    CaptureBuffer capture_b;
    CaptureBuffer capture_acc;
    capture_a.allocate_and_upload();
    capture_b.allocate_and_upload();
    capture_acc.allocate_and_upload();

    hipLaunchKernelGGL(
        capture_kernel,
        dim3(1),
        dim3(WAVE),
        0,
        0,
        d_marker_a,
        d_marker_b,
        d_identity_a,
        d_output_b,
        capture_a.device_view(),
        capture_b.device_view(),
        capture_acc.device_view(),
        d_stored_output,
        d_diagnostics);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    capture_a.download();
    capture_b.download();
    capture_acc.download();

    std::vector<uint32_t> diagnostics(6, 0u);
    HIP_CHECK(hipMemcpy(
        diagnostics.data(),
        d_diagnostics,
        diagnostics.size() * sizeof(uint32_t),
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        stored_output.data(),
        d_stored_output,
        stored_output.size() * sizeof(float),
        hipMemcpyDeviceToHost));

    capture_a.release();
    capture_b.release();
    capture_acc.release();

    HIP_CHECK(hipFree(d_marker_a));
    HIP_CHECK(hipFree(d_marker_b));
    HIP_CHECK(hipFree(d_identity_a));
    HIP_CHECK(hipFree(d_output_b));
    HIP_CHECK(hipFree(d_stored_output));
    HIP_CHECK(hipFree(d_diagnostics));

    const uint32_t a_slots = diagnostics[3];
    const uint32_t b_slots = diagnostics[4];
    const uint32_t acc_slots = diagnostics[5];

    const bool context_ok =
        std::string(props.gcnArchName).rfind("gfx1201", 0) == 0 &&
        props.warpSize == static_cast<int>(WAVE) &&
        diagnostics[0] == WAVE &&
        diagnostics[1] == WAVE &&
        diagnostics[2] == 1;

    const bool geometry_ok =
        a_slots == 8 &&
        b_slots == 8 &&
        acc_slots == 8;

    const bool guards_ok =
        buffer_guards_intact(capture_a) &&
        buffer_guards_intact(capture_b) &&
        buffer_guards_intact(capture_acc);

    const bool ownership_ok =
        slot_ownership_valid(capture_a, a_slots) &&
        slot_ownership_valid(capture_b, b_slots) &&
        slot_ownership_valid(capture_acc, acc_slots);

    const bool map_a_ok =
        validate_half_map(capture_a, a_slots, A_BASE);
    const bool map_b_ok =
        validate_half_map(capture_b, b_slots, B_BASE);
    const bool map_acc_ok =
        validate_float_map(capture_acc, acc_slots, ACC_BASE);
    const bool output_ok = stored_output_valid(stored_output);

    const bool passed =
        context_ok &&
        geometry_ok &&
        guards_ok &&
        ownership_ok &&
        map_a_ok &&
        map_b_ok &&
        map_acc_ok &&
        output_ok;

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "OUTPUT_ERROR: cannot write " << output_path << std::endl;
        return 3;
    }

    out << "META\tmarker\t" << MARKER << '\n';
    out << "META\tdecision\t" << (passed ? "PASS" : "FAIL") << '\n';
    out << "META\tdevice\t" << props.name << '\n';
    out << "META\tarch\t" << props.gcnArchName << '\n';
    out << "META\thip_runtime_version\t" << runtime_version << '\n';
    out << "META\thip_driver_version\t" << driver_version << '\n';
#ifdef __clang_version__
    out << "META\tcompiler_version\t" << __clang_version__ << '\n';
#endif
    out << "META\twarp_size\t" << diagnostics[0] << '\n';
    out << "META\tmatrix_a_device_slots_per_lane\t" << a_slots << '\n';
    out << "META\tmatrix_b_device_slots_per_lane\t" << b_slots << '\n';
    out << "META\taccumulator_device_slots_per_lane\t" << acc_slots << '\n';
    out << "META\tcapture_capacity_slots_per_lane\t" << MAX_SLOTS << '\n';

    out << "GATE\twave32_context\t" << context_ok << '\n';
    out << "GATE\tdevice_register_geometry\t" << geometry_ok << '\n';
    out << "GATE\tguard_regions\t" << guards_ok << '\n';
    out << "GATE\tunique_slot_write_ownership\t" << ownership_ok << '\n';
    out << "GATE\tmatrix_a_marker_bijection\t" << map_a_ok << '\n';
    out << "GATE\tmatrix_b_marker_bijection\t" << map_b_ok << '\n';
    out << "GATE\taccumulator_marker_bijection\t" << map_acc_ok << '\n';
    out << "GATE\tstored_output\t" << output_ok << '\n';

    if (a_slots <= MAX_SLOTS) {
        write_map_rows(
            out,
            "MAP_HALF",
            "matrix_a",
            capture_a,
            a_slots,
            A_BASE,
            true);
    }

    if (b_slots <= MAX_SLOTS) {
        write_map_rows(
            out,
            "MAP_HALF",
            "matrix_b",
            capture_b,
            b_slots,
            B_BASE,
            true);
    }

    if (acc_slots <= MAX_SLOTS) {
        write_map_rows(
            out,
            "MAP_FLOAT",
            "accumulator",
            capture_acc,
            acc_slots,
            ACC_BASE,
            false);
    }

    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            out << "OUTPUT\t"
                << row << '\t'
                << col << '\t'
                << stored_output[row * N + col] << '\t'
                << (ACC_BASE + row * N + col) << '\n';
        }
    }

    std::cout << "marker: " << MARKER << '\n';
    std::cout << "device_matrix_a_slots_per_lane: " << a_slots << '\n';
    std::cout << "device_matrix_b_slots_per_lane: " << b_slots << '\n';
    std::cout << "device_accumulator_slots_per_lane: " << acc_slots << '\n';
    std::cout << "ROCWMMA_P2_DEVICE_GEOMETRY_CAPTURE: "
              << (geometry_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_WAVE32_CONTEXT: "
              << (context_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_GUARD_REGIONS: "
              << (guards_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_WRITE_OWNERSHIP: "
              << (ownership_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_MATRIX_A_REGISTER_MAP: "
              << (map_a_ok ? "CAPTURED" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_MATRIX_B_REGISTER_MAP: "
              << (map_b_ok ? "CAPTURED" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_ACCUMULATOR_REGISTER_MAP: "
              << (map_acc_ok ? "CAPTURED" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P2_STORED_OUTPUT_VALIDATION: "
              << (output_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_PROCESS: "
              << (passed ? "PASS" : "FAIL") << '\n';

    return passed ? 0 : 1;
}
