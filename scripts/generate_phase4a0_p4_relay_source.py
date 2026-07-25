#!/usr/bin/env python3
"""Generate the Phase 4A0-P4 accumulator-to-matrix-A relay probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A0_P4_ACC_TO_A_RELAY_001"
P3_DECISION = "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_array(name: str, values: list[int]) -> str:
    lines = []
    for start in range(0, len(values), 16):
        chunk = ", ".join(str(value) for value in values[start : start + 16])
        lines.append(f"    {chunk},")
    return (
        f"__device__ __constant__ unsigned char {name}[256] = {{\n"
        + "\n".join(lines)
        + "\n};"
    )


def load_and_validate(path: Path) -> tuple[dict[str, Any], list[int], list[int], list[int]]:
    data = json.loads(path.read_text())

    assert data["decision"] == P3_DECISION
    assert all(bool(value) for value in data["gates"].values())

    context = data["context"]
    assert context["arch"].startswith("gfx1201")
    assert int(context["warp_size"]) == 32
    assert int(context["matrix_a_device_slots_per_lane"]) == 8
    assert int(context["accumulator_device_slots_per_lane"]) == 8

    pair = data["pairwise_role_equivalence"]["matrix_a_to_accumulator"]
    assert not pair["direct_same_slot_elementwise_safe"]

    entries = pair["coordinate_preserving_reindex"]["entries"]
    assert len(entries) == 256

    acc_inverse = data["role_models"]["accumulator"]["coordinate_to_lane_slot"]
    a_inverse = data["role_models"]["matrix_a"]["coordinate_to_lane_slot"]
    assert len(acc_inverse) == 256
    assert len(a_inverse) == 256

    source_lane_for_a = [-1] * 256
    source_slot_for_a = [-1] * 256
    acc_col_by_slot = [-1] * 256

    seen_a: set[tuple[int, int]] = set()
    seen_acc: set[tuple[int, int]] = set()

    for entry in entries:
        row = int(entry["matrix_row"])
        col = int(entry["matrix_col"])
        a_lane = int(entry["source_lane"])
        a_slot = int(entry["source_slot"])
        acc_lane = int(entry["target_lane"])
        acc_slot = int(entry["target_slot"])

        assert 0 <= row < 16 and 0 <= col < 16
        assert 0 <= a_lane < 32 and 0 <= a_slot < 8
        assert 0 <= acc_lane < 32 and 0 <= acc_slot < 8

        coord_key = f"{row},{col}"
        assert a_inverse[coord_key] == {"lane": a_lane, "slot": a_slot}
        assert acc_inverse[coord_key] == {"lane": acc_lane, "slot": acc_slot}

        a_index = a_lane * 8 + a_slot
        source_lane_for_a[a_index] = acc_lane
        source_slot_for_a[a_index] = acc_slot
        seen_a.add((a_lane, a_slot))
        seen_acc.add((acc_lane, acc_slot))

    assert len(seen_a) == 256
    assert len(seen_acc) == 256
    assert all(0 <= value < 32 for value in source_lane_for_a)
    assert all(0 <= value < 8 for value in source_slot_for_a)

    for coord_key, lane_slot in acc_inverse.items():
        row_text, col_text = coord_key.split(",")
        row = int(row_text)
        col = int(col_text)
        lane = int(lane_slot["lane"])
        slot = int(lane_slot["slot"])
        assert 0 <= row < 16 and 0 <= col < 16
        index = lane * 8 + slot
        assert acc_col_by_slot[index] == -1
        acc_col_by_slot[index] = col

    assert all(0 <= value < 16 for value in acc_col_by_slot)

    moved = sum(
        1
        for a_index, (src_lane, src_slot) in enumerate(
            zip(source_lane_for_a, source_slot_for_a)
        )
        if (a_index // 8, a_index % 8) != (src_lane, src_slot)
    )
    assert moved > 0

    return data, source_lane_for_a, source_slot_for_a, acc_col_by_slot


def generate_source(
    p3_path: Path,
    source_lane_for_a: list[int],
    source_slot_for_a: list[int],
    acc_col_by_slot: list[int],
) -> str:
    p3_sha = sha256(p3_path)

    arrays = "\n\n".join(
        (
            format_array("kAccLaneForATargetA", source_lane_for_a),
            format_array("kAccSlotForATargetA", source_slot_for_a),
            format_array("kAccumulatorColumn", acc_col_by_slot),
        )
    )

    template = r"""// __MARKER__
// Generated from Phase 4A0-P3 SHA256: __P3_SHA__

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
constexpr uint32_t SLOTS = 8;
constexpr uint32_t ELEMENTS = 256;
constexpr uint32_t GUARD = 16;

constexpr const char* MARKER = "__MARKER__";
constexpr const char* P3_SHA256 = "__P3_SHA__";

using Half = rocwmma::float16_t;
using F32 = rocwmma::float32_t;

using FragA = rocwmma::fragment<
    rocwmma::matrix_a, M, N, K, Half, rocwmma::row_major>;
using FragB = rocwmma::fragment<
    rocwmma::matrix_b, M, N, K, Half, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<
    rocwmma::accumulator, M, N, K, F32>;
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

__ARRAYS__

__global__ void relay_kernel(
    const Half* identity_a,
    const Half* source_b,
    const float* bias,
    Half* relay_output,
    uint32_t* diagnostics) {

    if (threadIdx.x == 0) {
        diagnostics[0] = static_cast<uint32_t>(warpSize);
        diagnostics[1] = static_cast<uint32_t>(blockDim.x);
        diagnostics[2] = static_cast<uint32_t>(gridDim.x);
        diagnostics[3] = RegAcc::num_elements;
        diagnostics[4] = RegA::num_elements;
    }

    if (warpSize != WAVE ||
        blockDim.x != WAVE ||
        gridDim.x != 1 ||
        RegAcc::num_elements != SLOTS ||
        RegA::num_elements != SLOTS) {
        if (threadIdx.x == 0) {
            diagnostics[5] = 1u;
        }
        return;
    }

    const uint32_t lane = threadIdx.x;

    FragA frag_identity;
    FragB frag_source;
    FragAcc frag_acc;

    rocwmma::load_matrix_sync(frag_identity, identity_a, K);
    rocwmma::load_matrix_sync(frag_source, source_b, K);
    rocwmma::fill_fragment(frag_acc, 0.0f);
    rocwmma::mma_sync(frag_acc, frag_identity, frag_source, frag_acc);

    auto const& reg_acc = rocwmma::to_register_file(frag_acc);

    F32 epilogue[SLOTS];
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        const uint32_t acc_index = lane * SLOTS + slot;
        const uint32_t col = kAccumulatorColumn[acc_index];
        const F32 biased = reg_acc[slot] + bias[col];
        epilogue[slot] = biased > 0.0f ? biased : 0.0f;
    }

    RegA relay_reg_a;

    for (uint32_t target_slot = 0; target_slot < SLOTS; ++target_slot) {
        const uint32_t target_index = lane * SLOTS + target_slot;
        const uint32_t desired_lane = kAccLaneForATargetA[target_index];
        const uint32_t desired_slot = kAccSlotForATargetA[target_index];

        if (desired_lane >= WAVE || desired_slot >= SLOTS) {
            atomicAdd(&diagnostics[6], 1u);
        }

        F32 selected = 0.0f;
        for (uint32_t candidate_slot = 0; candidate_slot < SLOTS; ++candidate_slot) {
            const F32 candidate = epilogue[candidate_slot];
            const F32 shuffled = __shfl(candidate, desired_lane, WAVE);
            if (candidate_slot == desired_slot) {
                selected = shuffled;
            }
        }

        relay_reg_a[target_slot] = static_cast<Half>(selected);
    }

    auto const& relayed_frag_a =
        rocwmma::from_register_file<FragA>(relay_reg_a);

    rocwmma::store_matrix_sync(
        relay_output,
        relayed_frag_a,
        K,
        rocwmma::mem_row_major);
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
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <result.json> <matrix.csv>" << std::endl;
        return 64;
    }

    const std::filesystem::path json_path = argv[1];
    const std::filesystem::path csv_path = argv[2];
    std::filesystem::create_directories(json_path.parent_path());
    std::filesystem::create_directories(csv_path.parent_path());

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    int runtime_version = 0;
    int driver_version = 0;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));

    std::vector<Half> identity_a(ELEMENTS);
    std::vector<Half> source_b(ELEMENTS);
    std::vector<float> bias(N);

    uint32_t input_quantization_changed = 0;
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < K; ++col) {
            identity_a[row * K + col] =
                static_cast<Half>(row == col ? 1.0f : 0.0f);
        }
    }

    for (uint32_t row = 0; row < K; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const int code =
                static_cast<int>((row * 17u + col * 13u + 5u) % 47u) - 23;
            const float source = static_cast<float>(code) * 0.0173f;
            const Half quantized = static_cast<Half>(source);
            source_b[col * K + row] = quantized;
            if (static_cast<float>(quantized) != source) {
                ++input_quantization_changed;
            }
        }
    }

    for (uint32_t col = 0; col < N; ++col) {
        bias[col] =
            (static_cast<int>(col) - 7) * 0.0137f + 0.0031f;
    }

    const Half guard_value = static_cast<Half>(-123.5f);
    const uint16_t guard_bits = half_bits(guard_value);

    std::vector<Half> output(ELEMENTS + 2 * GUARD, guard_value);
    std::vector<Half> reference(ELEMENTS);

    uint32_t relu_clamped_count = 0;
    uint32_t positive_count = 0;
    uint32_t fp16_cast_changed_count = 0;

    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const float accumulator =
                static_cast<float>(source_b[col * K + row]);
            const float biased = accumulator + bias[col];
            const float activated = biased > 0.0f ? biased : 0.0f;
            const Half casted = static_cast<Half>(activated);

            if (biased <= 0.0f) {
                ++relu_clamped_count;
            } else {
                ++positive_count;
            }
            if (static_cast<float>(casted) != activated) {
                ++fp16_cast_changed_count;
            }

            reference[row * N + col] = casted;
        }
    }

    Half* d_identity_a = nullptr;
    Half* d_source_b = nullptr;
    float* d_bias = nullptr;
    Half* d_output = nullptr;
    uint32_t* d_diagnostics = nullptr;

    HIP_CHECK(hipMalloc(&d_identity_a, identity_a.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_source_b, source_b.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_bias, bias.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(Half)));
    HIP_CHECK(hipMalloc(&d_diagnostics, 8 * sizeof(uint32_t)));

    HIP_CHECK(hipMemcpy(
        d_identity_a,
        identity_a.data(),
        identity_a.size() * sizeof(Half),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_source_b,
        source_b.data(),
        source_b.size() * sizeof(Half),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_bias,
        bias.data(),
        bias.size() * sizeof(float),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_output,
        output.data(),
        output.size() * sizeof(Half),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_diagnostics, 0, 8 * sizeof(uint32_t)));

    hipLaunchKernelGGL(
        relay_kernel,
        dim3(1),
        dim3(WAVE),
        0,
        0,
        d_identity_a,
        d_source_b,
        d_bias,
        d_output + GUARD,
        d_diagnostics);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> diagnostics(8, 0u);
    HIP_CHECK(hipMemcpy(
        output.data(),
        d_output,
        output.size() * sizeof(Half),
        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        diagnostics.data(),
        d_diagnostics,
        diagnostics.size() * sizeof(uint32_t),
        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_identity_a));
    HIP_CHECK(hipFree(d_source_b));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_diagnostics));

    bool guards_ok = true;
    for (uint32_t index = 0; index < GUARD; ++index) {
        guards_ok &=
            half_bits(output[index]) == guard_bits &&
            half_bits(output[GUARD + ELEMENTS + index]) == guard_bits;
    }

    uint32_t mismatch_count = 0;
    uint32_t max_mismatch_index = 0;
    double max_abs = 0.0;

    for (uint32_t index = 0; index < ELEMENTS; ++index) {
        const Half gpu = output[GUARD + index];
        const Half cpu = reference[index];

        if (half_bits(gpu) != half_bits(cpu)) {
            ++mismatch_count;
            const double diff = std::abs(
                static_cast<double>(static_cast<float>(gpu)) -
                static_cast<double>(static_cast<float>(cpu)));
            if (diff > max_abs) {
                max_abs = diff;
                max_mismatch_index = index;
            }
        }
    }

    const bool context_ok =
        std::string(props.gcnArchName).rfind("gfx1201", 0) == 0 &&
        props.warpSize == static_cast<int>(WAVE) &&
        diagnostics[0] == WAVE &&
        diagnostics[1] == WAVE &&
        diagnostics[2] == 1 &&
        diagnostics[3] == SLOTS &&
        diagnostics[4] == SLOTS &&
        diagnostics[5] == 0;

    const bool mapping_ok = diagnostics[6] == 0;
    const bool epilogue_exercised =
        relu_clamped_count > 0 &&
        positive_count > 0;
    const bool cast_exercised =
        input_quantization_changed > 0 &&
        fp16_cast_changed_count > 0;
    const bool matrix_ok =
        mismatch_count == 0 &&
        max_abs == 0.0;
    const bool passed =
        context_ok &&
        mapping_ok &&
        guards_ok &&
        epilogue_exercised &&
        cast_exercised &&
        matrix_ok;

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "OUTPUT_ERROR: cannot write " << csv_path << std::endl;
        return 3;
    }
    csv << "row,col,gpu,gpu_bits,cpu,cpu_bits\n";
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const Half gpu = output[GUARD + row * N + col];
            const Half cpu = reference[row * N + col];
            csv << row << ','
                << col << ','
                << static_cast<float>(gpu) << ','
                << half_bits(gpu) << ','
                << static_cast<float>(cpu) << ','
                << half_bits(cpu) << '\n';
        }
    }

    std::ofstream json(json_path);
    if (!json) {
        std::cerr << "OUTPUT_ERROR: cannot write " << json_path << std::endl;
        return 3;
    }

    json << std::setprecision(17);
    json << "{\n";
    json << "  \"marker\": \"" << MARKER << "\",\n";
    json << "  \"decision\": \""
         << (passed
                ? "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS"
                : "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_FAIL")
         << "\",\n";
    json << "  \"p3_sha256\": \"" << P3_SHA256 << "\",\n";
    json << "  \"context\": {\n";
    json << "    \"device\": \"" << json_escape(props.name) << "\",\n";
    json << "    \"arch\": \"" << json_escape(props.gcnArchName) << "\",\n";
    json << "    \"hip_runtime_version\": " << runtime_version << ",\n";
    json << "    \"hip_driver_version\": " << driver_version << ",\n";
#ifdef __clang_version__
    json << "    \"compiler_version\": \"" << json_escape(__clang_version__) << "\",\n";
#endif
    json << "    \"warp_size\": " << diagnostics[0] << ",\n";
    json << "    \"accumulator_slots_per_lane\": " << diagnostics[3] << ",\n";
    json << "    \"matrix_a_slots_per_lane\": " << diagnostics[4] << "\n";
    json << "  },\n";
    json << "  \"pipeline\": [\n";
    json << "    \"identity_fp16_x_source_b_fp16_to_accumulator_fp32\",\n";
    json << "    \"output_column_bias_fp32\",\n";
    json << "    \"relu_fp32\",\n";
    json << "    \"p3_accumulator_to_matrix_a_wave_shuffle_reindex\",\n";
    json << "    \"fp32_to_fp16_cast\",\n";
    json << "    \"from_register_file_matrix_a\",\n";
    json << "    \"store_matrix_sync_row_major\"\n";
    json << "  ],\n";
    json << "  \"metrics\": {\n";
    json << "    \"input_quantization_changed\": "
         << input_quantization_changed << ",\n";
    json << "    \"relu_clamped_count\": "
         << relu_clamped_count << ",\n";
    json << "    \"positive_count\": "
         << positive_count << ",\n";
    json << "    \"fp16_cast_changed_count\": "
         << fp16_cast_changed_count << ",\n";
    json << "    \"mismatch_count\": "
         << mismatch_count << ",\n";
    json << "    \"max_abs\": "
         << max_abs << ",\n";
    json << "    \"max_mismatch_row\": "
         << (max_mismatch_index / N) << ",\n";
    json << "    \"max_mismatch_col\": "
         << (max_mismatch_index % N) << "\n";
    json << "  },\n";
    json << "  \"gates\": {\n";
    json << "    \"wave32_context\": "
         << (context_ok ? "true" : "false") << ",\n";
    json << "    \"p3_mapping_embedded\": "
         << (mapping_ok ? "true" : "false") << ",\n";
    json << "    \"guard_regions\": "
         << (guards_ok ? "true" : "false") << ",\n";
    json << "    \"accumulator_epilogue_exercised\": "
         << (epilogue_exercised ? "true" : "false") << ",\n";
    json << "    \"fp16_cast_exercised\": "
         << (cast_exercised ? "true" : "false") << ",\n";
    json << "    \"stored_matrix_bitwise_equal\": "
         << (matrix_ok ? "true" : "false") << "\n";
    json << "  }\n";
    json << "}\n";

    std::cout << "marker: " << MARKER << '\n';
    std::cout << "p3_sha256: " << P3_SHA256 << '\n';
    std::cout << "input_quantization_changed: "
              << input_quantization_changed << '\n';
    std::cout << "relu_clamped_count: "
              << relu_clamped_count << '\n';
    std::cout << "positive_count: "
              << positive_count << '\n';
    std::cout << "fp16_cast_changed_count: "
              << fp16_cast_changed_count << '\n';
    std::cout << "mismatch_count: "
              << mismatch_count << '\n';
    std::cout << "max_abs: "
              << max_abs << '\n';

    std::cout << "ROCWMMA_P4_WAVE32_CONTEXT: "
              << (context_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P4_P3_MAPPING_EMBEDDED: "
              << (mapping_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P4_GUARD_REGIONS: "
              << (guards_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_P4_ACCUMULATOR_EPILOGUE: "
              << (epilogue_exercised ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_ACC_TO_A_REINDEX: "
              << (matrix_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_ACC_TO_A_FP16_CAST: "
              << (cast_exercised && matrix_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "ROCWMMA_ACC_TO_A_STORED_MATRIX: "
              << (matrix_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PROCESS: "
              << (passed ? "PASS" : "FAIL") << '\n';

    return passed ? 0 : 1;
}
"""

    return (
        template
        .replace("__MARKER__", MARKER)
        .replace("__P3_SHA__", p3_sha)
        .replace("__ARRAYS__", arrays)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-json", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    data, source_lanes, source_slots, acc_cols = load_and_validate(args.p3_json)
    source = generate_source(
        args.p3_json,
        source_lanes,
        source_slots,
        acc_cols,
    )

    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.write_text(source)

    moved = sum(
        1
        for index, pair in enumerate(zip(source_lanes, source_slots))
        if (index // 8, index % 8) != pair
    )

    manifest = {
        "marker": MARKER,
        "decision": "PHASE4A0_P4_SOURCE_GENERATION_PASS",
        "p3_json": str(args.p3_json.resolve()),
        "p3_sha256": sha256(args.p3_json),
        "generated_source": str(args.output_source.resolve()),
        "generated_source_sha256": sha256(args.output_source),
        "mapping": {
            "target_role": "matrix_a",
            "source_role": "accumulator",
            "entries": 256,
            "moved_entries": moved,
            "fixed_entries": 256 - moved,
            "source_lane_sha256": hashlib.sha256(bytes(source_lanes)).hexdigest(),
            "source_slot_sha256": hashlib.sha256(bytes(source_slots)).hexdigest(),
            "accumulator_column_sha256": hashlib.sha256(bytes(acc_cols)).hexdigest(),
        },
        "context": data["context"],
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    assert moved > 0
    print(f"p3_sha256: {manifest['p3_sha256']}")
    print(f"generated_source_sha256: {manifest['generated_source_sha256']}")
    print(f"relay_moved_entries: {moved}")
    print("PHASE4A0_P4_SOURCE_GENERATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
