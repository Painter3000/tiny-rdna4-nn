#!/usr/bin/env python3
"""Generate the Phase 4A0-P5 two-layer fused-forward correctness probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A0_P5_TWO_LAYER_FUSED_FORWARD_001"
P3_DECISION = "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"
P4_DECISION = "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS"


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


def load_mapping(
    p3_path: Path,
    p4_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[int], list[int], list[int]]:
    p3 = json.loads(p3_path.read_text())
    p4 = json.loads(p4_path.read_text())

    assert p3["decision"] == P3_DECISION
    assert all(bool(value) for value in p3["gates"].values())

    assert p4["decision"] == P4_DECISION
    assert all(bool(value) for value in p4["gates"].values())
    assert p4["p3_sha256"] == sha256(p3_path)

    context = p3["context"]
    assert context["arch"].startswith("gfx1201")
    assert int(context["warp_size"]) == 32
    assert int(context["matrix_a_device_slots_per_lane"]) == 8
    assert int(context["accumulator_device_slots_per_lane"]) == 8

    pair = p3["pairwise_role_equivalence"]["matrix_a_to_accumulator"]
    assert not pair["direct_same_slot_elementwise_safe"]
    entries = pair["coordinate_preserving_reindex"]["entries"]
    assert len(entries) == 256

    acc_inverse = p3["role_models"]["accumulator"]["coordinate_to_lane_slot"]
    a_inverse = p3["role_models"]["matrix_a"]["coordinate_to_lane_slot"]
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
        _, col_text = coord_key.split(",")
        col = int(col_text)
        lane = int(lane_slot["lane"])
        slot = int(lane_slot["slot"])
        index = lane * 8 + slot
        assert acc_col_by_slot[index] == -1
        acc_col_by_slot[index] = col

    assert all(0 <= value < 16 for value in acc_col_by_slot)

    moved = sum(
        1
        for index, pair_value in enumerate(
            zip(source_lane_for_a, source_slot_for_a)
        )
        if (index // 8, index % 8) != pair_value
    )
    assert moved == 240

    return p3, p4, source_lane_for_a, source_slot_for_a, acc_col_by_slot


def generate_source(
    p3_path: Path,
    p4_path: Path,
    source_lane_for_a: list[int],
    source_slot_for_a: list[int],
    acc_col_by_slot: list[int],
) -> str:
    p3_sha = sha256(p3_path)
    p4_sha = sha256(p4_path)

    arrays = "\n\n".join(
        (
            format_array("kAccLaneForATargetA", source_lane_for_a),
            format_array("kAccSlotForATargetA", source_slot_for_a),
            format_array("kAccumulatorColumn", acc_col_by_slot),
        )
    )

    template = r"""// __MARKER__
// Generated from Phase 4A0-P3 SHA256: __P3_SHA__
// Qualified by Phase 4A0-P4 SHA256: __P4_SHA__

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

constexpr double OUTPUT_MAX_ABS_TOLERANCE = 5.0e-5;
constexpr double OUTPUT_NORMALIZED_L2_TOLERANCE = 2.0e-5;

constexpr const char* MARKER = "__MARKER__";
constexpr const char* P3_SHA256 = "__P3_SHA__";
constexpr const char* P4_SHA256 = "__P4_SHA__";

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

__ARRAYS__

__global__ void two_layer_fused_forward_kernel(
    const Half* input_a,
    const Half* weight_1,
    const Half* weight_2,
    const float* bias_1,
    const float* bias_2,
    Half* hidden_diagnostic,
    float* output,
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

    FragA frag_input;
    FragB frag_weight_1;
    FragAcc frag_acc_1;

    rocwmma::load_matrix_sync(frag_input, input_a, K);
    rocwmma::load_matrix_sync(frag_weight_1, weight_1, K);
    rocwmma::fill_fragment(frag_acc_1, 0.0f);
    rocwmma::mma_sync(
        frag_acc_1,
        frag_input,
        frag_weight_1,
        frag_acc_1);

    auto const& reg_acc_1 =
        rocwmma::to_register_file(frag_acc_1);

    F32 hidden_epilogue[SLOTS];
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        const uint32_t acc_index = lane * SLOTS + slot;
        const uint32_t col = kAccumulatorColumn[acc_index];
        const F32 biased = reg_acc_1[slot] + bias_1[col];
        hidden_epilogue[slot] = biased > 0.0f ? biased : 0.0f;
    }

    RegA hidden_reg_a;

    // P3-derived accumulator -> matrix-A relay. Every lane executes all
    // candidate-slot shuffles, so shuffle participation remains uniform.
    for (uint32_t target_slot = 0; target_slot < SLOTS; ++target_slot) {
        const uint32_t target_index = lane * SLOTS + target_slot;
        const uint32_t desired_lane =
            kAccLaneForATargetA[target_index];
        const uint32_t desired_slot =
            kAccSlotForATargetA[target_index];

        if (desired_lane >= WAVE || desired_slot >= SLOTS) {
            atomicAdd(&diagnostics[6], 1u);
        }

        F32 selected = 0.0f;
        for (uint32_t candidate_slot = 0;
             candidate_slot < SLOTS;
             ++candidate_slot) {
            const F32 candidate = hidden_epilogue[candidate_slot];
            const F32 shuffled =
                __shfl(candidate, desired_lane, WAVE);
            if (candidate_slot == desired_slot) {
                selected = shuffled;
            }
        }

        hidden_reg_a[target_slot] =
            static_cast<Half>(selected);
    }

    auto const& hidden_frag_a =
        rocwmma::from_register_file<FragA>(hidden_reg_a);

    // Diagnostic store only. Layer 2 consumes hidden_frag_a directly and
    // never reloads hidden_diagnostic.
    rocwmma::store_matrix_sync(
        hidden_diagnostic,
        hidden_frag_a,
        K,
        rocwmma::mem_row_major);

    FragB frag_weight_2;
    FragAcc frag_acc_2;

    rocwmma::load_matrix_sync(frag_weight_2, weight_2, K);
    rocwmma::fill_fragment(frag_acc_2, 0.0f);
    rocwmma::mma_sync(
        frag_acc_2,
        hidden_frag_a,
        frag_weight_2,
        frag_acc_2);

    auto const& reg_acc_2 =
        rocwmma::to_register_file(frag_acc_2);

    RegAcc final_reg_acc;
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        const uint32_t acc_index = lane * SLOTS + slot;
        const uint32_t col = kAccumulatorColumn[acc_index];
        final_reg_acc[slot] = reg_acc_2[slot] + bias_2[col];
    }

    auto const& final_frag_acc =
        rocwmma::from_register_file<FragAcc>(final_reg_acc);

    rocwmma::store_matrix_sync(
        output,
        final_frag_acc,
        N,
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
                    out << "\\u" << std::hex
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <result.json> <hidden.csv> <output.csv>"
                  << std::endl;
        return 64;
    }

    const std::filesystem::path json_path = argv[1];
    const std::filesystem::path hidden_csv_path = argv[2];
    const std::filesystem::path output_csv_path = argv[3];

    std::filesystem::create_directories(json_path.parent_path());
    std::filesystem::create_directories(hidden_csv_path.parent_path());
    std::filesystem::create_directories(output_csv_path.parent_path());

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    int runtime_version = 0;
    int driver_version = 0;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));

    std::vector<Half> input_a(ELEMENTS);
    std::vector<Half> weight_1(ELEMENTS);
    std::vector<Half> weight_2(ELEMENTS);
    std::vector<float> bias_1(N);
    std::vector<float> bias_2(N);

    uint32_t input_quantization_changed = 0;
    uint32_t weight_1_quantization_changed = 0;
    uint32_t weight_2_quantization_changed = 0;

    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < K; ++col) {
            const int code =
                static_cast<int>(
                    (row * 7u + col * 11u + 3u) % 31u
                ) - 15;
            const float source =
                static_cast<float>(code) * 0.0313f;
            const Half quantized = static_cast<Half>(source);
            input_a[row * K + col] = quantized;
            if (static_cast<float>(quantized) != source) {
                ++input_quantization_changed;
            }
        }
    }

    // Weights are physically column-major.
    for (uint32_t row = 0; row < K; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const int code_1 =
                static_cast<int>(
                    (row * 13u + col * 5u + 7u) % 29u
                ) - 14;
            const float source_1 =
                static_cast<float>(code_1) * 0.0271f;
            const Half quantized_1 =
                static_cast<Half>(source_1);
            weight_1[col * K + row] = quantized_1;
            if (static_cast<float>(quantized_1) != source_1) {
                ++weight_1_quantization_changed;
            }

            const int code_2 =
                static_cast<int>(
                    (row * 17u + col * 3u + 11u) % 23u
                ) - 11;
            const float source_2 =
                static_cast<float>(code_2) * 0.0297f;
            const Half quantized_2 =
                static_cast<Half>(source_2);
            weight_2[col * K + row] = quantized_2;
            if (static_cast<float>(quantized_2) != source_2) {
                ++weight_2_quantization_changed;
            }
        }
    }

    for (uint32_t col = 0; col < N; ++col) {
        bias_1[col] =
            (static_cast<int>(col) - 7) * 0.0191f
            + 0.0047f;
        bias_2[col] =
            (static_cast<int>((col * 5u) % 13u) - 6)
            * 0.0113f
            - 0.0021f;
    }

    std::vector<Half> hidden_reference(ELEMENTS);
    std::vector<double> output_reference(ELEMENTS);

    uint32_t hidden_relu_clamped_count = 0;
    uint32_t hidden_positive_count = 0;
    uint32_t hidden_fp16_cast_changed_count = 0;

    // Independent CPU-FP64 reference from the exact FP16-rounded
    // input and weight values.
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            double accumulator = 0.0;
            for (uint32_t inner = 0; inner < K; ++inner) {
                const double input_value =
                    static_cast<double>(
                        static_cast<float>(
                            input_a[row * K + inner]
                        )
                    );
                const double weight_value =
                    static_cast<double>(
                        static_cast<float>(
                            weight_1[col * K + inner]
                        )
                    );
                accumulator += input_value * weight_value;
            }

            const double biased =
                accumulator
                + static_cast<double>(bias_1[col]);
            const double activated =
                biased > 0.0 ? biased : 0.0;
            const Half casted =
                static_cast<Half>(activated);

            if (biased <= 0.0) {
                ++hidden_relu_clamped_count;
            } else {
                ++hidden_positive_count;
            }

            if (static_cast<double>(
                    static_cast<float>(casted)
                ) != activated) {
                ++hidden_fp16_cast_changed_count;
            }

            hidden_reference[row * N + col] = casted;
        }
    }

    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            double accumulator = 0.0;
            for (uint32_t inner = 0; inner < K; ++inner) {
                const double hidden_value =
                    static_cast<double>(
                        static_cast<float>(
                            hidden_reference[row * K + inner]
                        )
                    );
                const double weight_value =
                    static_cast<double>(
                        static_cast<float>(
                            weight_2[col * K + inner]
                        )
                    );
                accumulator += hidden_value * weight_value;
            }

            output_reference[row * N + col] =
                accumulator
                + static_cast<double>(bias_2[col]);
        }
    }

    const Half hidden_guard =
        static_cast<Half>(-123.5f);
    const uint16_t hidden_guard_bits =
        half_bits(hidden_guard);
    const float output_guard = -12345.25f;
    const uint32_t output_guard_bits =
        float_bits(output_guard);

    std::vector<Half> hidden_output(
        ELEMENTS + 2 * GUARD,
        hidden_guard
    );
    std::vector<float> output(
        ELEMENTS + 2 * GUARD,
        output_guard
    );

    Half* d_input_a = nullptr;
    Half* d_weight_1 = nullptr;
    Half* d_weight_2 = nullptr;
    float* d_bias_1 = nullptr;
    float* d_bias_2 = nullptr;
    Half* d_hidden = nullptr;
    float* d_output = nullptr;
    uint32_t* d_diagnostics = nullptr;

    HIP_CHECK(hipMalloc(
        &d_input_a,
        input_a.size() * sizeof(Half)
    ));
    HIP_CHECK(hipMalloc(
        &d_weight_1,
        weight_1.size() * sizeof(Half)
    ));
    HIP_CHECK(hipMalloc(
        &d_weight_2,
        weight_2.size() * sizeof(Half)
    ));
    HIP_CHECK(hipMalloc(
        &d_bias_1,
        bias_1.size() * sizeof(float)
    ));
    HIP_CHECK(hipMalloc(
        &d_bias_2,
        bias_2.size() * sizeof(float)
    ));
    HIP_CHECK(hipMalloc(
        &d_hidden,
        hidden_output.size() * sizeof(Half)
    ));
    HIP_CHECK(hipMalloc(
        &d_output,
        output.size() * sizeof(float)
    ));
    HIP_CHECK(hipMalloc(
        &d_diagnostics,
        8 * sizeof(uint32_t)
    ));

    HIP_CHECK(hipMemcpy(
        d_input_a,
        input_a.data(),
        input_a.size() * sizeof(Half),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemcpy(
        d_weight_1,
        weight_1.data(),
        weight_1.size() * sizeof(Half),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemcpy(
        d_weight_2,
        weight_2.data(),
        weight_2.size() * sizeof(Half),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemcpy(
        d_bias_1,
        bias_1.data(),
        bias_1.size() * sizeof(float),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemcpy(
        d_bias_2,
        bias_2.data(),
        bias_2.size() * sizeof(float),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemcpy(
        d_hidden,
        hidden_output.data(),
        hidden_output.size() * sizeof(Half),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemcpy(
        d_output,
        output.data(),
        output.size() * sizeof(float),
        hipMemcpyHostToDevice
    ));
    HIP_CHECK(hipMemset(
        d_diagnostics,
        0,
        8 * sizeof(uint32_t)
    ));

    hipLaunchKernelGGL(
        two_layer_fused_forward_kernel,
        dim3(1),
        dim3(WAVE),
        0,
        0,
        d_input_a,
        d_weight_1,
        d_weight_2,
        d_bias_1,
        d_bias_2,
        d_hidden + GUARD,
        d_output + GUARD,
        d_diagnostics
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> diagnostics(8, 0u);

    HIP_CHECK(hipMemcpy(
        hidden_output.data(),
        d_hidden,
        hidden_output.size() * sizeof(Half),
        hipMemcpyDeviceToHost
    ));
    HIP_CHECK(hipMemcpy(
        output.data(),
        d_output,
        output.size() * sizeof(float),
        hipMemcpyDeviceToHost
    ));
    HIP_CHECK(hipMemcpy(
        diagnostics.data(),
        d_diagnostics,
        diagnostics.size() * sizeof(uint32_t),
        hipMemcpyDeviceToHost
    ));

    HIP_CHECK(hipFree(d_input_a));
    HIP_CHECK(hipFree(d_weight_1));
    HIP_CHECK(hipFree(d_weight_2));
    HIP_CHECK(hipFree(d_bias_1));
    HIP_CHECK(hipFree(d_bias_2));
    HIP_CHECK(hipFree(d_hidden));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_diagnostics));

    bool hidden_guards_ok = true;
    bool output_guards_ok = true;

    for (uint32_t index = 0; index < GUARD; ++index) {
        hidden_guards_ok &=
            half_bits(hidden_output[index])
                == hidden_guard_bits
            && half_bits(
                hidden_output[GUARD + ELEMENTS + index]
            ) == hidden_guard_bits;

        output_guards_ok &=
            float_bits(output[index])
                == output_guard_bits
            && float_bits(
                output[GUARD + ELEMENTS + index]
            ) == output_guard_bits;
    }

    uint32_t hidden_mismatch_count = 0;
    uint32_t hidden_first_mismatch = 0;

    for (uint32_t index = 0; index < ELEMENTS; ++index) {
        const Half gpu =
            hidden_output[GUARD + index];
        const Half cpu =
            hidden_reference[index];

        if (half_bits(gpu) != half_bits(cpu)) {
            if (hidden_mismatch_count == 0) {
                hidden_first_mismatch = index;
            }
            ++hidden_mismatch_count;
        }
    }

    uint32_t output_nonfinite_count = 0;
    uint32_t output_max_error_index = 0;
    double output_max_abs = 0.0;
    double output_sum_diff_sq = 0.0;
    double output_sum_ref_sq = 0.0;

    for (uint32_t index = 0; index < ELEMENTS; ++index) {
        const double gpu =
            static_cast<double>(output[GUARD + index]);
        const double cpu =
            output_reference[index];

        if (!std::isfinite(gpu)) {
            ++output_nonfinite_count;
            continue;
        }

        const double diff = std::abs(gpu - cpu);
        if (diff > output_max_abs) {
            output_max_abs = diff;
            output_max_error_index = index;
        }

        output_sum_diff_sq += diff * diff;
        output_sum_ref_sq += cpu * cpu;
    }

    const double output_normalized_l2 =
        std::sqrt(
            output_sum_diff_sq
            / std::max(
                output_sum_ref_sq,
                std::numeric_limits<double>::min()
            )
        );

    const bool context_ok =
        std::string(props.gcnArchName).rfind(
            "gfx1201",
            0
        ) == 0
        && props.warpSize == static_cast<int>(WAVE)
        && diagnostics[0] == WAVE
        && diagnostics[1] == WAVE
        && diagnostics[2] == 1
        && diagnostics[3] == SLOTS
        && diagnostics[4] == SLOTS
        && diagnostics[5] == 0;

    const bool mapping_ok =
        diagnostics[6] == 0;
    const bool guards_ok =
        hidden_guards_ok && output_guards_ok;
    const bool quantization_exercised =
        input_quantization_changed > 0
        && weight_1_quantization_changed > 0
        && weight_2_quantization_changed > 0;
    const bool hidden_epilogue_exercised =
        hidden_relu_clamped_count > 0
        && hidden_positive_count > 0
        && hidden_fp16_cast_changed_count > 0;
    const bool hidden_ok =
        hidden_mismatch_count == 0;
    const bool output_ok =
        output_nonfinite_count == 0
        && output_max_abs
            <= OUTPUT_MAX_ABS_TOLERANCE
        && output_normalized_l2
            <= OUTPUT_NORMALIZED_L2_TOLERANCE;

    const bool passed =
        context_ok
        && mapping_ok
        && guards_ok
        && quantization_exercised
        && hidden_epilogue_exercised
        && hidden_ok
        && output_ok;

    std::ofstream hidden_csv(hidden_csv_path);
    if (!hidden_csv) {
        std::cerr << "OUTPUT_ERROR: cannot write "
                  << hidden_csv_path
                  << std::endl;
        return 3;
    }

    hidden_csv
        << "row,col,gpu,gpu_bits,cpu,cpu_bits\n";
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const Half gpu =
                hidden_output[GUARD + row * N + col];
            const Half cpu =
                hidden_reference[row * N + col];

            hidden_csv
                << row << ','
                << col << ','
                << static_cast<float>(gpu) << ','
                << half_bits(gpu) << ','
                << static_cast<float>(cpu) << ','
                << half_bits(cpu) << '\n';
        }
    }

    std::ofstream output_csv(output_csv_path);
    if (!output_csv) {
        std::cerr << "OUTPUT_ERROR: cannot write "
                  << output_csv_path
                  << std::endl;
        return 3;
    }

    output_csv
        << "row,col,gpu,cpu_fp64,abs_diff\n";
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            const uint32_t index = row * N + col;
            const double gpu =
                static_cast<double>(
                    output[GUARD + index]
                );
            const double cpu =
                output_reference[index];

            output_csv
                << row << ','
                << col << ','
                << std::setprecision(17) << gpu << ','
                << cpu << ','
                << std::abs(gpu - cpu) << '\n';
        }
    }

    std::ofstream json(json_path);
    if (!json) {
        std::cerr << "OUTPUT_ERROR: cannot write "
                  << json_path
                  << std::endl;
        return 3;
    }

    json << std::setprecision(17);
    json << "{\n";
    json << "  \"marker\": \"" << MARKER << "\",\n";
    json << "  \"decision\": \""
         << (
                passed
                    ? "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PASS"
                    : "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_FAIL"
            )
         << "\",\n";
    json << "  \"p3_sha256\": \"" << P3_SHA256 << "\",\n";
    json << "  \"p4_sha256\": \"" << P4_SHA256 << "\",\n";
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
    json << "    \"accumulator_slots_per_lane\": "
         << diagnostics[3]
         << ",\n";
    json << "    \"matrix_a_slots_per_lane\": "
         << diagnostics[4]
         << "\n";
    json << "  },\n";
    json << "  \"topology\": {\n";
    json << "    \"batch_rows\": 16,\n";
    json << "    \"input_width\": 16,\n";
    json << "    \"hidden_width\": 16,\n";
    json << "    \"output_width\": 16,\n";
    json << "    \"hidden_activation\": \"ReLU\",\n";
    json << "    \"input_type\": \"FP16\",\n";
    json << "    \"weight_type\": \"FP16\",\n";
    json << "    \"accumulator_type\": \"FP32\",\n";
    json << "    \"hidden_type\": \"FP16\",\n";
    json << "    \"output_type\": \"FP32\",\n";
    json << "    \"kernel_launches\": 1,\n";
    json << "    \"intermediate_global_reload\": false,\n";
    json << "    \"hidden_global_store\": "
         << "\"diagnostic_only\"\n";
    json << "  },\n";
    json << "  \"metrics\": {\n";
    json << "    \"input_quantization_changed\": "
         << input_quantization_changed
         << ",\n";
    json << "    \"weight_1_quantization_changed\": "
         << weight_1_quantization_changed
         << ",\n";
    json << "    \"weight_2_quantization_changed\": "
         << weight_2_quantization_changed
         << ",\n";
    json << "    \"hidden_relu_clamped_count\": "
         << hidden_relu_clamped_count
         << ",\n";
    json << "    \"hidden_positive_count\": "
         << hidden_positive_count
         << ",\n";
    json << "    \"hidden_fp16_cast_changed_count\": "
         << hidden_fp16_cast_changed_count
         << ",\n";
    json << "    \"hidden_mismatch_count\": "
         << hidden_mismatch_count
         << ",\n";
    json << "    \"hidden_first_mismatch_row\": "
         << (hidden_first_mismatch / N)
         << ",\n";
    json << "    \"hidden_first_mismatch_col\": "
         << (hidden_first_mismatch % N)
         << ",\n";
    json << "    \"output_nonfinite_count\": "
         << output_nonfinite_count
         << ",\n";
    json << "    \"output_max_abs\": "
         << output_max_abs
         << ",\n";
    json << "    \"output_normalized_l2\": "
         << output_normalized_l2
         << ",\n";
    json << "    \"output_max_error_row\": "
         << (output_max_error_index / N)
         << ",\n";
    json << "    \"output_max_error_col\": "
         << (output_max_error_index % N)
         << "\n";
    json << "  },\n";
    json << "  \"tolerances\": {\n";
    json << "    \"output_max_abs\": "
         << OUTPUT_MAX_ABS_TOLERANCE
         << ",\n";
    json << "    \"output_normalized_l2\": "
         << OUTPUT_NORMALIZED_L2_TOLERANCE
         << "\n";
    json << "  },\n";
    json << "  \"gates\": {\n";
    json << "    \"wave32_context\": "
         << (context_ok ? "true" : "false")
         << ",\n";
    json << "    \"p3_mapping_embedded\": "
         << (mapping_ok ? "true" : "false")
         << ",\n";
    json << "    \"guard_regions\": "
         << (guards_ok ? "true" : "false")
         << ",\n";
    json << "    \"quantization_exercised\": "
         << (
                quantization_exercised
                    ? "true"
                    : "false"
            )
         << ",\n";
    json << "    \"hidden_epilogue_exercised\": "
         << (
                hidden_epilogue_exercised
                    ? "true"
                    : "false"
            )
         << ",\n";
    json << "    \"hidden_bitwise_equal\": "
         << (hidden_ok ? "true" : "false")
         << ",\n";
    json << "    \"output_vs_cpu_fp64\": "
         << (output_ok ? "true" : "false")
         << ",\n";
    json << "    \"single_kernel_no_intermediate_reload\": true\n";
    json << "  }\n";
    json << "}\n";

    std::cout << "marker: " << MARKER << '\n';
    std::cout << "p3_sha256: " << P3_SHA256 << '\n';
    std::cout << "p4_sha256: " << P4_SHA256 << '\n';
    std::cout << "input_quantization_changed: "
              << input_quantization_changed
              << '\n';
    std::cout << "weight_1_quantization_changed: "
              << weight_1_quantization_changed
              << '\n';
    std::cout << "weight_2_quantization_changed: "
              << weight_2_quantization_changed
              << '\n';
    std::cout << "hidden_relu_clamped_count: "
              << hidden_relu_clamped_count
              << '\n';
    std::cout << "hidden_positive_count: "
              << hidden_positive_count
              << '\n';
    std::cout << "hidden_fp16_cast_changed_count: "
              << hidden_fp16_cast_changed_count
              << '\n';
    std::cout << "hidden_mismatch_count: "
              << hidden_mismatch_count
              << '\n';
    std::cout << "output_nonfinite_count: "
              << output_nonfinite_count
              << '\n';
    std::cout << "output_max_abs: "
              << output_max_abs
              << '\n';
    std::cout << "output_normalized_l2: "
              << output_normalized_l2
              << '\n';

    std::cout << "ROCWMMA_P5_WAVE32_CONTEXT: "
              << (context_ok ? "PASS" : "FAIL")
              << '\n';
    std::cout << "ROCWMMA_P5_P3_MAPPING_EMBEDDED: "
              << (mapping_ok ? "PASS" : "FAIL")
              << '\n';
    std::cout << "ROCWMMA_P5_GUARD_REGIONS: "
              << (guards_ok ? "PASS" : "FAIL")
              << '\n';
    std::cout << "ROCWMMA_P5_QUANTIZATION_EXERCISED: "
              << (
                    quantization_exercised
                        ? "PASS"
                        : "FAIL"
                 )
              << '\n';
    std::cout << "ROCWMMA_P5_HIDDEN_EPILOGUE: "
              << (
                    hidden_epilogue_exercised
                        ? "PASS"
                        : "FAIL"
                 )
              << '\n';
    std::cout << "ROCWMMA_P5_HIDDEN_BITWISE_CORRECTNESS: "
              << (hidden_ok ? "PASS" : "FAIL")
              << '\n';
    std::cout << "ROCWMMA_P5_NO_INTERMEDIATE_GLOBAL_RELOAD: PASS\n";
    std::cout << "ROCWMMA_P5_OUTPUT_VS_CPU_FP64: "
              << (output_ok ? "PASS" : "FAIL")
              << '\n';
    std::cout << "RDNA4_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: "
              << (passed ? "PASS" : "FAIL")
              << '\n';
    std::cout << "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PROCESS: "
              << (passed ? "PASS" : "FAIL")
              << '\n';

    return passed ? 0 : 1;
}
"""

    return (
        template
        .replace("__MARKER__", MARKER)
        .replace("__P3_SHA__", p3_sha)
        .replace("__P4_SHA__", p4_sha)
        .replace("__ARRAYS__", arrays)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-json", type=Path, required=True)
    parser.add_argument("--p4-json", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    (
        p3,
        p4,
        source_lanes,
        source_slots,
        acc_cols,
    ) = load_mapping(args.p3_json, args.p4_json)

    source = generate_source(
        args.p3_json,
        args.p4_json,
        source_lanes,
        source_slots,
        acc_cols,
    )

    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.write_text(source)

    moved = sum(
        1
        for index, pair_value in enumerate(
            zip(source_lanes, source_slots)
        )
        if (index // 8, index % 8) != pair_value
    )

    manifest = {
        "marker": MARKER,
        "decision": "PHASE4A0_P5_SOURCE_GENERATION_PASS",
        "p3_json": str(args.p3_json.resolve()),
        "p3_sha256": sha256(args.p3_json),
        "p4_json": str(args.p4_json.resolve()),
        "p4_sha256": sha256(args.p4_json),
        "generated_source": str(args.output_source.resolve()),
        "generated_source_sha256": sha256(args.output_source),
        "mapping": {
            "target_role": "matrix_a",
            "source_role": "accumulator",
            "entries": 256,
            "moved_entries": moved,
            "fixed_entries": 256 - moved,
            "source_lane_sha256": hashlib.sha256(
                bytes(source_lanes)
            ).hexdigest(),
            "source_slot_sha256": hashlib.sha256(
                bytes(source_slots)
            ).hexdigest(),
            "accumulator_column_sha256": hashlib.sha256(
                bytes(acc_cols)
            ).hexdigest(),
        },
        "topology": {
            "batch_rows": 16,
            "input_width": 16,
            "hidden_width": 16,
            "output_width": 16,
            "hidden_activation": "ReLU",
            "kernel_launches": 1,
            "intermediate_global_reload": False,
        },
        "context": p3["context"],
        "p4_context": p4["context"],
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    assert moved == 240
    print(f"p3_sha256: {manifest['p3_sha256']}")
    print(f"p4_sha256: {manifest['p4_sha256']}")
    print(
        "generated_source_sha256: "
        f"{manifest['generated_source_sha256']}"
    )
    print(f"relay_moved_entries: {moved}")
    print("PHASE4A0_P5_SOURCE_GENERATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
