#!/usr/bin/env python3
"""Phase 4A1-P0: Width-64 fused-MLP tile plan and CPU-FP64 oracle.

Marker:
    TCNN_RDNA4_P4A1_P0_WIDTH64_TILE_PLAN_CPU_ORACLE_001

No GPU kernel is compiled or launched in this phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable

MARKER = "TCNN_RDNA4_P4A1_P0_WIDTH64_TILE_PLAN_CPU_ORACLE_001"
P5_DECISION = "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PASS"
P0_DECISION = "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"

BATCH = 16
WIDTH = 64
TILE = 16
WAVES = 4
WAVE_SIZE = 32
THREADS = WAVES * WAVE_SIZE
N_TILES = WIDTH // TILE
K_TILES = WIDTH // TILE
LAYERS = 3
LDS_BYTES = BATCH * WIDTH * 2

PROBE_COORDINATES = (
    (0, 0),
    (0, 15),
    (0, 16),
    (0, 63),
    (1, 7),
    (2, 31),
    (3, 32),
    (4, 47),
    (5, 48),
    (7, 9),
    (8, 55),
    (10, 21),
    (12, 42),
    (14, 62),
    (15, 0),
    (15, 63),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def quantize_fp16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def quantize_fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fp64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def pack_fp16(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<e", value) for value in values)


def pack_fp32(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def pack_fp64(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<d", value) for value in values)


def matrix_index(row: int, col: int) -> int:
    return row * WIDTH + col


def logical_to_col_major(matrix: list[float]) -> list[float]:
    return [
        matrix[matrix_index(row, col)]
        for col in range(WIDTH)
        for row in range(WIDTH)
    ]


def generate_tensors() -> dict[str, Any]:
    input_values: list[float] = []
    weights = {
        "weight_1": [],
        "weight_2": [],
        "weight_3": [],
    }
    biases = {
        "bias_1": [],
        "bias_2": [],
        "bias_3": [],
    }

    quantization_counts = {
        "input": 0,
        "weight_1": 0,
        "weight_2": 0,
        "weight_3": 0,
    }

    for row in range(BATCH):
        for col in range(WIDTH):
            code = ((row * 17 + col * 11 + 5) % 53) - 26
            source = (
                code * 0.0217
                + (((row + col) % 3) - 1) * 0.00031
            )
            value = quantize_fp16(source)
            input_values.append(value)
            quantization_counts["input"] += int(value != source)

    for row in range(WIDTH):
        for col in range(WIDTH):
            code_1 = ((row * 13 + col * 7 + 3) % 47) - 23
            source_1 = (
                code_1 * 0.0143
                + (((row + 2 * col) % 5) - 2) * 0.00017
            )
            value_1 = quantize_fp16(source_1)
            weights["weight_1"].append(value_1)
            quantization_counts["weight_1"] += int(value_1 != source_1)

            code_2 = ((row * 19 + col * 5 + 9) % 43) - 21
            source_2 = (
                code_2 * 0.0131
                + (((3 * row + col) % 7) - 3) * 0.00013
            )
            value_2 = quantize_fp16(source_2)
            weights["weight_2"].append(value_2)
            quantization_counts["weight_2"] += int(value_2 != source_2)

            code_3 = ((row * 23 + col * 3 + 1) % 41) - 20
            source_3 = (
                code_3 * 0.0127
                + (((row + 4 * col) % 6) - 2) * 0.00011
            )
            value_3 = quantize_fp16(source_3)
            weights["weight_3"].append(value_3)
            quantization_counts["weight_3"] += int(value_3 != source_3)

    for col in range(WIDTH):
        biases["bias_1"].append(
            quantize_fp32(
                (col - 31) * 0.0029
                + (((col * 7) % 9) - 4) * 0.0007
            )
        )
        biases["bias_2"].append(
            quantize_fp32(
                (((col * 5) % 17) - 8) * 0.0041
                + (col - 31) * 0.00037
            )
        )
        biases["bias_3"].append(
            quantize_fp32(
                (((col * 3) % 19) - 9) * 0.0033
                - 0.0011
            )
        )

    assert all(count > 0 for count in quantization_counts.values())

    return {
        "input": input_values,
        **weights,
        **biases,
        "quantization_counts": quantization_counts,
    }


def scalar_layer(
    input_matrix: list[float],
    weight_matrix: list[float],
    bias: list[float],
    relu_and_fp16: bool,
) -> tuple[list[float], dict[str, int]]:
    output: list[float] = []
    stats = {
        "relu_clamped": 0,
        "relu_positive": 0,
        "fp16_cast_changed": 0,
    }

    for row in range(BATCH):
        for col in range(WIDTH):
            accumulator = 0.0
            for inner in range(WIDTH):
                accumulator += (
                    input_matrix[matrix_index(row, inner)]
                    * weight_matrix[matrix_index(inner, col)]
                )

            value = accumulator + bias[col]
            if relu_and_fp16:
                if value > 0.0:
                    stats["relu_positive"] += 1
                else:
                    stats["relu_clamped"] += 1
                    value = 0.0

                quantized = quantize_fp16(value)
                stats["fp16_cast_changed"] += int(quantized != value)
                output.append(quantized)
            else:
                output.append(value)

    return output, stats


def tiled_layer(
    input_matrix: list[float],
    weight_matrix: list[float],
    bias: list[float],
    relu_and_fp16: bool,
) -> tuple[list[float], dict[str, int]]:
    output = [math.nan] * (BATCH * WIDTH)
    stats = {
        "relu_clamped": 0,
        "relu_positive": 0,
        "fp16_cast_changed": 0,
    }

    for n_tile in range(N_TILES):
        col_begin = n_tile * TILE
        tile_accumulators = [
            [0.0 for _ in range(TILE)]
            for _ in range(BATCH)
        ]

        for k_tile in range(K_TILES):
            inner_begin = k_tile * TILE

            for row in range(BATCH):
                for local_col in range(TILE):
                    col = col_begin + local_col
                    accumulator = tile_accumulators[row][local_col]

                    for local_inner in range(TILE):
                        inner = inner_begin + local_inner
                        accumulator += (
                            input_matrix[matrix_index(row, inner)]
                            * weight_matrix[matrix_index(inner, col)]
                        )

                    tile_accumulators[row][local_col] = accumulator

        for row in range(BATCH):
            for local_col in range(TILE):
                col = col_begin + local_col
                value = tile_accumulators[row][local_col] + bias[col]

                if relu_and_fp16:
                    if value > 0.0:
                        stats["relu_positive"] += 1
                    else:
                        stats["relu_clamped"] += 1
                        value = 0.0

                    quantized = quantize_fp16(value)
                    stats["fp16_cast_changed"] += int(quantized != value)
                    output[matrix_index(row, col)] = quantized
                else:
                    output[matrix_index(row, col)] = value

    assert all(math.isfinite(value) for value in output)
    return output, stats


def compare_fp16(left: list[float], right: list[float]) -> dict[str, Any]:
    mismatches = [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if fp16_bits(a) != fp16_bits(b)
    ]
    return {
        "count": len(mismatches),
        "first_index": mismatches[0] if mismatches else 0,
        "bitwise_equal": not mismatches,
    }


def compare_fp64(left: list[float], right: list[float]) -> dict[str, Any]:
    mismatch_indices = [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if fp64_bits(a) != fp64_bits(b)
    ]
    max_abs = max(
        (abs(a - b) for a, b in zip(left, right)),
        default=0.0,
    )
    sum_diff_sq = sum(
        (a - b) ** 2 for a, b in zip(left, right)
    )
    sum_ref_sq = sum(value * value for value in left)
    normalized_l2 = math.sqrt(
        sum_diff_sq / max(sum_ref_sq, float.fromhex("0x0.0000000000001p-1022"))
    )
    return {
        "count": len(mismatch_indices),
        "first_index": mismatch_indices[0] if mismatch_indices else 0,
        "bitwise_equal": not mismatch_indices,
        "max_abs": max_abs,
        "normalized_l2": normalized_l2,
    }


def build_tile_plan() -> dict[str, Any]:
    operations: list[dict[str, Any]] = []

    layer_definitions = (
        {
            "layer": 1,
            "name": "input_to_hidden_1",
            "a_source": "global_input_row_major",
            "weight": "weight_1_col_major",
            "epilogue": "bias_1_fp32_relu_fp32_cast_fp16",
            "destination": "lds_hidden_buffer_fp16",
        },
        {
            "layer": 2,
            "name": "hidden_1_to_hidden_2",
            "a_source": "lds_hidden_buffer_fp16",
            "weight": "weight_2_col_major",
            "epilogue": "bias_2_fp32_relu_fp32_cast_fp16",
            "destination": "same_lds_hidden_buffer_fp16_after_read_barrier",
        },
        {
            "layer": 3,
            "name": "hidden_2_to_output",
            "a_source": "lds_hidden_buffer_fp16",
            "weight": "weight_3_col_major",
            "epilogue": "bias_3_fp32",
            "destination": "global_output_fp32",
        },
    )

    for layer in layer_definitions:
        for wave in range(WAVES):
            n_tile = wave
            for k_tile in range(K_TILES):
                operations.append({
                    "layer": layer["layer"],
                    "layer_name": layer["name"],
                    "wave": wave,
                    "output_n_tile": n_tile,
                    "k_tile": k_tile,
                    "a_source": layer["a_source"],
                    "a_tile": {
                        "row_begin": 0,
                        "col_begin": k_tile * TILE,
                        "shape": [TILE, TILE],
                        "leading_dimension": WIDTH,
                        "pointer_offset_elements": k_tile * TILE,
                    },
                    "b_source": layer["weight"],
                    "b_tile": {
                        "row_begin": k_tile * TILE,
                        "col_begin": n_tile * TILE,
                        "shape": [TILE, TILE],
                        "leading_dimension": WIDTH,
                        "column_major_pointer_offset_elements": (
                            k_tile * TILE
                            + n_tile * TILE * WIDTH
                        ),
                    },
                    "output_tile": {
                        "row_begin": 0,
                        "col_begin": n_tile * TILE,
                        "shape": [TILE, TILE],
                        "leading_dimension": WIDTH,
                        "row_major_pointer_offset_elements": n_tile * TILE,
                    },
                    "accumulator": (
                        "one persistent FP32 16x16 fragment per wave "
                        "across k_tile=0..3"
                    ),
                    "mma_sequence_index_within_wave": (
                        (layer["layer"] - 1) * K_TILES + k_tile
                    ),
                })

    barriers = [
        {
            "index": 0,
            "position": "after_hidden_1_lds_store",
            "purpose": (
                "publish all four hidden-1 column tiles before layer 2 reads"
            ),
        },
        {
            "index": 1,
            "position": "after_layer_2_all_k_tiles_accumulated",
            "purpose": (
                "ensure no wave still reads hidden 1 before the single LDS "
                "buffer is overwritten"
            ),
        },
        {
            "index": 2,
            "position": "after_hidden_2_lds_store",
            "purpose": (
                "publish all four hidden-2 column tiles before layer 3 reads"
            ),
        },
    ]

    return {
        "marker": MARKER,
        "decision": "WIDTH64_FUSED_MLP_TILE_PLAN_LOCKED",
        "topology": {
            "batch_rows": BATCH,
            "input_width": WIDTH,
            "hidden_1_width": WIDTH,
            "hidden_2_width": WIDTH,
            "output_width": WIDTH,
            "layers": LAYERS,
            "hidden_activation": "ReLU",
            "input_type": "FP16",
            "weight_type": "FP16",
            "bias_type": "FP32",
            "accumulator_type": "FP32",
            "hidden_type": "FP16",
            "output_type": "FP32",
        },
        "execution": {
            "kernel_launches_planned": 1,
            "thread_blocks": 1,
            "waves_per_block": WAVES,
            "wave_size": WAVE_SIZE,
            "threads_per_block": THREADS,
            "wave_output_ownership": {
                "wave_0": "columns 0..15",
                "wave_1": "columns 16..31",
                "wave_2": "columns 32..47",
                "wave_3": "columns 48..63",
            },
            "output_tiles_per_layer": N_TILES,
            "k_tiles_per_output_tile": K_TILES,
            "mma_sync_calls_per_wave": LAYERS * K_TILES,
            "mma_sync_calls_per_block": (
                WAVES * LAYERS * K_TILES
            ),
            "intermediate_global_store": False,
            "intermediate_global_reload": False,
        },
        "lds": {
            "buffers": 1,
            "layout": "row_major_16x64_fp16",
            "elements": BATCH * WIDTH,
            "bytes": LDS_BYTES,
            "alignment_bytes": 16,
            "reuse_policy": (
                "hidden1 is overwritten by hidden2 only after the explicit "
                "read-complete barrier"
            ),
        },
        "barriers": barriers,
        "operations": operations,
        "pointer_formulas": {
            "a_row_major_tile": (
                "base + k_tile*16, ldm=64"
            ),
            "b_col_major_tile": (
                "base + k_tile*16 + n_tile*16*64, ldm=64"
            ),
            "output_row_major_tile": (
                "base + n_tile*16, ldm=64"
            ),
            "bias_tile": (
                "bias + n_tile*16"
            ),
        },
        "scope_boundary": {
            "proven_here": [
                "complete tile coverage",
                "wave ownership",
                "K-tile accumulation order",
                "single-LDS-buffer synchronization plan",
                "independent scalar and tiled CPU oracle equality",
            ],
            "not_proven_here": [
                "GPU compilation",
                "rocWMMA multi-wave correctness",
                "register occupancy",
                "performance",
                "tiny-cuda-nn integration",
            ],
        },
    }


def validate_tile_plan(plan: dict[str, Any]) -> dict[str, bool]:
    operations = plan["operations"]

    unique_keys = {
        (
            operation["layer"],
            operation["wave"],
            operation["output_n_tile"],
            operation["k_tile"],
        )
        for operation in operations
    }

    expected_keys = {
        (layer, wave, wave, k_tile)
        for layer in range(1, LAYERS + 1)
        for wave in range(WAVES)
        for k_tile in range(K_TILES)
    }

    per_layer_counts = {
        layer: sum(
            operation["layer"] == layer
            for operation in operations
        )
        for layer in range(1, LAYERS + 1)
    }

    per_wave_counts = {
        wave: sum(
            operation["wave"] == wave
            for operation in operations
        )
        for wave in range(WAVES)
    }

    pointer_offsets_valid = all(
        operation["a_tile"]["pointer_offset_elements"]
            == operation["k_tile"] * TILE
        and operation["b_tile"]["column_major_pointer_offset_elements"]
            == (
                operation["k_tile"] * TILE
                + operation["output_n_tile"] * TILE * WIDTH
            )
        and operation["output_tile"]["row_major_pointer_offset_elements"]
            == operation["output_n_tile"] * TILE
        for operation in operations
    )

    return {
        "operation_count_48": len(operations) == 48,
        "operation_keys_unique": len(unique_keys) == 48,
        "operation_coverage_exact": unique_keys == expected_keys,
        "sixteen_operations_per_layer": all(
            count == 16 for count in per_layer_counts.values()
        ),
        "twelve_operations_per_wave": all(
            count == 12 for count in per_wave_counts.values()
        ),
        "pointer_offsets_valid": pointer_offsets_valid,
        "single_lds_buffer_2048_bytes": (
            plan["lds"]["buffers"] == 1
            and plan["lds"]["bytes"] == 2048
        ),
        "three_barriers": len(plan["barriers"]) == 3,
        "one_block_four_waves": (
            plan["execution"]["thread_blocks"] == 1
            and plan["execution"]["waves_per_block"] == 4
            and plan["execution"]["threads_per_block"] == 128
        ),
        "no_intermediate_global_traffic": (
            not plan["execution"]["intermediate_global_store"]
            and not plan["execution"]["intermediate_global_reload"]
        ),
    }


def tensor_hashes(
    tensors: dict[str, Any],
    hidden_1: list[float],
    hidden_2: list[float],
    output: list[float],
) -> dict[str, Any]:
    result = {
        "input_fp16_row_major": {
            "bytes": len(pack_fp16(tensors["input"])),
            "sha256": sha256_bytes(pack_fp16(tensors["input"])),
        },
        "hidden_1_fp16_row_major": {
            "bytes": len(pack_fp16(hidden_1)),
            "sha256": sha256_bytes(pack_fp16(hidden_1)),
        },
        "hidden_2_fp16_row_major": {
            "bytes": len(pack_fp16(hidden_2)),
            "sha256": sha256_bytes(pack_fp16(hidden_2)),
        },
        "output_fp64_row_major": {
            "bytes": len(pack_fp64(output)),
            "sha256": sha256_bytes(pack_fp64(output)),
        },
    }

    for index in range(1, 4):
        weight = tensors[f"weight_{index}"]
        logical = pack_fp16(weight)
        physical = pack_fp16(logical_to_col_major(weight))
        bias = pack_fp32(tensors[f"bias_{index}"])

        result[f"weight_{index}_fp16_logical_row_major"] = {
            "bytes": len(logical),
            "sha256": sha256_bytes(logical),
        }
        result[f"weight_{index}_fp16_physical_col_major"] = {
            "bytes": len(physical),
            "sha256": sha256_bytes(physical),
        }
        result[f"bias_{index}_fp32"] = {
            "bytes": len(bias),
            "sha256": sha256_bytes(bias),
        }

    return result


def write_binary_files(
    output_dir: Path,
    tensors: dict[str, Any],
    hidden_1: list[float],
    hidden_2: list[float],
    output: list[float],
) -> None:
    (output_dir / "input_fp16_row_major.bin").write_bytes(
        pack_fp16(tensors["input"])
    )
    (output_dir / "hidden_1_fp16_row_major.bin").write_bytes(
        pack_fp16(hidden_1)
    )
    (output_dir / "hidden_2_fp16_row_major.bin").write_bytes(
        pack_fp16(hidden_2)
    )
    (output_dir / "output_fp64_row_major.bin").write_bytes(
        pack_fp64(output)
    )

    for index in range(1, 4):
        weight = tensors[f"weight_{index}"]
        (output_dir / f"weight_{index}_fp16_col_major.bin").write_bytes(
            pack_fp16(logical_to_col_major(weight))
        )
        (output_dir / f"bias_{index}_fp32.bin").write_bytes(
            pack_fp32(tensors[f"bias_{index}"])
        )


def make_probes(
    hidden_1: list[float],
    hidden_2: list[float],
    output: list[float],
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []

    for row, col in PROBE_COORDINATES:
        index = matrix_index(row, col)
        probes.append({
            "row": row,
            "col": col,
            "hidden_1": hidden_1[index],
            "hidden_1_bits": f"0x{fp16_bits(hidden_1[index]):04X}",
            "hidden_2": hidden_2[index],
            "hidden_2_bits": f"0x{fp16_bits(hidden_2[index]):04X}",
            "output_fp64": output[index],
            "output_fp64_bits": f"0x{fp64_bits(output[index]):016X}",
        })

    return probes


def write_probe_csv(path: Path, probes: list[dict[str, Any]]) -> None:
    fields = (
        "row",
        "col",
        "hidden_1",
        "hidden_1_bits",
        "hidden_2",
        "hidden_2_bits",
        "output_fp64",
        "output_fp64_bits",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(probes)


def run_oracle(p5_json_path: Path, output_dir: Path) -> dict[str, Any]:
    p5 = json.loads(p5_json_path.read_text())
    assert p5["decision"] == P5_DECISION
    assert all(bool(value) for value in p5["gates"].values())
    assert p5["topology"]["kernel_launches"] == 1
    assert not p5["topology"]["intermediate_global_reload"]
    assert p5["context"]["arch"].startswith("gfx1201")
    assert int(p5["context"]["warp_size"]) == 32

    tensors = generate_tensors()

    scalar_hidden_1, scalar_stats_1 = scalar_layer(
        tensors["input"],
        tensors["weight_1"],
        tensors["bias_1"],
        relu_and_fp16=True,
    )
    scalar_hidden_2, scalar_stats_2 = scalar_layer(
        scalar_hidden_1,
        tensors["weight_2"],
        tensors["bias_2"],
        relu_and_fp16=True,
    )
    scalar_output, scalar_stats_3 = scalar_layer(
        scalar_hidden_2,
        tensors["weight_3"],
        tensors["bias_3"],
        relu_and_fp16=False,
    )

    tiled_hidden_1, tiled_stats_1 = tiled_layer(
        tensors["input"],
        tensors["weight_1"],
        tensors["bias_1"],
        relu_and_fp16=True,
    )
    tiled_hidden_2, tiled_stats_2 = tiled_layer(
        tiled_hidden_1,
        tensors["weight_2"],
        tensors["bias_2"],
        relu_and_fp16=True,
    )
    tiled_output, tiled_stats_3 = tiled_layer(
        tiled_hidden_2,
        tensors["weight_3"],
        tensors["bias_3"],
        relu_and_fp16=False,
    )

    hidden_1_compare = compare_fp16(
        scalar_hidden_1,
        tiled_hidden_1,
    )
    hidden_2_compare = compare_fp16(
        scalar_hidden_2,
        tiled_hidden_2,
    )
    output_compare = compare_fp64(
        scalar_output,
        tiled_output,
    )

    assert scalar_stats_1 == tiled_stats_1
    assert scalar_stats_2 == tiled_stats_2
    assert scalar_stats_3 == tiled_stats_3

    plan = build_tile_plan()
    plan_gates = validate_tile_plan(plan)

    oracle_gates = {
        "phase4a0_p5_prerequisite": True,
        "input_and_weight_quantization_exercised": all(
            count > 0
            for count in tensors["quantization_counts"].values()
        ),
        "hidden_1_relu_branches_exercised": (
            scalar_stats_1["relu_clamped"] > 0
            and scalar_stats_1["relu_positive"] > 0
        ),
        "hidden_2_relu_branches_exercised": (
            scalar_stats_2["relu_clamped"] > 0
            and scalar_stats_2["relu_positive"] > 0
        ),
        "hidden_1_fp16_rounding_exercised": (
            scalar_stats_1["fp16_cast_changed"] > 0
        ),
        "hidden_2_fp16_rounding_exercised": (
            scalar_stats_2["fp16_cast_changed"] > 0
        ),
        "hidden_1_scalar_vs_tiled_bitwise": (
            hidden_1_compare["bitwise_equal"]
        ),
        "hidden_2_scalar_vs_tiled_bitwise": (
            hidden_2_compare["bitwise_equal"]
        ),
        "output_scalar_vs_tiled_fp64_bitwise": (
            output_compare["bitwise_equal"]
        ),
        "output_finite": all(
            math.isfinite(value) for value in scalar_output
        ),
    }

    all_gates = {
        **{f"tile_plan.{name}": value for name, value in plan_gates.items()},
        **{f"cpu_oracle.{name}": value for name, value in oracle_gates.items()},
    }
    passed = all(all_gates.values())

    hashes = tensor_hashes(
        tensors,
        scalar_hidden_1,
        scalar_hidden_2,
        scalar_output,
    )
    probes = make_probes(
        scalar_hidden_1,
        scalar_hidden_2,
        scalar_output,
    )

    output_l2 = math.sqrt(
        sum(value * value for value in scalar_output)
    )

    report = {
        "marker": MARKER,
        "decision": (
            P0_DECISION
            if passed
            else "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_FAIL"
        ),
        "phase4a0_baseline": {
            "p5_json": str(p5_json_path.resolve()),
            "p5_json_sha256": sha256(p5_json_path),
            "p5_decision": p5["decision"],
            "p5_context": p5["context"],
        },
        "tile_plan": plan,
        "cpu_oracle": {
            "primary_reference": (
                "scalar CPU-FP64 accumulation from exact FP16-rounded "
                "inputs, weights, and hidden tensors"
            ),
            "independent_check": (
                "4x4 tiled CPU-FP64 implementation following the locked "
                "K-tile schedule"
            ),
            "hidden_contract": (
                "FP32 bias + ReLU in FP64 reference arithmetic, then "
                "explicit IEEE FP16 quantization after hidden layers 1 and 2"
            ),
            "output_contract": (
                "FP64 output retained after layer-3 FP32 bias values are "
                "promoted exactly to FP64"
            ),
            "quantization_counts": tensors["quantization_counts"],
            "layer_1_stats": scalar_stats_1,
            "layer_2_stats": scalar_stats_2,
            "layer_3_stats": scalar_stats_3,
            "scalar_vs_tiled": {
                "hidden_1": hidden_1_compare,
                "hidden_2": hidden_2_compare,
                "output": output_compare,
            },
            "output_statistics": {
                "minimum": min(scalar_output),
                "maximum": max(scalar_output),
                "mean": sum(scalar_output) / len(scalar_output),
                "l2": output_l2,
                "nonfinite_count": sum(
                    not math.isfinite(value)
                    for value in scalar_output
                ),
            },
            "tensor_hashes": hashes,
            "probes": probes,
        },
        "gates": all_gates,
        "scope_boundary": {
            "this_phase_is_cpu_only": True,
            "gpu_correctness_claimed": False,
            "performance_claimed": False,
            "next_gpu_checkpoint": (
                "4A1-P1: WIDTH64_SINGLE_LAYER_FOUR_K_TILE_ACCUMULATION"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tile_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "cpu_oracle.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "tensor_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    write_probe_csv(output_dir / "oracle_probes.csv", probes)
    write_binary_files(
        output_dir,
        tensors,
        scalar_hidden_1,
        scalar_hidden_2,
        scalar_output,
    )

    markdown = (
        "# Phase 4A1-P0 — Width-64 tile plan and CPU oracle\n\n"
        f"Decision: **`{report['decision']}`**\n\n"
        "## Locked execution plan\n\n"
        "- Topology: `16 × 64 → 64 → 64 → 64`\n"
        "- Block: `4` Wave32 waves / `128` threads\n"
        "- Each wave owns one `16×16` output-column tile\n"
        "- Each output tile accumulates `4` ordered K tiles\n"
        "- `12` rocWMMA operations planned per wave\n"
        "- `48` rocWMMA operations planned per block\n"
        "- Intermediate storage: one `16×64` FP16 LDS buffer (`2048` bytes)\n"
        "- Barriers: `3`\n"
        "- Intermediate global store/reload: none\n\n"
        "## CPU oracle\n\n"
        f"- Hidden 1 ReLU: `{scalar_stats_1['relu_clamped']}` clamped, "
        f"`{scalar_stats_1['relu_positive']}` positive\n"
        f"- Hidden 2 ReLU: `{scalar_stats_2['relu_clamped']}` clamped, "
        f"`{scalar_stats_2['relu_positive']}` positive\n"
        f"- Hidden 1 FP16 rounding events: "
        f"`{scalar_stats_1['fp16_cast_changed']}`\n"
        f"- Hidden 2 FP16 rounding events: "
        f"`{scalar_stats_2['fp16_cast_changed']}`\n"
        f"- Scalar/tiled hidden 1 bitwise equal: "
        f"`{hidden_1_compare['bitwise_equal']}`\n"
        f"- Scalar/tiled hidden 2 bitwise equal: "
        f"`{hidden_2_compare['bitwise_equal']}`\n"
        f"- Scalar/tiled final FP64 bitwise equal: "
        f"`{output_compare['bitwise_equal']}`\n"
        f"- Output range: `{min(scalar_output)}` to `{max(scalar_output)}`\n\n"
        "This checkpoint is CPU-only. It locks the schedule and oracle but "
        "does not claim GPU correctness or performance.\n"
    )
    (output_dir / "PHASE4A1_P0_REPORT.md").write_text(markdown)

    print(f"p5_sha256: {report['phase4a0_baseline']['p5_json_sha256']}")
    print(f"tile_operations: {len(plan['operations'])}")
    print(f"threads_per_block: {THREADS}")
    print(f"lds_bytes: {LDS_BYTES}")
    print(
        "hidden_1_relu: "
        f"clamped={scalar_stats_1['relu_clamped']} "
        f"positive={scalar_stats_1['relu_positive']}"
    )
    print(
        "hidden_2_relu: "
        f"clamped={scalar_stats_2['relu_clamped']} "
        f"positive={scalar_stats_2['relu_positive']}"
    )
    print(
        "hidden_1_fp16_cast_changed: "
        f"{scalar_stats_1['fp16_cast_changed']}"
    )
    print(
        "hidden_2_fp16_cast_changed: "
        f"{scalar_stats_2['fp16_cast_changed']}"
    )
    print(
        "output_range: "
        f"{min(scalar_output):.17g} .. {max(scalar_output):.17g}"
    )
    print("WIDTH64_TILE_COVERAGE: PASS" if all(plan_gates.values()) else "WIDTH64_TILE_COVERAGE: FAIL")
    print("WIDTH64_LDS_SINGLE_BUFFER_PLAN: PASS" if (
        plan_gates["single_lds_buffer_2048_bytes"]
        and plan_gates["three_barriers"]
    ) else "WIDTH64_LDS_SINGLE_BUFFER_PLAN: FAIL")
    print("WIDTH64_SCALAR_VS_TILED_ORACLE: PASS" if (
        hidden_1_compare["bitwise_equal"]
        and hidden_2_compare["bitwise_equal"]
        and output_compare["bitwise_equal"]
    ) else "WIDTH64_SCALAR_VS_TILED_ORACLE: FAIL")
    print("WIDTH64_HIDDEN1_FP16_ORACLE: RECORDED")
    print("WIDTH64_HIDDEN2_FP16_ORACLE: RECORDED")
    print("WIDTH64_OUTPUT_FP64_ORACLE: RECORDED")
    print(
        "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PROCESS: "
        + ("PASS" if passed else "FAIL")
    )

    return report


def run_self_test() -> None:
    values = (
        0.0,
        1.0,
        -2.0,
        0.33325,
        65504.0,
        2 ** -14,
    )
    for value in values:
        round_trip = quantize_fp16(value)
        assert fp16_bits(round_trip) == fp16_bits(value)

    assert quantize_fp32(1.0) == 1.0
    assert fp32_bits(-0.0) == 0x80000000
    assert BATCH == TILE
    assert WIDTH % TILE == 0
    assert THREADS == 128
    assert LDS_BYTES == 2048

    plan = build_tile_plan()
    assert all(validate_tile_plan(plan).values())

    print("PHASE4A1_P0_ORACLE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p5-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.p5_json is None or args.output_dir is None:
        parser.error(
            "--p5-json and --output-dir are required unless --self-test is used"
        )

    report = run_oracle(args.p5_json, args.output_dir)
    return 0 if report["decision"] == P0_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
