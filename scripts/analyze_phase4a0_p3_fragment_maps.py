#!/usr/bin/env python3
"""Phase 4A0-P3: interpret rocWMMA lane/register-file maps.

Marker:
    TCNN_RDNA4_P4A0_P3_FRAGMENT_MAP_INTERPRETATION_001

This is a pure host-side analysis of the qualified Phase 4A0-P2 evidence.
It does not compile or launch a GPU kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MARKER = "TCNN_RDNA4_P4A0_P3_FRAGMENT_MAP_INTERPRETATION_001"
P2_DECISION = "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_PASS"
P3_DECISION = "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"

ROLE_NAMES = ("matrix_a", "matrix_b", "accumulator")
PAIR_NAMES = (
    ("matrix_a", "matrix_b"),
    ("matrix_a", "accumulator"),
    ("matrix_b", "accumulator"),
)
INPUT_NAMES_SLOT = (
    "lane_bit_0",
    "lane_bit_1",
    "lane_bit_2",
    "lane_bit_3",
    "lane_bit_4",
    "slot_bit_0",
    "slot_bit_1",
    "slot_bit_2",
)
INPUT_NAMES_COORD = (
    "source_row_bit_0",
    "source_row_bit_1",
    "source_row_bit_2",
    "source_row_bit_3",
    "source_col_bit_0",
    "source_col_bit_1",
    "source_col_bit_2",
    "source_col_bit_3",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parity(value: int) -> int:
    return value.bit_count() & 1


@dataclass(frozen=True)
class AffineBit:
    mask: int
    bias: int
    input_names: tuple[str, ...]

    def evaluate(self, packed_input: int) -> int:
        return parity(self.mask & packed_input) ^ self.bias

    @property
    def terms(self) -> list[str]:
        return [
            name
            for index, name in enumerate(self.input_names)
            if self.mask & (1 << index)
        ]

    def text(self) -> str:
        terms = self.terms[:]
        if self.bias:
            terms.append("1")
        return " xor ".join(terms) if terms else "0"

    def cpp(self, variable_terms: tuple[str, ...]) -> str:
        terms = [
            variable_terms[index]
            for index in range(len(variable_terms))
            if self.mask & (1 << index)
        ]
        if self.bias:
            terms.append("1u")
        return " ^ ".join(f"({term})" for term in terms) if terms else "0u"

    def as_json(self) -> dict[str, Any]:
        return {
            "mask": self.mask,
            "bias": self.bias,
            "terms": self.terms,
            "formula": self.text(),
        }


def fit_affine_bit(
    packed_inputs: list[int],
    outputs: list[int],
    input_names: tuple[str, ...],
) -> AffineBit | None:
    assert len(packed_inputs) == len(outputs)
    masks = sorted(
        range(1 << len(input_names)),
        key=lambda mask: (mask.bit_count(), mask),
    )
    for mask in masks:
        for bias in (0, 1):
            if all(
                (parity(mask & x) ^ bias) == y
                for x, y in zip(packed_inputs, outputs)
            ):
                return AffineBit(mask=mask, bias=bias, input_names=input_names)
    return None


def fit_affine_vector(
    packed_inputs: list[int],
    packed_outputs: list[int],
    output_width: int,
    input_names: tuple[str, ...],
    output_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    bits: list[AffineBit] = []
    for output_bit in range(output_width):
        model = fit_affine_bit(
            packed_inputs,
            [(value >> output_bit) & 1 for value in packed_outputs],
            input_names,
        )
        if model is None:
            return {
                "exact": False,
                "output_bits": [],
                "reason": f"output bit {output_bit} is not affine over GF(2)",
            }
        bits.append(model)

    exact = all(
        sum(model.evaluate(x) << bit for bit, model in enumerate(bits)) == y
        for x, y in zip(packed_inputs, packed_outputs)
    )

    result_bits = []
    for bit_index, model in enumerate(bits):
        prefix = output_prefixes[bit_index]
        result_bits.append({
            "name": prefix,
            **model.as_json(),
        })

    return {
        "exact": exact,
        "output_bits": result_bits,
    }


def cpp_coordinate_expression(
    models: list[dict[str, Any]],
    bit_offset: int,
    variable_terms: tuple[str, ...],
) -> str:
    parts = []
    for bit_index, model_json in enumerate(models):
        model = AffineBit(
            mask=int(model_json["mask"]),
            bias=int(model_json["bias"]),
            input_names=INPUT_NAMES_SLOT,
        )
        parts.append(f"(({model.cpp(variable_terms)}) << {bit_index + bit_offset})")
    return " | ".join(parts)


def validate_p2(data: dict[str, Any]) -> None:
    assert data["decision"] == P2_DECISION
    assert all(bool(value) for value in data["gates"].values())

    context = data["context"]
    assert context["arch"].startswith("gfx1201")
    assert int(context["warp_size"]) == 32
    assert int(context["matrix_a_device_slots_per_lane"]) == 8
    assert int(context["matrix_b_device_slots_per_lane"]) == 8
    assert int(context["accumulator_device_slots_per_lane"]) == 8

    for role in ROLE_NAMES:
        rows = data["maps"][role]
        assert len(rows) == 256
        slot_keys = {
            (int(row["lane"]), int(row["register_file_row"]))
            for row in rows
        }
        coordinates = {
            (int(row["matrix_row"]), int(row["matrix_col"]))
            for row in rows
        }
        assert len(slot_keys) == 256
        assert len(coordinates) == 256
        assert all(int(row["write_count"]) == 1 for row in rows)
        assert all(0 <= int(row["lane"]) < 32 for row in rows)
        assert all(0 <= int(row["register_file_row"]) < 8 for row in rows)
        assert all(0 <= int(row["matrix_row"]) < 16 for row in rows)
        assert all(0 <= int(row["matrix_col"]) < 16 for row in rows)

    output = data["stored_output"]
    assert len(output) == 256
    assert all(float(row["value"]) == float(row["expected"]) for row in output)


def normalize_role(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {
            "lane": int(row["lane"]),
            "slot": int(row["register_file_row"]),
            "marker": int(row["marker"]),
            "row": int(row["matrix_row"]),
            "col": int(row["matrix_col"]),
            "raw_bits": str(row["raw_bits"]),
            "write_count": int(row["write_count"]),
        }
        for row in rows
    ]
    return sorted(normalized, key=lambda row: (row["slot"], row["lane"]))


def slot_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["lane"]), int(row["slot"])


def coordinate(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["row"]), int(row["col"])


def pack_lane_slot(lane: int, slot: int) -> int:
    return lane | (slot << 5)


def pack_coordinate(row: int, col: int) -> int:
    return row | (col << 4)


def derive_role_model(role: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    packed_inputs = [
        pack_lane_slot(int(row["lane"]), int(row["slot"]))
        for row in rows
    ]
    packed_outputs = [
        pack_coordinate(int(row["row"]), int(row["col"]))
        for row in rows
    ]

    output_names = tuple(
        [f"row_bit_{bit}" for bit in range(4)]
        + [f"col_bit_{bit}" for bit in range(4)]
    )
    affine = fit_affine_vector(
        packed_inputs,
        packed_outputs,
        output_width=8,
        input_names=INPUT_NAMES_SLOT,
        output_prefixes=output_names,
    )

    row_models = affine.get("output_bits", [])[:4]
    col_models = affine.get("output_bits", [])[4:8]

    cpp_lane_terms = (
        "((lane >> 0) & 1u)",
        "((lane >> 1) & 1u)",
        "((lane >> 2) & 1u)",
        "((lane >> 3) & 1u)",
        "((lane >> 4) & 1u)",
        "((slot >> 0) & 1u)",
        "((slot >> 1) & 1u)",
        "((slot >> 2) & 1u)",
    )

    cpp = None
    if affine["exact"]:
        cpp = {
            "row_expression": cpp_coordinate_expression(
                row_models, 0, cpp_lane_terms
            ),
            "col_expression": cpp_coordinate_expression(
                col_models, 0, cpp_lane_terms
            ),
        }

    inverse = {
        f"{row['row']},{row['col']}": {
            "lane": row["lane"],
            "slot": row["slot"],
        }
        for row in rows
    }

    return {
        "role": role,
        "entry_count": len(rows),
        "map_shape": {
            "register_file_rows": 8,
            "lanes": 32,
            "logical_elements": 256,
        },
        "affine_gf2_lane_slot_to_coordinate": affine,
        "cpp_coordinate_decoder": cpp,
        "coordinate_to_lane_slot": inverse,
        "map_sha256": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
    }


def candidate_transform_name(
    source_coordinates: list[tuple[int, int]],
    target_coordinates: list[tuple[int, int]],
) -> str | None:
    candidates = {
        "identity": lambda r, c: (r, c),
        "transpose": lambda r, c: (c, r),
        "row_flip": lambda r, c: (15 - r, c),
        "col_flip": lambda r, c: (r, 15 - c),
        "row_col_flip": lambda r, c: (15 - r, 15 - c),
        "transpose_row_flip": lambda r, c: (15 - c, r),
        "transpose_col_flip": lambda r, c: (c, 15 - r),
        "transpose_row_col_flip": lambda r, c: (15 - c, 15 - r),
    }
    for name, transform in candidates.items():
        if all(
            transform(*source) == target
            for source, target in zip(source_coordinates, target_coordinates)
        ):
            return name
    return None


def permutation_cycles(permutation: dict[int, int]) -> list[list[int]]:
    remaining = set(permutation)
    cycles: list[list[int]] = []
    while remaining:
        start = min(remaining)
        current = start
        cycle = []
        while current not in cycle:
            cycle.append(current)
            remaining.discard(current)
            current = permutation[current]
        cycles.append(cycle)
    return cycles


def derive_pairwise(
    source_role: str,
    target_role: str,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_slot = {slot_key(row): row for row in source_rows}
    target_by_slot = {slot_key(row): row for row in target_rows}
    target_slot_by_coordinate = {
        coordinate(row): slot_key(row) for row in target_rows
    }

    ordered_slots = sorted(source_by_slot)
    source_coordinates = [
        coordinate(source_by_slot[key]) for key in ordered_slots
    ]
    target_coordinates = [
        coordinate(target_by_slot[key]) for key in ordered_slots
    ]

    identity_count = sum(
        source == target
        for source, target in zip(source_coordinates, target_coordinates)
    )
    transpose_count = sum(
        (source[1], source[0]) == target
        for source, target in zip(source_coordinates, target_coordinates)
    )

    packed_inputs = [
        pack_coordinate(*source) for source in source_coordinates
    ]
    packed_outputs = [
        pack_coordinate(*target) for target in target_coordinates
    ]
    output_names = tuple(
        [f"target_row_bit_{bit}" for bit in range(4)]
        + [f"target_col_bit_{bit}" for bit in range(4)]
    )
    affine = fit_affine_vector(
        packed_inputs,
        packed_outputs,
        output_width=8,
        input_names=INPUT_NAMES_COORD,
        output_prefixes=output_names,
    )

    simple_transform = candidate_transform_name(
        source_coordinates, target_coordinates
    )

    coordinate_preserving_reindex = []
    permutation: dict[int, int] = {}
    fixed_points = 0
    same_lane = 0
    same_slot = 0

    for source_index, key in enumerate(ordered_slots):
        coord = coordinate(source_by_slot[key])
        target_key = target_slot_by_coordinate[coord]
        target_index = ordered_slots.index(target_key)
        permutation[source_index] = target_index

        if target_key == key:
            fixed_points += 1
        if target_key[0] == key[0]:
            same_lane += 1
        if target_key[1] == key[1]:
            same_slot += 1

        coordinate_preserving_reindex.append({
            "matrix_row": coord[0],
            "matrix_col": coord[1],
            "source_lane": key[0],
            "source_slot": key[1],
            "target_lane": target_key[0],
            "target_slot": target_key[1],
        })

    cycles = permutation_cycles(permutation)

    same_slot_identity = identity_count == 256
    same_slot_transpose = transpose_count == 256

    if same_slot_identity:
        classification = "SAME_SLOT_IDENTITY"
    elif same_slot_transpose:
        classification = "SAME_SLOT_TRANSPOSE"
    elif simple_transform is not None:
        classification = f"SAME_SLOT_{simple_transform.upper()}"
    elif affine["exact"]:
        classification = "SAME_SLOT_AFFINE_GF2_REINTERPRETATION"
    else:
        classification = "GENERAL_SLOT_REINTERPRETATION"

    return {
        "source_role": source_role,
        "target_role": target_role,
        "classification": classification,
        "direct_same_slot_elementwise_safe": same_slot_identity,
        "same_slot_identity_count": identity_count,
        "same_slot_transpose_count": transpose_count,
        "simple_same_slot_transform": simple_transform,
        "same_slot_affine_gf2_transform": affine,
        "coordinate_preserving_reindex": {
            "required": not same_slot_identity,
            "fixed_points": fixed_points,
            "same_lane_count": same_lane,
            "same_register_file_row_count": same_slot,
            "cycle_count": len(cycles),
            "max_cycle_length": max(len(cycle) for cycle in cycles),
            "cycle_length_histogram": {
                str(length): sum(len(cycle) == length for cycle in cycles)
                for length in sorted({len(cycle) for cycle in cycles})
            },
            "entries": coordinate_preserving_reindex,
        },
    }


def write_slot_map_csv(
    path: Path,
    roles: dict[str, list[dict[str, Any]]],
) -> None:
    fields = ["role", "register_file_row", "lane", "matrix_row", "matrix_col", "marker", "raw_bits"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for role in ROLE_NAMES:
            for row in roles[role]:
                writer.writerow({
                    "role": role,
                    "register_file_row": row["slot"],
                    "lane": row["lane"],
                    "matrix_row": row["row"],
                    "matrix_col": row["col"],
                    "marker": row["marker"],
                    "raw_bits": row["raw_bits"],
                })


def write_pairwise_csv(path: Path, pairwise: dict[str, Any]) -> None:
    fields = [
        "pair",
        "matrix_row",
        "matrix_col",
        "source_lane",
        "source_slot",
        "target_lane",
        "target_slot",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair_name, pair in pairwise.items():
            for row in pair["coordinate_preserving_reindex"]["entries"]:
                writer.writerow({"pair": pair_name, **row})


def formulas_markdown(role_models: dict[str, Any]) -> str:
    lines = []
    for role in ROLE_NAMES:
        model = role_models[role]["affine_gf2_lane_slot_to_coordinate"]
        lines.append(f"### `{role}`")
        lines.append("")
        if not model["exact"]:
            lines.append("No exact GF(2)-affine formula was found; use the full lookup table.")
            lines.append("")
            continue

        for bit in model["output_bits"]:
            lines.append(f"- `{bit['name']} = {bit['formula']}`")
        lines.append("")
        cpp = role_models[role]["cpp_coordinate_decoder"]
        lines.append("Generated C++ decoder:")
        lines.append("")
        lines.append("```cpp")
        lines.append(f"uint32_t row = {cpp['row_expression']};")
        lines.append(f"uint32_t col = {cpp['col_expression']};")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def create_report(
    source_path: Path,
    p2: dict[str, Any],
) -> dict[str, Any]:
    validate_p2(p2)

    roles = {
        role: normalize_role(p2["maps"][role])
        for role in ROLE_NAMES
    }

    role_models = {
        role: derive_role_model(role, roles[role])
        for role in ROLE_NAMES
    }

    pairwise = {}
    for source_role, target_role in PAIR_NAMES:
        key = f"{source_role}_to_{target_role}"
        pairwise[key] = derive_pairwise(
            source_role,
            target_role,
            roles[source_role],
            roles[target_role],
        )

    role_models_exact_or_lookup = all(
        model["affine_gf2_lane_slot_to_coordinate"]["exact"]
        or len(model["coordinate_to_lane_slot"]) == 256
        for model in role_models.values()
    )
    pairwise_complete = all(
        len(pair["coordinate_preserving_reindex"]["entries"]) == 256
        for pair in pairwise.values()
    )
    pairwise_permutations_valid = all(
        len({
            (entry["target_lane"], entry["target_slot"])
            for entry in pair["coordinate_preserving_reindex"]["entries"]
        }) == 256
        for pair in pairwise.values()
    )

    direct_safe_pairs = [
        key
        for key, pair in pairwise.items()
        if pair["direct_same_slot_elementwise_safe"]
    ]
    reindex_required_pairs = [
        key
        for key, pair in pairwise.items()
        if not pair["direct_same_slot_elementwise_safe"]
    ]

    all_role_formulas_affine = all(
        model["affine_gf2_lane_slot_to_coordinate"]["exact"]
        for model in role_models.values()
    )
    all_pair_transforms_affine = all(
        pair["same_slot_affine_gf2_transform"]["exact"]
        for pair in pairwise.values()
    )

    gates = {
        "p2_prerequisite": True,
        "role_maps_complete": all(
            model["entry_count"] == 256 for model in role_models.values()
        ),
        "role_models_exact_or_lookup_complete": role_models_exact_or_lookup,
        "pairwise_reindex_tables_complete": pairwise_complete,
        "pairwise_reindex_tables_are_permutations": pairwise_permutations_valid,
        "kernel_policy_recorded": True,
    }

    passed = all(gates.values())

    return {
        "marker": MARKER,
        "decision": P3_DECISION if passed else "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_FAIL",
        "source": {
            "p2_json": str(source_path.resolve()),
            "p2_json_sha256": sha256(source_path),
            "p2_marker": p2["marker"],
            "p2_decision": p2["decision"],
        },
        "context": p2["context"],
        "scope": {
            "architecture": "gfx1201",
            "wave_size": 32,
            "tile": "16x16x16",
            "matrix_a": "FP16 row-major",
            "matrix_b": "FP16 column-major",
            "accumulator": "FP32",
            "register_file_geometry": "8 rows x 32 lanes",
            "interpretation_boundary": (
                "rocwmma::to_register_file geometry; not physical VGPR numbering "
                "and not a stable ABI"
            ),
        },
        "role_models": role_models,
        "pairwise_role_equivalence": pairwise,
        "kernel_policy": {
            "direct_same_slot_cross_role_elementwise_safe_pairs": direct_safe_pairs,
            "coordinate_reindex_required_pairs": reindex_required_pairs,
            "accumulator_epilogue_rule": (
                "Use the accumulator lane/slot-to-(row,col) decoder or the exact "
                "lookup table for output-column bias and elementwise activation."
            ),
            "cross_role_rule": (
                "Never assume matrix_a, matrix_b, and accumulator share the same "
                "coordinate at the same lane/register-file row unless the pair is "
                "listed as direct-safe."
            ),
        },
        "diagnostics": {
            "all_role_formulas_affine_gf2": all_role_formulas_affine,
            "all_pair_same_slot_transforms_affine_gf2": all_pair_transforms_affine,
        },
        "gates": gates,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    pair_lines = []
    for key, pair in report["pairwise_role_equivalence"].items():
        pair_lines.extend([
            f"### `{key}`",
            "",
            f"- Classification: `{pair['classification']}`",
            f"- Direct same-slot elementwise use: `{pair['direct_same_slot_elementwise_safe']}`",
            f"- Same-slot identity matches: `{pair['same_slot_identity_count']}/256`",
            f"- Same-slot transpose matches: `{pair['same_slot_transpose_count']}/256`",
            f"- Coordinate-preserving fixed points: "
            f"`{pair['coordinate_preserving_reindex']['fixed_points']}/256`",
            f"- Reindex cycle count: `{pair['coordinate_preserving_reindex']['cycle_count']}`",
            f"- Maximum cycle length: `{pair['coordinate_preserving_reindex']['max_cycle_length']}`",
            "",
        ])

    direct_pairs = report["kernel_policy"][
        "direct_same_slot_cross_role_elementwise_safe_pairs"
    ]
    reindex_pairs = report["kernel_policy"]["coordinate_reindex_required_pairs"]

    text = (
        "# Phase 4A0-P3 — Fragment-map interpretation and role equivalence\n\n"
        f"Decision: **`{report['decision']}`**\n\n"
        "## Qualified scope\n\n"
        "- `gfx1201`, Wave32, rocWMMA register-file transform\n"
        "- Tile: `16×16×16`\n"
        "- Matrix A: FP16 row-major\n"
        "- Matrix B: FP16 column-major\n"
        "- Accumulator: FP32\n"
        "- Geometry: `8 register-file rows × 32 lanes`\n\n"
        "> The analysis describes the public `rocwmma::to_register_file` "
        "geometry. It does not claim physical VGPR numbers or a stable ABI.\n\n"
        "## Exact lane/slot decoders\n\n"
        + formulas_markdown(report["role_models"])
        + "\n## Pairwise role equivalence\n\n"
        + "\n".join(pair_lines)
        + "\n## Kernel policy\n\n"
        + f"- Direct-safe same-slot pairs: `{direct_pairs}`\n"
        + f"- Pairs requiring coordinate reindexing: `{reindex_pairs}`\n"
        + "- Accumulator epilogues must decode accumulator output coordinates "
        "before applying output-column bias.\n"
        + "- Cross-role lane/slot arithmetic is forbidden unless its pair is "
        "explicitly classified direct-safe.\n\n"
        "## Gates\n\n"
        + "\n".join(
            f"- `{name}`: `{value}`"
            for name, value in report["gates"].items()
        )
        + "\n"
    )
    path.write_text(text)


def run_self_test() -> None:
    rows = []
    for slot in range(8):
        for lane in range(32):
            packed = pack_lane_slot(lane, slot)
            row = packed & 0xF
            col = (packed >> 4) & 0xF
            rows.append({
                "lane": lane,
                "slot": slot,
                "marker": 1 + row * 16 + col,
                "row": row,
                "col": col,
                "raw_bits": "0x0000",
                "write_count": 1,
            })

    model = derive_role_model("synthetic", rows)
    assert model["affine_gf2_lane_slot_to_coordinate"]["exact"]

    identity = derive_pairwise("a", "b", rows, rows)
    assert identity["classification"] == "SAME_SLOT_IDENTITY"
    assert identity["direct_same_slot_elementwise_safe"]

    transposed = [
        {
            **row,
            "row": row["col"],
            "col": row["row"],
        }
        for row in rows
    ]
    transpose = derive_pairwise("a", "b", rows, transposed)
    assert transpose["classification"] == "SAME_SLOT_TRANSPOSE"
    assert not transpose["direct_same_slot_elementwise_safe"]

    print("PHASE4A0_P3_ANALYZER_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless --self-test is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    p2 = json.loads(args.input.read_text())
    report = create_report(args.input, p2)

    json_path = args.output_dir / "phase4a0_p3_analysis.json"
    markdown_path = args.output_dir / "PHASE4A0_P3_ANALYSIS.md"
    slot_csv = args.output_dir / "fragment_role_slot_maps.csv"
    pair_csv = args.output_dir / "fragment_role_reindex_tables.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(markdown_path, report)

    roles = {
        role: normalize_role(p2["maps"][role])
        for role in ROLE_NAMES
    }
    write_slot_map_csv(slot_csv, roles)
    write_pairwise_csv(pair_csv, report["pairwise_role_equivalence"])

    if report["decision"] != P3_DECISION:
        print("ROCWMMA_P3_MAP_INTERPRETATION: FAIL")
        print("PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION: FAIL")
        return 1

    affine_roles = report["diagnostics"]["all_role_formulas_affine_gf2"]
    affine_pairs = report["diagnostics"][
        "all_pair_same_slot_transforms_affine_gf2"
    ]

    print(f"source_p2_sha256: {report['source']['p2_json_sha256']}")
    print(
        "direct_safe_pairs: "
        + json.dumps(
            report["kernel_policy"][
                "direct_same_slot_cross_role_elementwise_safe_pairs"
            ],
            sort_keys=True,
        )
    )
    print(
        "reindex_required_pairs: "
        + json.dumps(
            report["kernel_policy"]["coordinate_reindex_required_pairs"],
            sort_keys=True,
        )
    )
    print(
        "ROCWMMA_P3_ROLE_AFFINE_GF2_MODELS: "
        + ("DERIVED" if affine_roles else "LOOKUP_FALLBACK")
    )
    print(
        "ROCWMMA_P3_PAIRWISE_AFFINE_GF2_MODELS: "
        + ("DERIVED" if affine_pairs else "LOOKUP_FALLBACK")
    )
    print("ROCWMMA_P3_ROLE_EQUIVALENCE: CLASSIFIED")
    print("ROCWMMA_P3_KERNEL_POLICY: RECORDED")
    print("PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
