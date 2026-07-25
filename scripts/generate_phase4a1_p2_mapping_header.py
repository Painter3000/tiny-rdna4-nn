#!/usr/bin/env python3
"""Generate the exact Phase 4A1-P2 rocWMMA relay-map header."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P2_HIDDEN_EPILOGUE_LDS_001"

P0_DECISION = "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
P1_DECISION = "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
P3_DECISION = "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"
P4_DECISION = "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_array(name: str, values: list[int]) -> str:
    rows = []
    for start in range(0, len(values), 16):
        rows.append(
            "    "
            + ", ".join(str(value) for value in values[start : start + 16])
            + ","
        )

    return (
        f"__device__ __constant__ unsigned char {name}[256] = {{\n"
        + "\n".join(rows)
        + "\n};"
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_and_build(
    p0_path: Path,
    p1_path: Path,
    p3_path: Path,
    p4_path: Path,
) -> tuple[dict[str, Any], list[int], list[int], list[int]]:
    p0 = load_json(p0_path)
    p1 = load_json(p1_path)
    p3 = load_json(p3_path)
    p4 = load_json(p4_path)

    assert p0["decision"] == P0_DECISION
    assert all(bool(value) for value in p0["gates"].values())

    assert p1["decision"] == P1_DECISION
    assert all(bool(value) for value in p1["gates"].values())
    assert p1["p0_json_sha256"] == sha256(p0_path)
    assert p1["result"]["decision"] == (
        "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_PASS"
    )
    assert all(bool(value) for value in p1["result"]["gates"].values())

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

    acc_inverse = p3["role_models"]["accumulator"][
        "coordinate_to_lane_slot"
    ]
    a_inverse = p3["role_models"]["matrix_a"][
        "coordinate_to_lane_slot"
    ]

    assert len(acc_inverse) == 256
    assert len(a_inverse) == 256

    source_lane_for_a = [-1] * 256
    source_slot_for_a = [-1] * 256
    accumulator_col_by_slot = [-1] * 256

    source_slots_seen: set[tuple[int, int]] = set()
    target_slots_seen: set[tuple[int, int]] = set()

    for entry in entries:
        row = int(entry["matrix_row"])
        col = int(entry["matrix_col"])

        a_lane = int(entry["source_lane"])
        a_slot = int(entry["source_slot"])
        acc_lane = int(entry["target_lane"])
        acc_slot = int(entry["target_slot"])

        coordinate_key = f"{row},{col}"

        assert a_inverse[coordinate_key] == {
            "lane": a_lane,
            "slot": a_slot,
        }
        assert acc_inverse[coordinate_key] == {
            "lane": acc_lane,
            "slot": acc_slot,
        }

        target_index = a_lane * 8 + a_slot
        source_lane_for_a[target_index] = acc_lane
        source_slot_for_a[target_index] = acc_slot

        target_slots_seen.add((a_lane, a_slot))
        source_slots_seen.add((acc_lane, acc_slot))

    assert len(target_slots_seen) == 256
    assert len(source_slots_seen) == 256
    assert all(0 <= value < 32 for value in source_lane_for_a)
    assert all(0 <= value < 8 for value in source_slot_for_a)

    for coordinate_key, lane_slot in acc_inverse.items():
        _, col_text = coordinate_key.split(",")
        col = int(col_text)
        lane = int(lane_slot["lane"])
        slot = int(lane_slot["slot"])
        index = lane * 8 + slot

        assert accumulator_col_by_slot[index] == -1
        accumulator_col_by_slot[index] = col

    assert all(0 <= value < 16 for value in accumulator_col_by_slot)

    moved_entries = sum(
        1
        for index, source in enumerate(
            zip(source_lane_for_a, source_slot_for_a)
        )
        if (index // 8, index % 8) != source
    )
    assert moved_entries == 240

    metadata = {
        "marker": MARKER,
        "decision": "PHASE4A1_P2_MAPPING_HEADER_GENERATION_PASS",
        "prerequisites": {
            "p0_json": str(p0_path.resolve()),
            "p0_sha256": sha256(p0_path),
            "p1_json": str(p1_path.resolve()),
            "p1_sha256": sha256(p1_path),
            "p3_json": str(p3_path.resolve()),
            "p3_sha256": sha256(p3_path),
            "p4_json": str(p4_path.resolve()),
            "p4_sha256": sha256(p4_path),
        },
        "mapping": {
            "entries": 256,
            "moved_entries": moved_entries,
            "fixed_entries": 256 - moved_entries,
            "source_lane_sha256": hashlib.sha256(
                bytes(source_lane_for_a)
            ).hexdigest(),
            "source_slot_sha256": hashlib.sha256(
                bytes(source_slot_for_a)
            ).hexdigest(),
            "accumulator_column_sha256": hashlib.sha256(
                bytes(accumulator_col_by_slot)
            ).hexdigest(),
        },
        "context": context,
    }

    return (
        metadata,
        source_lane_for_a,
        source_slot_for_a,
        accumulator_col_by_slot,
    )


def make_header(
    metadata: dict[str, Any],
    source_lanes: list[int],
    source_slots: list[int],
    accumulator_cols: list[int],
) -> str:
    prerequisites = metadata["prerequisites"]

    arrays = "\n\n".join(
        (
            format_array("kAccLaneForATargetA", source_lanes),
            format_array("kAccSlotForATargetA", source_slots),
            format_array("kAccumulatorColumn", accumulator_cols),
        )
    )

    return f"""// {MARKER}
// Generated file. Do not edit manually.
#pragma once

namespace phase4a1_p2_generated {{

constexpr const char* kMarker = "{MARKER}";
constexpr const char* kP0Sha256 = "{prerequisites["p0_sha256"]}";
constexpr const char* kP1Sha256 = "{prerequisites["p1_sha256"]}";
constexpr const char* kP3Sha256 = "{prerequisites["p3_sha256"]}";
constexpr const char* kP4Sha256 = "{prerequisites["p4_sha256"]}";
constexpr unsigned int kRelayMovedEntries = 240u;

{arrays}

}} // namespace phase4a1_p2_generated
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-json", type=Path, required=True)
    parser.add_argument("--p1-json", type=Path, required=True)
    parser.add_argument("--p3-json", type=Path, required=True)
    parser.add_argument("--p4-json", type=Path, required=True)
    parser.add_argument("--output-header", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    (
        metadata,
        source_lanes,
        source_slots,
        accumulator_cols,
    ) = validate_and_build(
        args.p0_json,
        args.p1_json,
        args.p3_json,
        args.p4_json,
    )

    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    args.output_header.write_text(
        make_header(
            metadata,
            source_lanes,
            source_slots,
            accumulator_cols,
        )
    )

    metadata["generated_header"] = str(args.output_header.resolve())
    metadata["generated_header_sha256"] = sha256(args.output_header)

    args.output_manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print("p0_sha256: " + metadata["prerequisites"]["p0_sha256"])
    print("p1_sha256: " + metadata["prerequisites"]["p1_sha256"])
    print("p3_sha256: " + metadata["prerequisites"]["p3_sha256"])
    print("p4_sha256: " + metadata["prerequisites"]["p4_sha256"])
    print(
        "generated_header_sha256: "
        + metadata["generated_header_sha256"]
    )
    print(
        "relay_moved_entries: "
        + str(metadata["mapping"]["moved_entries"])
    )
    print("PHASE4A1_P2_MAPPING_HEADER_GENERATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
