#!/usr/bin/env python3
"""Finalize TCNN_RDNA4_P4A0_P2_RAW_LANE_FRAGMENT_MAP_003."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A0_P2_RAW_LANE_FRAGMENT_MAP_003"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_capture(path: Path) -> tuple[
    dict[str, str],
    dict[str, bool],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    meta: dict[str, str] = {}
    gates: dict[str, bool] = {}
    maps: dict[str, list[dict[str, Any]]] = {
        "matrix_a": [],
        "matrix_b": [],
        "accumulator": [],
    }
    output: list[dict[str, Any]] = []

    for raw_line in path.read_text().splitlines():
        parts = raw_line.split("\t")
        if not parts:
            continue

        if parts[0] == "META":
            meta[parts[1]] = parts[2]
        elif parts[0] == "GATE":
            gates[parts[1]] = parts[2] == "1"
        elif parts[0] == "MAP_HALF":
            maps[parts[1]].append({
                "register_file_row": int(parts[2]),
                "lane": int(parts[3]),
                "marker": int(parts[4]),
                "matrix_row": int(parts[5]),
                "matrix_col": int(parts[6]),
                "raw_bits": parts[7],
                "write_count": int(parts[8]),
            })
        elif parts[0] == "MAP_FLOAT":
            maps[parts[1]].append({
                "register_file_row": int(parts[2]),
                "lane": int(parts[3]),
                "marker": int(parts[4]),
                "matrix_row": int(parts[5]),
                "matrix_col": int(parts[6]),
                "raw_bits": parts[7],
                "write_count": int(parts[8]),
            })
        elif parts[0] == "OUTPUT":
            output.append({
                "row": int(parts[1]),
                "col": int(parts[2]),
                "value": float(parts[3]),
                "expected": int(parts[4]),
            })

    assert meta["marker"] == MARKER
    assert meta["decision"] == "PASS"
    assert all(gates.values())
    assert meta["matrix_a_device_slots_per_lane"] == "8"
    assert meta["matrix_b_device_slots_per_lane"] == "8"
    assert meta["accumulator_device_slots_per_lane"] == "8"
    assert all(len(rows) == 256 for rows in maps.values())
    assert len(output) == 256

    return meta, gates, maps, output


def write_csv(path: Path, role: str, rows: list[dict[str, Any]]) -> None:
    fields = [
        "role",
        "register_file_row",
        "lane",
        "marker",
        "matrix_row",
        "matrix_col",
        "raw_bits",
        "write_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"role": role, **row})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-1", type=Path, required=True)
    parser.add_argument("--process-2", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    meta1, gates1, maps1, output1 = parse_capture(args.process_1)
    meta2, gates2, maps2, output2 = parse_capture(args.process_2)

    context_keys = (
        "arch",
        "hip_runtime_version",
        "hip_driver_version",
        "compiler_version",
        "warp_size",
        "matrix_a_device_slots_per_lane",
        "matrix_b_device_slots_per_lane",
        "accumulator_device_slots_per_lane",
        "capture_capacity_slots_per_lane",
    )
    context_equal = all(
        meta1.get(key) == meta2.get(key) for key in context_keys
    )
    maps_equal = maps1 == maps2 and output1 == output2
    write_counts_one = all(
        entry["write_count"] == 1
        for rows in maps1.values()
        for entry in rows
    )
    expected_geometry = (
        meta1["matrix_a_device_slots_per_lane"] == "8"
        and meta1["matrix_b_device_slots_per_lane"] == "8"
        and meta1["accumulator_device_slots_per_lane"] == "8"
    )

    passed = (
        context_equal
        and maps_equal
        and write_counts_one
        and expected_geometry
        and all(gates1.values())
        and all(gates2.values())
    )

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_PASS"
            if passed
            else "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_FAIL"
        ),
        "context": meta1,
        "interpretation": {
            "source": "rocwmma::to_register_file",
            "geometry_source": "device-compiled kernel diagnostics",
            "matrix_a_geometry": "8 FP16 register-file elements per Wave32 lane",
            "matrix_b_geometry": "8 FP16 register-file elements per Wave32 lane",
            "accumulator_geometry": "8 FP32 register-file elements per Wave32 lane",
            "logical_map_shape": "8 register-file rows x 32 lanes",
            "status": "version-bound diagnostic map, not a stable ABI",
            "tile": "16x16x16",
            "host_trait_warning": (
                "rocWMMA host compilation uses a Wave64 fallback; "
                "host-side fragment trait sizes are not the gfx1201 device geometry"
            ),
        },
        "processes": [
            {
                "path": str(args.process_1.resolve()),
                "sha256": sha256(args.process_1),
            },
            {
                "path": str(args.process_2.resolve()),
                "sha256": sha256(args.process_2),
            },
        ],
        "gates": {
            **gates1,
            "fresh_process_context_equal": context_equal,
            "fresh_process_maps_equal": maps_equal,
            "all_write_counts_one": write_counts_one,
            "expected_device_geometry": expected_geometry,
        },
        "maps": maps1,
        "stored_output": output1,
    }

    args.evidence.mkdir(parents=True, exist_ok=True)
    json_path = args.evidence / "phase4a0_p2_fragment_map.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    for role, rows in maps1.items():
        write_csv(
            args.evidence / f"{role}_register_file_map.csv",
            role,
            rows,
        )

    (args.evidence / "PHASE4A0_P2_FRAGMENT_MAP.md").write_text(
        "# Phase 4A0-P2 — rocWMMA raw lane/register-file map\n\n"
        f"Decision: **`{result['decision']}`**\n\n"
        "- Target: `gfx1201`, Wave32, tile `16×16×16`\n"
        "- Geometry source: device-compiled kernel diagnostics\n"
        "- Matrix A: `8` FP16 register-file elements per lane\n"
        "- Matrix B: `8` FP16 register-file elements per lane\n"
        "- Accumulator: `8` FP32 register-file elements per lane\n"
        "- Map geometry: `8 × 32 = 256` elements per role\n"
        f"- Fresh-process maps identical: `{maps_equal}`\n"
        f"- Every active lane/slot written once: `{write_counts_one}`\n\n"
        "rocWMMA's host compilation uses a Wave64 fallback. Therefore "
        "host-side fragment trait sizes are not used as evidence for the "
        "gfx1201 device layout.\n\n"
        "This is a version-bound diagnostic map produced through "
        "`rocwmma::to_register_file`, not a stable ABI guarantee.\n"
    )

    if not passed:
        print("ROCWMMA_P2_FRESH_PROCESS_REPRODUCIBILITY: FAIL")
        print("PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP: FAIL")
        return 1

    print("ROCWMMA_P2_DEVICE_GEOMETRY_CAPTURE: PASS")
    print("ROCWMMA_P2_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("ROCWMMA_P2_MAP_CONTEXT: RECORDED")
    print("ROCWMMA_RAW_LANE_FRAGMENT_MAP: CAPTURED")
    print("PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
