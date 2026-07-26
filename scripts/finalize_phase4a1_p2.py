#!/usr/bin/env python3
"""Finalize Phase 4A1-P2 fresh-process evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P2_HIDDEN_EPILOGUE_LDS_001"
PROCESS_DECISION = "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_PASS"
FINAL_DECISION = "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_CONSOLIDATED_PASS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    assert data["marker"] == MARKER
    assert data["decision"] == PROCESS_DECISION
    assert all(bool(value) for value in data["gates"].values())

    assert data["context"]["arch"].startswith("gfx1201")
    assert data["context"]["warp_size"] == 32
    assert data["context"]["threads_per_block"] == 128
    assert data["context"]["waves_per_block"] == 4
    assert data["context"]["k_tiles"] == 4
    assert data["context"]["accumulator_slots_per_lane"] == 8
    assert data["context"]["matrix_a_slots_per_lane"] == 8

    assert data["topology"]["hidden_width"] == 64
    assert data["topology"]["lds_bytes"] == 2048
    assert data["topology"]["publication_barriers"] == 1
    assert data["topology"]["kernel_launches"] == 1

    assert data["diagnostics"]["wave_entry_counts"] == [1, 1, 1, 1]
    assert data["diagnostics"]["publication_counts"] == [1, 1, 1, 1]
    assert data["diagnostics"]["consumer_counts"] == [1, 1, 1, 1]
    assert data["diagnostics"]["producer_visibility_counts"] == [1, 1, 1, 1]
    assert data["diagnostics"]["mapping_error_count"] == 0

    assert data["cross_wave_readback"] == [
        {"consumer_wave": 0, "producer_wave": 1},
        {"consumer_wave": 1, "producer_wave": 2},
        {"consumer_wave": 2, "producer_wave": 3},
        {"consumer_wave": 3, "producer_wave": 0},
    ]

    assert data["metrics"]["mismatch_count"] == 0
    assert data["metrics"]["nonfinite_count"] == 0
    assert data["metrics"]["zero_count"] > 0
    assert data["metrics"]["positive_count"] > 0
    assert data["metrics"]["max_abs"] == 0
    assert data["metrics"]["relay_moved_entries_per_tile"] == 240

    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-1-json", type=Path, required=True)
    parser.add_argument("--process-2-json", type=Path, required=True)
    parser.add_argument("--process-1-csv", type=Path, required=True)
    parser.add_argument("--process-2-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--p0-json", type=Path, required=True)
    parser.add_argument("--p1-json", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    process_1 = load_result(args.process_1_json)
    process_2 = load_result(args.process_2_json)

    manifest = json.loads(args.manifest.read_text())
    p0 = json.loads(args.p0_json.read_text())
    p1 = json.loads(args.p1_json.read_text())

    assert manifest["decision"] == (
        "PHASE4A1_P2_MAPPING_HEADER_GENERATION_PASS"
    )
    assert manifest["mapping"]["moved_entries"] == 240

    assert p0["decision"] == (
        "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
    )
    assert all(p0["gates"].values())

    assert p1["decision"] == (
        "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
    )
    assert all(p1["gates"].values())

    p0_layer_1 = p0["cpu_oracle"]["layer_1_stats"]
    assert p0_layer_1["relu_clamped"] > 0
    assert p0_layer_1["relu_positive"] > 0
    assert p0_layer_1["fp16_cast_changed"] > 0

    json_identical = (
        args.process_1_json.read_bytes()
        == args.process_2_json.read_bytes()
    )
    csv_identical = (
        args.process_1_csv.read_bytes()
        == args.process_2_csv.read_bytes()
    )
    parsed_identical = process_1 == process_2

    passed = json_identical and csv_identical and parsed_identical

    consolidated = {
        "marker": MARKER,
        "decision": (
            FINAL_DECISION
            if passed
            else "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_CONSOLIDATED_FAIL"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
            "data": manifest,
        },
        "p0_oracle": {
            "path": str(args.p0_json.resolve()),
            "sha256": sha256(args.p0_json),
            "layer_1_stats": p0_layer_1,
        },
        "p1_evidence": {
            "path": str(args.p1_json.resolve()),
            "sha256": sha256(args.p1_json),
        },
        "fresh_process_reproducibility": {
            "json_identical": json_identical,
            "csv_identical": csv_identical,
            "parsed_data_identical": parsed_identical,
        },
        "result": process_1,
        "processes": [
            {
                "json": str(args.process_1_json.resolve()),
                "json_sha256": sha256(args.process_1_json),
                "csv": str(args.process_1_csv.resolve()),
                "csv_sha256": sha256(args.process_1_csv),
            },
            {
                "json": str(args.process_2_json.resolve()),
                "json_sha256": sha256(args.process_2_json),
                "csv": str(args.process_2_csv.resolve()),
                "csv_sha256": sha256(args.process_2_csv),
            },
        ],
        "gates": {
            "process_1_pass": True,
            "process_2_pass": True,
            "p0_hidden_epilogue_exercised": True,
            "p1_four_k_tile_prerequisite": True,
            "fresh_process_json_identical": json_identical,
            "fresh_process_csv_identical": csv_identical,
            "fresh_process_parsed_data_identical": parsed_identical,
        },
    }

    args.evidence.mkdir(parents=True, exist_ok=True)

    output_json = (
        args.evidence / "phase4a1_p2_width64_hidden_lds.json"
    )

    output_json.write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n"
    )

    shutil.copy2(
        args.process_1_csv,
        args.evidence / "phase4a1_p2_hidden_readback.csv",
    )

    metrics = process_1["metrics"]

    markdown = (
        "# Phase 4A1-P2 — Width-64 hidden epilogue and LDS publication\n\n"
        f"Decision: **`{consolidated['decision']}`**\n\n"
        "- Target: `gfx1201`, Wave32\n"
        "- Block: `4` waves / `128` threads\n"
        "- Width-64 K accumulation: `4` K tiles per wave\n"
        "- Epilogue: FP32 bias + ReLU + FP32→FP16\n"
        "- Relay: P3-derived accumulator→matrix-A permutation\n"
        "- Shared hidden buffer: `16×64` FP16 / `2048` bytes\n"
        "- Publication barrier: `1`\n"
        "- Visibility test: every wave reads a tile written by another wave\n"
        "- Consumer→producer mapping: `0→1, 1→2, 2→3, 3→0`\n"
        f"- Hidden mismatches: `{metrics['mismatch_count']}`\n"
        f"- Hidden nonfinite values: `{metrics['nonfinite_count']}`\n"
        f"- Hidden zero values: `{metrics['zero_count']}`\n"
        f"- Hidden positive values: `{metrics['positive_count']}`\n"
        f"- Fresh-process JSON identical: `{json_identical}`\n"
        f"- Fresh-process CSV identical: `{csv_identical}`\n"
    )

    (
        args.evidence / "PHASE4A1_P2_REPORT.md"
    ).write_text(markdown)

    if not passed:
        print(
            "WIDTH64_HIDDEN_LDS_FRESH_PROCESS_REPRODUCIBILITY: FAIL"
        )
        print("PHASE4A1_P2_WIDTH64_HIDDEN_LDS: FAIL")
        return 1

    print(
        "WIDTH64_HIDDEN_LDS_FRESH_PROCESS_REPRODUCIBILITY: PASS"
    )
    print("PHASE4A1_P2_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A1_P2_WIDTH64_HIDDEN_LDS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
