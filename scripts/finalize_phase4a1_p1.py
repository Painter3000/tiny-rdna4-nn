#!/usr/bin/env python3
"""Finalize Phase 4A1-P1 fresh-process evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P1_WIDTH64_FOUR_K_TILE_001"
DECISION = "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_PASS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    assert data["marker"] == MARKER
    assert data["decision"] == DECISION
    assert all(bool(value) for value in data["gates"].values())

    assert data["context"]["arch"].startswith("gfx1201")
    assert data["context"]["warp_size"] == 32
    assert data["context"]["threads_per_block"] == 128
    assert data["context"]["waves_per_block"] == 4
    assert data["context"]["k_tiles"] == 4

    assert data["topology"]["output_tiles"] == 4
    assert data["topology"]["k_tiles_per_output_tile"] == 4
    assert data["topology"]["mma_sync_calls_per_wave"] == 4
    assert data["topology"]["mma_sync_calls_per_block"] == 16
    assert data["topology"]["kernel_launches"] == 1

    assert len(data["stages"]) == 4
    assert [stage["k_terms"] for stage in data["stages"]] == [
        16,
        32,
        48,
        64,
    ]
    assert all(stage["passed"] for stage in data["stages"])

    assert len(data["wave_tiles"]) == 16
    assert all(tile["passed"] for tile in data["wave_tiles"])

    assert data["diagnostics"]["wave_entry_counts"] == [1, 1, 1, 1]
    assert data["diagnostics"]["wave_exit_counts"] == [1, 1, 1, 1]

    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-1-json", type=Path, required=True)
    parser.add_argument("--process-2-json", type=Path, required=True)
    parser.add_argument("--process-1-csv", type=Path, required=True)
    parser.add_argument("--process-2-csv", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--p0-json", type=Path, required=True)
    args = parser.parse_args()

    process_1 = load(args.process_1_json)
    process_2 = load(args.process_2_json)

    json_equal = (
        args.process_1_json.read_bytes()
        == args.process_2_json.read_bytes()
    )
    csv_equal = (
        args.process_1_csv.read_bytes()
        == args.process_2_csv.read_bytes()
    )
    data_equal = process_1 == process_2

    passed = json_equal and csv_equal and data_equal

    consolidated = {
        "marker": MARKER,
        "decision": (
            "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
            if passed
            else "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_FAIL"
        ),
        "p0_json": str(args.p0_json.resolve()),
        "p0_json_sha256": sha256(args.p0_json),
        "fresh_process_reproducibility": {
            "json_identical": json_equal,
            "csv_identical": csv_equal,
            "parsed_data_identical": data_equal,
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
            "fresh_process_json_identical": json_equal,
            "fresh_process_csv_identical": csv_equal,
            "fresh_process_parsed_data_identical": data_equal,
        },
    }

    args.evidence.mkdir(parents=True, exist_ok=True)

    consolidated_path = (
        args.evidence / "phase4a1_p1_width64_single_layer.json"
    )
    consolidated_path.write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n"
    )

    shutil.copy2(
        args.process_1_csv,
        args.evidence / "phase4a1_p1_stage_outputs.csv",
    )

    stages = process_1["stages"]

    markdown = (
        "# Phase 4A1-P1 — Width-64 single-layer four-K-tile accumulation\n\n"
        f"Decision: **`{consolidated['decision']}`**\n\n"
        "- Target: `gfx1201`, Wave32\n"
        "- Block: `4` waves / `128` threads\n"
        "- Output: `16×64` FP32\n"
        "- Each wave owns one `16×16` output tile\n"
        "- Each output tile accumulates four ordered `16×16×16` K tiles\n"
        "- GPU launches: `1`\n"
        "- CPU reference: FP64 from exact P0 FP16 tensors\n"
        "- Fresh process JSON identical: "
        f"`{json_equal}`\n"
        "- Fresh process CSV identical: "
        f"`{csv_equal}`\n\n"
        "## Partial accumulation stages\n\n"
        + "\n".join(
            (
                f"- Stage {stage['stage']}: "
                f"`{stage['k_terms']}` K terms, "
                f"max abs `{stage['max_abs']}`, "
                f"normalized L2 `{stage['normalized_l2']}`, "
                f"PASS `{stage['passed']}`"
            )
            for stage in stages
        )
        + "\n"
    )

    (
        args.evidence / "PHASE4A1_P1_REPORT.md"
    ).write_text(markdown)

    if not passed:
        print(
            "WIDTH64_SINGLE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: FAIL"
        )
        print("PHASE4A1_P1_WIDTH64_SINGLE_LAYER: FAIL")
        return 1

    print(
        "WIDTH64_SINGLE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS"
    )
    print("PHASE4A1_P1_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A1_P1_WIDTH64_SINGLE_LAYER: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
