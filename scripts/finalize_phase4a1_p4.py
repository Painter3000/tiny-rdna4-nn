#!/usr/bin/env python3
"""Finalize Phase 4A1-P4 fresh-process evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P4_WIDTH64_THREE_LAYER_FUSED_001"
PROCESS_DECISION = "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_PASS"
FINAL_DECISION = (
    "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_CONSOLIDATED_PASS"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    assert data["marker"] == MARKER
    assert data["decision"] == PROCESS_DECISION
    assert all(bool(value) for value in data["gates"].values())

    context = data["context"]
    assert context["arch"].startswith("gfx1201")
    assert context["warp_size"] == 32
    assert context["threads_per_block"] == 128
    assert context["waves_per_block"] == 4
    assert context["k_tiles_per_layer"] == 4
    assert context["accumulator_slots_per_lane"] == 8
    assert context["matrix_a_slots_per_lane"] == 8

    topology = data["topology"]
    assert topology["layers"] == 3
    assert topology["mma_sync_calls_per_wave"] == 12
    assert topology["mma_sync_calls_per_block"] == 48
    assert topology["kernel_launches"] == 1
    assert topology["lds_buffers"] == 1
    assert topology["lds_bytes"] == 2048
    assert topology["barriers"] == 3
    assert topology["hidden_1_transport"] == "LDS_only"
    assert topology["hidden_2_transport"] == (
        "same_LDS_buffer_after_read_complete_barrier"
    )
    assert not topology["hidden_1_global_store"]
    assert not topology["hidden_1_global_reload"]
    assert not topology["hidden_2_global_store"]
    assert not topology["hidden_2_global_reload"]

    diagnostics = data["diagnostics"]
    assert diagnostics["wave_entry_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_1_publication_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_1_cross_wave_check_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_2_read_complete_counts"] == [1, 1, 1, 1]
    assert diagnostics["mapping_error_count"] == 0
    assert diagnostics["hidden_1_lds_mismatch_count"] == 0
    assert diagnostics["single_lds_buffer_count"] == 1
    assert diagnostics["layer_1_cross_wave_sources"] == [1, 2, 3, 0]
    assert diagnostics["layer_2_k_tile_counts"] == [4, 4, 4, 4]
    assert diagnostics["layer_2_overwrite_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_2_cross_wave_check_counts"] == [1, 1, 1, 1]
    assert diagnostics["hidden_2_lds_mismatch_count"] == 0
    assert diagnostics["layer_3_k_tile_counts"] == [4, 4, 4, 4]
    assert diagnostics["final_output_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_2_cross_wave_sources"] == [1, 2, 3, 0]

    metrics = data["metrics"]
    assert metrics["output_nonfinite_count"] == 0
    assert metrics["output_positive_count"] > 0
    assert metrics["output_negative_count"] > 0
    assert metrics["output_max_abs"] <= metrics["max_abs_tolerance"]
    assert metrics["output_normalized_l2"] <= (
        metrics["normalized_l2_tolerance"]
    )
    assert metrics["relay_moved_entries_per_tile"] == 240
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-1-json", type=Path, required=True)
    parser.add_argument("--process-2-json", type=Path, required=True)
    parser.add_argument("--process-1-csv", type=Path, required=True)
    parser.add_argument("--process-2-csv", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--p0-json", type=Path, required=True)
    parser.add_argument("--p3-json", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    process_1 = load_result(args.process_1_json)
    process_2 = load_result(args.process_2_json)
    preparation = json.loads(args.preparation_manifest.read_text())
    p0 = json.loads(args.p0_json.read_text())
    p3 = json.loads(args.p3_json.read_text())

    assert preparation["decision"] == "PHASE4A1_P4_PREPARATION_PASS"
    contract = preparation["contract"]
    assert contract["layers"] == 3
    assert contract["lds_buffers"] == 1
    assert contract["barriers"] == 3
    assert contract["hidden_1_transport"] == "single_LDS_buffer"
    assert contract["hidden_2_transport"] == (
        "same_single_LDS_buffer_after_safe_reuse"
    )
    assert not contract["hidden_1_global_store"]
    assert not contract["hidden_1_global_reload"]
    assert not contract["hidden_2_global_store"]
    assert not contract["hidden_2_global_reload"]

    assert p0["decision"] == (
        "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
    )
    assert all(p0["gates"].values())
    assert p3["decision"] == (
        "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_CONSOLIDATED_PASS"
    )
    assert all(p3["gates"].values())

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
            else "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_CONSOLIDATED_FAIL"
        ),
        "preparation_manifest": {
            "path": str(args.preparation_manifest.resolve()),
            "sha256": sha256(args.preparation_manifest),
            "data": preparation,
        },
        "p0_oracle": {
            "path": str(args.p0_json.resolve()),
            "sha256": sha256(args.p0_json),
            "output_statistics": p0["cpu_oracle"]["output_statistics"],
        },
        "p3_evidence": {
            "path": str(args.p3_json.resolve()),
            "sha256": sha256(args.p3_json),
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
            "p3_two_layer_prerequisite": True,
            "single_lds_reuse_proven": True,
            "three_layer_output_vs_cpu_fp64": True,
            "fresh_process_json_identical": json_identical,
            "fresh_process_csv_identical": csv_identical,
            "fresh_process_parsed_data_identical": parsed_identical,
        },
    }

    args.evidence.mkdir(parents=True, exist_ok=True)
    output_json = (
        args.evidence / "phase4a1_p4_width64_three_layer_fused.json"
    )
    output_json.write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(
        args.process_1_csv,
        args.evidence / "phase4a1_p4_final_output.csv",
    )

    metrics = process_1["metrics"]
    markdown = (
        "# Phase 4A1-P4 — Width-64 single-LDS reuse and three-layer fused forward\n\n"
        f"Decision: **`{consolidated['decision']}`**\n\n"
        "- Target: `gfx1201`, Wave32\n"
        "- Block: `4` waves / `128` threads\n"
        "- Layers: `3`\n"
        "- K tiles per layer and wave: `4`\n"
        "- rocWMMA operations: `12` per wave / `48` per block\n"
        "- LDS buffers: `1` (`2048` bytes)\n"
        "- Barriers: `3`\n"
        "- Hidden 1 and hidden 2 global store/reload: none\n"
        "- Layer 2 read-complete barrier before LDS overwrite: PASS\n"
        "- Hidden 2 published through the same physical LDS buffer: PASS\n"
        f"- Final output nonfinite count: `{metrics['output_nonfinite_count']}`\n"
        f"- Final output max abs: `{metrics['output_max_abs']}`\n"
        f"- Final output normalized L2: `{metrics['output_normalized_l2']}`\n"
        f"- Fresh-process JSON identical: `{json_identical}`\n"
        f"- Fresh-process CSV identical: `{csv_identical}`\n"
    )
    (args.evidence / "PHASE4A1_P4_REPORT.md").write_text(markdown)

    if not passed:
        print("WIDTH64_THREE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: FAIL")
        print("PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED: FAIL")
        return 1

    print("WIDTH64_THREE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("PHASE4A1_P4_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
