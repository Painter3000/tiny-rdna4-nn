#!/usr/bin/env python3
"""Finalize Phase 4A1-P3 fresh-process evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P3_WIDTH64_TWO_LAYER_FUSED_001"
PROCESS_DECISION = "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_PASS"
FINAL_DECISION = "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_CONSOLIDATED_PASS"


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
    assert topology["layers"] == 2
    assert topology["mma_sync_calls_per_wave"] == 8
    assert topology["mma_sync_calls_per_block"] == 32
    assert topology["kernel_launches"] == 1
    assert topology["lds_bytes"] == 2048
    assert topology["hidden_1_transport"] == "LDS_only"
    assert not topology["hidden_1_global_store"]
    assert not topology["hidden_1_global_reload"]
    diagnostics = data["diagnostics"]
    assert diagnostics["wave_entry_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_1_publication_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_1_cross_wave_check_counts"] == [1, 1, 1, 1]
    assert diagnostics["layer_2_output_counts"] == [1, 1, 1, 1]
    assert diagnostics["mapping_error_count"] == 0
    assert diagnostics["hidden_1_lds_mismatch_count"] == 0
    assert diagnostics["cross_wave_sources"] == [1, 2, 3, 0]
    assert diagnostics["layer_2_k_tile_counts"] == [4, 4, 4, 4]
    metrics = data["metrics"]
    assert metrics["hidden_2_mismatch_count"] == 0
    assert metrics["nonfinite_count"] == 0
    assert metrics["zero_count"] > 0
    assert metrics["positive_count"] > 0
    assert metrics["max_abs"] == 0
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
    parser.add_argument("--p2-json", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    process_1 = load_result(args.process_1_json)
    process_2 = load_result(args.process_2_json)
    preparation = json.loads(args.preparation_manifest.read_text())
    p0 = json.loads(args.p0_json.read_text())
    p2 = json.loads(args.p2_json.read_text())
    assert preparation["decision"] == "PHASE4A1_P3_PREPARATION_PASS"
    assert preparation["contract"]["hidden_1_transport"] == "LDS_only"
    assert not preparation["contract"]["hidden_1_global_store"]
    assert not preparation["contract"]["hidden_1_global_reload"]
    assert p0["decision"] == "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
    assert all(p0["gates"].values())
    assert p2["decision"] == "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_CONSOLIDATED_PASS"
    assert all(p2["gates"].values())
    layer_2_stats = p0["cpu_oracle"]["layer_2_stats"]
    assert layer_2_stats["relu_clamped"] > 0
    assert layer_2_stats["relu_positive"] > 0
    assert layer_2_stats["fp16_cast_changed"] > 0

    json_identical = args.process_1_json.read_bytes() == args.process_2_json.read_bytes()
    csv_identical = args.process_1_csv.read_bytes() == args.process_2_csv.read_bytes()
    parsed_identical = process_1 == process_2
    passed = json_identical and csv_identical and parsed_identical

    consolidated = {
        "marker": MARKER,
        "decision": FINAL_DECISION if passed else "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_CONSOLIDATED_FAIL",
        "preparation_manifest": {
            "path": str(args.preparation_manifest.resolve()),
            "sha256": sha256(args.preparation_manifest),
            "data": preparation,
        },
        "p0_oracle": {
            "path": str(args.p0_json.resolve()),
            "sha256": sha256(args.p0_json),
            "layer_2_stats": layer_2_stats,
        },
        "p2_evidence": {
            "path": str(args.p2_json.resolve()),
            "sha256": sha256(args.p2_json),
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
            "p2_hidden_lds_prerequisite": True,
            "layer_2_epilogue_exercised": True,
            "fresh_process_json_identical": json_identical,
            "fresh_process_csv_identical": csv_identical,
            "fresh_process_parsed_data_identical": parsed_identical,
        },
    }

    args.evidence.mkdir(parents=True, exist_ok=True)
    output_json = args.evidence / "phase4a1_p3_width64_two_layer_fused.json"
    output_json.write_text(json.dumps(consolidated, indent=2, sort_keys=True) + "\n")
    shutil.copy2(args.process_1_csv, args.evidence / "phase4a1_p3_hidden_2_output.csv")

    metrics = process_1["metrics"]
    markdown = (
        "# Phase 4A1-P3 — Width-64 two-layer fused forward\n\n"
        f"Decision: **`{consolidated['decision']}`**\n\n"
        "- Target: `gfx1201`, Wave32\n"
        "- Block: `4` waves / `128` threads\n"
        "- Layers: `2`\n"
        "- K tiles per layer and wave: `4`\n"
        "- rocWMMA operations: `8` per wave / `32` per block\n"
        "- Hidden-1 transport: LDS only\n"
        "- Hidden-1 global store/reload: none\n"
        "- Layer-1 publication barrier: `1`\n"
        "- Layer-1 cross-wave validation: `0→1, 1→2, 2→3, 3→0`\n"
        f"- Hidden-1 LDS mismatches: `{process_1['diagnostics']['hidden_1_lds_mismatch_count']}`\n"
        f"- Hidden-2 mismatches: `{metrics['hidden_2_mismatch_count']}`\n"
        f"- Hidden-2 nonfinite values: `{metrics['nonfinite_count']}`\n"
        f"- Hidden-2 zero values: `{metrics['zero_count']}`\n"
        f"- Hidden-2 positive values: `{metrics['positive_count']}`\n"
        f"- Fresh-process JSON identical: `{json_identical}`\n"
        f"- Fresh-process CSV identical: `{csv_identical}`\n"
    )
    (args.evidence / "PHASE4A1_P3_REPORT.md").write_text(markdown)

    if not passed:
        print("WIDTH64_TWO_LAYER_FRESH_PROCESS_REPRODUCIBILITY: FAIL")
        print("PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED: FAIL")
        return 1
    print("WIDTH64_TWO_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("PHASE4A1_P3_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
