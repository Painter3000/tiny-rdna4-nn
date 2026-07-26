#!/usr/bin/env python3
"""Prepare and bind Phase 4A1-P3 to exact prerequisite evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P3_WIDTH64_TWO_LAYER_FUSED_001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-json", type=Path, required=True)
    parser.add_argument("--p1-json", type=Path, required=True)
    parser.add_argument("--p2-json", type=Path, required=True)
    parser.add_argument("--p2-manifest", type=Path, required=True)
    parser.add_argument("--p2-header", type=Path, required=True)
    parser.add_argument("--input-bin", type=Path, required=True)
    parser.add_argument("--weight-1-bin", type=Path, required=True)
    parser.add_argument("--weight-2-bin", type=Path, required=True)
    parser.add_argument("--bias-1-bin", type=Path, required=True)
    parser.add_argument("--bias-2-bin", type=Path, required=True)
    parser.add_argument("--expected-hidden-1-bin", type=Path, required=True)
    parser.add_argument("--expected-hidden-2-bin", type=Path, required=True)
    parser.add_argument("--tensor-hashes", type=Path, required=True)
    parser.add_argument("--output-header", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    p0 = load(args.p0_json)
    p1 = load(args.p1_json)
    p2 = load(args.p2_json)
    p2_manifest = load(args.p2_manifest)
    hashes = load(args.tensor_hashes)

    assert p0["decision"] == (
        "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
    )
    assert all(bool(value) for value in p0["gates"].values())

    assert p1["decision"] == (
        "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
    )
    assert all(bool(value) for value in p1["gates"].values())
    assert p1["p0_json_sha256"] == sha256(args.p0_json)

    assert p2["decision"] == (
        "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_CONSOLIDATED_PASS"
    )
    assert all(bool(value) for value in p2["gates"].values())
    assert p2["p0_oracle"]["sha256"] == sha256(args.p0_json)
    assert p2["p1_evidence"]["sha256"] == sha256(args.p1_json)
    assert p2["result"]["decision"] == (
        "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_PASS"
    )
    assert all(bool(value) for value in p2["result"]["gates"].values())
    assert p2["result"]["metrics"]["mismatch_count"] == 0

    assert p2_manifest["decision"] == (
        "PHASE4A1_P2_MAPPING_HEADER_GENERATION_PASS"
    )
    assert p2_manifest["mapping"]["entries"] == 256
    assert p2_manifest["mapping"]["moved_entries"] == 240
    assert p2_manifest["generated_header_sha256"] == sha256(args.p2_header)
    assert p2["manifest"]["sha256"] == sha256(args.p2_manifest)

    expected_sizes = {
        args.input_bin: 16 * 64 * 2,
        args.weight_1_bin: 64 * 64 * 2,
        args.weight_2_bin: 64 * 64 * 2,
        args.bias_1_bin: 64 * 4,
        args.bias_2_bin: 64 * 4,
        args.expected_hidden_1_bin: 16 * 64 * 2,
        args.expected_hidden_2_bin: 16 * 64 * 2,
    }
    for path, size in expected_sizes.items():
        assert path.stat().st_size == size, (path, path.stat().st_size, size)

    hash_bindings = {
        args.input_bin: hashes["input_fp16_row_major"]["sha256"],
        args.weight_1_bin: hashes[
            "weight_1_fp16_physical_col_major"
        ]["sha256"],
        args.weight_2_bin: hashes[
            "weight_2_fp16_physical_col_major"
        ]["sha256"],
        args.bias_1_bin: hashes["bias_1_fp32"]["sha256"],
        args.bias_2_bin: hashes["bias_2_fp32"]["sha256"],
        args.expected_hidden_1_bin: hashes[
            "hidden_1_fp16_row_major"
        ]["sha256"],
        args.expected_hidden_2_bin: hashes[
            "hidden_2_fp16_row_major"
        ]["sha256"],
    }
    for path, expected_hash in hash_bindings.items():
        assert sha256(path) == expected_hash, path

    layer_1 = p0["cpu_oracle"]["layer_1_stats"]
    layer_2 = p0["cpu_oracle"]["layer_2_stats"]
    for stats in (layer_1, layer_2):
        assert stats["relu_clamped"] > 0
        assert stats["relu_positive"] > 0
        assert stats["fp16_cast_changed"] > 0

    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.p2_header, args.output_header)

    manifest = {
        "marker": MARKER,
        "decision": "PHASE4A1_P3_PREPARATION_PASS",
        "prerequisites": {
            "p0_json": str(args.p0_json.resolve()),
            "p0_sha256": sha256(args.p0_json),
            "p1_json": str(args.p1_json.resolve()),
            "p1_sha256": sha256(args.p1_json),
            "p2_json": str(args.p2_json.resolve()),
            "p2_sha256": sha256(args.p2_json),
            "p2_manifest": str(args.p2_manifest.resolve()),
            "p2_manifest_sha256": sha256(args.p2_manifest),
            "p2_header": str(args.p2_header.resolve()),
            "p2_header_sha256": sha256(args.p2_header),
        },
        "tensors": {
            path.name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in hash_bindings
        },
        "copied_mapping_header": {
            "path": str(args.output_header.resolve()),
            "sha256": sha256(args.output_header),
        },
        "contract": {
            "batch_rows": 16,
            "width": 64,
            "layers": 2,
            "waves_per_block": 4,
            "wave_size": 32,
            "threads_per_block": 128,
            "k_tiles_per_layer": 4,
            "mma_sync_calls_per_wave": 8,
            "mma_sync_calls_per_block": 32,
            "hidden_1_transport": "LDS_only",
            "hidden_1_global_store": False,
            "hidden_1_global_reload": False,
            "final_output": "hidden_2_fp16_diagnostic_store",
        },
        "p0_layer_1_stats": layer_1,
        "p0_layer_2_stats": layer_2,
    }

    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("p0_sha256: " + manifest["prerequisites"]["p0_sha256"])
    print("p1_sha256: " + manifest["prerequisites"]["p1_sha256"])
    print("p2_sha256: " + manifest["prerequisites"]["p2_sha256"])
    print(
        "mapping_header_sha256: "
        + manifest["copied_mapping_header"]["sha256"]
    )
    print("PHASE4A1_P3_PREPARATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
