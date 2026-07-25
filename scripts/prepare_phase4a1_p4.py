#!/usr/bin/env python3
"""Prepare Phase 4A1-P4 and bind it to exact prerequisite evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P4_WIDTH64_THREE_LAYER_FUSED_001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-json", type=Path, required=True)
    parser.add_argument("--p1-json", type=Path, required=True)
    parser.add_argument("--p2-json", type=Path, required=True)
    parser.add_argument("--p3-json", type=Path, required=True)
    parser.add_argument("--p3-preparation", type=Path, required=True)
    parser.add_argument("--p3-header", type=Path, required=True)
    parser.add_argument("--tensor-hashes", type=Path, required=True)
    parser.add_argument("--input-bin", type=Path, required=True)
    parser.add_argument("--weight-1-bin", type=Path, required=True)
    parser.add_argument("--weight-2-bin", type=Path, required=True)
    parser.add_argument("--weight-3-bin", type=Path, required=True)
    parser.add_argument("--bias-1-bin", type=Path, required=True)
    parser.add_argument("--bias-2-bin", type=Path, required=True)
    parser.add_argument("--bias-3-bin", type=Path, required=True)
    parser.add_argument("--expected-hidden-1-bin", type=Path, required=True)
    parser.add_argument("--expected-hidden-2-bin", type=Path, required=True)
    parser.add_argument("--expected-output-bin", type=Path, required=True)
    parser.add_argument("--output-header", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    p0 = load(args.p0_json)
    p1 = load(args.p1_json)
    p2 = load(args.p2_json)
    p3 = load(args.p3_json)
    p3_preparation = load(args.p3_preparation)
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

    assert p3["decision"] == (
        "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_CONSOLIDATED_PASS"
    )
    assert all(bool(value) for value in p3["gates"].values())
    assert p3["p0_oracle"]["sha256"] == sha256(args.p0_json)
    assert p3["p2_evidence"]["sha256"] == sha256(args.p2_json)
    assert p3["result"]["decision"] == (
        "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_PASS"
    )
    assert all(bool(value) for value in p3["result"]["gates"].values())
    assert p3["result"]["diagnostics"]["hidden_1_lds_mismatch_count"] == 0
    assert p3["result"]["metrics"]["hidden_2_mismatch_count"] == 0

    assert p3_preparation["decision"] == "PHASE4A1_P3_PREPARATION_PASS"
    assert p3["preparation_manifest"]["sha256"] == sha256(
        args.p3_preparation
    )
    assert p3_preparation["copied_mapping_header"]["sha256"] == sha256(
        args.p3_header
    )

    expected_sizes = {
        args.input_bin: 16 * 64 * 2,
        args.weight_1_bin: 64 * 64 * 2,
        args.weight_2_bin: 64 * 64 * 2,
        args.weight_3_bin: 64 * 64 * 2,
        args.bias_1_bin: 64 * 4,
        args.bias_2_bin: 64 * 4,
        args.bias_3_bin: 64 * 4,
        args.expected_hidden_1_bin: 16 * 64 * 2,
        args.expected_hidden_2_bin: 16 * 64 * 2,
        args.expected_output_bin: 16 * 64 * 8,
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
        args.weight_3_bin: hashes[
            "weight_3_fp16_physical_col_major"
        ]["sha256"],
        args.bias_1_bin: hashes["bias_1_fp32"]["sha256"],
        args.bias_2_bin: hashes["bias_2_fp32"]["sha256"],
        args.bias_3_bin: hashes["bias_3_fp32"]["sha256"],
        args.expected_hidden_1_bin: hashes[
            "hidden_1_fp16_row_major"
        ]["sha256"],
        args.expected_hidden_2_bin: hashes[
            "hidden_2_fp16_row_major"
        ]["sha256"],
        args.expected_output_bin: hashes[
            "output_fp64_row_major"
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

    output_stats = p0["cpu_oracle"]["output_statistics"]
    assert output_stats["nonfinite_count"] == 0
    assert output_stats["minimum"] < 0.0
    assert output_stats["maximum"] > 0.0

    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    bindings = {
        "p0": sha256(args.p0_json),
        "p1": sha256(args.p1_json),
        "p2": sha256(args.p2_json),
        "p3": sha256(args.p3_json),
        "p3_preparation": sha256(args.p3_preparation),
        "p3_header": sha256(args.p3_header),
    }

    header_text = args.p3_header.read_text()
    header_text += f"""\nnamespace phase4a1_p4_generated {{\n\nconstexpr const char* kMarker = "{MARKER}";\nconstexpr const char* kP0Sha256 = "{bindings["p0"]}";\nconstexpr const char* kP1Sha256 = "{bindings["p1"]}";\nconstexpr const char* kP2Sha256 = "{bindings["p2"]}";\nconstexpr const char* kP3Sha256 = "{bindings["p3"]}";\nconstexpr const char* kP3PreparationSha256 = "{bindings["p3_preparation"]}";\nconstexpr const char* kP3HeaderSha256 = "{bindings["p3_header"]}";\n\n}} // namespace phase4a1_p4_generated\n"""
    args.output_header.write_text(header_text)

    manifest = {
        "marker": MARKER,
        "decision": "PHASE4A1_P4_PREPARATION_PASS",
        "prerequisites": {
            "p0_json": str(args.p0_json.resolve()),
            "p0_sha256": bindings["p0"],
            "p1_json": str(args.p1_json.resolve()),
            "p1_sha256": bindings["p1"],
            "p2_json": str(args.p2_json.resolve()),
            "p2_sha256": bindings["p2"],
            "p3_json": str(args.p3_json.resolve()),
            "p3_sha256": bindings["p3"],
            "p3_preparation": str(args.p3_preparation.resolve()),
            "p3_preparation_sha256": bindings["p3_preparation"],
            "p3_header": str(args.p3_header.resolve()),
            "p3_header_sha256": bindings["p3_header"],
        },
        "tensors": {
            path.name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in hash_bindings
        },
        "generated_header": {
            "path": str(args.output_header.resolve()),
            "sha256": sha256(args.output_header),
        },
        "contract": {
            "batch_rows": 16,
            "width": 64,
            "layers": 3,
            "waves_per_block": 4,
            "wave_size": 32,
            "threads_per_block": 128,
            "k_tiles_per_layer": 4,
            "mma_sync_calls_per_wave": 12,
            "mma_sync_calls_per_block": 48,
            "lds_buffers": 1,
            "lds_bytes": 2048,
            "barriers": 3,
            "hidden_1_transport": "single_LDS_buffer",
            "hidden_2_transport": "same_single_LDS_buffer_after_safe_reuse",
            "hidden_1_global_store": False,
            "hidden_1_global_reload": False,
            "hidden_2_global_store": False,
            "hidden_2_global_reload": False,
            "final_output": "FP32_global_store_after_bias_3",
        },
        "p0_layer_1_stats": layer_1,
        "p0_layer_2_stats": layer_2,
        "p0_output_statistics": output_stats,
    }

    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("p0_sha256: " + bindings["p0"])
    print("p1_sha256: " + bindings["p1"])
    print("p2_sha256: " + bindings["p2"])
    print("p3_sha256: " + bindings["p3"])
    print("generated_header_sha256: " + sha256(args.output_header))
    print("PHASE4A1_P4_PREPARATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
