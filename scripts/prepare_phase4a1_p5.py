#!/usr/bin/env python3
"""Bind Phase 4A1-P5 to the exact validated P4 source and evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P5_ISA_RESOURCE_GLOBAL_TRAFFIC_001"
P4_DECISION = "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_CONSOLIDATED_PASS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def hashes_for_basename(path: Path, basename: str) -> set[str]:
    results: set[str] = set()
    for raw_line in path.read_text().splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, recorded_path = parts
        if Path(recorded_path.lstrip("*")).name == basename:
            results.add(digest)
    return results


def extract_kernel(source_text: str, kernel_name: str) -> str:
    anchor = source_text.find(f"__global__ void {kernel_name}")
    assert anchor >= 0, kernel_name
    opening = source_text.find("{", anchor)
    assert opening >= 0
    depth = 0
    for index in range(opening, len(source_text)):
        char = source_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source_text[anchor:index + 1]
    raise AssertionError("unterminated kernel body")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4-json", type=Path, required=True)
    parser.add_argument("--p4-preparation", type=Path, required=True)
    parser.add_argument("--p4-header", type=Path, required=True)
    parser.add_argument("--p4-sha256s", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    p4 = load(args.p4_json)
    preparation = load(args.p4_preparation)

    assert p4["decision"] == P4_DECISION
    assert all(bool(value) for value in p4["gates"].values())
    assert p4["result"]["decision"] == (
        "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_PASS"
    )
    assert all(bool(value) for value in p4["result"]["gates"].values())

    topology = p4["result"]["topology"]
    assert topology["layers"] == 3
    assert topology["kernel_launches"] == 1
    assert topology["lds_buffers"] == 1
    assert topology["lds_bytes"] == 2048
    assert topology["barriers"] == 3
    assert topology["mma_sync_calls_per_wave"] == 12
    assert topology["mma_sync_calls_per_block"] == 48
    assert topology["hidden_1_transport"] == "LDS_only"
    assert topology["hidden_2_transport"] == (
        "same_LDS_buffer_after_read_complete_barrier"
    )
    assert not topology["hidden_1_global_store"]
    assert not topology["hidden_1_global_reload"]
    assert not topology["hidden_2_global_store"]
    assert not topology["hidden_2_global_reload"]

    assert preparation["decision"] == "PHASE4A1_P4_PREPARATION_PASS"
    assert p4["preparation_manifest"]["sha256"] == sha256(
        args.p4_preparation
    )
    assert preparation["generated_header"]["sha256"] == sha256(
        args.p4_header
    )

    source_hashes = hashes_for_basename(
        args.p4_sha256s,
        args.source.name,
    )
    assert source_hashes == {sha256(args.source)}, source_hashes

    header_hashes = hashes_for_basename(
        args.p4_sha256s,
        args.p4_header.name,
    )
    assert sha256(args.p4_header) in header_hashes, header_hashes

    source_text = args.source.read_text()
    kernel = extract_kernel(
        source_text,
        "width64_three_layer_fused_kernel",
    )

    source_contract = {
        "global_kernel_declarations": source_text.count(
            "__global__ void width64_three_layer_fused_kernel"
        ),
        "single_hidden_lds_declarations": kernel.count(
            "__shared__ __align__(16) Half hidden_lds[ELEMENTS]"
        ),
        "block_barriers": kernel.count("__syncthreads()"),
        "hidden_lds_matrix_stores": kernel.count(
            "hidden_lds + output_col_begin,"
        ),
        "layer_2_3_lds_matrix_inputs": kernel.count(
            "const Half* a_tile = hidden_lds + k_tile * TILE;"
        ),
        "final_output_epilogue_calls": kernel.count(
            "accumulator_bias_to_fp32_output("
        ),
        "expected_hidden_oracle_arguments": (
            kernel.count("const Half* expected_hidden_1")
            + kernel.count("const Half* expected_hidden_2")
        ),
        "explicit_hidden_global_output_arguments": (
            kernel.count("Half* hidden_1_output")
            + kernel.count("Half* hidden_2_output")
            + kernel.count("Half* hidden_output")
        ),
    }

    assert source_contract == {
        "global_kernel_declarations": 1,
        "single_hidden_lds_declarations": 1,
        "block_barriers": 3,
        "hidden_lds_matrix_stores": 2,
        "layer_2_3_lds_matrix_inputs": 2,
        "final_output_epilogue_calls": 1,
        "expected_hidden_oracle_arguments": 2,
        "explicit_hidden_global_output_arguments": 0,
    }

    manifest = {
        "marker": MARKER,
        "decision": "PHASE4A1_P5_PREPARATION_PASS",
        "p4": {
            "json": str(args.p4_json.resolve()),
            "json_sha256": sha256(args.p4_json),
            "preparation": str(args.p4_preparation.resolve()),
            "preparation_sha256": sha256(args.p4_preparation),
            "header": str(args.p4_header.resolve()),
            "header_sha256": sha256(args.p4_header),
            "sha256s": str(args.p4_sha256s.resolve()),
            "sha256s_sha256": sha256(args.p4_sha256s),
        },
        "source": {
            "path": str(args.source.resolve()),
            "sha256": sha256(args.source),
            "recorded_p4_sha256": next(iter(source_hashes)),
        },
        "compile_contract": {
            "language_standard": "c++17",
            "optimization": "O2",
            "debug": "gline-tables-only",
            "offload_arch": "gfx1201",
            "device_only_companion_builds": 2,
            "functional_replay_processes": 2,
        },
        "source_contract": source_contract,
        "p4_topology": topology,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("p4_json_sha256: " + manifest["p4"]["json_sha256"])
    print("p4_source_sha256: " + manifest["source"]["sha256"])
    print("p4_header_sha256: " + manifest["p4"]["header_sha256"])
    print("PHASE4A1_P5_SOURCE_CONTRACT: PASS")
    print("PHASE4A1_P5_PREPARATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
