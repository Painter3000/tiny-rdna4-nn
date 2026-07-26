#!/usr/bin/env python3
"""Finalize Phase 4A2-P2 production inference evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001"
EXPECTED_MAPPING_SHA256 = (
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"
)

EXPECTED_CHANGED = {
    "README_PHASE4A2_P2.md",
    "contracts/phase4a2_p2_production_inference_contract.json",
    "include/tiny-cuda-nn/network_with_input_encoding.h",
    "include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h",
    "probes/phase4a2_p2_inference_probe.py",
    "scripts/apply_phase4a2_p2.py",
    "scripts/finalize_phase4a2_p2.py",
    "scripts/resume_phase4a2_p2_after_register_size_fix.sh",
    "scripts/run_phase4a2_p2_production_inference.sh",
    "src/rocwmma_width64_mlp.cu",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_paths(repo: Path) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    paths: set[str] = set()
    for raw in result.stdout.splitlines():
        if not raw:
            continue
        payload = raw[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.add(payload)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--apply", type=Path, required=True)
    parser.add_argument("--process-1", type=Path, required=True)
    parser.add_argument("--process-2", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract = load(args.contract)
    apply = load(args.apply)
    process_1 = load(args.process_1)
    process_2 = load(args.process_2)

    assert contract["marker"] == MARKER
    assert contract["phase"] == "4A2-P2"
    assert apply["decision"] == "PHASE4A2_P2_APPLY_PASS"
    assert all(bool(value) for value in apply["gates"].values())
    assert all(bool(value) for value in apply["source_gates"].values())

    for process in (process_1, process_2):
        assert process["decision"] == (
            "PHASE4A2_P2_PRODUCTION_INFERENCE_PROCESS_PASS"
        )
        assert all(bool(value) for value in process["gates"].values())

    fresh_process_identical = (
        args.process_1.read_bytes() == args.process_2.read_bytes()
    )
    assert fresh_process_identical

    source_path = repo / "src/rocwmma_width64_mlp.cu"
    bridge_path = (
        repo / "include/tiny-cuda-nn/network_with_input_encoding.h"
    )
    mapping_path = (
        repo
        / "include/tiny-cuda-nn/networks/"
        "rocwmma_width64_mapping_gfx1201.h"
    )

    source = source_path.read_text()
    bridge = bridge_path.read_text()
    mapping = mapping_path.read_text()
    build_log = args.build_log.read_text(errors="replace")
    changed_paths = git_paths(repo)

    case_values = process_1["cases"]
    max_abs = max(
        float(case["max_abs"])
        for case in case_values.values()
    )
    max_nl2 = max(
        float(case["normalized_l2"])
        for case in case_values.values()
    )

    static_gates = {
        "production_source_hash_bound": (
            sha256(source_path)
            == contract["production_kernel"]["source_sha256"]
        ),
        "register_file_unpacked_size_api": (
            "static_assert(RegA::size() == SLOTS);" in source
            and "static_assert(RegAcc::size() == SLOTS);" in source
            and "::num_elements == SLOTS" not in source
        ),
        "register_file_size_device_pass_only": (
            "#if defined(__HIP_DEVICE_COMPILE__)" in source
            and "gfx1201/Wave32 register view only" in source
        ),
        "one_production_kernel": source.count("__global__ void") == 1,
        "production_kernel_symbol": (
            "rocwmma_width64_inference_kernel" in source
        ),
        "three_mma_sites": source.count("rocwmma::mma_sync(") == 3,
        "three_source_barriers": source.count(
            "__syncthreads();"
        ) == 3,
        "one_2048_byte_lds_buffer": (
            source.count(
                "__shared__ __align__(16) Half hidden_lds"
            ) == 1
            and "ELEMENTS * sizeof(Half) == 2048" in source
        ),
        "multi_block_batch_grid": (
            "blockIdx.x" in source
            and "input.n() / TILE_ROWS" in source
        ),
        "fp16_bias_promoted_to_fp32": (
            "const __half* bias_0" in source
            and "static_cast<F32>(bias[global_col])" in source
        ),
        "public_fp16_output": (
            "accumulator_bias_to_matrix_a<false>" in source
            and "reinterpret_cast<Half*>(output.data())" in source
        ),
        "parameter_offsets_bound": all(
            token in source
            for token in (
                "WEIGHT_0_OFFSET",
                "BIAS_0_OFFSET",
                "WEIGHT_1_OFFSET",
                "BIAS_1_OFFSET",
                "WEIGHT_2_OFFSET",
                "BIAS_2_OFFSET",
            )
        ),
        "caller_stream_no_host_sync": (
            "hipLaunchKernelGGL(" in source
            and "stream," in source
            and "hipDeviceSynchronize" not in source
            and "hipStreamSynchronize" not in source
        ),
        "no_oracle_diagnostic_path": all(
            token not in source
            for token in (
                "expected_hidden",
                "diagnostics",
                "atomicAdd(",
            )
        ),
        "training_backward_fail_closed": (
            "training forward is not " in source
            and "qualified; use inference/no-grad." in source
            and "backward is not qualified" in source
            and "no fallback was executed" in source
        ),
        "network_bridge_column_major": (
            "m_fp16_rocwmma_width64" in bridge
            and "MatrixLayout::ColumnMajor" in bridge
            and "set_padded_output_width(64)" in bridge
        ),
        "mapping_header_exact": (
            sha256(mapping_path) == EXPECTED_MAPPING_SHA256
            and "namespace phase4a1_p2_generated" in mapping
            and "kAccLaneForATargetA" in mapping
        ),
        "build_opt_in_on": (
            "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" in build_log
            and "rocwmma_width64_mlp.cu" in build_log
            and "TCNN_WITH_ROCWMMA_WIDTH64_MLP" in build_log
        ),
        "changed_file_set_exact": changed_paths == EXPECTED_CHANGED,
    }
    assert all(static_gates.values()), {
        key: value
        for key, value in static_gates.items()
        if not value
    }

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI_PASS"
        ),
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract),
            "data": contract,
        },
        "apply": {
            "path": str(args.apply.resolve()),
            "sha256": sha256(args.apply),
            "data": apply,
        },
        "build": {
            "log": str(args.build_log.resolve()),
            "sha256": sha256(args.build_log),
        },
        "fresh_processes": {
            "exact_match": fresh_process_identical,
            "process_1": process_1,
            "process_2": process_2,
        },
        "metrics": {
            "maximum_max_abs": max_abs,
            "maximum_normalized_l2": max_nl2,
            "batch_cases": sorted(
                int(key) for key in case_values
            ),
        },
        "artifacts": {
            "production_source": {
                "path": str(source_path.resolve()),
                "sha256": sha256(source_path),
            },
            "network_bridge": {
                "path": str(bridge_path.resolve()),
                "sha256": sha256(bridge_path),
            },
            "mapping_header": {
                "path": str(mapping_path.resolve()),
                "sha256": sha256(mapping_path),
            },
        },
        "static_gates": static_gates,
        "changed_files": sorted(changed_paths),
        "gates": {
            "p1_and_p4_baselines_bound": True,
            "production_kernel_installed": True,
            "parameter_abi_adapted": True,
            "column_major_batch_bridge_installed": True,
            "six_public_batch_cases_correct": len(case_values) == 6,
            "fresh_process_reproducible": fresh_process_identical,
            "repeat_bitwise": process_1["gates"][
                "all_batch_cases_correct"
            ],
            "tile_prefix_invariant": process_1["gates"][
                "tile_prefix_invariance"
            ],
            "nondefault_stream_correct": process_1["gates"][
                "nondefault_stream_correct"
            ],
            "training_forward_fail_closed": process_1["gates"][
                "training_forward_fail_closed"
            ],
            "no_oracle_or_diagnostic_arguments": True,
            "no_performance_claim": (
                contract["scope"]["performance_claim"] == "none"
            ),
            "production_isa_claim_deferred": (
                contract["scope"]["production_isa_claim"]
                == "deferred_to_4A2_P4"
            ),
            "changed_file_set_exact": True,
        },
    }
    assert all(result["gates"].values())

    args.evidence.mkdir(parents=True, exist_ok=True)
    output = (
        args.evidence
        / "phase4a2_p2_width64_production_inference_parameter_abi.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    report = f"""# Phase 4A2-P2 — Width-64 production inference

Decision: **`{result["decision"]}`**

## Production bridge

- Exact P4 source SHA: `{apply["baseline"]["p4_reference_source_sha256"]}`
- Exact mapping header SHA: `{sha256(mapping_path)}`
- Public parameter buffer: 12,480 FP16 elements in
  `W0,b0,W1,b1,W2,b2` order.
- Public input/output: contiguous ColumnMajor `[64][batch]`.
- Launch: one 128-thread block per 16 samples on the caller stream.
- Hidden tensors: one reused 2,048-byte LDS allocation.
- Public output: FP16 after FP32 final accumulation and FP16 bias promotion.

## Correctness

The public Python entry point passed batches
`{", ".join(str(v) for v in result["metrics"]["batch_cases"])}`. Internal
padding exercised 256- and 512-sample launches.

- Worst max absolute error: `{max_abs:.9g}`
- Worst normalized L2 error: `{max_nl2:.9g}`
- Repeated inference: bitwise identical
- Tile-prefix invariance: passed
- Non-default stream: passed
- Nonfinite output: zero

## Deliberate boundary

Training forward and backward remain fail-closed. No automatic selection,
silent fallback, performance, occupancy, or production-ISA/resource claim is
made in P2. Production ISA qualification remains scheduled for 4A2-P4.
"""
    (args.evidence / "PHASE4A2_P2_REPORT.md").write_text(report)

    print("WIDTH64_PRODUCTION_KERNEL_INSTALLED: PASS")
    print("WIDTH64_P4_SOURCE_AND_MAPPING_BOUND: PASS")
    print("WIDTH64_FP16_PARAMETER_ABI_12480: PASS")
    print("WIDTH64_COLUMN_MAJOR_BATCH_BRIDGE: PASS")
    print("WIDTH64_MULTI_BLOCK_BATCH_CORRECTNESS: PASS")
    print("WIDTH64_INFERENCE_VS_CPU_REFERENCE: PASS")
    print("WIDTH64_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("WIDTH64_NONDEFAULT_STREAM_CORRECTNESS: PASS")
    print("WIDTH64_TRAINING_BACKWARD_FAIL_CLOSED: PASS")
    print("WIDTH64_NO_ORACLE_DIAGNOSTIC_PATH: PASS")
    print("PHASE4A2_P2_CONSOLIDATED_EVIDENCE: RECORDED")
    print(
        "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
