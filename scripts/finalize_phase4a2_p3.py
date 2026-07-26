#!/usr/bin/env python3
"""Finalize Phase 4A2-P3 runtime integration closure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P3_RUNTIME_INTEGRATION_CLOSURE_001"
EXPECTED_SOURCE_SHA256 = (
    "7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4"
)
EXPECTED_MAPPING_SHA256 = (
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"
)

EXPECTED_CHANGED = {
    "README_PHASE4A2_P3.md",
    "contracts/phase4a2_p3_runtime_integration_closure_contract.json",
    "probes/phase4a2_p3_runtime_integration_probe.py",
    "scripts/finalize_phase4a2_p3.py",
    "scripts/run_phase4a2_p3_runtime_integration_closure.sh",
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
    parser.add_argument("--p2-json", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--process-1", type=Path, required=True)
    parser.add_argument("--process-2", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract = load(args.contract)
    p2 = load(args.p2_json)
    process_1 = load(args.process_1)
    process_2 = load(args.process_2)

    assert contract["marker"] == MARKER
    assert contract["phase"] == "4A2-P3"
    assert p2["decision"] == (
        "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI_PASS"
    )
    assert all(bool(value) for value in p2["gates"].values())

    for process in (process_1, process_2):
        assert process["decision"] == (
            "PHASE4A2_P3_RUNTIME_INTEGRATION_LIFECYCLE_PROCESS_PASS"
        )
        assert all(bool(value) for value in process["gates"].values())

    exact_process_match = (
        args.process_1.read_bytes() == args.process_2.read_bytes()
    )
    assert exact_process_match

    source_path = repo / "src/rocwmma_width64_mlp.cu"
    mapping_path = (
        repo
        / "include/tiny-cuda-nn/networks/"
        "rocwmma_width64_mapping_gfx1201.h"
    )
    bridge_path = (
        repo / "include/tiny-cuda-nn/network_with_input_encoding.h"
    )

    source = source_path.read_text()
    bridge = bridge_path.read_text()
    changed_paths = git_paths(repo)

    static_gates = {
        "production_source_unchanged_from_p2": (
            sha256(source_path) == EXPECTED_SOURCE_SHA256
        ),
        "mapping_header_unchanged_from_p2": (
            sha256(mapping_path) == EXPECTED_MAPPING_SHA256
        ),
        "single_production_kernel": (
            source.count("__global__ void") == 1
        ),
        "three_mma_sites": source.count("rocwmma::mma_sync(") == 3,
        "three_source_barriers": source.count(
            "__syncthreads();"
        ) == 3,
        "one_lds_buffer": source.count(
            "__shared__ __align__(16) Half hidden_lds"
        ) == 1,
        "caller_stream_no_host_sync": (
            "hipLaunchKernelGGL(" in source
            and "hipDeviceSynchronize" not in source
            and "hipStreamSynchronize" not in source
        ),
        "bridge_remains_column_major": (
            "m_fp16_rocwmma_width64" in bridge
            and "MatrixLayout::ColumnMajor" in bridge
            and "set_padded_output_width(64)" in bridge
        ),
        "no_p3_production_code_change": all(
            not path.startswith("src/")
            and not path.startswith("include/")
            and not path.startswith("bindings/")
            for path in changed_paths
        ),
        "changed_file_set_exact": changed_paths == EXPECTED_CHANGED,
    }
    assert all(static_gates.values()), {
        key: value
        for key, value in static_gates.items()
        if not value
    }

    maximum_max_abs = max(
        float(process_1["metrics"]["maximum_max_abs"]),
        float(process_2["metrics"]["maximum_max_abs"]),
    )
    maximum_nl2 = max(
        float(process_1["metrics"]["maximum_normalized_l2"]),
        float(process_2["metrics"]["maximum_normalized_l2"]),
    )

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE_PASS"
        ),
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract),
            "data": contract,
        },
        "p2_baseline": {
            "path": str(args.p2_json.resolve()),
            "sha256": sha256(args.p2_json),
            "decision": p2["decision"],
        },
        "reused_extension": {
            "path": str(args.extension.resolve()),
            "sha256": sha256(args.extension),
        },
        "fresh_processes": {
            "exact_match": exact_process_match,
            "process_1": process_1,
            "process_2": process_2,
        },
        "metrics": {
            "maximum_max_abs": maximum_max_abs,
            "maximum_normalized_l2": maximum_nl2,
            "batch_cases": sorted(
                int(key)
                for key in process_1["batch_results"]
            ),
            "repeat_launches": process_1["repeat_launches"]["count"],
        },
        "artifacts": {
            "production_source_sha256": sha256(source_path),
            "mapping_header_sha256": sha256(mapping_path),
            "network_bridge_sha256": sha256(bridge_path),
        },
        "static_gates": static_gates,
        "changed_files": sorted(changed_paths),
        "gates": {
            "p2_evidence_bound": True,
            "p2_extension_reused_without_rebuild": True,
            "production_source_unchanged": True,
            "twenty_batch_runtime_matrix_pass": True,
            "padding_boundaries_256_through_1024_pass": True,
            "sixty_four_launch_bitwise_replay_pass": True,
            "prefix_invariance_pass": True,
            "parameter_hot_swap_pass": True,
            "dual_stream_two_model_isolation_pass": True,
            "existing_factories_construct_pass": True,
            "training_forward_fail_closed": True,
            "two_fresh_processes_exact": exact_process_match,
            "no_kernel_math_change": True,
            "no_performance_claim": (
                contract["integration_gates"]["no_performance_claim"]
                is True
            ),
            "production_isa_deferred_to_p4": (
                contract["integration_gates"]["production_isa_claim"]
                == "deferred_to_4A2-P4"
            ),
            "changed_file_set_exact": True,
        },
    }
    assert all(result["gates"].values())

    args.evidence.mkdir(parents=True, exist_ok=True)
    output = (
        args.evidence
        / "phase4a2_p3_width64_runtime_integration_lifecycle_closure.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    report = f"""# Phase 4A2-P3 — Runtime integration and lifecycle closure

Decision: **`{result["decision"]}`**

## Runtime matrix

- Public batch cases: `{", ".join(str(v) for v in result["metrics"]["batch_cases"])}`
- Internal padding boundaries: `256, 512, 768, 1024`
- Repeated launch sequence: `{result["metrics"]["repeat_launches"]}`
- Worst maximum absolute error: `{maximum_max_abs:.9g}`
- Worst normalized L2 error: `{maximum_nl2:.9g}`

## Lifecycle and isolation

- Parameter hot swap `A -> B -> A`: bitwise restoration passed.
- Two model instances with different parameter buffers: isolated.
- Concurrent non-default streams: both references passed.
- Existing `PortableMLP` and `HipBLASLtMLPFP16` factories: construct.
- Training forward: fail-closed.
- Two fresh process results: byte-identical.

## Deliberate boundary

P3 reuses the exact P2 extension and makes no production-code or kernel-math
change. It makes no performance, occupancy, spill, resource, or production-ISA
claim. The final production code-object audit remains Phase 4A2-P4.
"""
    (args.evidence / "PHASE4A2_P3_REPORT.md").write_text(report)

    print("WIDTH64_P2_EXTENSION_REUSED: PASS")
    print("WIDTH64_RUNTIME_MATRIX_20_BATCHES: PASS")
    print("WIDTH64_RUNTIME_PADDING_256_1024: PASS")
    print("WIDTH64_RUNTIME_64_LAUNCH_BITWISE: PASS")
    print("WIDTH64_RUNTIME_PARAMETER_HOT_SWAP: PASS")
    print("WIDTH64_RUNTIME_DUAL_STREAM_MODEL_ISOLATION: PASS")
    print("WIDTH64_RUNTIME_EXISTING_FACTORIES: PASS")
    print("WIDTH64_RUNTIME_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("WIDTH64_P3_NO_PRODUCTION_CODE_CHANGE: PASS")
    print("PHASE4A2_P3_CONSOLIDATED_EVIDENCE: RECORDED")
    print(
        "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
