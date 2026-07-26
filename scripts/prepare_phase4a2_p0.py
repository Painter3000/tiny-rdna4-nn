#!/usr/bin/env python3
"""Prepare the deterministic Phase 4A2-P0 integration-surface inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_001"
P5_DECISION = "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT_PASS"

REQUIRED_SURFACES: dict[str, tuple[str, ...]] = {
    "CMakeLists.txt": ("project",),
    "src/CMakeLists.txt": ("network.cu",),
    "src/network.cu": ("create_network",),
    "src/portable_network.cu": ("PortableMLP",),
    "src/portable_mlp.cu": ("PortableMLP",),
    "include/tiny-cuda-nn/network.h": ("class Network",),
    "include/tiny-cuda-nn/gpu_matrix.h": ("MatrixLayout",),
    "include/tiny-cuda-nn/networks/portable_mlp.h": ("PortableMLP",),
    "bindings/torch/tinycudann/bindings.cpp": ("tiny-cuda-nn",),
}

OPTIONAL_SURFACES = (
    "src/hipblaslt_mlp.cu",
    "include/tiny-cuda-nn/networks/hipblaslt_mlp_fp16.h",
    "include/tiny-cuda-nn/network_with_input_encoding.h",
    "src/fully_fused_mlp.cu",
    "src/cutlass_mlp.cu",
)

COLLISION_TOKENS = (
    "RocWMMAWidth64MLP",
    "rocwmma_width64_mlp",
    "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP",
    "TCNN_WITH_ROCWMMA_WIDTH64_MLP",
    "TCNN_RDNA4_P4A2_",
)

SEARCH_ROOTS = (
    "CMakeLists.txt",
    "src",
    "include",
    "bindings",
    "samples",
    "tests",
)

TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".cmake",
    ".txt",
    ".py",
    ".json",
}


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def line_hits(text: str, tokens: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for token in tokens:
            if token in line:
                rows.append(
                    {
                        "line": number,
                        "token": token,
                        "text": line.strip()[:240],
                    }
                )
                break
        if len(rows) >= 16:
            break
    return rows


def scan_collisions(repo: Path) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []

    candidates: list[Path] = []
    for item in SEARCH_ROOTS:
        path = repo / item
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(p for p in path.rglob("*") if p.is_file()))

    for path in sorted(set(candidates)):
        if any(part in {".git", "build", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        relative = str(path.relative_to(repo))
        for number, line in enumerate(text.splitlines(), start=1):
            for token in COLLISION_TOKENS:
                if token in line:
                    collisions.append(
                        {
                            "path": relative,
                            "line": number,
                            "token": token,
                            "text": line.strip()[:240],
                        }
                    )
    return collisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--p5-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract = load_json(args.contract)
    p5 = load_json(args.p5_json)

    assert contract["marker"] == MARKER
    assert contract["phase"] == "4A2-P0"
    assert p5["decision"] == P5_DECISION
    assert all(bool(value) for value in p5["gates"].values())

    p5_audit = p5["isa_audit"]["data"]
    assert p5_audit["decision"] == "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT_PASS"
    assert all(bool(value) for value in p5_audit["gates"].values())

    baseline = contract["baseline"]
    resources = p5_audit["resources"]
    assert resources["wavefront_size32"] == 1
    assert resources["group_segment_fixed_size"] == baseline["required_lds_bytes"]
    assert resources["private_segment_fixed_size"] == baseline["required_private_segment_bytes"]
    assert p5_audit["scratch_instruction_count"] == baseline["required_scratch_instruction_count"]

    tag = baseline["required_tag"]
    head = run_git(repo, "rev-parse", "HEAD")
    tag_commit = run_git(repo, "rev-parse", f"{tag}^{{}}")
    tag_object = run_git(repo, "rev-parse", tag)
    branch = run_git(repo, "branch", "--show-current")
    assert head == tag_commit, (head, tag_commit)

    production_diff = run_git(
        repo,
        "status",
        "--short",
        "--",
        "CMakeLists.txt",
        "src",
        "include",
        "bindings",
        "samples",
        "tests",
    )
    assert production_diff == "", production_diff

    surfaces: dict[str, dict[str, Any]] = {}
    for relative, tokens in REQUIRED_SURFACES.items():
        path = repo / relative
        assert path.is_file(), relative
        text = path.read_text(errors="replace")
        for token in tokens:
            assert token in text, (relative, token)
        surfaces[relative] = {
            "required": True,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "required_tokens": list(tokens),
            "hits": line_hits(
                text,
                tuple(
                    sorted(
                        set(tokens)
                        | {
                            "PortableMLP",
                            "HipBLASLtMLPFP16",
                            "create_network",
                            "MatrixLayout",
                            "NetworkWithInputEncoding",
                        }
                    )
                ),
            ),
        }

    for relative in OPTIONAL_SURFACES:
        path = repo / relative
        if not path.is_file():
            surfaces[relative] = {"required": False, "present": False}
            continue
        text = path.read_text(errors="replace")
        surfaces[relative] = {
            "required": False,
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "hits": line_hits(
                text,
                (
                    "PortableMLP",
                    "HipBLASLtMLPFP16",
                    "FullyFusedMLP",
                    "CutlassMLP",
                    "NetworkWithInputEncoding",
                ),
            ),
        }

    collisions = scan_collisions(repo)
    assert collisions == [], collisions

    api = contract["public_api"]
    eligibility = contract["eligibility"]
    parameter_abi = contract["parameter_abi"]
    execution = contract["execution_contract"]
    scope = contract["scope"]

    contract_gates = {
        "explicit_otype_only": api["selection"] == "explicit_otype_only",
        "default_disabled": api["default_enabled"] is False,
        "no_silent_fallback": api["silent_fallback"] is False,
        "gfx1201_only": eligibility["runtime_architecture"] == ["gfx1201"],
        "width64_topology": all(
            eligibility[key] == 64
            for key in ("input_width", "hidden_width", "output_width")
        ),
        "two_hidden_layers": eligibility["n_hidden_layers"] == 2,
        "relu_none_activations": (
            eligibility["hidden_activation"] == "ReLU"
            and eligibility["output_activation"] == "None"
        ),
        "batch_tile_16": (
            eligibility["minimum_batch_rows"] == 16
            and eligibility["batch_rows_multiple"] == 16
        ),
        "fp16_parameter_abi": (
            parameter_abi["storage_type"] == "__half"
            and parameter_abi["total_parameter_elements"] == 12480
        ),
        "single_2048_byte_lds": (
            execution["single_reused_lds_buffer"] is True
            and execution["lds_bytes"] == 2048
        ),
        "caller_stream_no_host_sync": (
            eligibility["stream"] == "caller_supplied_hipStream_t"
            and execution["host_synchronization"] is False
        ),
        "inference_only": scope["phase_4a2_initial_capability"] == "inference_only",
        "existing_backends_unchanged": len(scope["existing_backends_unchanged"]) >= 4,
        "no_performance_claim": scope["performance_claim"] == "none",
        "bridge_delta_ledger_complete": len(contract["p4_to_production_bridge_deltas"]) == 6,
    }
    assert all(contract_gates.values())

    result = {
        "marker": MARKER,
        "decision": "PHASE4A2_P0_PREPARATION_PASS",
        "baseline": {
            "branch": branch,
            "head": head,
            "tag": tag,
            "tag_object": tag_object,
            "tag_commit": tag_commit,
            "head_equals_tag_commit": head == tag_commit,
            "p5_json": str(args.p5_json.resolve()),
            "p5_json_sha256": sha256(args.p5_json),
            "p5_decision": p5["decision"],
            "p5_resources": {
                "wavefront_size32": resources["wavefront_size32"],
                "next_free_vgpr": resources["next_free_vgpr"],
                "next_free_sgpr": resources["next_free_sgpr"],
                "group_segment_fixed_size": resources["group_segment_fixed_size"],
                "private_segment_fixed_size": resources["private_segment_fixed_size"],
                "scratch_instruction_count": p5_audit["scratch_instruction_count"],
                "mfma_or_wmma": p5_audit["instructions"]["mfma_or_wmma"],
                "lds_reads": p5_audit["instructions"]["lds_reads"],
                "lds_writes": p5_audit["instructions"]["lds_writes"],
                "block_barriers": p5_audit["instructions"]["block_barriers"],
            },
        },
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract),
            "data": contract,
            "gates": contract_gates,
        },
        "production_source_diff": production_diff,
        "integration_surfaces": surfaces,
        "reserved_identifier_collisions": collisions,
        "gates": {
            "phase4a1_tag_bound": head == tag_commit,
            "phase4a1_p5_evidence_bound": p5["decision"] == P5_DECISION,
            "production_sources_clean": production_diff == "",
            "required_surfaces_present": all(
                record.get("required") is True
                for path, record in surfaces.items()
                if path in REQUIRED_SURFACES
            ),
            "reserved_identifiers_collision_free": collisions == [],
            "contract_gates_pass": all(contract_gates.values()),
        },
    }
    assert all(result["gates"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("head: " + head)
    print("phase4a1_tag_commit: " + tag_commit)
    print("p5_json_sha256: " + result["baseline"]["p5_json_sha256"])
    print("contract_sha256: " + result["contract"]["sha256"])
    print("required_surface_count: " + str(len(REQUIRED_SURFACES)))
    print("reserved_identifier_collisions: 0")
    print("PHASE4A1_PASS_TAG_BOUND: PASS")
    print("PHASE4A1_P5_EVIDENCE_BOUND: PASS")
    print("WIDTH64_PRODUCTION_SURFACES_INVENTORIED: PASS")
    print("WIDTH64_PUBLIC_OTYPE_COLLISION_FREE: PASS")
    print("WIDTH64_OPT_IN_FAIL_CLOSED_CONTRACT: PASS")
    print("WIDTH64_NETWORK_ABI_CONTRACT: PASS")
    print("WIDTH64_PARAMETER_LAYOUT_CONTRACT: PASS")
    print("WIDTH64_BATCH_LAYOUT_CONTRACT: PASS")
    print("WIDTH64_INFERENCE_ONLY_SCOPE_LOCKED: PASS")
    print("WIDTH64_EXISTING_BACKENDS_UNCHANGED_CONTRACT: PASS")
    print("PHASE4A2_P0_PREPARATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
