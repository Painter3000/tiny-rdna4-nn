#!/usr/bin/env python3
"""Finalize Phase 4A2-P4 production code-object audit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_001"
EXPECTED_SOURCE_SHA256 = (
    "7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4"
)
EXPECTED_MAPPING_SHA256 = (
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"
)

EXPECTED_CHANGED = {
    "README_PHASE4A2_P4.md",
    "contracts/phase4a2_p4_production_code_object_audit_contract.json",
    "scripts/audit_phase4a2_p4_code_object.py",
    "scripts/finalize_phase4a2_p4.py",
    "scripts/resume_phase4a2_p4_after_symbol_tool_fix.sh",
    "scripts/run_phase4a2_p4_production_code_object_audit.sh",
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


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def git_diff_tree_paths(
    repo: Path,
    older: str,
    newer: str,
) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            older,
            newer,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }


def resolve_git_context(repo: Path) -> dict[str, Any]:
    head = git_output(repo, "rev-parse", "HEAD")
    subject = git_output(repo, "show", "-s", "--format=%s", "HEAD")
    parent = git_output(repo, "rev-parse", "HEAD^")
    parent_subject = git_output(
        repo,
        "show",
        "-s",
        "--format=%s",
        "HEAD^",
    )

    if (
        head.startswith("de76469")
        and subject
        == "Add Phase 4A2-P3 rocWMMA Width-64 runtime integration closure"
    ):
        mode = "p3_bundle_precommit"
        changed_paths = git_paths(repo)
    elif (
        subject
        == "Add Phase 4A2-P4 rocWMMA Width-64 production code-object audit"
        and parent.startswith("de76469")
        and parent_subject
        == "Add Phase 4A2-P3 rocWMMA Width-64 runtime integration closure"
    ):
        mode = "p4_release_commit"
        worktree_paths = git_paths(repo)
        assert not worktree_paths, {
            "release_worktree_not_clean": sorted(worktree_paths)
        }
        changed_paths = git_diff_tree_paths(repo, parent, head)
    else:
        raise AssertionError(
            {
                "unsupported_git_context": {
                    "head": head,
                    "subject": subject,
                    "parent": parent,
                    "parent_subject": parent_subject,
                }
            }
        )

    return {
        "mode": mode,
        "head": head,
        "subject": subject,
        "parent": parent,
        "parent_subject": parent_subject,
        "changed_paths": changed_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--p3-json", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--replay-1", type=Path, required=True)
    parser.add_argument("--replay-2", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--prior-p4-json", type=Path)
    parser.add_argument("--prior-replay-1", type=Path)
    parser.add_argument("--prior-replay-2", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract = load(args.contract)
    p3 = load(args.p3_json)
    audit = load(args.audit_json)
    replay_1 = load(args.replay_1)
    replay_2 = load(args.replay_2)

    assert contract["marker"] == MARKER
    assert contract["phase"] == "4A2-P4"
    assert p3["decision"] == (
        "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE_PASS"
    )
    assert all(bool(value) for value in p3["gates"].values())

    assert audit["decision"] == (
        "PHASE4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_PASS"
    )
    assert all(bool(value) for value in audit["gates"].values())

    for replay in (replay_1, replay_2):
        assert replay["decision"] == (
            "PHASE4A2_P3_RUNTIME_INTEGRATION_LIFECYCLE_PROCESS_PASS"
        )
        assert all(bool(value) for value in replay["gates"].values())

    replay_exact = args.replay_1.read_bytes() == args.replay_2.read_bytes()
    assert replay_exact

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
    build_log = args.build_log.read_text(errors="replace")
    git_context = resolve_git_context(repo)
    changed_paths = git_context["changed_paths"]

    static_gates = {
        "production_source_exact": (
            sha256(source_path) == EXPECTED_SOURCE_SHA256
        ),
        "mapping_header_exact": (
            sha256(mapping_path) == EXPECTED_MAPPING_SHA256
        ),
        "single_production_kernel_source": (
            source.count("__global__ void") == 1
        ),
        "three_source_mma_sites": (
            source.count("rocwmma::mma_sync(") == 3
        ),
        "three_source_barriers": (
            source.count("__syncthreads();") == 3
        ),
        "one_source_lds_buffer": (
            source.count(
                "__shared__ __align__(16) Half hidden_lds"
            )
            == 1
        ),
        "caller_stream_no_host_sync_source": (
            "hipLaunchKernelGGL(" in source
            and "hipDeviceSynchronize" not in source
            and "hipStreamSynchronize" not in source
        ),
        "column_major_bridge_preserved": (
            "m_fp16_rocwmma_width64" in bridge
            and "MatrixLayout::ColumnMajor" in bridge
            and "set_padded_output_width(64)" in bridge
        ),
        "fresh_explicit_on_build": (
            "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" in build_log
            and "rocwmma_width64_mlp.cu" in build_log
            and "TCNN_WITH_ROCWMMA_WIDTH64_MLP" in build_log
        ),
        "p4_no_production_code_change": all(
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

    resources = audit["resources"]
    isa = audit["isa"]

    prior_equivalence = {
        "provided": False,
        "all_equal": None,
        "gates": {},
    }
    prior_args = (
        args.prior_p4_json,
        args.prior_replay_1,
        args.prior_replay_2,
    )
    if any(path is not None for path in prior_args):
        assert all(path is not None for path in prior_args)
        prior = load(args.prior_p4_json)
        prior_gates = {
            "prior_decision_pass": (
                prior["decision"]
                == "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT_PASS"
            ),
            "object_sha256_equal": (
                prior["fresh_build"]["object_sha256"]
                == sha256(args.object)
            ),
            "extension_sha256_equal": (
                prior["fresh_build"]["extension_sha256"]
                == sha256(args.extension)
            ),
            "code_object_sha256_equal": (
                prior["production_artifacts"]["code_object_sha256"]
                == audit["extraction"]["code_object_sha256"]
            ),
            "source_sha256_equal": (
                prior["production_artifacts"]["source_sha256"]
                == sha256(source_path)
            ),
            "mapping_header_sha256_equal": (
                prior["production_artifacts"]["mapping_header_sha256"]
                == sha256(mapping_path)
            ),
            "network_bridge_sha256_equal": (
                prior["production_artifacts"]["network_bridge_sha256"]
                == sha256(bridge_path)
            ),
            "replay_1_bytes_equal": (
                args.prior_replay_1.read_bytes()
                == args.replay_1.read_bytes()
            ),
            "replay_2_bytes_equal": (
                args.prior_replay_2.read_bytes()
                == args.replay_2.read_bytes()
            ),
        }
        assert all(prior_gates.values()), {
            key: value
            for key, value in prior_gates.items()
            if not value
        }
        prior_equivalence = {
            "provided": True,
            "prior_p4_json": str(
                args.prior_p4_json.resolve()
            ),
            "prior_p4_json_sha256": sha256(
                args.prior_p4_json
            ),
            "all_equal": True,
            "gates": prior_gates,
        }

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT_PASS"
        ),
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract),
            "data": contract,
        },
        "git_context": {
            "mode": git_context["mode"],
            "head": git_context["head"],
            "subject": git_context["subject"],
            "parent": git_context["parent"],
            "parent_subject": git_context["parent_subject"],
        },
        "prior_p4_equivalence": prior_equivalence,
        "p3_baseline": {
            "path": str(args.p3_json.resolve()),
            "sha256": sha256(args.p3_json),
            "decision": p3["decision"],
        },
        "fresh_build": {
            "log": str(args.build_log.resolve()),
            "log_sha256": sha256(args.build_log),
            "object": str(args.object.resolve()),
            "object_sha256": sha256(args.object),
            "extension": str(args.extension.resolve()),
            "extension_sha256": sha256(args.extension),
        },
        "code_object_audit": {
            "path": str(args.audit_json.resolve()),
            "sha256": sha256(args.audit_json),
            "data": audit,
        },
        "runtime_replays": {
            "exact_match": replay_exact,
            "process_1": replay_1,
            "process_2": replay_2,
        },
        "production_artifacts": {
            "source_sha256": sha256(source_path),
            "mapping_header_sha256": sha256(mapping_path),
            "network_bridge_sha256": sha256(bridge_path),
            "code_object_sha256": audit["extraction"][
                "code_object_sha256"
            ],
        },
        "resource_summary": {
            "next_free_vgpr_or_vgpr_count": resources[
                "next_free_vgpr_or_vgpr_count"
            ],
            "next_free_sgpr_or_sgpr_count": resources[
                "next_free_sgpr_or_sgpr_count"
            ],
            "group_segment_fixed_size": resources[
                "group_segment_fixed_size"
            ],
            "private_segment_fixed_size": resources[
                "private_segment_fixed_size"
            ],
            "mfma_or_wmma_instructions": isa[
                "mfma_or_wmma_instructions"
            ],
            "lds_load_instructions": isa[
                "lds_load_instructions"
            ],
            "lds_store_instructions": isa[
                "lds_store_instructions"
            ],
            "ds_bpermute_b32_instructions": isa[
                "ds_bpermute_b32_instructions"
            ],
            "block_barriers": isa["block_barriers"],
            "scratch_instruction_count": isa[
                "scratch_instruction_count"
            ],
            "ds_mnemonics": isa["ds_mnemonics"],
            "global_memory_mnemonics": isa[
                "global_memory_mnemonics"
            ],
        },
        "claim_boundaries": {
            "register_counts_do_not_prove_occupancy": True,
            "global_mnemonics_do_not_prove_pointer_provenance": True,
            "no_performance_measurement": True,
            "source_dataflow_has_no_hidden_global_roundtrip": (
                "hidden_lds" in source
                and "expected_hidden" not in source
                and "diagnostics" not in source
            ),
            "isa_has_no_scratch_instructions": (
                isa["scratch_instruction_count"] == 0
            ),
        },
        "static_gates": static_gates,
        "changed_files": sorted(changed_paths),
        "gates": {
            "p3_evidence_bound": True,
            "fresh_explicit_on_build_pass": True,
            "gfx1201_code_object_extracted": True,
            "exact_production_kernel_symbol_pass": True,
            "resource_metadata_recorded": True,
            "group_segment_2048_pass": (
                resources["group_segment_fixed_size"] == 2048
            ),
            "private_segment_zero_pass": (
                resources["private_segment_fixed_size"] == 0
            ),
            "twelve_mfma_or_wmma_pass": (
                isa["mfma_or_wmma_instructions"] == 12
            ),
            "lds_loads_8_stores_2_pass": (
                isa["lds_load_instructions"] == 8
                and isa["lds_store_instructions"] == 2
            ),
            "ds_bpermute_b32_192_pass": (
                isa["ds_bpermute_b32_instructions"] == 192
            ),
            "exact_ds_mnemonic_inventory_pass": (
                isa["ds_mnemonics"]
                == [
                    "ds_bpermute_b32",
                    "ds_load_b128",
                    "ds_store_b128",
                ]
            ),
            "six_barriers_pass": (
                isa["block_barriers"] == 6
            ),
            "zero_scratch_instructions_pass": (
                isa["scratch_instruction_count"] == 0
            ),
            "runtime_replay_processes_exact": replay_exact,
            "runtime_replay_all_gates_pass": (
                all(replay_1["gates"].values())
                and all(replay_2["gates"].values())
            ),
            "production_source_unchanged": True,
            "no_performance_claim": (
                contract["claim_boundaries"]["performance"]
                == "not_measured"
            ),
            "no_occupancy_claim": (
                contract["claim_boundaries"]["occupancy"]
                == "not_claimed"
            ),
            "no_pointer_provenance_overclaim": (
                contract["claim_boundaries"][
                    "per_instruction_pointer_provenance"
                ]
                == "not_proven"
            ),
            "changed_file_set_exact": True,
            "supported_git_context": (
                git_context["mode"]
                in {
                    "p3_bundle_precommit",
                    "p4_release_commit",
                }
            ),
            "prior_p4_equivalence_pass_or_not_requested": (
                not prior_equivalence["provided"]
                or prior_equivalence["all_equal"] is True
            ),
        },
    }
    assert all(result["gates"].values())

    args.evidence.mkdir(parents=True, exist_ok=True)
    output = (
        args.evidence
        / "phase4a2_p4_width64_production_code_object_resource_audit.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    summary = result["resource_summary"]
    report = f"""# Phase 4A2-P4 — Production code-object audit

Decision: **`{result["decision"]}`**

## Extracted production kernel

- Bundle target: `{audit["extraction"]["bundle_target"]}`
- Extraction method: `{audit["extraction"]["method"]}`
- Raw symbol: `{audit["kernel"]["raw_symbol"]}`
- Code-object SHA-256: `{audit["extraction"]["code_object_sha256"]}`

## Resource and ISA facts

- VGPR field: `{summary["next_free_vgpr_or_vgpr_count"]}`
- SGPR field: `{summary["next_free_sgpr_or_sgpr_count"]}`
- Group segment: `{summary["group_segment_fixed_size"]}` bytes
- Private segment: `{summary["private_segment_fixed_size"]}` bytes
- MFMA/WMMA instructions: `{summary["mfma_or_wmma_instructions"]}`
- Static LDS load instructions: `{summary["lds_load_instructions"]}`
- Static LDS store instructions: `{summary["lds_store_instructions"]}`
- Static `ds_bpermute_b32` instructions: `{summary["ds_bpermute_b32_instructions"]}`
- Barrier instructions: `{summary["block_barriers"]}`
- Scratch instructions: `{summary["scratch_instruction_count"]}`
- DS mnemonics: `{", ".join(summary["ds_mnemonics"])}`
- Global-memory mnemonics: `{json.dumps(summary["global_memory_mnemonics"], sort_keys=True)}`

## Runtime replay

The full P3 runtime matrix passed twice against the fresh P4 build, and both
JSON results were byte-identical.

## Release reproducibility

- Git context: `{git_context["mode"]}`
- HEAD: `{git_context["head"]}`
- Parent: `{git_context["parent"]}`
- Prior P4 evidence supplied: `{prior_equivalence["provided"]}`
- Prior production artifacts and replay bytes identical:
  `{prior_equivalence["all_equal"]}`

## Claim boundary

The register fields are recorded but do not by themselves establish occupancy.
The global-memory mnemonic inventory does not provide formal per-instruction
pointer provenance. The source/dataflow audit establishes one reused LDS hidden
buffer and no compiler-generated scratch instructions were found. No
performance measurement or speed claim is made.
"""
    (args.evidence / "PHASE4A2_P4_REPORT.md").write_text(report)

    print("WIDTH64_FRESH_PRODUCTION_BUILD: PASS")
    print("WIDTH64_GFX1201_CODE_OBJECT_BOUND: PASS")
    print("WIDTH64_EXACT_PRODUCTION_KERNEL_BOUND: PASS")
    print("WIDTH64_RESOURCE_FIELDS_RECORDED: PASS")
    print("WIDTH64_GROUP_2048_PRIVATE_ZERO: PASS")
    print("WIDTH64_MFMA12_LDSLOAD8_STORE2_BARRIER6: PASS")
    print("WIDTH64_DS_BPERMUTE_B32_192: PASS")
    print("WIDTH64_SCRATCH_ZERO: PASS")
    print("WIDTH64_FRESH_RUNTIME_REPLAY_TWICE: PASS")
    print("WIDTH64_P4_NO_PRODUCTION_CODE_CHANGE: PASS")
    print("WIDTH64_P4_RELEASE_GIT_CONTEXT: PASS")
    if prior_equivalence["provided"]:
        print(
            "WIDTH64_PRIOR_P4_PRODUCTION_ARTIFACTS_BYTE_IDENTICAL: PASS"
        )
        print(
            "WIDTH64_PRIOR_P4_RUNTIME_REPLAYS_BYTE_IDENTICAL: PASS"
        )
    print("PHASE4A2_P4_CONSOLIDATED_EVIDENCE: RECORDED")
    print(
        "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
