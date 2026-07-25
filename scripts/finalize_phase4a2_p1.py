#!/usr/bin/env python3
"""Finalize Phase 4A2-P1 build, factory, and source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001"

EXPECTED_CHANGED = {
    "README_PHASE4A2_P1.md",
    "bindings/torch/setup.py",
    "contracts/phase4a2_p1_integration_surface_addendum.json",
    "include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h",
    "probes/phase4a2_p1_factory_probe.py",
    "scripts/apply_phase4a2_p1.py",
    "scripts/finalize_phase4a2_p1.py",
    "scripts/resume_phase4a2_p1_after_probe_fix.sh",
    "scripts/run_phase4a2_p1_opt_in_class_build_factory_skeleton.sh",
    "src/cpp_api.cu",
    "src/portable_network.cu",
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
    parser.add_argument("--apply", type=Path, required=True)
    parser.add_argument("--disabled", type=Path, required=True)
    parser.add_argument("--enabled-1", type=Path, required=True)
    parser.add_argument("--enabled-2", type=Path, required=True)
    parser.add_argument("--build-off-log", type=Path, required=True)
    parser.add_argument("--build-on-log", type=Path, required=True)
    parser.add_argument("--surface-addendum", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    apply = load(args.apply)
    disabled = load(args.disabled)
    enabled_1 = load(args.enabled_1)
    enabled_2 = load(args.enabled_2)
    addendum = load(args.surface_addendum)

    assert apply["decision"] == "PHASE4A2_P1_APPLY_PASS"
    assert all(bool(value) for value in apply["gates"].values())
    assert disabled["decision"] == "PHASE4A2_P1_FACTORY_DISABLED_PASS"
    assert enabled_1["decision"] == (
        "PHASE4A2_P1_FACTORY_ENABLED_SKELETON_PASS"
    )
    assert enabled_2["decision"] == (
        "PHASE4A2_P1_FACTORY_ENABLED_SKELETON_PASS"
    )
    assert all(bool(value) for value in disabled["gates"].values())
    assert all(bool(value) for value in enabled_1["gates"].values())
    assert all(bool(value) for value in enabled_2["gates"].values())
    assert args.enabled_1.read_bytes() == args.enabled_2.read_bytes()

    assert addendum["marker"] == "TCNN_RDNA4_P4A2_P1_SURFACE_ADDENDUM_001"
    assert addendum["p0_contract_preserved"] is True
    assert len(addendum["additional_required_surfaces"]) == 2

    setup = (repo / "bindings/torch/setup.py").read_text()
    portable_factory = (repo / "src/portable_network.cu").read_text()
    cpp_api = (repo / "src/cpp_api.cu").read_text()
    header = (
        repo
        / "include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h"
    ).read_text()
    source = (repo / "src/rocwmma_width64_mlp.cu").read_text()

    off_log = args.build_off_log.read_text(errors="replace")
    on_log = args.build_on_log.read_text(errors="replace")

    current_paths = git_paths(repo)
    static_gates = {
        "setup_default_off": (
            '"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP",\n\tFalse,' in setup
        ),
        "setup_conditional_source": (
            "if is_rocm and include_networks and "
            "enable_rocwmma_width64_mlp:" in setup
            and 'base_source_files.append("../../src/rocwmma_width64_mlp.cu")'
            in setup
        ),
        "setup_conditional_definition": (
            'definitions.append("-DTCNN_WITH_ROCWMMA_WIDTH64_MLP")'
            in setup
        ),
        "factory_explicit_otype": portable_factory.count(
            "RocWMMAWidth64MLP"
        ) >= 8,
        "factory_backend_not_compiled_guard": (
            "RocWMMAWidth64MLP was not compiled" in portable_factory
        ),
        "cpp_api_fp16_dispatch": (
            'equals_case_insensitive(requested_otype, "RocWMMAWidth64MLP")'
            in cpp_api
        ),
        "class_is_final_network_half": (
            "class RocWMMAWidth64MLP final : public Network<__half>"
            in header
        ),
        "parameter_abi_static_assert": (
            "TOTAL_PARAMETER_ELEMENTS == 12480" in header
        ),
        "no_production_kernel": (
            "__global__" not in source
            and "rocwmma::" not in source
            and "mma_sync" not in source
        ),
        "fail_closed_message": (
            "no fallback was executed" in source
        ),
        "disabled_build_excludes_source": (
            "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: OFF" in off_log
            and "rocwmma_width64_mlp.cu" not in off_log
            and "TCNN_WITH_ROCWMMA_WIDTH64_MLP" not in off_log
        ),
        "enabled_build_includes_source": (
            "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" in on_log
            and "rocwmma_width64_mlp.cu" in on_log
            and "TCNN_WITH_ROCWMMA_WIDTH64_MLP" in on_log
        ),
        "changed_file_set_exact": current_paths == EXPECTED_CHANGED,
    }
    assert all(static_gates.values()), {
        key: value for key, value in static_gates.items() if not value
    }

    result = {
        "marker": MARKER,
        "decision": "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON_PASS",
        "apply": {
            "path": str(args.apply.resolve()),
            "sha256": sha256(args.apply),
            "data": apply,
        },
        "builds": {
            "default_off": {
                "log": str(args.build_off_log.resolve()),
                "sha256": sha256(args.build_off_log),
                "factory_result": disabled,
            },
            "explicit_on": {
                "log": str(args.build_on_log.resolve()),
                "sha256": sha256(args.build_on_log),
                "fresh_process_1": enabled_1,
                "fresh_process_2": enabled_2,
                "fresh_process_exact_match": True,
            },
        },
        "surface_addendum": addendum,
        "static_gates": static_gates,
        "changed_files": sorted(current_paths),
        "gates": {
            "p0_bound_and_anchor_patch_pass": True,
            "default_off_build_pass": True,
            "default_off_factory_fails_closed": True,
            "existing_factories_regression_pass": True,
            "explicit_on_build_pass": True,
            "explicit_factory_constructs": True,
            "parameter_abi_12480": True,
            "invalid_configs_fail_closed": True,
            "inference_and_forward_fail_before_kernel": True,
            "enabled_fresh_process_reproducible": True,
            "no_production_kernel_installed": True,
            "surface_addendum_recorded": True,
            "changed_file_set_exact": True,
        },
    }
    assert all(result["gates"].values())

    args.evidence.mkdir(parents=True, exist_ok=True)
    output = (
        args.evidence
        / "phase4a2_p1_opt_in_class_build_factory_skeleton.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    report = """# Phase 4A2-P1 — Opt-in class, build and factory skeleton

Decision: **`PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON_PASS`**

## Proven

- the PyTorch ROCm build remains default-OFF;
- the OFF build excludes the new source and compile definition;
- an explicit request in the OFF build fails with “was not compiled”;
- existing PortableMLP and HipBLASLtMLPFP16 factories still construct;
- the ON build compiles the new source with the dedicated definition;
- the explicit `RocWMMAWidth64MLP` factory constructs on gfx1201;
- the parameter ABI is exactly 12,480 FP16 elements;
- wrong shape, activation, precision, and bias requests fail closed;
- inference and forward fail before any production kernel or fallback;
- two enabled fresh processes produce identical evidence.

## Deliberate boundary

There is no `__global__` production kernel, rocWMMA call, automatic backend
selection, training implementation, backward implementation, or performance
claim in P1.
"""
    (args.evidence / "PHASE4A2_P1_REPORT.md").write_text(report)

    print("WIDTH64_DEFAULT_OFF_BUILD: PASS")
    print("WIDTH64_DEFAULT_OFF_FACTORY_FAIL_CLOSED: PASS")
    print("WIDTH64_EXISTING_FACTORY_REGRESSION: PASS")
    print("WIDTH64_EXPLICIT_ON_BUILD: PASS")
    print("WIDTH64_EXPLICIT_FACTORY_CONSTRUCTION: PASS")
    print("WIDTH64_PARAMETER_ABI_12480: PASS")
    print("WIDTH64_INVALID_CONFIG_FAIL_CLOSED: PASS")
    print("WIDTH64_INFERENCE_FORWARD_PREKERNEL_FAIL_CLOSED: PASS")
    print("WIDTH64_ENABLED_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("WIDTH64_NO_PRODUCTION_KERNEL_INSTALLED: PASS")
    print("PHASE4A2_P1_SURFACE_ADDENDUM: RECORDED")
    print("PHASE4A2_P1_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
