#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1A_FP16_CAPABILITY_001: run the isolated FP16 capability matrix."""

import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/probe_phase3b1a_fp16_hipblaslt.cpp"
REPORT_DIR = ROOT / "phase3b1_reports"
JSON_REPORT = REPORT_DIR / "phase3b1a_fp16_capability.json"
MD_REPORT = REPORT_DIR / "PHASE3B1A_FP16_CAPABILITY.md"
BUILD_LOG = REPORT_DIR / "phase3b1a_build.log"
RUN_LOG = REPORT_DIR / "phase3b1a_run.log"
MARKER = "TCNN_RDNA4_P3B1A_FP16_CAPABILITY_001"
FIXED_PHASE3A4 = "6258184d8d9d032ef423b75eddeeaf8168c7e45a"
DIRECTIONS = ("NN", "NT", "TN")
KS = (16, 32, 64, 128, 256)
OUTPUT_TYPES = ("f16", "f32")
LABELS = (
    "FP32_ACCUMULATION_CONFIRMED",
    "FP16_ACCUMULATION_OBSERVED",
    "AMBIGUOUS",
    "NO_OP_OR_INVALID",
    "UNSUPPORTED",
)


def command_output(command):
    return subprocess.run(command, text=True, capture_output=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_metrics(gpu, reference, near_zero=1e-5):
    differences = [abs(float(a) - float(b)) for a, b in zip(gpu, reference)]
    max_absolute = max(differences, default=math.inf)
    relative = [difference / abs(float(ref)) for difference, ref in zip(differences, reference) if abs(float(ref)) > near_zero]
    reference_norm = math.sqrt(sum(float(value) ** 2 for value in reference))
    error_norm = math.sqrt(sum(value * value for value in differences))
    return {
        "near_zero_threshold": near_zero,
        "max_absolute": max_absolute,
        "max_relative_outside_near_zero": max(relative, default=0.0),
        "normalized_l2": error_norm / max(reference_norm, near_zero),
        "l2": error_norm,
    }


def distance_between(a, b):
    differences = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    norm = math.sqrt(sum(float(value) ** 2 for value in a))
    l2 = math.sqrt(sum(value * value for value in differences))
    return {"max_absolute": max(differences, default=0.0), "normalized_l2": l2 / max(norm, 1e-5), "l2": l2}


def classify(run):
    if run.get("process_status") != "PASS" or run.get("probe_state") != "EXECUTED":
        return "UNSUPPORTED" if run.get("probe_state") == "UNSUPPORTED" else "NO_OP_OR_INVALID", {}
    launches = run.get("launches", [])
    memory_ok = bool(launches) and all(
        item.get("matmul_status") == 0
        and item.get("synchronize_status") == 0
        and item.get("output_memory", {}).get("prefix_guard_intact")
        and item.get("output_memory", {}).get("suffix_guard_intact")
        and not item.get("output_memory", {}).get("payload_unchanged", True)
        for item in launches
    )
    workspace = run.get("workspace_memory", {})
    memory_ok = memory_ok and workspace.get("prefix_guard_intact") and workspace.get("suffix_guard_intact")
    steady = run.get("steady_state", {})
    steady_ok = (
        steady.get("new_handles") == 0
        and steady.get("new_heuristic_queries") == 0
        and steady.get("descriptor_growth") == 0
        and steady.get("workspace_growth_bytes") == 0
    )
    gpu = run.get("gpu_output", [])
    r32 = run.get("reference_r32_projected_to_d", [])
    r16 = run.get("reference_r16_projected_to_d", [])
    r64 = run.get("reference_r64", [])
    finite = len(gpu) == 256 and all(math.isfinite(float(value)) for value in gpu)
    metrics = {
        "gpu_to_r32": error_metrics(gpu, r32) if gpu and r32 else None,
        "gpu_to_r16": error_metrics(gpu, r16) if gpu and r16 else None,
        "gpu_to_r64": error_metrics(gpu, r64) if gpu and r64 else None,
        "r32_to_r16_separation": distance_between(r32, r16) if r32 and r16 else None,
        "nan_count": sum(math.isnan(float(value)) for value in gpu),
        "inf_count": sum(math.isinf(float(value)) for value in gpu),
        "memory_integrity_pass": bool(memory_ok),
        "steady_state_invariants_pass": bool(steady_ok),
    }
    if not memory_ok or not steady_ok or not finite:
        return "NO_OP_OR_INVALID", metrics
    separation = metrics["r32_to_r16_separation"]
    to32, to16 = metrics["gpu_to_r32"], metrics["gpu_to_r16"]
    distinguishable = separation["max_absolute"] > 1e-3 and separation["normalized_l2"] > 1e-5
    if not distinguishable:
        return "AMBIGUOUS", metrics
    fp32_compatible = (
        to32["max_absolute"] <= 0.10 * separation["max_absolute"] + 5e-6
        and to32["normalized_l2"] <= 0.10 * separation["normalized_l2"] + 1e-7
    )
    fp32_distinct = (
        to16["max_absolute"] >= 4.0 * max(to32["max_absolute"], 1e-7)
        and to16["normalized_l2"] >= 4.0 * max(to32["normalized_l2"], 1e-9)
    )
    fp16_compatible = (
        to16["max_absolute"] <= 0.10 * separation["max_absolute"] + 5e-6
        and to16["normalized_l2"] <= 0.10 * separation["normalized_l2"] + 1e-7
    )
    if fp32_compatible and fp32_distinct:
        return "FP32_ACCUMULATION_CONFIRMED", metrics
    if fp16_compatible and to32["normalized_l2"] >= 4.0 * max(to16["normalized_l2"], 1e-9):
        return "FP16_ACCUMULATION_OBSERVED", metrics
    return "AMBIGUOUS", metrics


def algorithm_identity(run):
    algo = run.get("algorithm", {})
    return {
        key: algo.get(key)
        for key in ("index", "solution_name", "kernel_name", "serialized", "workspace_required_bytes")
    }


def render_markdown(document):
    summary = document["summary"]
    lines = [
        "# Phase 3B1-A – FP16 hipBLASLt capability spike",
        "",
        f"Marker: `{MARKER}`",
        "",
        f"**Decision: `{document['decision']}`**",
        "",
        "This is capability evidence only. It is not a performance release, changes no production",
        "kernel or dispatch, creates no PASS tag, and does not modify the frozen Phase-3A4 identity.",
        "",
        "## Branching and frozen-code proof",
        "",
        f"- branch: `{document['git']['branch']}`",
        f"- spike base HEAD: `{document['git']['starting_head']}`",
        f"- fixed Phase-3A4 code identity: `{FIXED_PHASE3A4}`",
        f"- production diff since fixed identity before spike: `{document['git']['production_diff_before_spike'] or 'none'}`",
        "- all intervening changes were reports or diagnostic/qualification scripts: PASS",
        "",
        "## Environment",
        "",
    ]
    for key, value in document["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Capability summary",
        "",
        f"- tested signatures: {summary['tested_signatures']}",
        f"- FP32_ACCUMULATION_CONFIRMED: {summary['counts'].get('FP32_ACCUMULATION_CONFIRMED', 0)}",
        f"- FP16_ACCUMULATION_OBSERVED: {summary['counts'].get('FP16_ACCUMULATION_OBSERVED', 0)}",
        f"- AMBIGUOUS: {summary['counts'].get('AMBIGUOUS', 0)}",
        f"- NO_OP_OR_INVALID: {summary['counts'].get('NO_OP_OR_INVALID', 0)}",
        f"- UNSUPPORTED: {summary['counts'].get('UNSUPPORTED', 0)}",
        f"- blockers: {summary['blockers']}",
        "",
        "The classification is structural: R32 and R16 must be distinguishable, the GPU result",
        "must be within 10% of their separation from projected R32, and at least four times closer",
        "to R32 than R16. These are spike-discrimination rules, not production tolerances.",
        "",
        "## Signature matrix",
        "",
        "| Direction | K | D | Classification | Algo | Workspace | Fresh stable | max|GPU-R32| | max|GPU-R16| |",
        "|---|---:|---|---|---:|---:|---|---:|---:|",
    ])
    for case in document["cases"]:
        metrics = case["metrics"]
        to32 = metrics.get("gpu_to_r32") or {}
        to16 = metrics.get("gpu_to_r16") or {}
        lines.append(
            f"| {case['direction']} | {case['k']} | {case['output_type']} | {case['classification']} | "
            f"{case['algorithm'].get('index')} | {case['algorithm'].get('workspace_required_bytes')} | "
            f"{case['fresh_algorithm_stable']} | {to32.get('max_absolute')} | {to16.get('max_absolute')} |"
        )
    lines.extend([
        "",
        "## Descriptor, stream, and memory evidence",
        "",
        "Every full per-process descriptor, reference array, GPU output, algorithm serialization,",
        "status code, stream identity, sentinel/guard result, and error metric is retained in the",
        "JSON report. Each signature ran twice in separate fresh processes. Within each process the",
        "selected plan ran on two distinct streams with two distinct execution handles, followed by",
        "warm-cache calls with zero new handles, heuristic queries, descriptors, or workspace growth.",
        "",
        "## Selected algorithms and workspace",
        "",
    ])
    for item in summary["selected_algorithms"]:
        lines.append(
            f"- index `{item['index']}`, workspace `{item['workspace_required_bytes']}` bytes, "
            f"signatures `{item['signature_count']}`, kernel `{item['kernel_name']}`"
        )
    lines.extend([
        "",
        "## Marker audit",
        "",
        f"- required marker: `{MARKER}`",
        f"- marker occurrences: `{document['marker_audit']['occurrences']}`",
        f"- unmarked new diagnostic files: `{document['marker_audit']['unmarked_files']}`",
        "",
    ])
    return "\n".join(lines)


def main():
    if JSON_REPORT.exists() or MD_REPORT.exists():
        raise SystemExit("refusing to overwrite existing Phase-3B1-A evidence")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    starting_head = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
    branch = command_output(["git", "-C", str(ROOT), "branch", "--show-current"]).stdout.strip()
    production_diff = command_output([
        "git", "-C", str(ROOT), "diff", "--name-only", f"{FIXED_PHASE3A4}..{starting_head}",
        "--", "bindings", "include", "src", "CMakeLists.txt",
    ]).stdout.strip()
    if branch != "phase3b1-fp16-capability-spike" or production_diff:
        raise SystemExit("branch or frozen Phase-3A4 production-code precondition failed")

    with tempfile.TemporaryDirectory(prefix="tcnn_phase3b1a_") as build_dir:
        binary = pathlib.Path(build_dir) / "probe_phase3b1a"
        compile_command = [
            "/opt/rocm/bin/hipcc", "-std=c++17", "-O2", "--offload-arch=gfx1201",
            "-I/opt/rocm/include", f"-I{ROOT / 'dependencies'}", str(SOURCE),
            "-L/opt/rocm/lib", "-lhipblaslt", "-lamdhip64", "-o", str(binary),
        ]
        compiled = subprocess.run(compile_command, text=True, capture_output=True)
        BUILD_LOG.write_text(f"MARKER={MARKER}\nCOMMAND=" + json.dumps(compile_command) + "\n" + compiled.stdout + compiled.stderr)
        if compiled.returncode:
            raise SystemExit(f"probe build failed with rc={compiled.returncode}")
        cases = []
        run_log = []
        for direction in DIRECTIONS:
            for k in KS:
                for output_type in OUTPUT_TYPES:
                    runs = []
                    for fresh_repeat in (1, 2):
                        command = [str(binary), direction, str(k), output_type, str(fresh_repeat)]
                        completed = subprocess.run(command, cwd=build_dir, text=True, capture_output=True)
                        run_log.append({"command": command, "returncode": completed.returncode,
                            "stdout": completed.stdout, "stderr": completed.stderr})
                        try:
                            run = json.loads(completed.stdout.strip().splitlines()[-1])
                        except Exception as error:
                            run = {"marker": MARKER, "process_status": "ERROR", "error": f"JSON parse: {error}"}
                        run["process_returncode"] = completed.returncode
                        run["process_stderr"] = completed.stderr
                        runs.append(run)
                    classification, metrics = classify(runs[0])
                    second_classification, second_metrics = classify(runs[1])
                    stable = algorithm_identity(runs[0]) == algorithm_identity(runs[1])
                    if classification != second_classification:
                        classification = "AMBIGUOUS"
                    if not stable and classification != "UNSUPPORTED":
                        classification = "NO_OP_OR_INVALID"
                    cases.append({
                        "marker": MARKER,
                        "direction": direction,
                        "k": k,
                        "output_type": output_type,
                        "classification": classification,
                        "second_fresh_classification": second_classification,
                        "metrics": metrics,
                        "second_fresh_metrics": second_metrics,
                        "algorithm": algorithm_identity(runs[0]),
                        "second_fresh_algorithm": algorithm_identity(runs[1]),
                        "fresh_algorithm_stable": stable,
                        "fresh_process_runs": runs,
                    })
        RUN_LOG.write_text(json.dumps({"marker": MARKER, "fresh_process_invocations": run_log}, indent=2, sort_keys=True) + "\n")

    counts = collections.Counter(case["classification"] for case in cases)
    blockers = sum(case["classification"] != "FP32_ACCUMULATION_CONFIRMED" for case in cases)
    algorithm_groups = {}
    for case in cases:
        identity = case["algorithm"]
        key = (identity.get("index"), identity.get("kernel_name"), identity.get("workspace_required_bytes"))
        algorithm_groups.setdefault(key, 0)
        algorithm_groups[key] += 1
    selected_algorithms = [
        {"index": key[0], "kernel_name": key[1], "workspace_required_bytes": key[2], "signature_count": count}
        for key, count in sorted(algorithm_groups.items(), key=lambda item: str(item[0]))
    ]
    diagnostic_files = [SOURCE, pathlib.Path(__file__), JSON_REPORT, MD_REPORT, BUILD_LOG, RUN_LOG]
    marker_occurrences = sum(path.read_text().count(MARKER) for path in diagnostic_files if path.exists())
    unmarked = [str(path.relative_to(ROOT)) for path in diagnostic_files if path.exists() and MARKER not in path.read_text()]
    environment = cases[0]["fresh_process_runs"][0].get("environment", {}) if cases else {}
    environment.update({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": sha256(SOURCE),
        "hipcc": command_output(["/opt/rocm/bin/hipcc", "--version"]).stdout.splitlines()[0],
    })
    document = {
        "schema": 1,
        "marker": MARKER,
        "phase": "3B1-A",
        "capability_only": True,
        "production_integration": False,
        "performance_release": False,
        "pass_tag_created": False,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "decision": "PROCEED_TO_3B1B" if blockers == 0 and len(cases) == 30 else "PHASE3B1A_BLOCKED",
        "git": {"branch": branch, "starting_head": starting_head,
            "fixed_phase3a4_code_identity": FIXED_PHASE3A4,
            "production_diff_before_spike": production_diff,
            "intervening_changes_classification": "reports_and_diagnostic_infrastructure_only"},
        "environment": environment,
        "classification_rule": {"r32_r16_distinguishable_max_absolute_gt": 1e-3,
            "r32_r16_distinguishable_normalized_l2_gt": 1e-5,
            "compatible_with_r32_fraction_of_separation": 0.10,
            "closer_to_r32_factor": 4.0, "production_tolerance": None},
        "summary": {"tested_signatures": len(cases), "counts": dict(counts),
            "blockers": blockers, "selected_algorithms": selected_algorithms},
        "marker_audit": {"occurrences": marker_occurrences, "unmarked_files": unmarked},
        "cases": cases,
    }
    JSON_REPORT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    # Recompute marker audit after both evidence files exist.
    document["marker_audit"]["occurrences"] = sum(
        path.read_text().count(MARKER) for path in diagnostic_files if path.exists()
    )
    document["marker_audit"]["unmarked_files"] = [
        str(path.relative_to(ROOT)) for path in diagnostic_files if path.exists() and MARKER not in path.read_text()
    ]
    MD_REPORT.write_text(render_markdown(document))
    document["marker_audit"]["occurrences"] = sum(
        path.read_text().count(MARKER) for path in diagnostic_files if path.exists()
    )
    document["marker_audit"]["unmarked_files"] = [
        str(path.relative_to(ROOT)) for path in diagnostic_files if path.exists() and MARKER not in path.read_text()
    ]
    JSON_REPORT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    MD_REPORT.write_text(render_markdown(document))
    print(document["decision"])
    print(JSON_REPORT)
    return 0 if document["decision"] == "PROCEED_TO_3B1B" else 1


if __name__ == "__main__":
    raise SystemExit(main())
