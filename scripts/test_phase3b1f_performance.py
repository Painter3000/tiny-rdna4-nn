#!/usr/bin/env python3
"""Protocol audit and explicit F1 orchestrator for Phase 3B1-F."""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "phase3b1_reports/phase3b1f_protocol_manifest.json"
WORKER = ROOT / "scripts/phase3b1f_benchmark_worker.py"
FINALIZER = ROOT / "scripts/finalize_phase3b1f_report.py"
RAW = pathlib.Path("/tmp/phase3b1f_fp16_performance_raw.json")
MARKER = "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001"
CONFIRMATION = "RUN_PHASE3B1F1_FULL_MEASUREMENT"
EXPECTED_MANIFEST_SHA256 = "cf097d672595e72ca58a4090bff1135882f2181756f8e8ffb70281ea2fcfe0e3"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cases():
    network = 4 * 4 * 3
    encoding = 5 * 3 * 5
    return network, encoding, network + encoding


def protocol_audit():
    manifest = json.loads(MANIFEST.read_text())
    network, encoding, total = expected_cases()
    ids = [case["id"] for case in manifest["cases"]]
    required_top = {
        "base_commit", "marker", "comparators", "inputs", "seeds", "timer", "calibration",
        "warmup", "measurement", "statistics", "valid_process", "system_stability",
        "cold_start", "resources", "backend_evidence", "profiling", "performance_gates",
        "numerical_gates", "correctness_pre_post", "cases", "expected_counts",
    }
    changed = set(subprocess.check_output(["git", "diff", "--name-only", manifest["base_commit"]], cwd=ROOT, text=True).splitlines())
    changed.update(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines())
    production = [path for path in changed if path.startswith(("src/", "include/", "bindings/"))]
    historical_equal = True
    for path in subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", manifest["base_commit"], "phase3b1_reports"], cwd=ROOT, text=True
    ).splitlines():
        historical_equal &= (ROOT / path).read_bytes() == subprocess.check_output(
            ["git", "show", f"{manifest['base_commit']}:{path}"], cwd=ROOT
        )
    required_mutations = (
        "primary_case_removed", "batch_changed", "fp16_fp32_time_swapped", "speedup_increased",
        "process_median_manipulated", "invalid_process_removed", "bootstrap_interval_manipulated",
        "warmup_as_steady_state", "candidate_reference_iterations_differ", "fallback_enabled",
        "numerical_maximum_increased", "scratch_live_nonzero", "descriptor_count_growth",
        "algorithm_id_missing", "manifest_hash_changed", "fewer_than_seven_processes",
        "two_invalid_processes", "network_geomean_below_gate", "encoding_geomean_below_gate",
        "historical_report_changed",
    )
    finalizer_source = FINALIZER.read_text()
    checks = {
        "base": manifest.get("base_commit") == "3265070edbef35969f569972eaf0731d9dab2fe3",
        "marker": manifest.get("marker") == MARKER,
        "manifest_frozen_hash": sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256,
        "schema": required_top <= set(manifest),
        "case_count": len(ids) == total and len(set(ids)) == total,
        "family_counts": (
            sum(x["family"] == "network_only" for x in manifest["cases"]) == network
            and sum(x["family"] == "encoding" for x in manifest["cases"]) == encoding
        ),
        "process_count": manifest["expected_counts"]["fresh_primary_processes"] == total * 7,
        "fresh_processes": manifest["measurement"]["fresh_processes_per_primary_case"] == 7,
        "paired_blocks": manifest["measurement"]["paired_rounds_per_process"] >= 5,
        "calibration": (
            manifest["calibration"]["target_min_ms"] == 250
            and manifest["calibration"]["target_max_ms_approx"] == 1500
            and manifest["calibration"]["min_iterations"] == 50
            and manifest["calibration"]["max_iterations"] == 5000
        ),
        "bootstrap": (
            manifest["statistics"]["bootstrap"]["paired"] is True
            and manifest["statistics"]["bootstrap"]["resamples"] >= 10000
        ),
        "invalid_limit": manifest["valid_process"]["max_invalid_per_case"] == 1,
        "no_outlier_deletion": manifest["measurement"]["no_value_based_outlier_removal"] is True,
        "gates_frozen": manifest["performance_gates"]["network_large_batch_geomean"] == {
            "batch_min": 1024, "forward": 1.20, "forward_backward": 1.15, "adam_training_step": 1.10
        },
        "numerics_frozen": manifest["numerical_gates"]["phase3b1f_starting_baseline"]["dinput"]["max_abs"] == 0.02745274268090725,
        "no_production_changes": not production,
        "historical_reports_byte_equal": historical_equal,
        "f1_manipulations_declared": all(name in finalizer_source for name in required_mutations),
        "worker_explicit_gate": "--execute-primary" in WORKER.read_text(),
        "full_run_explicit_gate": CONFIRMATION in pathlib.Path(__file__).read_text(),
        "python_syntax": all(ast.parse(path.read_text()) for path in (WORKER, pathlib.Path(__file__), FINALIZER)),
    }
    decision = "PHASE3B1F0_PROTOCOL_READY" if all(checks.values()) else "PHASE3B1F0_BLOCKED"
    return {
        "marker": MARKER,
        "decision": decision,
        "checks": checks,
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_primary_cases": total,
        "expected_fresh_processes": total * 7,
        "estimated_operation_iterations": manifest["estimated_operation_iterations"],
    }


def full_measurement(manifest_sha):
    manifest = json.loads(MANIFEST.read_text())
    if sha256(MANIFEST) != manifest_sha:
        raise SystemExit("Manifest changed before F1")
    root = pathlib.Path("/tmp/phase3b1f_processes")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    binding_root = ROOT / "bindings/torch/build/lib.linux-x86_64-cpython-312"
    python_path = f"{binding_root}:{ROOT / 'bindings/torch'}:{ROOT / 'scripts'}"
    for case in manifest["cases"]:
        for process_index in range(manifest["measurement"]["fresh_processes_per_primary_case"]):
            output = root / f"{case['id'].replace('.', '_')}.p{process_index}.json"
            log = output.with_suffix(".hipblaslt.log")
            env = os.environ.copy()
            env["PYTHONPATH"] = python_path + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env.update(manifest["backend_evidence"]["environment"])
            env["HIPBLASLT_LOG_FILE"] = str(log.resolve())
            command = [
                sys.executable, str(WORKER), "--manifest", str(MANIFEST),
                "--manifest-sha256", manifest_sha, "--case-id", case["id"],
                "--process-index", str(process_index), "--output", str(output), "--execute-primary",
            ]
            started = time.time_ns()
            completed = subprocess.run(command, cwd="/tmp", env=env, capture_output=True, text=True)
            record = json.loads(output.read_text()) if output.exists() else {
                "case": case, "process_index": process_index, "valid": False,
                "invalid_reasons": ["missing_process_output"],
            }
            record["returncode"] = completed.returncode
            record["stdout"] = completed.stdout
            record["stderr"] = completed.stderr
            record["orchestrator_elapsed_ns"] = time.time_ns() - started
            if completed.returncode != 0:
                record["valid"] = False
                if "nonzero_returncode" not in record["invalid_reasons"]:
                    record["invalid_reasons"].append("nonzero_returncode")
            records.append(record)
            # No retry: every launched process remains part of the evidence.
            RAW.write_text(json.dumps({
                "marker": MARKER, "manifest_sha256": manifest_sha,
                "complete": False, "primary_processes": records,
            }, indent=2) + "\n")
    cold = []
    for stage in manifest["cold_start"]["stages"]:
        for process_index in range(manifest["cold_start"]["fresh_processes_per_case"]):
            output = root / f"cold_{stage}.p{process_index}.json"
            command = [sys.executable, str(WORKER), "--cold-stage", stage, "--output", str(output)]
            completed = subprocess.run(command, cwd="/tmp", env=env, capture_output=True, text=True)
            item = json.loads(output.read_text()) if output.exists() else {"requested_stage": stage}
            item.update({"process_index": process_index, "returncode": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr,
                         "valid": completed.returncode == 0})
            cold.append(item)
    profiles = []
    profiler = next((tool for tool in manifest["profiling"]["tool_preference"] if shutil.which(tool)), None)
    for profile_index, case_id in enumerate(manifest["profiling"]["cases"]):
        output = root / f"profile_{profile_index}.json"
        profile_dir = root / f"profile_{profile_index}_rocprof"
        worker_command = [
            sys.executable, str(WORKER), "--manifest", str(MANIFEST),
            "--manifest-sha256", manifest_sha, "--case-id", case_id,
            "--process-index", "99", "--output", str(output), "--execute-primary",
        ]
        if profiler == "rocprofv3":
            command = [profiler, "--output-directory", str(profile_dir), "--"] + worker_command
        elif profiler == "rocprof":
            command = [profiler, "--stats", "-d", str(profile_dir)] + worker_command
        else:
            profiles.append({"case_id": case_id, "valid": False, "reason": "profiler_unavailable"})
            continue
        completed = subprocess.run(command, cwd="/tmp", env=env, capture_output=True, text=True)
        profiles.append({"case_id": case_id, "tool": profiler, "returncode": completed.returncode,
                         "output_directory": str(profile_dir), "worker_output": str(output),
                         "stdout": completed.stdout, "stderr": completed.stderr,
                         "valid": completed.returncode == 0})
    raw = {
        "marker": MARKER, "manifest_sha256": manifest_sha, "complete": True,
        "primary_processes": records,
        "expected_primary_cases": manifest["expected_counts"]["supported_primary_cases"],
        "expected_primary_processes": manifest["expected_counts"]["fresh_primary_processes"],
        "cold_start": {"status": "complete", "results": cold},
        "profiling": {"status": "complete", "results": profiles},
        "correctness_pre": None,
        "correctness_post": None,
    }
    RAW.write_text(json.dumps(raw, indent=2) + "\n")
    return subprocess.run([sys.executable, str(FINALIZER), "--raw", str(RAW)], cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-audit", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    audit = protocol_audit()
    if args.output:
        args.output.write_text(json.dumps(audit, indent=2) + "\n")
    if not args.execute_full:
        print(audit["decision"])
        return 0 if audit["decision"] == "PHASE3B1F0_PROTOCOL_READY" else 1
    if audit["decision"] != "PHASE3B1F0_PROTOCOL_READY":
        raise SystemExit("F0 protocol is not ready")
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Full F1 requires --confirm {CONFIRMATION}")
    return full_measurement(audit["manifest_sha256"])


if __name__ == "__main__":
    raise SystemExit(main())
