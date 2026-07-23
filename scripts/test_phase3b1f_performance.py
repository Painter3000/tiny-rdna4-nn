#!/usr/bin/env python3
"""Protocol audit and explicit F1 orchestrator for Phase 3B1-F."""
import argparse
import ast
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "phase3b1_reports/phase3b1f_protocol_manifest.json"
WORKER = ROOT / "scripts/phase3b1f_benchmark_worker.py"
FINALIZER = ROOT / "scripts/finalize_phase3b1f_report.py"
CONTRACT = ROOT / "phase3b1_reports/phase3b1f0a_harness_contract.json"
RAW = pathlib.Path("/tmp/phase3b1f_fp16_performance_raw.json")
MARKER = "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001"
CONFIRMATION = "RUN_PHASE3B1F1_FULL_MEASUREMENT"
EXPECTED_MANIFEST_SHA256 = "cf097d672595e72ca58a4090bff1135882f2181756f8e8ffb70281ea2fcfe0e3"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_profile_directory(path):
    rows = []
    for csv_path in sorted(path.rglob("*.csv")) if path.exists() else []:
        try:
            with csv_path.open(errors="replace") as handle:
                for row in csv.DictReader(handle):
                    lowered = {str(key).lower(): value for key, value in row.items()}
                    name = next((lowered[key] for key in lowered if "kernel" in key and "name" in key), None)
                    duration = next((lowered[key] for key in lowered if "duration" in key or "totalduration" in key), None)
                    if name and duration:
                        rows.append((name, float(str(duration).replace(",", ""))))
        except (OSError, ValueError, TypeError, csv.Error):
            continue
    totals = {}
    for name, duration in rows:
        entry = totals.setdefault(name, {"kernel_name": name, "kernel_count": 0, "kernel_time": 0.0})
        entry["kernel_count"] += 1
        entry["kernel_time"] += duration
    kernels = sorted(totals.values(), key=lambda item: item["kernel_time"], reverse=True)
    total_time = sum(item["kernel_time"] for item in kernels)
    def share(pattern):
        return sum(item["kernel_time"] for item in kernels if re.search(pattern, item["kernel_name"], re.I)) / total_time if total_time else 0.0
    return {
        "kernels": kernels,
        "gemm_share": share(r"gemm|hipblas|matmul"),
        "encoding_share": share(r"grid|encoding|frequency|spherical"),
        "gradient_scratch_share": share(r"gradient|backward|reduce|scratch"),
        "optimizer_share": share(r"adam|optimizer"),
        "unexpected_copies": sum(item["kernel_count"] for item in kernels if re.search(r"copy|memcpy", item["kernel_name"], re.I)),
        "host_synchronizations": sum(item["kernel_count"] for item in kernels if re.search(r"synchron", item["kernel_name"], re.I)),
        "parsed_kernel_rows": len(rows),
    }


def expected_cases():
    network = 4 * 4 * 3
    encoding = 5 * 3 * 5
    return network, encoding, network + encoding


def parse_algorithm_log(path, contract, case_id, process_index):
    text = path.read_text(errors="replace") if path.exists() else ""
    ids = []
    for expression in contract["backend_evidence"]["gemm"]["log_id_regexes"]:
        ids.extend(int(value) for value in re.findall(expression, text))
    workspace = [int(value) for value in re.findall(r"(?i)workspace(?:_size|Size| bytes)?[^0-9]*([0-9]+)", text)]
    records = [line for line in text.splitlines() if re.search(r"(?i)(matmul|gemm|algo|solution)", line)]
    trans_a = sorted(set(re.findall(r"--transA\s+([NTC])", text)))
    trans_b = sorted(set(re.findall(r"--transB\s+([NTC])", text)))
    data_types = sorted(set(re.findall(r"--[abcd]_type\s+(\S+)", text)))
    compute_types = sorted(set(re.findall(r"--compute_type\s+(\S+)", text)))
    epilogues = sorted(set(re.findall(r"--activation_type\s+(\S+)", text)))
    return {
        "case_id": case_id, "process_index": process_index, "path": str(path),
        "sha256": sha256(path) if path.exists() else None, "size_bytes": len(text.encode()),
        "algorithm_ids": sorted(set(ids)), "workspace_bytes": max(workspace, default=None),
        "record_count": len(records), "sample_records": records[:20],
        "trans_a": trans_a, "trans_b": trans_b, "data_types": data_types,
        "compute_types": compute_types, "epilogues": epilogues,
        "fallback_observed": bool(re.search(r"(?i)fallback", text)),
    }


def launch_worker(case, process_index, output, manifest_sha, contract, extra_args=()):
    log = output.with_suffix(".hipblaslt.log")
    output.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    binding_root = ROOT / "bindings/torch/build/lib.linux-x86_64-cpython-312"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{binding_root}:{ROOT / 'bindings/torch'}:{ROOT / 'scripts'}" + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.update(json.loads(MANIFEST.read_text())["backend_evidence"]["environment"])
    env["HIPBLASLT_LOG_FILE"] = str(log.resolve())
    command = [
        sys.executable, str(WORKER), "--manifest", str(MANIFEST),
        "--manifest-sha256", manifest_sha, "--harness-contract", str(CONTRACT),
        "--harness-contract-sha256", sha256(CONTRACT),
        "--case-id", case["id"], "--process-index", str(process_index),
        "--output", str(output), "--execute-primary", *extra_args,
    ]
    started = time.time_ns()
    try:
        completed = subprocess.run(
            command, cwd="/tmp", env=env, capture_output=True, text=True,
            timeout=contract["process_execution"]["timeout_seconds"],
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        completed = subprocess.CompletedProcess(command, 124, error.stdout or "", error.stderr or "")
        timed_out = True
    fresh = output.exists() and output.stat().st_mtime_ns >= started
    record = json.loads(output.read_text()) if fresh else {
        "marker": MARKER, "manifest_sha256": manifest_sha, "case": case,
        "process_index": process_index, "valid": False, "invalid_reasons": ["missing_or_stale_process_output"],
    }
    evidence = parse_algorithm_log(log, contract, case["id"], process_index)
    gemm_expected = record.get("backend_evidence", {}).get("gemm_expected")
    if gemm_expected is True:
        record["backend_evidence"]["algorithm_ids"] = evidence["algorithm_ids"]
        if evidence["workspace_bytes"] is not None:
            record["backend_evidence"]["workspace_bytes"] = evidence["workspace_bytes"]
        record["backend_evidence"]["fallback"] = evidence["fallback_observed"]
        record["backend_evidence"]["fallback_measurement_source"] = "orchestrator_hipblaslt_log"
        timed_iterations = sum(block.get("iterations", 0) for block in record.get("timing_blocks", []))
        record["backend_evidence"]["gemm_call_count"] = evidence["record_count"]
        record["backend_evidence"]["gemm_calls_per_iteration"] = evidence["record_count"] / timed_iterations if timed_iterations else None
        record["backend_evidence"]["candidate"].update({
            "trans_a": evidence["trans_a"], "trans_b": evidence["trans_b"],
            "matrix_layouts": "ColumnMajor", "epilogues": evidence["epilogues"],
            "logged_data_types": evidence["data_types"], "logged_compute_types": evidence["compute_types"],
        })
        if not evidence["algorithm_ids"]:
            record.setdefault("invalid_reasons", []).append("missing_algorithm_id")
    elif gemm_expected is False:
        record["backend_evidence"].update({
            "backend_kind": "encoding_kernel", "algorithm_ids": "not_applicable",
            "workspace_bytes": 0, "fallback": "not_applicable",
            "gemm_call_count": 0, "gemm_calls_per_iteration": 0,
        })
    record["backend_evidence"]["algorithm_log"] = evidence
    identity_ok = (
        record.get("marker") == MARKER
        and record.get("manifest_sha256") == manifest_sha
        and record.get("harness_contract_sha256") == sha256(CONTRACT)
        and record.get("case", {}).get("id") == case["id"]
        and record.get("process_index") == process_index
    )
    if not identity_ok:
        record.setdefault("invalid_reasons", []).append("worker_identity_mismatch")
    if timed_out:
        record.setdefault("invalid_reasons", []).append("process_timeout")
    if completed.returncode not in (0, 2):
        record.setdefault("invalid_reasons", []).append("nonzero_returncode")
    record.update({
        "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
        "orchestrator_elapsed_ns": time.time_ns() - started, "output_fresh": fresh,
        "process_file_sha256": sha256(output) if fresh else None,
        "process_exit_verified": True,
    })
    record["valid"] = record.get("valid") is True and identity_ok and not timed_out and not record["invalid_reasons"]
    output.write_text(json.dumps(record, indent=2) + "\n")
    record["process_file_sha256"] = sha256(output)
    return record


def load_resumable_record(output, index_entry, case, process_index, manifest_sha, contract_sha, run_root):
    if output.resolve().parent != run_root.resolve() or index_entry.get("path") != str(output):
        raise RuntimeError(f"Unsafe resume path for {output}")
    if not output.is_file() or sha256(output) != index_entry.get("sha256"):
        raise RuntimeError(f"Resume SHA256 mismatch for {output}")
    record = json.loads(output.read_text())
    if not all((
        record.get("marker") == MARKER,
        record.get("case", {}).get("id") == case["id"],
        record.get("process_index") == process_index,
        record.get("manifest_sha256") == manifest_sha,
        record.get("harness_contract_sha256") == contract_sha,
    )):
        raise RuntimeError(f"Resume identity mismatch for {output}")
    record["resumed_after_full_validation"] = True
    return record


def run_correctness_process(script, output, timeout):
    output.unlink(missing_ok=True)
    started = time.time_ns()
    command = [sys.executable, str(script), "--correctness-snapshot", str(output)]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        completed = subprocess.CompletedProcess(command, 124, error.stdout or "", error.stderr or "")
        timed_out = True
    fresh = output.exists() and output.stat().st_mtime_ns >= started
    result = json.loads(output.read_text()) if fresh else {"passed": False, "error": "missing_or_stale_correctness"}
    result.update({"returncode": completed.returncode, "output_fresh": fresh, "timed_out": timed_out})
    result["passed"] = result.get("passed") is True and completed.returncode == 0 and fresh and not timed_out
    return result


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


def full_measurement(manifest_sha, resume_run_dir=None):
    manifest = json.loads(MANIFEST.read_text())
    if sha256(MANIFEST) != manifest_sha:
        raise SystemExit("Manifest changed before F1")
    contract = json.loads(CONTRACT.read_text())
    contract_sha = sha256(CONTRACT)
    if resume_run_dir:
        root = resume_run_dir.resolve()
        if root.parent != pathlib.Path(contract["process_execution"]["root"]).resolve() or not root.is_dir():
            raise SystemExit("Unsafe or missing resume run directory")
        run_id = root.name
    else:
        run_id = time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
        root = pathlib.Path(contract["process_execution"]["root"]) / run_id
        root.mkdir(parents=True, exist_ok=False)
    index_path = root / contract["process_execution"]["append_only_index"]
    state_path = root / contract["process_execution"]["state_file"]
    resume_entries = {}
    if resume_run_dir and index_path.exists():
        for line in index_path.read_text().splitlines():
            entry = json.loads(line)
            key = (entry["case_id"], entry["process_index"])
            if key in resume_entries:
                raise SystemExit(f"Duplicate resume index entry: {key}")
            resume_entries[key] = entry
    snapshot_script = ROOT / "scripts/test_phase3b1f0a_harness.py"
    pre_path = root / "correctness_pre.json"
    correctness_pre = run_correctness_process(
        snapshot_script, pre_path, contract["process_execution"]["timeout_seconds"]
    )
    records = []
    for case in manifest["cases"]:
        for process_index in range(manifest["measurement"]["fresh_processes_per_primary_case"]):
            output = root / f"{case['id'].replace('.', '_')}.p{process_index}.json"
            key = (case["id"], process_index)
            record = (
                load_resumable_record(output, resume_entries[key], case, process_index, manifest_sha, contract_sha, root)
                if key in resume_entries else launch_worker(case, process_index, output, manifest_sha, contract)
            )
            records.append(record)
            if key not in resume_entries:
                with index_path.open("a") as index:
                    index.write(json.dumps({"case_id": case["id"], "process_index": process_index,
                                            "path": str(output), "sha256": record["process_file_sha256"],
                                            "valid": record["valid"]}) + "\n")
            state_path.write_text(json.dumps({"run_id": run_id, "completed": len(records),
                                              "expected": manifest["expected_counts"]["fresh_primary_processes"]}, indent=2) + "\n")
    cold = []
    env = os.environ.copy()
    binding_root = ROOT / "bindings/torch/build/lib.linux-x86_64-cpython-312"
    env["PYTHONPATH"] = f"{binding_root}:{ROOT / 'bindings/torch'}:{ROOT / 'scripts'}" + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for stage in manifest["cold_start"]["stages"]:
        for process_index in range(manifest["cold_start"]["fresh_processes_per_case"]):
            output = root / f"cold_{stage}.p{process_index}.json"
            output.unlink(missing_ok=True)
            command = [sys.executable, str(WORKER), "--cold-stage", stage, "--output", str(output)]
            started = time.time_ns()
            try:
                completed = subprocess.run(
                    command, cwd="/tmp", env=env, capture_output=True, text=True,
                    timeout=contract["process_execution"]["timeout_seconds"],
                )
                timed_out = False
            except subprocess.TimeoutExpired as error:
                completed = subprocess.CompletedProcess(command, 124, error.stdout or "", error.stderr or "")
                timed_out = True
            fresh = output.exists() and output.stat().st_mtime_ns >= started
            item = json.loads(output.read_text()) if fresh else {"requested_stage": stage}
            item.update({"process_index": process_index, "returncode": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr,
                         "output_fresh": fresh, "timed_out": timed_out,
                         "valid": completed.returncode == 0 and fresh and not timed_out})
            cold.append(item)
    profiles = []
    profiler = next((tool for tool in manifest["profiling"]["tool_preference"] if shutil.which(tool)), None)
    for profile_index, case_id in enumerate(manifest["profiling"]["cases"]):
        output = root / f"profile_{profile_index}.json"
        profile_dir = root / f"profile_{profile_index}_rocprof"
        output.unlink(missing_ok=True)
        worker_command = [
            sys.executable, str(WORKER), "--manifest", str(MANIFEST),
            "--manifest-sha256", manifest_sha, "--harness-contract", str(CONTRACT),
            "--harness-contract-sha256", contract_sha, "--case-id", case_id,
            "--process-index", "99", "--output", str(output), "--execute-primary",
        ]
        if profiler == "rocprofv3":
            command = [profiler, "--output-directory", str(profile_dir), "--"] + worker_command
        elif profiler == "rocprof":
            command = [profiler, "--stats", "-d", str(profile_dir)] + worker_command
        else:
            profiles.append({"case_id": case_id, "valid": False, "reason": "profiler_unavailable"})
            continue
        started = time.time_ns()
        try:
            completed = subprocess.run(
                command, cwd="/tmp", env=env, capture_output=True, text=True,
                timeout=contract["process_execution"]["timeout_seconds"],
            )
            timed_out = False
        except subprocess.TimeoutExpired as error:
            completed = subprocess.CompletedProcess(command, 124, error.stdout or "", error.stderr or "")
            timed_out = True
        fresh = output.exists() and output.stat().st_mtime_ns >= started
        parsed = parse_profile_directory(profile_dir)
        profiles.append({"case_id": case_id, "tool": profiler, "returncode": completed.returncode,
                         "output_directory": str(profile_dir), "worker_output": str(output),
                         "stdout": completed.stdout, "stderr": completed.stderr,
                         "parsed": parsed, "output_fresh": fresh, "timed_out": timed_out,
                         "valid": completed.returncode == 0 and fresh and not timed_out
                                  and parsed["parsed_kernel_rows"] > 0})
    raw = {
        "marker": MARKER, "manifest_sha256": manifest_sha,
        "harness_contract_sha256": contract_sha, "complete": True,
        "primary_processes": records,
        "expected_primary_cases": manifest["expected_counts"]["supported_primary_cases"],
        "expected_primary_processes": manifest["expected_counts"]["fresh_primary_processes"],
        "cold_start": {"status": "complete", "results": cold},
        "profiling": {"status": "complete", "results": profiles},
        "correctness_pre": correctness_pre,
        "correctness_post": None,
    }
    post_path = root / "correctness_post.json"
    raw["correctness_post"] = run_correctness_process(
        snapshot_script, post_path, contract["process_execution"]["timeout_seconds"]
    )
    RAW.write_text(json.dumps(raw, indent=2) + "\n")
    return subprocess.run([sys.executable, str(FINALIZER), "--raw", str(RAW)], cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-audit", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--resume-run-dir", type=pathlib.Path)
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
    return full_measurement(audit["manifest_sha256"], args.resume_run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
