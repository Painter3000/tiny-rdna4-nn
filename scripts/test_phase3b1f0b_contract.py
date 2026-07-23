#!/usr/bin/env python3
"""F0b four-case smoke and local 2/3 validity contract test."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "phase3b1_reports/phase3b1f0b_reproducible_benchmark_manifest.json"
REPORT = ROOT / "phase3b1_reports/phase3b1f0b_smoke.json"
MARKER = "TCNN_RDNA4_P3B1F0B_REPRODUCIBLE_BENCHMARK_001"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-only", action="store_true")
    parser.add_argument("--single-worker-smoke", action="store_true")
    args = parser.parse_args()
    import finalize_phase3b1f_report as finalizer
    import test_phase3b1f_performance as orchestrator
    manifest = json.loads(MANIFEST.read_text())
    reader = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/phase3b1f_benchmark_worker.py"),
            "--manifest", str(MANIFEST), "--manifest-sha256", sha256(MANIFEST),
            "--print-measurement-config",
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    worker_config = json.loads(reader.stdout.strip()) if reader.returncode == 0 else {}
    reader_contract = {
        "active_manifest_loaded": True,
        "paired_rounds_per_process": worker_config.get("manifest_paired_rounds_per_process"),
        "paired_rounds_equal_five": worker_config.get("manifest_paired_rounds_per_process") == 5,
        "worker_reader_succeeded_without_keyerror": reader.returncode == 0,
        "worker_reader_stderr": reader.stderr,
    }
    if args.reader_only:
        print(json.dumps(reader_contract, indent=2))
        return 0 if all((
            reader_contract["active_manifest_loaded"],
            reader_contract["paired_rounds_equal_five"],
            reader_contract["worker_reader_succeeded_without_keyerror"],
        )) else 2
    cases = {case["id"]: case for case in manifest["cases"]}
    run_id = "f0b_smoke_" + time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    root = pathlib.Path(manifest["process_execution"]["root"]) / run_id
    root.mkdir(parents=True, exist_ok=False)
    snapshot_script = ROOT / "scripts/test_phase3b1f0a_harness.py"
    pre = orchestrator.run_correctness_process(
        snapshot_script, root / "correctness_pre.json", manifest["process_execution"]["timeout_seconds"]
    )
    records = []
    selected_smoke_cases = (
        ["network.N32-L2.b1024.forward"]
        if args.single_worker_smoke else manifest["smoke_cases"]
    )
    for case_id in selected_smoke_cases:
        case = cases[case_id]
        output = root / f"{case_id.replace('.', '_')}.json"
        record = orchestrator.launch_worker(
            case, 0, output, sha256(MANIFEST), manifest,
            ("--paired-rounds", "1", "--fixed-iterations", "3", "--harness-smoke"),
        )
        valid, checks = finalizer.process_valid(record, case, manifest, smoke=True)
        records.append({
            "case_id": case_id, "path": str(output), "sha256": sha256(output),
            "valid": valid, "checks": checks,
        })
    post = orchestrator.run_correctness_process(
        snapshot_script, root / "correctness_post.json", manifest["process_execution"]["timeout_seconds"]
    )
    local = finalizer.local_validity_contract_test(manifest)
    historical = {
        "phase3b1_reports/PHASE3B1F_PROTOCOL.md": "f88163eeaeadb8785dcfdc31e8db49df7e07f97718333da637ab4145ff041b4a",
        "phase3b1_reports/phase3b1f_protocol_manifest.json": "cf097d672595e72ca58a4090bff1135882f2181756f8e8ffb70281ea2fcfe0e3",
        "phase3b1_reports/PHASE3B1F0A_HARNESS_HARDENING.md": "4ed765b23588ca4ad0abd30d01d6a94fedf2a346291955e863f2627a41f7ea78",
        "phase3b1_reports/PHASE3B1F0A1_FINAL_HARNESS_CLOSURE.md": "711000b22667b165ce61dee8be8cf25b2ec569ca14800766a510eed7427b7ef2",
    }
    historical_equal = all(sha256(ROOT / path) == digest for path, digest in historical.items())
    production = subprocess.check_output(
        ["git", "diff", "--name-only", manifest["base_commit"], "--", "src", "include", "bindings"],
        cwd=ROOT, text=True,
    ).splitlines()
    decision = (
        "PROCEED_TO_3B1F1_72_PROCESS_MEASUREMENT"
        if pre.get("passed") is True and post.get("passed") is True
        and all(item["valid"] for item in records) and local["passed"]
        and historical_equal and not production
        else "PHASE3B1F0B_BLOCKED"
    )
    report = {
        "marker": MARKER, "decision": decision,
        "manifest_path": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
        "run_directory": str(root), "include_in_f1_statistics": False,
        "correctness_pre": pre, "correctness_post": post,
        "cases": records, "validity_contract_test": local,
        "worker_manifest_reader_contract": reader_contract,
        "historical_artifacts_byte_equal": historical_equal,
        "production_files_changed": production, "f1_full_measurement_executed": False,
    }
    report_path = pathlib.Path("/tmp/phase3b1f0b2_single_worker_smoke.json") if args.single_worker_smoke else REPORT
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(decision)
    return 0 if decision == "PROCEED_TO_3B1F1_72_PROCESS_MEASUREMENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
