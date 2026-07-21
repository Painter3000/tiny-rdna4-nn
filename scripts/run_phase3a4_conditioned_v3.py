#!/usr/bin/env python3
"""Run exactly ten alternating Width-128 Forward pairs under protocol v3."""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys

FORBIDDEN_MONITORS = {"amd-smi", "rocm-smi", "nvtop", "radeontop"}


def monitor_processes():
    found = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = pathlib.Path(cmdline.split()[0]).name if cmdline else comm
        if comm in FORBIDDEN_MONITORS or executable in FORBIDDEN_MONITORS:
            found.append({"pid": int(entry.name), "comm": comm, "cmdline": cmdline})
    return sorted(found, key=lambda item: item["pid"])


def write(document, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="phase3a4_reports/conditioned_raw_v3_exploratory")
    parser.add_argument("--output", default="phase3a4_reports/conditioned_width128_forward_pairs_v3.json")
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    root = pathlib.Path(args.output_dir)
    if output.exists() or root.exists():
        print("PHASE3A4_WIDTH128_FORWARD_V3=INFRASTRUCTURE_FAIL")
        return 2
    monitors = monitor_processes()
    if monitors:
        write({"schema": 3, "protocol": "conditioning_v3", "status": "INVALID_ENVIRONMENT",
            "forbidden_monitor_processes": monitors, "pairs": []}, output)
        print("PHASE3A4_WIDTH128_FORWARD_V3=INVALID_ENVIRONMENT")
        return 3
    bindings = json.loads(pathlib.Path(args.manifest).read_text())["bindings"]
    child = pathlib.Path(__file__).with_name("benchmark_phase3a4_conditioned_metric_v3.py")
    pairs = []
    invalid_environment = False
    infrastructure_failure = False
    for number in range(1, 11):
        order = ("phase3a3", "phase3a4") if number % 2 else ("phase3a4", "phase3a3")
        runs = {}
        for variant in order:
            raw = root / f"pair_{number:02d}_large_1024_w128_relu_forward_{variant}.json"
            command = [sys.executable, str(child), "--bindings", bindings[variant],
                "--variant", variant, "--output", str(raw)]
            completed = subprocess.run(command, text=True, capture_output=True)
            if raw.exists():
                run = json.loads(raw.read_text())
            else:
                run = {"schema": 3, "protocol": "conditioning_v3", "variant": variant,
                    "status": "INFRASTRUCTURE_FAIL", "valid": False, "missing_child_output": True}
            run["process_returncode"] = completed.returncode
            run["process_stdout"] = completed.stdout
            run["process_stderr"] = completed.stderr
            runs[variant] = run
            invalid_environment = invalid_environment or run.get("status") == "INVALID_ENVIRONMENT"
            infrastructure_failure = infrastructure_failure or not run.get("valid", False)
        valid = all(run.get("valid", False) for run in runs.values())
        base_measurement = runs["phase3a3"].get("measurement", {})
        candidate_measurement = runs["phase3a4"].get("measurement", {})
        base_stationarity = base_measurement.get("stationarity")
        candidate_stationarity = candidate_measurement.get("stationarity")
        base = base_stationarity.get("overall_median_per_operation_ms") if base_stationarity else None
        candidate = candidate_stationarity.get("overall_median_per_operation_ms") if candidate_stationarity else None
        pairs.append({"pair": number, "order": list(order), "runs": runs, "valid": valid,
            "phase3a3_per_operation_ms": base, "phase3a4_per_operation_ms": candidate,
            "ratio": base / candidate if base is not None and candidate is not None else None})
    all_valid = all(pair["valid"] for pair in pairs)
    base_values = [pair["phase3a3_per_operation_ms"] for pair in pairs if pair["phase3a3_per_operation_ms"] is not None]
    candidate_values = [pair["phase3a4_per_operation_ms"] for pair in pairs if pair["phase3a4_per_operation_ms"] is not None]
    ratio = statistics.median(base_values) / statistics.median(candidate_values) if all_valid else None
    performance_pass = all_valid and ratio is not None and ratio >= 0.99
    if invalid_environment:
        status = "INVALID_ENVIRONMENT"
    elif infrastructure_failure or not all_valid:
        status = "INFRASTRUCTURE_FAIL"
    elif performance_pass:
        status = "PASS"
    else:
        status = "PERFORMANCE_FAIL"
    document = {"schema": 3, "protocol": "conditioning_v3", "status": status,
        "case": "large_1024_w128_relu", "metric": "forward", "pair_count": 10,
        "pairs": pairs, "phase3a3_median_per_operation_ms": statistics.median(base_values) if len(base_values) == 10 else None,
        "phase3a4_median_per_operation_ms": statistics.median(candidate_values) if len(candidate_values) == 10 else None,
        "conditioned_ratio": ratio, "gate": {"threshold": 0.99,
            "comparison": "phase3a3_median_per_op / phase3a4_median_per_op",
            "all_pairs_valid": all_valid, "pass": performance_pass},
        "official_series_v3_planning_unlocked": performance_pass,
        "official_series_started": False, "forbidden_monitor_processes": monitors}
    write(document, output)
    print("PHASE3A4_WIDTH128_FORWARD_V3=" + status)
    return 0 if status == "PASS" else (3 if status == "INVALID_ENVIRONMENT" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
