#!/usr/bin/env python3
"""Run exactly ten alternating native-window Protocol-v4 pairs."""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, root = pathlib.Path(args.output), pathlib.Path(args.output_dir)
    if output.exists() or root.exists():
        print("PHASE3A4_WIDTH128_FORWARD_V4=INFRASTRUCTURE_FAIL")
        return 2
    bindings = json.loads(pathlib.Path(args.manifest).read_text())["bindings"]
    child = pathlib.Path(__file__).with_name("benchmark_phase3a4_conditioned_metric_v4.py")
    pairs, invalid_environment = [], False
    for number in range(1, 11):
        order = ("phase3a3", "phase3a4") if number % 2 else ("phase3a4", "phase3a3")
        runs = {}
        for variant in order:
            raw = root / f"pair_{number:02d}_large_1024_w128_relu_forward_{variant}.json"
            command = [sys.executable, str(child), "--bindings", bindings[variant], "--variant", variant, "--output", str(raw)]
            completed = subprocess.run(command, text=True, capture_output=True)
            run = json.loads(raw.read_text()) if raw.exists() else {"status": "INFRASTRUCTURE_FAIL", "valid": False}
            run.update({"process_returncode": completed.returncode, "process_stdout": completed.stdout,
                "process_stderr": completed.stderr})
            runs[variant] = run
            invalid_environment = invalid_environment or run.get("status") == "INVALID_ENVIRONMENT"
        valid = all(run.get("valid", False) for run in runs.values())
        values = {}
        for variant in ("phase3a3", "phase3a4"):
            stationarity = runs[variant].get("measurement", {}).get("stationarity")
            values[variant] = stationarity.get("overall_median_per_operation_ms") if stationarity else None
        pairs.append({"pair": number, "order": list(order), "runs": runs, "valid": valid,
            "phase3a3_per_operation_ms": values["phase3a3"], "phase3a4_per_operation_ms": values["phase3a4"],
            "ratio": values["phase3a3"] / values["phase3a4"] if all(value is not None for value in values.values()) else None})
    all_valid = all(pair["valid"] for pair in pairs)
    base = [pair["phase3a3_per_operation_ms"] for pair in pairs if pair["phase3a3_per_operation_ms"] is not None]
    candidate = [pair["phase3a4_per_operation_ms"] for pair in pairs if pair["phase3a4_per_operation_ms"] is not None]
    ratio = statistics.median(base) / statistics.median(candidate) if all_valid else None
    if invalid_environment:
        status = "INVALID_ENVIRONMENT"
    elif not all_valid:
        status = "INFRASTRUCTURE_FAIL"
    elif ratio >= 0.99:
        status = "PASS"
    else:
        status = "PERFORMANCE_FAIL"
    document = {"schema": 4, "protocol": "conditioning_v4", "status": status,
        "case": "large_1024_w128_relu", "metric": "forward", "pair_count": 10, "pairs": pairs,
        "phase3a3_median_per_operation_ms": statistics.median(base) if len(base) == 10 else None,
        "phase3a4_median_per_operation_ms": statistics.median(candidate) if len(candidate) == 10 else None,
        "conditioned_ratio": ratio, "gate": {"threshold": 0.99, "all_pairs_valid": all_valid,
            "pass": all_valid and ratio is not None and ratio >= 0.99}, "official_series_started": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print("PHASE3A4_WIDTH128_FORWARD_V4=" + status)
    return 0 if status == "PASS" else (3 if status == "INVALID_ENVIRONMENT" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
