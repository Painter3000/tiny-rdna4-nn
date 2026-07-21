#!/usr/bin/env python3
"""Orchestrate fresh-process conditioned Phase-3A3/3A4 comparisons."""
import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys

CASES = ("large_1024_w64_relu", "large_1024_w128_relu", "large_4096_w64_relu", "large_4096_w64_none")


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def child(script, binding, variant, case, output):
    command = [sys.executable, str(script), "--bindings", binding, "--variant", variant,
        "--case", case, "--output", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if not output.exists():
        raise RuntimeError({"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr})
    result = json.loads(output.read_text())
    result["process_returncode"] = completed.returncode
    result["process_stdout"] = completed.stdout
    result["process_stderr"] = completed.stderr
    return result


def metric_median(run, metric):
    measurement = run["metrics"][metric]["measurement"]
    return measurement["median_ms"] if measurement else None


def paired(script, bindings, case, pair_number, root):
    order = ("phase3a3", "phase3a4") if pair_number % 2 else ("phase3a4", "phase3a3")
    runs = {}
    for variant in order:
        out = root / f"pair_{pair_number:02d}_{variant}.json"
        runs[variant] = child(script, bindings[variant], variant, case, out)
    ratios = {"forward": None, "forward_backward": None}
    valid = all(run["valid"] for run in runs.values())
    for metric in ("forward", "forward_backward"):
        base = metric_median(runs["phase3a3"], metric)
        candidate = metric_median(runs["phase3a4"], metric)
        if base is not None and candidate is not None:
            ratios[metric] = base / candidate
    return {"pair": pair_number, "order": list(order), "case": case, "runs": runs,
        "ratios": ratios, "valid": valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("baseline", "none-pairs", "official"), required=True)
    parser.add_argument("--output-dir", default="phase3a4_reports/conditioned_raw")
    parser.add_argument("--baseline-output")
    parser.add_argument("--pairs-output", default="phase3a4_reports/conditioned_none_pairs.json")
    parser.add_argument("--baseline-repetitions", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    bindings = manifest["bindings"]
    script = pathlib.Path(__file__).with_name("benchmark_phase3a4_conditioned_case.py")
    root = pathlib.Path(args.output_dir) / args.mode
    root.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        if not args.baseline_output:
            raise SystemExit("--baseline-output is required in baseline mode")
        if args.baseline_repetitions < 1:
            raise SystemExit("baseline repetitions must be positive")
        cases = []
        for case in CASES:
            runs = [child(script, bindings["phase3a3"], "phase3a3", case,
                root / f"{case}_{index:02d}.json") for index in range(1, args.baseline_repetitions + 1)]
            valid = all(run["valid"] for run in runs)
            cases.append({"name": case, "runs": runs, "all_valid": valid,
                "forward": {"median_ms": statistics.median(metric_median(run, "forward") for run in runs) if valid else None},
                "forward_backward": {"median_ms": statistics.median(metric_median(run, "forward_backward") for run in runs) if valid else None}})
        result = {"schema": 1, "source": "unchanged phase3a3 blocked commit", "binding": bindings["phase3a3"],
            "conditioning_protocol": "adaptive-50x5-to-100", "repetitions": args.baseline_repetitions,
            "cases": cases, "valid": all(case["all_valid"] for case in cases)}
        baseline_output = pathlib.Path(args.baseline_output)
        baseline_output.parent.mkdir(parents=True, exist_ok=True)
        baseline_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("PHASE3A3_CONDITIONED_BASELINE=" + ("VALID" if result["valid"] else "INVALID"))
        raise SystemExit(0 if result["valid"] else 2)

    if args.mode == "none-pairs":
        if args.pairs < 10:
            raise SystemExit("At least ten exploratory pairs are required")
        pairs = [paired(script, bindings, "large_4096_w64_none", index, root)
            for index in range(1, args.pairs + 1)]
        valid = all(pair["valid"] for pair in pairs)
        base = [metric_median(pair["runs"]["phase3a3"], "forward_backward") for pair in pairs]
        candidate = [metric_median(pair["runs"]["phase3a4"], "forward_backward") for pair in pairs]
        ratio = statistics.median(base) / statistics.median(candidate) if valid else None
        result = {"schema": 1, "case": "large_4096_w64_none", "pair_count": len(pairs), "pairs": pairs,
            "phase3a3_conditioned_median_ms": statistics.median(base),
            "phase3a4_conditioned_median_ms": statistics.median(candidate), "conditioned_ratio": ratio,
            "gate": {"threshold": 0.99, "comparison": "phase3a3_median / phase3a4_median",
                "all_runs_valid": valid, "pass": valid and ratio is not None and ratio >= 0.99},
            "workload_order_sensitivity": {"release_gate": False,
                "report": "none_regression_bisect.json", "sequence": "relu-none"}}
        pathlib.Path(args.pairs_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("PHASE3A4_CONDITIONED_NONE=" + ("PASS" if result["gate"]["pass"] else "FAIL"))
        raise SystemExit(0 if result["gate"]["pass"] else 1)

    pairs_doc = json.loads(pathlib.Path(args.pairs_output).read_text())
    if not pairs_doc["gate"]["pass"]:
        raise SystemExit("Conditioned None gate is not PASS; official runs are locked")
    existing = [pathlib.Path(f"phase3a4_reports/conditioned_performance_run_{index}.json") for index in range(1, 5)]
    if any(path.exists() for path in existing):
        raise SystemExit("Official output already exists; refusing to overwrite or add runs")
    for run_number, output in enumerate(existing, 1):
        case_pairs = [paired(script, bindings, case, run_number, root / f"run_{run_number}") for case in CASES]
        rows = []
        for pair in case_pairs:
            rows.append({"name": pair["case"], "pair": pair, "forward_ratio": pair["ratios"]["forward"],
                "forward_backward_ratio": pair["ratios"]["forward_backward"]})
        relu = [row for row in rows if row["name"].endswith("_relu")]
        all_valid = all(pair["valid"] for pair in case_pairs)
        summary = {"forward_geomean": geomean([row["forward_ratio"] for row in rows]) if all_valid else None,
            "relu_forward_backward_geomean": geomean([row["forward_backward_ratio"] for row in relu]) if all_valid else None,
            "minimum_relu_forward_backward": min(row["forward_backward_ratio"] for row in relu) if all_valid else None,
            "maximum_relu_forward_backward": max(row["forward_backward_ratio"] for row in relu) if all_valid else None,
            "none_forward_backward": next(row["forward_backward_ratio"] for row in rows if row["name"].endswith("_none")) if all_valid else None}
        gates = {"all_cases_valid": all_valid,
            "forward_geomean_ge_0_99": all_valid and summary["forward_geomean"] >= 0.99,
            "minimum_relu_forward_backward_ge_0_98": all_valid and summary["minimum_relu_forward_backward"] >= 0.98,
            "relu_forward_backward_geomean_ge_1_02": all_valid and summary["relu_forward_backward_geomean"] >= 1.02,
            "maximum_relu_forward_backward_ge_1_04": all_valid and summary["maximum_relu_forward_backward"] >= 1.04,
            "none_forward_backward_ge_0_99": all_valid and summary["none_forward_backward"] >= 0.99}
        result = {"schema": 1, "official_run": run_number, "case_process_isolation": True,
            "alternating_pair_order": case_pairs[0]["order"], "cases": rows, "summary": summary,
            "gates": gates, "result": "PASS" if all(gates.values()) else "FAIL"}
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    all_pass = all(json.loads(path.read_text())["result"] == "PASS" for path in existing)
    print("PHASE3A4_CONDITIONED_OFFICIAL=" + ("PASS" if all_pass else "FAIL"))
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
