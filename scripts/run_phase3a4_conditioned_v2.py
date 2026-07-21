#!/usr/bin/env python3
"""Orchestrate exploratory and official Phase-3A4 conditioning protocol v2."""
import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys

CASES = ("large_1024_w64_relu", "large_1024_w128_relu", "large_4096_w64_relu", "large_4096_w64_none")
METRICS = ("forward", "forward_backward")


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def child(script, binding, variant, case, metric, output):
    command = [sys.executable, str(script), "--bindings", binding, "--variant", variant,
        "--case", case, "--metric", metric, "--output", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if not output.exists():
        raise RuntimeError({"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr})
    run = json.loads(output.read_text())
    run["process_returncode"] = completed.returncode
    run["process_stdout"] = completed.stdout
    run["process_stderr"] = completed.stderr
    return run


def pair(script, bindings, case, metrics, number, root):
    order = ("phase3a3", "phase3a4") if number % 2 else ("phase3a4", "phase3a3")
    runs = {variant: {} for variant in order}
    for variant in order:
        for metric in metrics:
            output = root / f"pair_{number:02d}_{case}_{metric}_{variant}.json"
            runs[variant][metric] = child(script, bindings[variant], variant, case, metric, output)
    valid = all(run["valid"] for variant in runs.values() for run in variant.values())
    ratios = {}
    for metric in metrics:
        base = runs["phase3a3"][metric]["measurement"]
        candidate = runs["phase3a4"][metric]["measurement"]
        ratios[metric] = base["median_ms"] / candidate["median_ms"] if base and candidate else None
    return {"pair": number, "case": case, "metrics": list(metrics), "order": list(order),
        "runs": runs, "ratios": ratios, "valid": valid}


def comparison_report(v1_paths, v2_paths, output):
    v1 = [json.loads(path.read_text()) for path in v1_paths]
    v2 = [json.loads(path.read_text()) for path in v2_paths]
    lines = ["# Conditioning Protocol v1 versus v2", "", "## Method", "",
        "Protocol v1 used one process per case and conditioned the complete forward-backward workload before measuring both metrics. "
        "Protocol v2 uses one fresh process per case, metric and variant; each metric conditions only itself and the 150-value measurement must pass six fixed 25-value stationarity blocks.", "",
        "## Official outcomes", "", "| Run | Protocol v1 | Protocol v2 | v1 forward geomean | v2 forward geomean |", "|---:|:---:|:---:|---:|---:|"]
    for index in range(4):
        lines.append(f"| {index+1} | {v1[index]['result']} | {v2[index]['result']} | {v1[index]['summary']['forward_geomean']:.10f} | " +
            (f"{v2[index]['summary']['forward_geomean']:.10f} |" if v2[index]["summary"]["forward_geomean"] is not None else "invalid |"))
    lines += ["", "Protocol v1 series 1 remains immutable and is not replaced. Protocol v2 series 2 is a separately predeclared four-run series; no best-result selection is performed.", ""]
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("width128-forward", "official"), required=True)
    parser.add_argument("--output-dir", default="phase3a4_reports/conditioned_raw_v2")
    parser.add_argument("--pairs-output", default="phase3a4_reports/conditioned_width128_forward_pairs_v2.json")
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    bindings = json.loads(pathlib.Path(args.manifest).read_text())["bindings"]
    script = pathlib.Path(__file__).with_name("benchmark_phase3a4_conditioned_metric_v2.py")
    root = pathlib.Path(args.output_dir) / args.mode
    root.mkdir(parents=True, exist_ok=True)

    if args.mode == "width128-forward":
        if args.pairs < 10:
            raise SystemExit("At least ten exploratory pairs are required")
        pairs = [pair(script, bindings, "large_1024_w128_relu", ("forward",), number, root)
            for number in range(1, args.pairs + 1)]
        valid = all(item["valid"] for item in pairs)
        base = [item["runs"]["phase3a3"]["forward"]["measurement"]["median_ms"] for item in pairs if item["runs"]["phase3a3"]["forward"]["measurement"]]
        candidate = [item["runs"]["phase3a4"]["forward"]["measurement"]["median_ms"] for item in pairs if item["runs"]["phase3a4"]["forward"]["measurement"]]
        ratio = statistics.median(base) / statistics.median(candidate) if valid else None
        result = {"schema": 2, "protocol": "conditioning_v2", "case": "large_1024_w128_relu",
            "metric": "forward", "pair_count": len(pairs), "pairs": pairs,
            "phase3a3_conditioned_median_ms": statistics.median(base) if len(base) == len(pairs) else None,
            "phase3a4_conditioned_median_ms": statistics.median(candidate) if len(candidate) == len(pairs) else None,
            "conditioned_ratio": ratio, "gate": {"threshold": 0.99,
                "comparison": "phase3a3_median / phase3a4_median", "all_runs_valid": valid,
                "pass": valid and ratio is not None and ratio >= 0.99}}
        pathlib.Path(args.pairs_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("PHASE3A4_WIDTH128_FORWARD_V2=" + ("PASS" if result["gate"]["pass"] else "FAIL"))
        raise SystemExit(0 if result["gate"]["pass"] else 1)

    exploration = json.loads(pathlib.Path(args.pairs_output).read_text())
    if not exploration["gate"]["pass"]:
        raise SystemExit("Protocol v2 exploratory gate is not PASS; official series 2 is locked")
    outputs = [pathlib.Path(f"phase3a4_reports/conditioned_performance_v2_run_{number}.json") for number in range(1, 5)]
    if any(path.exists() for path in outputs):
        raise SystemExit("Protocol v2 official output already exists; refusing replacement or additional runs")
    for number, output in enumerate(outputs, 1):
        case_pairs = [pair(script, bindings, case, METRICS, number, root / f"run_{number}") for case in CASES]
        valid = all(item["valid"] for item in case_pairs)
        rows = [{"name": item["case"], "pair": item, "forward_ratio": item["ratios"]["forward"],
            "forward_backward_ratio": item["ratios"]["forward_backward"]} for item in case_pairs]
        relu = [row for row in rows if row["name"].endswith("_relu")]
        summary = {"forward_geomean": geomean([row["forward_ratio"] for row in rows]) if valid else None,
            "relu_forward_backward_geomean": geomean([row["forward_backward_ratio"] for row in relu]) if valid else None,
            "minimum_relu_forward_backward": min(row["forward_backward_ratio"] for row in relu) if valid else None,
            "maximum_relu_forward_backward": max(row["forward_backward_ratio"] for row in relu) if valid else None,
            "none_forward_backward": next(row["forward_backward_ratio"] for row in rows if row["name"].endswith("_none")) if valid else None}
        gates = {"all_metric_processes_valid_and_stationary": valid,
            "forward_geomean_ge_0_99": valid and summary["forward_geomean"] >= 0.99,
            "minimum_relu_forward_backward_ge_0_98": valid and summary["minimum_relu_forward_backward"] >= 0.98,
            "relu_forward_backward_geomean_ge_1_02": valid and summary["relu_forward_backward_geomean"] >= 1.02,
            "maximum_relu_forward_backward_ge_1_04": valid and summary["maximum_relu_forward_backward"] >= 1.04,
            "none_forward_backward_ge_0_99": valid and summary["none_forward_backward"] >= 0.99}
        document = {"schema": 2, "protocol": "conditioning_v2", "official_series": 2,
            "official_run": number, "metric_process_isolation": True, "alternating_pair_order": case_pairs[0]["order"],
            "cases": rows, "summary": summary, "gates": gates,
            "result": "PASS" if all(gates.values()) else "FAIL"}
        output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    all_pass = all(json.loads(path.read_text())["result"] == "PASS" for path in outputs)
    v1_paths = [pathlib.Path(f"phase3a4_reports/conditioned_performance_run_{number}.json") for number in range(1, 5)]
    comparison_report(v1_paths, outputs, pathlib.Path("phase3a4_reports/BENCHMARK_CONDITIONING_V1_V2_COMPARISON.md"))
    print("PHASE3A4_CONDITIONED_OFFICIAL_V2=" + ("PASS" if all_pass else "FAIL"))
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
