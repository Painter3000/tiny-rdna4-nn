#!/usr/bin/env python3
"""Protocol v3 Width-128 Forward measurement in one fresh process."""
import argparse
import importlib
import json
import pathlib
import statistics
import sys
import time

FORBIDDEN_MONITORS = {"amd-smi", "rocm-smi", "nvtop", "radeontop"}
COUNTERS = {
    "handle_creations": "_hipblaslt_execution_handle_creations",
    "heuristic_misses": "_hipblaslt_mlp_cache_misses",
    "partial_live": "_hipblaslt_fused_partial_bytes_live",
    "partial_peak": "_hipblaslt_fused_partial_bytes_peak",
}


def monitor_processes():
    found = []
    proc = pathlib.Path("/proc")
    for entry in proc.iterdir():
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


def snapshot(native):
    result = {}
    for label, symbol in COUNTERS.items():
        function = getattr(native, symbol, None)
        result[label] = int(function()) if function else 0
    return result


def delta(after, before):
    return {key: after[key] - before[key] for key in before}


def convergence(values):
    recent = values[-5:]
    center = statistics.median(recent)
    spread = (max(recent) - min(recent)) / center
    nondecreasing = all(a <= b for a, b in zip(recent, recent[1:]))
    nonincreasing = all(a >= b for a, b in zip(recent, recent[1:]))
    drift = abs(recent[-1] - recent[0]) / center
    monotonic_bad = (nondecreasing or nonincreasing) and drift > 0.005
    return {"converged": spread <= 0.01 and not monotonic_bad,
        "last_5_per_operation_ms": recent, "spread_fraction": spread,
        "monotonic": nondecreasing or nonincreasing, "endpoint_drift_fraction": drift,
        "monotonic_drift_over_0_5_percent": monotonic_bad}


def window(operation, torch):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter_ns()
    for _ in range(128):
        operation()
    wall_stop = time.perf_counter_ns()
    stop.record()
    stop.synchronize()
    elapsed_ms = float(start.elapsed_time(stop))
    submission_ms = (wall_stop - wall_start) / 1_000_000.0
    return {"operations": 128, "elapsed_window_ms": elapsed_ms,
        "per_operation_ms": elapsed_ms / 128.0, "host_submission_ms": submission_ms,
        "host_to_gpu_elapsed_ratio": submission_ms / elapsed_ms,
        "probable_stream_starvation": submission_ms >= elapsed_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--variant", choices=("phase3a3", "phase3a4"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    monitors = monitor_processes()
    if monitors:
        output.write_text(json.dumps({"schema": 3, "protocol": "conditioning_v3",
            "status": "INVALID_ENVIRONMENT", "forbidden_monitor_processes": monitors}, indent=2, sort_keys=True) + "\n")
        raise SystemExit(3)

    import torch
    sys.path.insert(0, str(pathlib.Path(args.bindings).resolve()))
    import tinycudann as tcnn
    native = importlib.import_module("tinycudann_bindings._120_C")
    torch.manual_seed(20260721)
    model = tcnn.Network(8, 8, {"otype": "HipBLASLtMLP", "n_hidden_layers": 4,
        "n_neurons": 128, "activation": "ReLU", "output_activation": "None"}, seed=20260721)
    x = torch.randn(1024, 8, device="cuda") * 0.2

    def forward():
        with torch.no_grad():
            model(x)

    before_plan = snapshot(native)
    for _ in range(3):
        forward()
    torch.cuda.synchronize()
    after_plan = snapshot(native)
    plan = {"operations": 3, "counters_before": before_plan, "counters_after": after_plan,
        "counter_delta": delta(after_plan, before_plan)}

    warm_windows, warm_values, proof = [], [], None
    for index in range(1, 101):
        item = window(forward, torch)
        item["window"] = index
        warm_windows.append(item)
        warm_values.append(item["per_operation_ms"])
        if index >= 10:
            proof = convergence(warm_values)
            if proof["converged"]:
                break
    converged = bool(proof and proof["converged"])
    before_measurement = snapshot(native)
    measured = [dict(window(forward, torch), window=index) for index in range(1, 41)] if converged else []
    after_measurement = snapshot(native)
    changes = delta(after_measurement, before_measurement)

    stationarity = None
    if measured:
        per_operation = [item["per_operation_ms"] for item in measured]
        overall = statistics.median(per_operation)
        blocks = []
        for index in range(8):
            values = per_operation[index * 5:(index + 1) * 5]
            blocks.append({"block": index + 1, "windows": list(range(index * 5 + 1, index * 5 + 6)),
                "per_operation_ms": values, "median_ms": statistics.median(values)})
        block_medians = [block["median_ms"] for block in blocks]
        spread = (max(block_medians) - min(block_medians)) / overall
        deviations = [abs(value - overall) / overall for value in block_medians]
        stationarity = {"overall_median_per_operation_ms": overall, "blocks": blocks,
            "block_medians_ms": block_medians, "block_spread_fraction": spread,
            "maximum_block_deviation_fraction": max(deviations),
            "spread_at_most_2_percent": spread <= 0.02,
            "every_block_within_2_percent": all(value <= 0.02 for value in deviations)}
    invariants = {"no_handle_creations": changes["handle_creations"] == 0,
        "no_heuristic_misses": changes["heuristic_misses"] == 0,
        "no_scratch_peak_growth": changes["partial_peak"] == 0,
        "partial_live_zero_at_boundaries": before_measurement["partial_live"] == 0 and after_measurement["partial_live"] == 0}
    starvation_windows = [{"phase": "warmup", "window": item["window"],
        "host_submission_ms": item["host_submission_ms"], "elapsed_window_ms": item["elapsed_window_ms"]}
        for item in warm_windows if item["probable_stream_starvation"]]
    starvation_windows += [{"phase": "measurement", "window": item["window"],
        "host_submission_ms": item["host_submission_ms"], "elapsed_window_ms": item["elapsed_window_ms"]}
        for item in measured if item["probable_stream_starvation"]]
    stationary = bool(stationarity and stationarity["spread_at_most_2_percent"] and stationarity["every_block_within_2_percent"])
    valid = converged and stationary and all(invariants.values()) and not starvation_windows
    result = {"schema": 3, "protocol": "conditioning_v3", "status": "VALID" if valid else "INFRASTRUCTURE_FAIL",
        "fresh_process": True, "variant": args.variant, "case": "large_1024_w128_relu", "metric": "forward",
        "window_operations": 128, "binding": str(pathlib.Path(native.__file__).resolve()),
        "model_plan_warmup": plan, "steady_state_warmup": {"minimum_windows": 10, "maximum_windows": 100,
            "window_count": len(warm_windows), "windows": warm_windows, "convergence": proof},
        "measurement": {"window_count": len(measured), "total_operations": len(measured) * 128,
            "windows": measured, "stationarity": stationarity, "stationarity_pass": stationary,
            "counters_before": before_measurement, "counters_after": after_measurement,
            "counter_delta": changes, "invariants": invariants, "invariants_pass": all(invariants.values())},
        "host_submission": {"starvation_rule": "host_submission_ms >= elapsed_window_ms",
            "probable_stream_starvation": bool(starvation_windows), "failing_windows": starvation_windows},
        "forbidden_monitor_processes": [], "valid": valid}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
