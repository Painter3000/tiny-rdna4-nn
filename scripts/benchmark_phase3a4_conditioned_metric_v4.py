#!/usr/bin/env python3
"""Protocol v4: one native 128-Forward submission per Python window call."""
import argparse
import importlib
import json
import pathlib
import statistics
import sys

FORBIDDEN = {"amd-smi", "rocm-smi", "nvtop", "radeontop"}
COUNTERS = {"handle_creations": "_hipblaslt_execution_handle_creations",
    "heuristic_misses": "_hipblaslt_mlp_cache_misses", "partial_live": "_hipblaslt_fused_partial_bytes_live",
    "partial_peak": "_hipblaslt_fused_partial_bytes_peak"}


def monitors():
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
        if comm in FORBIDDEN or executable in FORBIDDEN:
            found.append({"pid": int(entry.name), "comm": comm, "cmdline": cmdline})
    return sorted(found, key=lambda item: item["pid"])


def snapshot(native):
    result = {}
    for key, symbol in COUNTERS.items():
        function = getattr(native, symbol, None)
        result[key] = int(function()) if function else 0
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
    return {"converged": spread <= 0.01 and not monotonic_bad, "last_5_per_operation_ms": recent,
        "spread_fraction": spread, "monotonic": nondecreasing or nonincreasing,
        "endpoint_drift_fraction": drift, "monotonic_drift_over_0_5_percent": monotonic_bad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--variant", choices=("phase3a3", "phase3a4"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forbidden = monitors()
    if forbidden:
        output.write_text(json.dumps({"schema": 4, "status": "INVALID_ENVIRONMENT",
            "forbidden_monitor_processes": forbidden}, indent=2, sort_keys=True) + "\n")
        raise SystemExit(3)
    import torch
    sys.path.insert(0, str(pathlib.Path(args.bindings).resolve()))
    import tinycudann as tcnn
    native = importlib.import_module("tinycudann_bindings._120_C")
    torch.manual_seed(20260721)
    model = tcnn.Network(8, 8, {"otype": "HipBLASLtMLP", "n_hidden_layers": 4,
        "n_neurons": 128, "activation": "ReLU", "output_activation": "None"}, seed=20260721)
    x = (torch.randn(1024, 8, device="cuda") * 0.2).contiguous()
    params = model.params.detach().contiguous()
    method = model.native_tcnn_module._benchmark_forward_window_128
    before_plan = snapshot(native)
    for _ in range(3):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    after_plan = snapshot(native)
    plan = {"normal_forward_operations": 3, "counter_delta": delta(after_plan, before_plan)}

    def native_window(index):
        host_ms, gpu_ms, output_tensor = method(x, params)
        del output_tensor
        host_ms, gpu_ms = float(host_ms), float(gpu_ms)
        ratio = host_ms / gpu_ms
        return {"window": index, "operations": 128, "native_host_enqueue_ms": host_ms,
            "hip_event_gpu_ms": gpu_ms, "per_operation_ms": gpu_ms / 128.0,
            "enqueue_to_gpu_ratio": ratio, "queue_headroom_fraction": 1.0 - ratio,
            "queue_headroom_at_least_20_percent": ratio <= 0.80}

    warmup, values, proof = [], [], None
    for index in range(1, 101):
        item = native_window(index)
        warmup.append(item)
        values.append(item["per_operation_ms"])
        if index >= 10:
            proof = convergence(values)
            if proof["converged"]:
                break
    converged = bool(proof and proof["converged"])
    before_measurement = snapshot(native)
    measured = [native_window(index) for index in range(1, 41)] if converged else []
    after_measurement = snapshot(native)
    changes = delta(after_measurement, before_measurement)
    stationarity = None
    if measured:
        per_operation = [item["per_operation_ms"] for item in measured]
        overall = statistics.median(per_operation)
        blocks = []
        for index in range(8):
            block_values = per_operation[index * 5:(index + 1) * 5]
            blocks.append({"block": index + 1, "windows": list(range(index * 5 + 1, index * 5 + 6)),
                "per_operation_ms": block_values, "median_ms": statistics.median(block_values)})
        medians = [block["median_ms"] for block in blocks]
        spread = (max(medians) - min(medians)) / overall
        deviations = [abs(value - overall) / overall for value in medians]
        stationarity = {"overall_median_per_operation_ms": overall, "blocks": blocks,
            "block_medians_ms": medians, "block_spread_fraction": spread,
            "maximum_block_deviation_fraction": max(deviations), "spread_at_most_2_percent": spread <= 0.02,
            "every_block_within_2_percent": all(value <= 0.02 for value in deviations)}
    stationary = bool(stationarity and stationarity["spread_at_most_2_percent"] and stationarity["every_block_within_2_percent"])
    invariants = {"no_handle_creations": changes["handle_creations"] == 0,
        "no_heuristic_misses": changes["heuristic_misses"] == 0,
        "no_scratch_peak_growth": changes["partial_peak"] == 0,
        "partial_live_zero_at_boundaries": before_measurement["partial_live"] == 0 and after_measurement["partial_live"] == 0}
    headroom_failures = [{"phase": phase, "window": item["window"], "enqueue_to_gpu_ratio": item["enqueue_to_gpu_ratio"]}
        for phase, windows in (("warmup", warmup), ("measurement", measured)) for item in windows
        if not item["queue_headroom_at_least_20_percent"]]
    valid = converged and stationary and all(invariants.values()) and not headroom_failures
    result = {"schema": 4, "protocol": "conditioning_v4", "status": "VALID" if valid else "INFRASTRUCTURE_FAIL",
        "fresh_process": True, "variant": args.variant, "case": "large_1024_w128_relu", "metric": "forward",
        "native_window": {"python_calls_per_window": 1, "normal_forward_operations": 128,
            "same_stream": True, "same_input_and_output_within_window": True, "hip_graph": False},
        "model_plan_warmup": plan, "steady_state_warmup": {"minimum_windows": 10, "maximum_windows": 100,
            "window_count": len(warmup), "windows": warmup, "convergence": proof},
        "measurement": {"window_count": len(measured), "total_forward_operations": len(measured) * 128,
            "windows": measured, "stationarity": stationarity, "stationarity_pass": stationary,
            "counters_before": before_measurement, "counters_after": after_measurement,
            "counter_delta": changes, "invariants": invariants, "invariants_pass": all(invariants.values())},
        "queue_headroom": {"minimum_required_fraction": 0.20,
            "rule": "native_host_enqueue_ms / hip_event_gpu_ms <= 0.80",
            "pass": not headroom_failures, "failing_windows": headroom_failures},
        "forbidden_monitor_processes": [], "valid": valid}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
