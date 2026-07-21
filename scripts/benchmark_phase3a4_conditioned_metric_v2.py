#!/usr/bin/env python3
"""Protocol v2: one fresh process measures one case/metric/variant combination."""
import argparse
import importlib
import json
import pathlib
import statistics
import sys

import torch

CASES = {
    "large_1024_w64_relu": (1024, 8, 64, 4, 3, "ReLU", "Sigmoid"),
    "large_1024_w128_relu": (1024, 8, 128, 4, 8, "ReLU", "None"),
    "large_4096_w64_relu": (4096, 32, 64, 2, 16, "ReLU", "None"),
    "large_4096_w64_none": (4096, 32, 64, 2, 16, "None", "None"),
}
COUNTERS = {
    "handle_creations": "_hipblaslt_execution_handle_creations",
    "heuristic_misses": "_hipblaslt_mlp_cache_misses",
    "fused_stage1": "_hipblaslt_fused_relu_biasgrad_stage1_launches",
    "fused_relu_only": "_hipblaslt_fused_relu_only_launches",
    "biasgrad_finalize": "_hipblaslt_biasgrad_finalize_launches",
    "fusion_fallbacks": "_hipblaslt_fused_relu_biasgrad_fallbacks",
    "partial_live": "_hipblaslt_fused_partial_bytes_live",
    "partial_peak": "_hipblaslt_fused_partial_bytes_peak",
}


def snapshot(native):
    values = {}
    for label, symbol in COUNTERS.items():
        fn = getattr(native, symbol, None)
        values[label] = int(fn()) if fn else 0
    return values


def delta(after, before):
    return {key: after[key] - before[key] for key in before}


def timed(operation, count):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, stop in zip(starts, stops):
        start.record()
        operation()
        stop.record()
    stops[-1].synchronize()
    return [float(start.elapsed_time(stop)) for start, stop in zip(starts, stops)]


def convergence(medians):
    recent = medians[-5:]
    center = statistics.median(recent)
    spread = (max(recent) - min(recent)) / center
    nondecreasing = all(a <= b for a, b in zip(recent, recent[1:]))
    nonincreasing = all(a >= b for a, b in zip(recent, recent[1:]))
    drift = abs(recent[-1] - recent[0]) / center
    monotonic_bad = (nondecreasing or nonincreasing) and drift > 0.005
    return {"converged": spread <= 0.01 and not monotonic_bad, "last_5_medians_ms": recent,
        "spread_fraction": spread, "monotonic": nondecreasing or nonincreasing,
        "endpoint_drift_fraction": drift, "monotonic_drift_over_0_5_percent": monotonic_bad}


def adaptive_warmup(operation, native):
    before = snapshot(native)
    windows, medians, proof = [], [], None
    for index in range(1, 101):
        raw = timed(operation, 50)
        median = statistics.median(raw)
        windows.append({"window": index, "raw_ms": raw, "median_ms": median})
        medians.append(median)
        if index >= 5:
            proof = convergence(medians)
            if proof["converged"]:
                break
    after = snapshot(native)
    return {"window_size": 50, "window_count": len(windows), "iterations": 50 * len(windows),
        "windows": windows, "window_medians_ms": medians, "convergence": proof,
        "counters_before": before, "counters_after": after, "counter_delta": delta(after, before)}


def measurement(operation, native):
    before = snapshot(native)
    raw = timed(operation, 150)
    after = snapshot(native)
    changes = delta(after, before)
    overall = statistics.median(raw)
    blocks = []
    for index in range(6):
        values = raw[index * 25:(index + 1) * 25]
        blocks.append({"block": index + 1, "raw_ms": values, "median_ms": statistics.median(values)})
    block_medians = [block["median_ms"] for block in blocks]
    spread = (max(block_medians) - min(block_medians)) / overall
    deviations = [abs(value - overall) / overall for value in block_medians]
    stationarity = {"block_size": 25, "block_count": 6, "block_medians_ms": block_medians,
        "block_spread_fraction": spread, "maximum_block_deviation_fraction": max(deviations),
        "spread_at_most_2_percent": spread <= 0.02,
        "every_block_within_2_percent": all(value <= 0.02 for value in deviations)}
    invariants = {"no_handle_creations": changes["handle_creations"] == 0,
        "no_heuristic_misses": changes["heuristic_misses"] == 0,
        "no_scratch_peak_growth": changes["partial_peak"] == 0,
        "partial_live_zero_at_boundaries": before["partial_live"] == 0 and after["partial_live"] == 0}
    return {"iterations": 150, "raw_ms": raw, "median_ms": overall, "blocks": blocks,
        "stationarity": stationarity, "stationarity_pass": all((stationarity["spread_at_most_2_percent"],
            stationarity["every_block_within_2_percent"])), "counters_before": before,
        "counters_after": after, "counter_delta": changes, "measurement_invariants": invariants,
        "measurement_invariants_pass": all(invariants.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--variant", choices=("phase3a3", "phase3a4"), required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--metric", choices=("forward", "forward_backward"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(pathlib.Path(args.bindings).resolve()))
    import tinycudann as tcnn
    native = importlib.import_module("tinycudann_bindings._120_C")
    batch, inp, width, hidden, out, activation, output_activation = CASES[args.case]
    torch.manual_seed(20260721)
    process_start = snapshot(native)
    model = tcnn.Network(inp, out, {"otype": "HipBLASLtMLP", "n_hidden_layers": hidden,
        "n_neurons": width, "activation": activation, "output_activation": output_activation}, seed=20260721)
    x = torch.randn(batch, inp, device="cuda") * 0.2
    xt = x.clone().requires_grad_()
    grad = torch.randn(batch, out, device="cuda") * 0.2

    def forward():
        with torch.no_grad():
            model(x)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        xt.grad = None
        model(xt).backward(grad)

    operation = forward if args.metric == "forward" else forward_backward
    before_plan = snapshot(native)
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    after_plan = snapshot(native)
    plan = {"operation": args.metric, "iterations": 3, "counters_before": before_plan,
        "counters_after": after_plan, "counter_delta": delta(after_plan, before_plan)}
    warmup = adaptive_warmup(operation, native)
    converged = bool(warmup["convergence"] and warmup["convergence"]["converged"])
    measured = measurement(operation, native) if converged else None
    final = snapshot(native)
    valid = converged and bool(measured and measured["stationarity_pass"] and measured["measurement_invariants_pass"])

    none_invariants, fallback_accounting = None, None
    if activation == "None":
        changes = delta(final, process_start)
        fallback_present = args.variant == "phase3a4" and hasattr(native, COUNTERS["fusion_fallbacks"])
        backward_calls = (3 + warmup["iterations"] + (150 if measured else 0)) if args.metric == "forward_backward" else 0
        expected_fallback = hidden * backward_calls if fallback_present else 0
        fallback_accounting = {"counter_present": fallback_present, "hidden_layers": hidden,
            "backward_invocations": backward_calls, "expected_legacy_path_delta": expected_fallback,
            "observed_delta": changes["fusion_fallbacks"], "unexpected_delta": changes["fusion_fallbacks"] - expected_fallback}
        none_invariants = {"fused_stage1_delta_zero": changes["fused_stage1"] == 0,
            "fused_relu_only_delta_zero": changes["fused_relu_only"] == 0,
            "biasgrad_finalize_delta_zero": changes["biasgrad_finalize"] == 0,
            "partial_scratch_live_zero": process_start["partial_live"] == 0 and final["partial_live"] == 0,
            "no_phase3a4_scratch_created": final["partial_peak"] == 0,
            "no_new_fusion_fallback_executed": changes["fusion_fallbacks"] == expected_fallback}
        valid = valid and all(none_invariants.values())

    result = {"schema": 2, "protocol": "conditioning_v2", "fresh_process": True,
        "variant": args.variant, "case": args.case, "metric": args.metric, "activation": activation,
        "binding": str(pathlib.Path(native.__file__).resolve()), "model_plan_warmup": plan,
        "steady_state_warmup": warmup, "measurement": measured, "none_invariants": none_invariants,
        "none_fallback_accounting": fallback_accounting, "valid": valid, "final_counters": final}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PHASE3A4_CONDITIONED_METRIC_V2=" + ("VALID" if valid else "INFRASTRUCTURE_FAIL"))
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
