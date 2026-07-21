#!/usr/bin/env python3
"""Measure exactly one Phase-3A3/3A4 case in a fresh process."""
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
    result = {}
    for label, symbol in COUNTERS.items():
        fn = getattr(native, symbol, None)
        result[label] = int(fn()) if fn else 0
    return result


def delta(after, before):
    return {key: after[key] - before[key] for key in before}


def timed_window(operation, count):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, stop in zip(starts, stops):
        start.record()
        operation()
        stop.record()
    stops[-1].synchronize()
    return [float(start.elapsed_time(stop)) for start, stop in zip(starts, stops)]


def convergence(window_medians):
    recent = window_medians[-5:]
    center = statistics.median(recent)
    spread = (max(recent) - min(recent)) / center
    nondecreasing = all(a <= b for a, b in zip(recent, recent[1:]))
    nonincreasing = all(a >= b for a, b in zip(recent, recent[1:]))
    drift = abs(recent[-1] - recent[0]) / center
    monotonic_drift = (nondecreasing or nonincreasing) and drift > 0.005
    return {
        "converged": spread <= 0.01 and not monotonic_drift,
        "last_5_medians_ms": recent,
        "spread_fraction": spread,
        "monotonic": nondecreasing or nonincreasing,
        "endpoint_drift_fraction": drift,
        "monotonic_drift_over_0_5_percent": monotonic_drift,
    }


def adaptive_warmup(operation, native):
    before = snapshot(native)
    medians, windows = [], []
    proof = None
    for index in range(1, 101):
        raw = timed_window(operation, 50)
        median = statistics.median(raw)
        windows.append({"window": index, "raw_ms": raw, "median_ms": median})
        medians.append(median)
        if index >= 5:
            proof = convergence(medians)
            if proof["converged"]:
                break
    after = snapshot(native)
    return {
        "window_size": 50,
        "window_count": len(windows),
        "iterations": len(windows) * 50,
        "windows": windows,
        "window_medians_ms": medians,
        "convergence": proof,
        "counters_before": before,
        "counters_after": after,
        "counter_delta": delta(after, before),
    }


def measure(operation, native, iterations):
    before = snapshot(native)
    raw = timed_window(operation, iterations)
    after = snapshot(native)
    changes = delta(after, before)
    invariants = {
        "no_handle_creations": changes["handle_creations"] == 0,
        "no_heuristic_misses": changes["heuristic_misses"] == 0,
        "no_scratch_peak_growth": changes["partial_peak"] == 0,
        "partial_scratch_live_returns_to_zero": before["partial_live"] == 0 and after["partial_live"] == 0,
    }
    return {
        "iterations": iterations,
        "raw_ms": raw,
        "median_ms": statistics.median(raw),
        "minimum_ms": min(raw),
        "maximum_ms": max(raw),
        "counters_before": before,
        "counters_after": after,
        "counter_delta": changes,
        "measurement_invariants": invariants,
        "measurement_invariants_pass": all(invariants.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--variant", choices=("phase3a3", "phase3a4"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--measurement-iterations", type=int, default=150)
    args = parser.parse_args()
    sys.path.insert(0, str(pathlib.Path(args.bindings).resolve()))
    import tinycudann as tcnn
    native = importlib.import_module("tinycudann_bindings._120_C")

    batch, inp, width, hidden, out, activation, output_activation = CASES[args.case]
    torch.manual_seed(20260721)
    before_model = snapshot(native)
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

    # Explicitly materialize plans, cache entries, handles and scratch before steady-state conditioning.
    for _ in range(3):
        forward()
        forward_backward()
    torch.cuda.synchronize()
    after_plan = snapshot(native)
    plan_warmup = {"iterations_per_operation": 3, "counters_before_model": before_model,
        "counters_after": after_plan, "counter_delta": delta(after_plan, before_model)}

    # One case-specific conditioning gate uses the complete training workload.
    # Both metric measurements begin only after this gate has converged.
    case_warmup = adaptive_warmup(forward_backward, native)
    converged = bool(case_warmup["convergence"] and case_warmup["convergence"]["converged"])
    metrics = {"forward": {"measurement": measure(forward, native, args.measurement_iterations) if converged else None},
        "forward_backward": {"measurement": measure(forward_backward, native, args.measurement_iterations) if converged else None}}
    valid = converged and all(bool(metric["measurement"] and metric["measurement"]["measurement_invariants_pass"])
        for metric in metrics.values())

    none_invariants = None
    none_fallback_accounting = None
    if activation == "None":
        total_before = before_model
        total_after = snapshot(native)
        change = delta(total_after, total_before)
        fallback_counter_present = hasattr(native, COUNTERS["fusion_fallbacks"])
        backward_iterations = 3 + case_warmup["iterations"]
        if metrics["forward_backward"]["measurement"]:
            backward_iterations += metrics["forward_backward"]["measurement"]["iterations"]
        expected_legacy_fallback_accounting = 2 * backward_iterations if fallback_counter_present else 0
        none_fallback_accounting = {"counter_present": fallback_counter_present,
            "hidden_layers": 2, "forward_backward_invocations": backward_iterations,
            "expected_legacy_path_delta": expected_legacy_fallback_accounting,
            "observed_delta": change["fusion_fallbacks"],
            "unexpected_delta": change["fusion_fallbacks"] - expected_legacy_fallback_accounting}
        none_invariants = {
            "fused_stage1_delta_zero": change["fused_stage1"] == 0,
            "fused_relu_only_delta_zero": change["fused_relu_only"] == 0,
            "biasgrad_finalize_delta_zero": change["biasgrad_finalize"] == 0,
            "partial_scratch_live_zero": total_before["partial_live"] == 0 and total_after["partial_live"] == 0,
            "no_phase3a4_scratch_created": total_after["partial_peak"] == 0,
            "no_new_fusion_fallback_executed": change["fusion_fallbacks"] == expected_legacy_fallback_accounting,
        }
        valid = valid and all(none_invariants.values())

    result = {"schema": 1, "variant": args.variant, "case": args.case, "activation": activation,
        "binding": str(pathlib.Path(native.__file__).resolve()), "fresh_process": True,
        "plan_cache_warmup": plan_warmup, "case_steady_state_warmup": case_warmup,
        "conditioning_operation": "forward_backward", "metrics": metrics, "none_invariants": none_invariants,
        "none_fallback_accounting": none_fallback_accounting,
        "valid": valid, "final_counters": snapshot(native)}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PHASE3A4_CONDITIONED_CASE=" + ("VALID" if valid else "INVALID"))
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
