#!/usr/bin/env python3
"""HIP-event Phase 3A4 benchmark against the captured Phase-3A3 baseline."""
import argparse
import importlib
import json
import math
import pathlib
import statistics
import sys

import torch

CASES = (
    # Keep the capture script's case order exactly so that every candidate is
    # compared at the same point in the device warm-up/clock sequence.
    ("large_1024_w64_relu", 1024, 8, 64, 4, 3, "ReLU", "Sigmoid"),
    ("large_1024_w128_relu", 1024, 8, 128, 4, 8, "ReLU", "None"),
    ("large_4096_w64_relu", 4096, 32, 64, 2, 16, "ReLU", "None"),
    ("large_4096_w64_none", 4096, 32, 64, 2, 16, "None", "None"),
)


def median_event(operation, native, warmup=30, iterations=150):
    for _ in range(warmup): operation()
    torch.cuda.synchronize()
    before = (int(native._hipblaslt_execution_handle_creations()),
              int(native._hipblaslt_mlp_cache_misses()))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, stop in zip(starts, stops):
        start.record(); operation(); stop.record()
    stops[-1].synchronize()
    median = statistics.median(start.elapsed_time(stop) for start, stop in zip(starts, stops))
    after = (int(native._hipblaslt_execution_handle_creations()),
             int(native._hipblaslt_mlp_cache_misses()))
    return median, {"handle_creations": after[0]-before[0], "heuristic_misses": after[1]-before[1]}


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-run", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    if args.bindings: sys.path.insert(0, str(pathlib.Path(args.bindings).resolve()))
    import tinycudann as tcnn
    native = importlib.import_module("tinycudann_bindings._120_C")
    baseline = json.loads(pathlib.Path(args.baseline).read_text())
    base = {case["name"]: case for case in baseline["cases"]}
    rows = []
    for name, batch, input_width, width, layers, output_width, activation, output_activation in CASES:
        model = tcnn.Network(input_width, output_width, {"otype":"HipBLASLtMLP", "n_hidden_layers":layers,
            "n_neurons":width, "activation":activation, "output_activation":output_activation}, seed=20260721)
        x = torch.randn(batch, input_width, device="cuda") * 0.2
        xt = x.clone().requires_grad_()
        grad = torch.randn(batch, output_width, device="cuda") * 0.2
        def fwd():
            with torch.no_grad(): model(x)
        forward, forward_counters = median_event(fwd, native)
        def fb():
            model.zero_grad(set_to_none=True); xt.grad=None; model(xt).backward(grad)
        forward_backward, backward_counters = median_event(fb, native)
        rows.append({"name":name,"activation":activation,"forward_ms":forward,
            "forward_backward_ms":forward_backward,"forward_ratio":base[name]["forward"]["median_ms"]/forward,
            "forward_backward_ratio":base[name]["forward_backward"]["median_ms"]/forward_backward,
            "warm_cache_counter_delta":{"handle_creations":forward_counters["handle_creations"]+backward_counters["handle_creations"],
                "heuristic_misses":forward_counters["heuristic_misses"]+backward_counters["heuristic_misses"]}})
    relu=[row for row in rows if row["activation"]=="ReLU"]
    summary={"forward_geomean":geomean([row["forward_ratio"] for row in rows]),
        "relu_forward_backward_geomean":geomean([row["forward_backward_ratio"] for row in relu]),
        "minimum_relu_forward_backward":min(row["forward_backward_ratio"] for row in relu),
        "maximum_relu_forward_backward":max(row["forward_backward_ratio"] for row in relu),
        "none_forward_backward":next(row["forward_backward_ratio"] for row in rows if row["activation"]=="None")}
    counters_clean=all(not row["warm_cache_counter_delta"]["handle_creations"] and not row["warm_cache_counter_delta"]["heuristic_misses"] for row in rows)
    passed=summary["forward_geomean"]>=.99 and summary["minimum_relu_forward_backward"]>=.98 and summary["relu_forward_backward_geomean"]>=1.02 and summary["maximum_relu_forward_backward"]>=1.04 and summary["none_forward_backward"]>=.99 and counters_clean
    result={"result":"PASS" if passed else "FAIL","official_run":args.official_run,"cases":rows,"summary":summary,
        "warm_cache_counters_clean":counters_clean,"counters":{"handle_creations":int(native._hipblaslt_execution_handle_creations()),
                    "heuristic_misses":int(native._hipblaslt_mlp_cache_misses())}}
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("PHASE3A4_BENCHMARK="+result["result"])
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
