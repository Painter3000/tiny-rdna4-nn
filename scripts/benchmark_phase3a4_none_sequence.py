#!/usr/bin/env python3
"""Measure one Phase-3A4 activation-order sequence in exactly one fresh process."""
import argparse, importlib, json, pathlib, statistics, sys
import torch

BASELINE_NAME = "large_4096_w64_none"
RELU_CASES = (
    ("large_1024_w64_relu", 1024, 8, 64, 4, 3, "Sigmoid"),
    ("large_1024_w128_relu", 1024, 8, 128, 4, 8, "None"),
    ("large_4096_w64_relu", 4096, 32, 64, 2, 16, "None"),
)

def counter(native, name):
    fn = getattr(native, name, None)
    return int(fn()) if fn else 0

def counters(native):
    return {
        "fused_stage1": counter(native, "_hipblaslt_fused_relu_biasgrad_stage1_launches"),
        "fused_relu_only": counter(native, "_hipblaslt_fused_relu_only_launches"),
        "biasgrad_finalize": counter(native, "_hipblaslt_biasgrad_finalize_launches"),
        "partial_live": counter(native, "_hipblaslt_fused_partial_bytes_live"),
        "partial_peak": counter(native, "_hipblaslt_fused_partial_bytes_peak"),
        "scratch_allocations": counter(native, "_hipblaslt_fused_scratch_allocations"),
        "fusion_fallbacks": counter(native, "_hipblaslt_fused_relu_biasgrad_fallbacks"),
        "handle_creations": counter(native, "_hipblaslt_execution_handle_creations"),
        "heuristic_misses": counter(native, "_hipblaslt_mlp_cache_misses"),
    }

def delta(after, before):
    return {key: after[key] - before[key] for key in before}

def median_event(operation, warmup=30, iterations=150):
    for _ in range(warmup): operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, stop in zip(starts, stops):
        start.record(); operation(); stop.record()
    stops[-1].synchronize()
    values = [start.elapsed_time(stop) for start, stop in zip(starts, stops)]
    return {"median_ms": statistics.median(values), "minimum_ms": min(values), "raw_ms": values}

def make_model(tcnn, batch, inp, width, hidden, out, activation, output_activation):
    model = tcnn.Network(inp, out, {"otype":"HipBLASLtMLP", "n_hidden_layers":hidden,
        "n_neurons":width, "activation":activation, "output_activation":output_activation}, seed=20260721)
    x = torch.randn(batch, inp, device="cuda") * 0.2
    xt = x.clone().requires_grad_()
    grad = torch.randn(batch, out, device="cuda") * 0.2
    return model, x, xt, grad

def run_relu(tcnn):
    rows=[]
    for name,batch,inp,width,hidden,out,outact in RELU_CASES:
        model,x,xt,grad=make_model(tcnn,batch,inp,width,hidden,out,"ReLU",outact)
        def fb():
            model.zero_grad(set_to_none=True); xt.grad=None; model(xt).backward(grad)
        rows.append({"name":name,"forward_backward":median_event(fb)})
    return rows

def run_none(tcnn, native, baseline_ms, label):
    model,x,xt,grad=make_model(tcnn,4096,32,64,2,16,"None","None")
    def fwd():
        with torch.no_grad(): model(x)
    forward=median_event(fwd)
    def fb():
        model.zero_grad(set_to_none=True); xt.grad=None; model(xt).backward(grad)
    before=counters(native); forward_backward=median_event(fb); after=counters(native)
    change=delta(after,before)
    invariants={
        "fused_stage1_delta_zero":change["fused_stage1"] == 0,
        "fused_relu_only_delta_zero":change["fused_relu_only"] == 0,
        "biasgrad_finalize_delta_zero":change["biasgrad_finalize"] == 0,
        "partial_live_stays_zero":before["partial_live"] == 0 and after["partial_live"] == 0,
        "scratch_allocations_delta_zero":change["scratch_allocations"] == 0,
        "fusion_fallback_delta_zero":change["fusion_fallbacks"] == 0,
    }
    return {"label":label,"forward":forward,"forward_backward":forward_backward,
        "forward_backward_ratio":baseline_ms/forward_backward["median_ms"],
        "counters_before":before,"counters_after":after,"counter_delta":change,
        "invariants":invariants,"invariants_pass":all(invariants.values())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bindings",required=True); p.add_argument("--baseline",required=True)
    p.add_argument("--sequence",choices=("none-only","relu-none","none-relu-none"),required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); sys.path.insert(0,str(pathlib.Path(a.bindings).resolve()))
    import tinycudann as tcnn
    native=importlib.import_module("tinycudann_bindings._120_C")
    baseline=json.loads(pathlib.Path(a.baseline).read_text())
    baseline_ms=next(x for x in baseline["cases"] if x["name"]==BASELINE_NAME)["forward_backward"]["median_ms"]
    result={"sequence":a.sequence,"binding":native.__file__,"none":[],"relu":[]}
    if a.sequence=="none-only": result["none"].append(run_none(tcnn,native,baseline_ms,"only"))
    elif a.sequence=="relu-none":
        result["relu"]=run_relu(tcnn); result["none"].append(run_none(tcnn,native,baseline_ms,"after_relu"))
    else:
        result["none"].append(run_none(tcnn,native,baseline_ms,"before_relu")); result["relu"]=run_relu(tcnn)
        result["none"].append(run_none(tcnn,native,baseline_ms,"after_relu"))
    result["pass"]=all(x["invariants_pass"] for x in result["none"])
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("PHASE3A4_NONE_SEQUENCE="+("PASS" if result["pass"] else "FAIL"))
    raise SystemExit(0 if result["pass"] else 1)
if __name__=="__main__": main()
