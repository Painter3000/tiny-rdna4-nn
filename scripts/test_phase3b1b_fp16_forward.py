#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1B_FP16_FORWARD_001: production FP16-forward regression."""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

import torch
import tinycudann as tcnn
from tinycudann.modules import _C

MARKER = "TCNN_RDNA4_P3B1B_FP16_FORWARD_001"
WIDTHS = (16, 32, 64, 128)
HIDDEN_LAYERS = (1, 2, 4)
BATCHES = (1, 16, 128, 1024, 4096)
ACTIVATIONS = ("None", "ReLU")
NEAR_ZERO = 1e-2
# Frozen before the final regression series. These are production gates, not
# the structural 3B1-A discrimination ratios.
TOLERANCES = {
    "gpu_to_cpu64_quantized": {"max_abs": 0.125, "max_rel": 0.075, "normalized_l2": 0.008},
    "gpu_to_torch_fp32_accum": {"max_abs": 0.0625, "max_rel": 0.05, "normalized_l2": 0.004},
    "gpu_to_frozen_tcnn_fp32": {"max_abs": 0.25, "max_rel": 0.15, "normalized_l2": 0.025},
}

def config(width, layers, activation, fp16=True):
    return {"otype": "HipBLASLtMLPFP16" if fp16 else "HipBLASLtMLP",
            **({"precision": "Fp16"} if fp16 else {"precision": "Fp32"}),
            "n_neurons": width, "n_hidden_layers": layers,
            "activation": activation, "output_activation": activation if fp16 else "None"}

def make_params(width, layers, seed, adversarial=False):
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    pieces = []
    scale = (0.002 if adversarial else 0.10) / math.sqrt(width)
    for layer in range(layers + 1):
        w = torch.randn(width * width, generator=gen, dtype=torch.float32) * scale
        b = torch.randn(width, generator=gen, dtype=torch.float32) * scale
        if adversarial:
            b[:8] = torch.tensor([0.0, 2**-14, -2**-14, 2**-10, -2**-10, 1.0, -1.0, 0.0])
        pieces += [w, b]
    return torch.cat(pieces).half().float()

def layers_from(params, width, layers, dtype):
    result=[]; offset=0
    for _ in range(layers+1):
        count=width*width
        w=params[offset:offset+count].reshape(width,width).t().to(dtype); offset+=count
        b=params[offset:offset+width].to(dtype); offset+=width
        result.append((w,b))
    return result

def refs(x, params, width, layers, activation):
    xq=x.half().double(); cpu_layers=[]
    for w,b in layers_from(params,width,layers,torch.float64):
        xq=xq@w+b
        if activation=="ReLU": xq=torch.relu(xq)
        xq=xq.half().double(); cpu_layers.append(xq.clone())
    tg=x.half().float().cuda(); torch_layers=[]
    for w,b in layers_from(params,width,layers,torch.float32):
        tg=tg@w.cuda()+b.cuda()
        if activation=="ReLU": tg=torch.relu(tg)
        tg=tg.half().float(); torch_layers.append(tg.cpu().double())
    return cpu_layers,torch_layers

def metrics(got, ref, relu=False):
    got=got.double().cpu(); ref=ref.double().cpu(); diff=(got-ref).abs()
    mask=ref.abs()>NEAR_ZERO
    rel=(diff[mask]/ref[mask].abs()).max().item() if mask.any() else 0.0
    denom=torch.linalg.vector_norm(ref).item()
    return {"max_abs":diff.max().item() if diff.numel() else 0.0,
            "max_rel_outside_near_zero":rel,
            "normalized_l2":torch.linalg.vector_norm(diff).item()/max(denom,1e-30),
            "nan_count":int(torch.isnan(got).sum()),"inf_count":int(torch.isinf(got).sum()),
            "relu_mask_mismatches":int(((got>0)!=(ref>0)).sum()) if relu else 0}

def gate(metric, name):
    t=TOLERANCES[name]
    return (metric["max_abs"]<=t["max_abs"] and metric["max_rel_outside_near_zero"]<=t["max_rel"]
            and metric["normalized_l2"]<=t["normalized_l2"] and metric["nan_count"]==0
            and metric["inf_count"]==0
            # The frozen FP32 path deliberately does not quantize hidden
            # activations. Its near-zero mask delta is a reported crosscheck,
            # while mask identity is gated against the two quantized oracles.
            and (name=="gpu_to_frozen_tcnn_fp32" or metric["relu_mask_mismatches"]==0))

def counters():
    names=("cache_hits","cache_misses","cache_size","heuristic_queries","execution_handle_count",
           "execution_handle_creations","execution_handle_reuses","descriptor_count","bias_launches","relu_bias_launches",
           "scratch_bytes_live","scratch_bytes_peak")
    return {n:getattr(_C,"_hipblaslt_fp16_"+n)() for n in names}

def one_case(width,layers,batch,activation,seed,adversarial=False,stream=None):
    params=make_params(width,layers,seed,adversarial)
    model=tcnn.Network(width,width,config(width,layers,activation,True))
    frozen=tcnn.Network(width,width,config(width,layers,activation,False))
    with torch.no_grad(): model.params.copy_(params); frozen.params.copy_(params)
    gen=torch.Generator(device="cpu");gen.manual_seed(seed+991)
    x=torch.randn(batch,width,generator=gen,dtype=torch.float32)*(.25 if not adversarial else 1e-2)
    if adversarial and batch:
        pattern=torch.tensor([0.0,2**-24,-2**-24,2**-14,-2**-14,65504.0,-65504.0,1.0],dtype=torch.float32)
        x[0,:min(width,8)]=pattern[:min(width,8)]
    cpu,pt=refs(x,params,width,layers,activation)
    target_stream=stream or torch.cuda.current_stream()
    with torch.no_grad(),torch.cuda.stream(target_stream):
        y=model(x.cuda()); y2=model(x.cuda()); yf=frozen(x.cuda())
        if activation=="ReLU": yf=torch.relu(yf)
    target_stream.synchronize()
    m_cpu=metrics(y,cpu[-1],activation=="ReLU");m_pt=metrics(y,pt[-1],activation=="ReLU");m_fp32=metrics(y,yf,activation=="ReLU")
    layer_cross=[metrics(pt[i],cpu[i],activation=="ReLU") for i in range(len(cpu))]
    return {"width":width,"hidden_layers":layers,"batch":batch,"activation":activation,
            "adversarial":adversarial,"output_dtype":str(y.dtype),"repeat_identical":bool(torch.equal(y,y2)),
            "gpu_to_cpu64_quantized":m_cpu,"gpu_to_torch_fp32_accum":m_pt,
            "gpu_to_frozen_tcnn_fp32":m_fp32,"per_layer_cpu64_to_torch":layer_cross,
            "passed":bool(torch.equal(y,y2) and y.dtype==torch.float16 and gate(m_cpu,"gpu_to_cpu64_quantized")
                          and gate(m_pt,"gpu_to_torch_fp32_accum") and gate(m_fp32,"gpu_to_frozen_tcnn_fp32"))}

def fresh(args):
    stream=torch.cuda.Stream()
    result=one_case(args.width,args.layers,args.batch,args.activation,7000+args.width+args.layers,stream=stream)
    print(json.dumps({"marker":MARKER,"case":result,"counters":counters()}));return 0 if result["passed"] else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fresh",action="store_true");parser.add_argument("--width",type=int);parser.add_argument("--layers",type=int);parser.add_argument("--batch",type=int);parser.add_argument("--activation");parser.add_argument("--output",type=pathlib.Path)
    args=parser.parse_args()
    if args.fresh:return fresh(args)
    before=counters();streams=[torch.cuda.Stream(),torch.cuda.Stream()];cases=[]
    index=0
    for layers in HIDDEN_LAYERS:
        for width in WIDTHS:
            for batch in BATCHES:
                for activation in ACTIVATIONS:
                    cases.append(one_case(width,layers,batch,activation,10000+index,stream=streams[index%2]));index+=1
    adversarial=[]
    for width in WIDTHS:
        for activation in ACTIVATIONS:
            adversarial.append(one_case(width,4,128,activation,30000+width,True,streams[width%2]))
    after_matrix=counters()
    # A true warm-cache rerun of an already planned signature.
    warm_before=counters();warm=one_case(16,1,16,"ReLU",10000,stream=streams[0]);warm_after=counters()
    warm_ok=(warm_after["cache_misses"]==warm_before["cache_misses"] and warm_after["heuristic_queries"]==warm_before["heuristic_queries"]
             and warm_after["execution_handle_creations"]==warm_before["execution_handle_creations"]
             and warm_after["descriptor_count"]==warm_before["descriptor_count"])
    fresh_runs=[]
    for width in WIDTHS:
        for layers in HIDDEN_LAYERS:
            cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),"--fresh","--width",str(width),"--layers",str(layers),"--batch","128","--activation","ReLU"]
            done=subprocess.run(cmd,text=True,capture_output=True,env=os.environ.copy())
            try:data=json.loads(done.stdout.strip().splitlines()[-1])
            except Exception as exc:data={"error":str(exc),"stdout":done.stdout}
            fresh_runs.append({"returncode":done.returncode,"stderr":done.stderr,"result":data})
    zero_batch_rejected=False;zero_error=""
    try:
        m=tcnn.Network(16,16,config(16,1,"None",True));m(torch.empty(0,16,device="cuda"))
    except Exception as exc:zero_batch_rejected=True;zero_error=str(exc)
    backward_rejected=False;backward_error=""
    try:
        m=tcnn.Network(16,16,config(16,1,"ReLU",True))
        m(torch.randn(16,16,device="cuda",requires_grad=True)).sum().backward()
    except Exception as exc:backward_rejected=True;backward_error=str(exc)
    stable_model=tcnn.Network(128,128,config(128,4,"ReLU",True));stable_x=torch.randn(1024,128,device="cuda")
    with torch.no_grad():
        for _ in range(20): stable_y=stable_model(stable_x)
    torch.cuda.synchronize();allocated_20=torch.cuda.memory_allocated()
    with torch.no_grad():
        for _ in range(80): stable_y=stable_model(stable_x)
    torch.cuda.synchronize();allocated_100=torch.cuda.memory_allocated()
    memory_growth=allocated_100-allocated_20
    memory_pass=memory_growth<=1024*1024 and counters()["scratch_bytes_live"]==0
    allcases=cases+adversarial
    maxima={}
    for category in TOLERANCES:
        maxima[category]={"max_abs":max(c[category]["max_abs"] for c in allcases),
                          "max_rel_outside_near_zero":max(c[category]["max_rel_outside_near_zero"] for c in allcases),
                          "normalized_l2":max(c[category]["normalized_l2"] for c in allcases),
                          "relu_mask_mismatches":sum(c[category]["relu_mask_mismatches"] for c in allcases)}
    doc={"marker":MARKER,"tolerances_frozen_before_final_series":TOLERANCES,"near_zero_threshold":NEAR_ZERO,
         "functional_cases":len(cases),"adversarial_cases":len(adversarial),"passed_cases":sum(c["passed"] for c in allcases),
         "maxima":maxima,"two_streams":{"stream_ids":[int(s.cuda_stream) for s in streams],"passed":streams[0].cuda_stream!=streams[1].cuda_stream},
         "counters":{"before":before,"after_matrix":after_matrix,"warm_before":warm_before,"warm_after":warm_after,"warm_cache_passed":warm_ok},
         "fresh_processes":{"count":len(fresh_runs),"passed":all(x["returncode"]==0 for x in fresh_runs),"runs":fresh_runs},
         "zero_batch":{"contract":"explicit rejection","passed":zero_batch_rejected,"error":zero_error},
         "backward":{"contract":"unsupported","passed":backward_rejected,"error":backward_error},
         "memory_stability":{"iterations":100,"allocated_after_20":allocated_20,"allocated_after_100":allocated_100,
                             "allocated_growth":memory_growth,"maximum_growth":1024*1024,"passed":memory_pass},
         "cases":cases,"adversarial":adversarial,"warm_case":warm}
    doc["decision"]="FUNCTIONAL_PASS" if (all(c["passed"] for c in allcases) and warm_ok and doc["fresh_processes"]["passed"] and zero_batch_rejected and backward_rejected and memory_pass and doc["two_streams"]["passed"]) else "PHASE3B1B_BLOCKED"
    serialized=json.dumps(doc,indent=2,sort_keys=True)+"\n"
    if args.output: args.output.write_text(serialized)
    print(doc["decision"]);return 0 if doc["decision"]=="FUNCTIONAL_PASS" else 1

if __name__=="__main__":raise SystemExit(main())
