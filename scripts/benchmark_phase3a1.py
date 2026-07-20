#!/usr/bin/env python3
"""TCNN_RDNA4_P3A1_HIPBLASLT_005: HIP-event end-to-end performance gate."""
import argparse, importlib, json, math, pathlib, statistics, sys
import torch
SEED=20260720
CASES=(("small",1,3,16,1,2,"ReLU","None"),("medium",64,8,32,2,3,"ReLU","None"),("target_257_w64",257,8,64,4,3,"ReLU","Sigmoid"),("large_1024_w64",1024,8,64,4,3,"ReLU","Sigmoid"),("large_1024_w128",1024,8,128,4,8,"ReLU","None"),("large_4096_w64",4096,32,64,2,16,"ReLU","None"))
def load(root):
 sys.path.insert(0,str(pathlib.Path(root).resolve())); import tinycudann as t
 return t,importlib.import_module("tinycudann_bindings._120_C")
def cfg(c,b): return {"otype":b,"n_neurons":c[3],"n_hidden_layers":c[4],"activation":c[6],"output_activation":c[7]}
def percentile(v,p):
 v=sorted(v); x=p*(len(v)-1); lo=math.floor(x); hi=math.ceil(x); return v[lo] if lo==hi else v[lo]*(hi-x)+v[hi]*(x-lo)
def timing(fn,warmup,iterations):
 for _ in range(warmup): fn()
 torch.cuda.synchronize(); starts=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]; stops=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
 for i in range(iterations): starts[i].record(); fn(); stops[i].record()
 stops[-1].synchronize(); values=[starts[i].elapsed_time(stops[i]) for i in range(iterations)]; return {"minimum_ms":min(values),"median_ms":statistics.median(values),"p95_ms":percentile(values,.95)}
def model_bench(t,native,c,backend,params,x,go,warmup,iterations):
 m=t.Network(c[2],c[5],cfg(c,backend),seed=SEED); m.params.data.copy_(params); xi=x.detach(); xb=x.clone().requires_grad_()
 def f():
  with torch.no_grad(): m(xi)
 def fb(): m.zero_grad(set_to_none=True); xb.grad=None; m(xb).backward(go)
 a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); f(); b.record(); b.synchronize()
 before={"hits":native._hipblaslt_cache_hits(),"misses":native._hipblaslt_cache_misses(),"size":native._hipblaslt_cache_size()}; forward=timing(f,warmup,iterations); backward=timing(fb,warmup,iterations); after={"hits":native._hipblaslt_cache_hits(),"misses":native._hipblaslt_cache_misses(),"size":native._hipblaslt_cache_size()}
 return {"backend":backend,"cold_forward_ms":a.elapsed_time(b),"forward":forward,"forward_backward":backward,"cache_before":before,"cache_after":after,"memory_allocated":torch.cuda.memory_allocated(),"memory_reserved":torch.cuda.memory_reserved()}
def gm(v): return math.exp(sum(math.log(x) for x in v)/len(v))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--bindings",required=True); ap.add_argument("--output",required=True); ap.add_argument("--warmup",type=int,default=20); ap.add_argument("--iterations",type=int,default=100); a=ap.parse_args(); t,n=load(a.bindings); out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True); rows=[]
 for c in CASES:
  g=torch.Generator(device="cuda"); g.manual_seed(SEED+len(c[0])); x=torch.randn(c[1],c[2],device="cuda",generator=g)*.25; go=torch.randn(c[1],c[5],device="cuda",generator=g)*.2; canonical=t.Network(c[2],c[5],cfg(c,"PortableMLP"),seed=SEED).params.detach().clone(); p=model_bench(t,n,c,"PortableMLP",canonical,x,go,a.warmup,a.iterations); h=model_bench(t,n,c,"HipBLASLtMLP",canonical,x,go,a.warmup,a.iterations); assert h["cache_after"]["hits"]>h["cache_before"]["hits"] and h["cache_after"]["size"]>=h["cache_before"]["size"]; row={"case":{"name":c[0],"batch":c[1],"input_dims":c[2],"width":c[3],"hidden":c[4],"output_dims":c[5]},"portable":p,"hipblaslt":h,"forward_speedup":p["forward"]["median_ms"]/h["forward"]["median_ms"],"forward_backward_speedup":p["forward_backward"]["median_ms"]/h["forward_backward"]["median_ms"]}; rows.append(row); print(f"{c[0]}: forward={row['forward_speedup']:.3f}x forward_backward={row['forward_backward_speedup']:.3f}x")
 large=[r for r in rows if r["case"]["name"].startswith("large_")]; fg=gm([r["forward_speedup"] for r in large]); bg=gm([r["forward_backward_speedup"] for r in large]); targets=[r for r in large if r["case"]["name"] in ("large_1024_w128","large_4096_w64")]; gates={"no_large_regression_over_15_percent":all(r["forward_speedup"]>=1/1.15 and r["forward_backward_speedup"]>=1/1.15 for r in targets),"one_large_forward_at_least_1_25x":max(r["forward_speedup"] for r in large)>=1.25,"large_forward_geomean_over_1":fg>1.,"large_forward_backward_geomean_at_least_0_95":bg>=.95}; result={"result":"PASS" if all(gates.values()) else "FAIL","environment":{"torch":torch.__version__,"hip":torch.version.hip,"device":torch.cuda.get_device_name(0),"arch":torch.cuda.get_device_properties(0).gcnArchName,"binding":n.__file__},"warmup":a.warmup,"iterations":a.iterations,"cases":rows,"large_forward_geomean":fg,"large_forward_backward_geomean":bg,"gates":gates}; (out/"phase3a1_benchmark.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 md=["# Phase 3A1 performance","",f"- Result: **{result['result']}**",f"- Unfused FP32 backend; {a.warmup} warmups, {a.iterations} HIP-event iterations.","", "| Case | Forward speedup | Forward+Backward speedup |","|---|---:|---:|"]+[f"| {r['case']['name']} | {r['forward_speedup']:.3f}× | {r['forward_backward_speedup']:.3f}× |" for r in rows]+["","## Gates",""]+[f"- `{k}`: {'PASS' if v else 'FAIL'}" for k,v in gates.items()]; (out/"PHASE3A1_PERFORMANCE.md").write_text("\n".join(md)+"\n"); print(f"PHASE3A1_PERFORMANCE={result['result']}"); raise SystemExit(0 if result["result"]=="PASS" else 2)
if __name__=="__main__": main()
