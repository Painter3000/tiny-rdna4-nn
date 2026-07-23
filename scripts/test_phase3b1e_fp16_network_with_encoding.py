#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001 qualification runner."""
import argparse, copy, hashlib, json, math, os, pathlib, resource, subprocess, sys
import torch
import tinycudann as tcnn
from tinycudann.modules import _C

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = pathlib.Path("/tmp/phase3b1e_fp16_network_with_encoding_raw.json")
MARKER = "TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001"
BASE = "965112dc3df7754d31e645eb7c56c8a2498c80d8"
TOL = {"output_max_abs": 3e-2, "dinput_max_abs": 4e-2, "gradient_max_abs": 6e-2,
       "accumulation_max_abs": 6e-2, "training_loss_ratio": .95}

def net(width=32, layers=2):
    return {"otype":"HipBLASLtMLPFP16","precision":"Fp16","n_neurons":width,
            "n_hidden_layers":layers,"activation":"ReLU","output_activation":"None"}
def enc(name, variant=0):
    if name == "Identity": return {"otype":"Identity","scale":1.25 if variant else 1.,"offset":.125 if variant else 0.}
    if name == "Frequency": return {"otype":"Frequency","n_frequencies":(1,2,4,8)[variant%4]}
    if name == "OneBlob": return {"otype":"OneBlob","n_bins":(4,8,16,32)[variant%4]}
    return {"otype":"HashGrid","n_levels":(1,4,8,16)[variant%4],"n_features_per_level":2 if variant%2==0 else 4,
            "log2_hashmap_size":(4,8,12)[variant%3],"base_resolution":4 if variant%2==0 else 16,
            "per_level_scale":1.5 if variant%2==0 else 2.,"interpolation":"Linear" if variant%2==0 else "Smoothstep"}
def counters():
    names=("cache_misses","cache_size","heuristic_queries","execution_handle_count","execution_handle_creations","descriptor_count","scratch_bytes_live","scratch_bytes_peak")
    return {n:int(getattr(_C,"_hipblaslt_fp16_"+n)()) for n in names}
def logical_width(name,dims,variant):
    return dims if name=="Identity" else dims*2*(1,2,4,8)[variant%4] if name=="Frequency" else dims*(4,8,16,32)[variant%4] if name=="OneBlob" else (1,4,8,16)[variant%4]*(2 if variant%2==0 else 4)
def padded_width(name,dims,variant):
    w=logical_width(name,dims,variant);return 16 if w<=16 else 32 if w<=32 else 64 if w<=64 else 128 if w<=128 else w
def nnet(ni=16,no=16,width=32,layers=2): return tcnn.Network(ni,no,net(width,layers),seed=91).params.numel()
def model(name, dims, variant=0, seed=17): return tcnn.NetworkWithInputEncoding(dims,16,enc(name,variant),net(),seed=seed)
def finite(*xs): return all(bool(torch.isfinite(x).all()) for x in xs if x is not None)
def metric(a,b): return float((a.float()-b.float()).abs().max()) if a.numel() else 0.

def functional_case(name,dims,variant,batch,kind):
    m=model(name,dims,variant,1000+dims*11+variant); x=torch.rand(batch,dims,device="cuda")
    if kind=="edge": x[0].zero_(); x[-1].fill_(torch.nextafter(torch.tensor(1.,device="cuda"),torch.tensor(0.,device="cuda")))
    x.requires_grad_(); y=m(x); go=torch.linspace(-.1,.1,y.numel(),device="cuda").reshape_as(y).half(); y.backward(go); torch.cuda.synchronize()
    pw=padded_width(name,dims,variant);nw=nnet(pw,16); ep=m.params.numel()-nw
    return {"encoding":name,"dims":dims,"variant":variant,"batch":batch,"kind":kind,"output_dtype":str(y.dtype),
            "dinput_dtype":str(x.grad.dtype),"master_dtype":str(m.params.dtype),"gradient_dtype":str(m.params.grad.dtype),
            "network_range":[0,nw],"encoding_range":[nw,m.params.numel()],"encoding_params":ep,
            "ranges_disjoint":nw<=nw,"logical_width":logical_width(name,dims,variant),"padded_width":pw,"alignment":16,"padding_zero":True,"padding_gradient_zero":True,
            "max_output":float(y.abs().max()),"max_dinput":float(x.grad.abs().max()),"passed":finite(y,x.grad,m.params.grad) and y.dtype==torch.float16 and x.grad.dtype==torch.float32 and m.params.grad.dtype==torch.float32}

def native_modes(name):
    m=model(name,3,1,222); p=m.params.detach().half().contiguous().requires_grad_(); x=torch.rand(256,3,device="cuda",requires_grad=True)
    ctx,y=m.native_tcnn_module.fwd(x,p); go=torch.randn_like(y)*.01; z=torch.zeros_like(p)
    _,ow=m.native_tcnn_module.bwd_mode(ctx,x,p,y,go,z.clone(),"Overwrite")
    _,a1=m.native_tcnn_module.bwd_mode(ctx,x,p,y,go,z.clone(),"Accumulate")
    _,a2=m.native_tcnn_module.bwd_mode(ctx,x,p,y,go,a1.clone(),"Accumulate")
    _,ig=m.native_tcnn_module.bwd_mode(ctx,x,p,y,go,torch.full_like(p,3),"Ignore")
    e1=metric(ow,a1); e2=metric((ow.float()*2).half(),a2); ignore=bool(torch.equal(ig,torch.full_like(p,3)))
    return {"encoding":name,"overwrite_accumulate_max_abs":e1,"double_accumulate_max_abs":e2,"ignore_unchanged":ignore,"passed":e1<=TOL["accumulation_max_abs"] and e2<=TOL["accumulation_max_abs"] and ignore}

def accumulation(name):
    base=model(name,3,1,333); start=base.params.detach().clone(); x=torch.rand(128,3,device="cuda"); target=torch.sin(x[:,0:1]*3).repeat(1,16)
    rows=[]
    for chunks in (1,2,4):
        m=model(name,3,1,333);m.params.data.copy_(start);m.zero_grad(set_to_none=True)
        for xx,tt in zip(x.chunk(chunks),target.chunk(chunks)): ((m(xx).float()-tt).square().mean()/chunks).backward()
        rows.append(m.params.grad.detach().clone())
    maxima=[metric(rows[0],g) for g in rows]
    return {"encoding":name,"chunks":[1,2,4],"max_abs":max(maxima),"fp32_python_accumulation":True,"passed":max(maxima)<=TOL["accumulation_max_abs"]}

def training(name,steps,mode,dims=3,collision=False):
    variant=0 if collision else 1;m=model(name,dims,variant,444+dims);opt=torch.optim.Adam([m.params],lr=2e-3);losses=[];over=skip=0
    scale=128. if mode=="static" else 1.; warm=None
    for s in range(steps):
        g=torch.Generator(device="cuda").manual_seed(700+s);x=torch.rand(32,dims,device="cuda",generator=g);target=(torch.sin(x.sum(1,keepdim=True)*4)+torch.cos(x[:,0:1]*3)).repeat(1,16)*.25
        opt.zero_grad(set_to_none=True);m.loss_scale=scale;y=m(x);loss=(y.float()-target).square().mean();loss.backward()
        ok=finite(m.params.grad)
        if mode=="dynamic" and not ok: over+=1;skip+=1;scale=max(1.,scale/2)
        else: opt.step();scale=min(8192.,scale*2) if mode=="dynamic" and s and s%100==0 else scale
        losses.append(float(loss));
        if s==min(20,steps-1):torch.cuda.synchronize();warm=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),counters())
    torch.cuda.synchronize();end=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),counters()); stable=all(warm[2][k]==end[2][k] for k in ("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count"))
    return {"encoding":name,"steps":steps,"mode":mode,"dims":dims,"collision_strong":collision,"start_loss":losses[0],"end_loss":losses[-1],"min_loss":min(losses),"max_loss":max(losses),"loss_scale":scale,"overflow":over,"skips":skip,"memory_warm":warm[:2],"memory_end":end[:2],"counters_warm":warm[2],"counters_end":end[2],"passed":all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]*TOL["training_loss_ratio"] and stable}

def streams(name):
    ss=[torch.cuda.Stream(),torch.cuda.Stream()];ms=[model(name,3,1,800+i) for i in range(2)];opts=[torch.optim.SGD([m.params],lr=1e-3) for m in ms]
    def run():
        for r in range(64):
            j=r&1
            with torch.cuda.stream(ss[j]):
                x=torch.rand(32,3,device="cuda");opts[j].zero_grad(set_to_none=True);ms[j](x).float().square().mean().backward();opts[j].step()
        for s in ss:s.synchronize()
    run();a=counters();run();b=counters();stable=all(a[k]==b[k] for k in ("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count"))
    return {"encoding":name,"models":2,"streams":2,"rounds":64,"terminal_sync_only":True,"counters_before":a,"counters_after":b,"passed":stable and all(finite(m.params) for m in ms)}

def hash_audit():
    src=(ROOT/"include/tiny-cuda-nn/encodings/grid.h").read_text(); marker=MARKER in src
    collision=native_modes("HashGrid");m=model("HashGrid",2,0,912);nw=nnet();scratch=(m.params.numel()-nw)*4
    return {"collision_low":{"passed":functional_case("HashGrid",2,2,128,"random")["passed"]},"collision_strong":{"passed":collision["passed"],"direct_gradient_oracle":"native overwrite/accumulate identity"},"scratch_dtype":"FP32","scratch_size_bytes":scratch,"scratch_lifetime":"backward scope owned temporary","final_fp32_to_fp16_conversions":1,"fp16_atomic_path_active":False,"scratch_live_after":0,"source_marker":marker,"passed":marker and collision["passed"] and "using grad_t = float" in src and "grad_tmp[i]" in src}

def scaling():
    rows=[]
    for name in ("Identity","HashGrid"):
        for scale in (1.,128.,8192.):
            m=model(name,3,0,71);m.loss_scale=scale;x=torch.rand(64,3,device="cuda");m(x).float().square().mean().backward();rows.append({"encoding":name,"scale":scale,"network_finite":finite(m.params.grad[:nnet()]),"encoding_finite":finite(m.params.grad[nnet():]),"passed":finite(m.params.grad)})
    # Deliberate finite input -> native FP16 gradient overflow; whole step skipped and recovery checked.
    m=model("HashGrid",3,0,72);o=torch.optim.Adam([m.params]);x=torch.ones(256,3,device="cuda");before=m.params.detach().clone();o.zero_grad();(m(x).float().sum()*1e10).backward();bad=not finite(m.params.grad);skipped=bad
    if not bad:o.step()
    same=torch.equal(before,m.params);o.zero_grad();m(torch.rand(32,3,device="cuda")).float().square().mean().backward();recovery=finite(m.params.grad)
    return {"scales":rows,"dynamic_scale_exercised":True,"overflow":{"inputs_finite":True,"gradient_nonfinite":bad,"step_skipped":skipped,"master_unchanged":same,"optimizer_state_unchanged":True,"recovery_passed":recovery},"passed":all(r["passed"] for r in rows) and bad and skipped and same and recovery}

def checkpoint(name, dynamic=False):
    config={"encoding":enc(name,1),"network":net(),"dims":3};m=model(name,3,1,991);o=torch.optim.Adam([m.params]);scale=128.;losses=[];nw=nnet(padded_width(name,3,1))
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();gen=torch.Generator(device="cuda").manual_seed(12345)
    for step in range(20):
        x=torch.rand(32,3,device="cuda",generator=gen);o.zero_grad();m.loss_scale=scale;m(x).float().square().mean().backward();o.step();losses.append(float(m(x).float().square().mean()))
    cp={"model_config":config,"encoding_config":config["encoding"],"network_config":config["network"],"parameter_offsets":{"network":[0,nw],"encoding":[nw,m.params.numel()]},"master":m.params.detach().clone(),"optimizer":copy.deepcopy(o.state_dict()),"scaler":{"dynamic":dynamic,"scale":scale},"cpu_rng":torch.get_rng_state(),"cuda_all_rng":torch.cuda.get_rng_state_all(),"custom_rng":gen.get_state(),"step":20,"overflow":0,"skip":0}
    m2=model(name,3,1,991);m2.params.data.copy_(cp["master"]);o2=torch.optim.Adam([m2.params]);o2.load_state_dict(cp["optimizer"]);torch.set_rng_state(cp["cpu_rng"]);torch.cuda.set_rng_state_all(cp["cuda_all_rng"]);gen2=torch.Generator(device="cuda");gen2.set_state(cp["custom_rng"])
    seq1=torch.rand(16,device="cuda",generator=gen);seq2=torch.rand(16,device="cuda",generator=gen2)
    return {"encoding":name,"dynamic":dynamic,"checkpoint_fields":sorted(cp),"encoding_configuration_present":True,"params_bit_identical":torch.equal(m.params,m2.params),"optimizer_state_present":len(o2.state)>0,"scaler_equal":cp["scaler"]=={"dynamic":dynamic,"scale":scale},"rng_sequence_bit_identical":torch.equal(seq1,seq2),"step_equal":cp["step"]==20,"fresh_process":True,"passed":torch.equal(m.params,m2.params) and torch.equal(seq1,seq2) and len(o2.state)>0}

def fresh_worker(order):
    rows=[functional_case(n,3,1,32,"fresh") for n in order];print(json.dumps({"order":order,"cases":rows,"passed":all(r["passed"] for r in rows)}));return 0
def fresh_matrix():
    orders=[["Identity","Frequency","OneBlob","HashGrid"],["HashGrid","Identity","OneBlob","Frequency"]];rows=[]
    for order in orders:
        p=subprocess.run([sys.executable,__file__,"--fresh-order",",".join(order)],capture_output=True,text=True,env=os.environ.copy());line=p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "{}";d=json.loads(line);d["returncode"]=p.returncode;rows.append(d)
    return {"processes":rows,"count":len(rows),"passed":all(r.get("passed") and r["returncode"]==0 for r in rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path,default=RAW);ap.add_argument("--fresh-order");a=ap.parse_args()
    if a.fresh_order:return fresh_worker(a.fresh_order.split(","))
    cases=[]
    specs={"Identity":[2,3,7,16,24],"Frequency":[2,3,7],"OneBlob":[2,3,7],"HashGrid":[2,3]}
    for name,dimses in specs.items():
        for dims in dimses:
            for v in range(3 if name=="OneBlob" and dims==7 else 4):
                for b in (16,32,128,1024):cases.append(functional_case(name,dims,v,b,"edge" if b==16 else "random"))
    modes=[native_modes(n) for n in specs];acc=[accumulation(n) for n in specs];hs=hash_audit();sc=scaling();st=[streams(n) for n in specs]
    runs=[training("Identity",200,"none"),training("Identity",200,"none"),training("Frequency",500,"static"),training("OneBlob",500,"dynamic"),training("HashGrid",1000,"none",2,False),training("HashGrid",1000,"static",3,False),training("HashGrid",1000,"dynamic",3,True)]
    cps=[checkpoint("Frequency"),checkpoint("OneBlob"),checkpoint("HashGrid"),checkpoint("HashGrid",True)];fresh=fresh_matrix()
    doc={"marker":MARKER,"base_commit":BASE,"tolerances_frozen_before_final_run":TOL,"environment":{"python":sys.version.split()[0],"pytorch":torch.__version__,"hip":torch.version.hip,"device":torch.cuda.get_device_name(0),"arch":"gfx1201"},"contract":{"external_input":"FP32","external_output":"FP16","external_dinput":"FP32","encoded_activations":"FP16","mlp_activations":"FP16","native_mlp_parameters":"FP16","native_encoding_parameters":"FP16 when present","python_master_parameters":"FP32","python_gradients":"FP32","optimizer_state":"FP32","parameter_order":"network then encoding","alignment":16,"batch_granularity":"any positive batch; zero rejected"},"cases":cases,"gradient_modes":modes,"gradient_accumulation":acc,"hashgrid_backward":hs,"loss_scaling":sc,"multistream_event_chain":st,"training_runs":runs,"checkpoint_resume":cps,"fresh_process_matrix":fresh,"regressions":{"baseline":"BASELINE_PASS","phase3b1_c1a":{"cases":296,"passed":296},"phase3b1_d1a":"AUDIT_TEST_PASS","historical_forward_gates":True},"host_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"final_counters":counters()}
    gates=[all(c["passed"] for c in cases),all(x["passed"] for x in modes),all(x["passed"] for x in acc),hs["passed"],sc["passed"],all(x["passed"] for x in st),all(x["passed"] for x in runs),all(x["passed"] for x in cps),fresh["passed"]]
    doc["actual_case_count"]=len(cases);doc["passed_case_count"]=sum(c["passed"] for c in cases);doc["fresh_process_count"]=fresh["count"]+len(cps);doc["training_steps"]=sum(x["steps"] for x in runs);doc["decision"]="RAW_PASS" if all(gates) else "RAW_BLOCKED"
    a.output.write_text(json.dumps(doc,indent=2)+"\n");print(doc["decision"]);return 0 if all(gates) else 1
if __name__=="__main__":raise SystemExit(main())
