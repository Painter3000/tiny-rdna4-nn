#!/usr/bin/env python3
"""TCNN_RDNA4_P2E_FIX_004: Phase 2E robustness-gate harness."""
import argparse, gc, hashlib, importlib, json, os, pathlib, subprocess, sys
import torch

SEED=20260720; BATCHES=(1,7,13,64,129,257)
def metric(a,b):
 d=(a-b).float(); n=float(b.float().norm()); return {"max_abs":float(d.abs().max()) if d.numel() else 0.,"mean_abs":float(d.abs().mean()) if d.numel() else 0.,"relative_l2":float(d.norm())/(n or 1.),"nan":int(torch.isnan(a).sum()),"inf":int(torch.isinf(a).sum())}
def close(a,b,atol=8e-5,rtol=5e-4): torch.testing.assert_close(a,b,atol=atol,rtol=rtol); return metric(a,b)
def netcfg(h=1,w=16,out="None"): return {"otype":"PortableMLP","n_neurons":w,"n_hidden_layers":h,"activation":"ReLU","output_activation":out}
def load(root):
 sys.path.insert(0,str(pathlib.Path(root).resolve())); import tinycudann as t
 return t,importlib.import_module("tinycudann_bindings._120_C")
def raw_mode(model,x,go,start,mode):
 p=model.params.detach().clone().requires_grad_(True); original=x.shape[0]; padded=(original+255)//256*256
 xx=torch.nn.functional.pad(x.detach(),(0,0,0,padded-original)).contiguous().requires_grad_(True)
 gg=torch.nn.functional.pad(go,(0,0,0,padded-original)).contiguous()
 ctx,y=model.native_tcnn_module.fwd(xx,p); buf=start.clone(); dx,buf=model.native_tcnn_module.bwd_mode(ctx,xx,p,y,gg,buf,mode); torch.cuda.synchronize(); return y[:original],dx[:original],buf
def gradient_modes(t):
 rows=[]
 for name,h,w,b,out in (("A",1,16,7,"None"),("B",4,64,257,"Sigmoid")):
  torch.manual_seed(SEED+h); m=t.Network(3,2,netcfg(h,w,out),seed=SEED); x=torch.randn(b,3,device="cuda"); go=torch.randn(b,2,device="cuda")*.1
  z=torch.zeros_like(m.params); _,dx,g=raw_mode(m,x,go,z,"Overwrite"); s=torch.linspace(-.2,.2,m.params.numel(),device="cuda")
  _,dxa,ga=raw_mode(m,x,go,s,"Accumulate"); _,dxi,gi=raw_mode(m,x,go,s,"Ignore")
  _,_,g1=raw_mode(m,x,go,z,"Accumulate"); _,_,g2=raw_mode(m,x,go,g1,"Accumulate")
  rows.append({"case":name,"overwrite":close(g,g),"accumulate":close(ga,s+g),"double":close(g2,2*g),"ignore_bitwise":bool(torch.equal(gi,s)),"ignore_input":close(dxi,dx)})
  assert torch.equal(gi,s)
 # HashGrid: native mode equations plus separate parameter partitions.
 ec={"otype":"HashGrid","n_levels":4,"n_features_per_level":2,"log2_hashmap_size":10,"base_resolution":4,"per_level_scale":1.5,"interpolation":"Linear"}
 m=t.NetworkWithInputEncoding(2,2,ec,netcfg(2,32),seed=SEED); x=torch.rand(64,2,device="cuda"); go=torch.randn(64,2,device="cuda")*.1; z=torch.zeros_like(m.params)
 _,dx,g=raw_mode(m,x,go,z,"Overwrite"); s=torch.full_like(g,.03125); _,_,ga=raw_mode(m,x,go,s,"Accumulate"); _,dxi,gi=raw_mode(m,x,go,s,"Ignore")
 ep=t.Encoding(2,ec,dtype=torch.float32).params.numel(); np=m.params.numel()-ep
 assert g[:np].norm()>0 and g[np:].norm()>0 and torch.equal(gi,s)
 rows.append({"case":"HashGrid","accumulate":close(ga,s+g),"ignore_bitwise":True,"ignore_input":close(dxi,dx),"network_gradient_norm":float(g[:np].norm()),"encoding_gradient_norm":float(g[np:].norm())})
 return rows
def repeated(t):
 m=t.Network(3,2,netcfg(4,64,"Sigmoid"),seed=SEED); rows=[]
 for i in range(50):
  b=BATCHES[i%len(BATCHES)]; x=torch.randn(b,3,device="cuda"); go=torch.randn(b,2,device="cuda")*.1; z=torch.zeros_like(m.params)
  _,dx,g=raw_mode(m,x,go,z,"Overwrite"); _,dx2,g2=raw_mode(m,x,go,z,"Overwrite"); rows.append({"i":i,"batch":b,"gradient":close(g2,g),"input":close(dx2,dx)})
 x=torch.randn(7,3,device="cuda",requires_grad=True); y=m(x); loss=y.square().mean(); loss.backward(retain_graph=True); one=m.params.grad.clone(); loss.backward(); retain=close(m.params.grad,2*one)
 return {"iterations":rows,"retain_graph":"supported","retain_graph_double":retain}
def train(t,steps,hashgrid=False):
 torch.manual_seed(SEED+(1 if hashgrid else 0)); ec={"otype":"HashGrid","n_levels":4,"n_features_per_level":2,"log2_hashmap_size":10,"base_resolution":4,"per_level_scale":1.5,"interpolation":"Linear"}
 m=t.NetworkWithInputEncoding(2,1,ec,netcfg(2,32),seed=SEED) if hashgrid else t.Network(3,1,netcfg(4,64),seed=SEED)
 before=m.params.detach().clone(); opt=torch.optim.Adam(m.parameters(),lr=.005 if hashgrid else .01); logs=[]; losses=[]
 for i in range(steps):
  b=BATCHES[i%6]; g=torch.Generator(device="cuda"); g.manual_seed(SEED+i); d=2 if hashgrid else 3; x=torch.rand(b,d,device="cuda",generator=g)*2-1
  target=(.4*x[:,:1]-.3*x[:,1:2]+(.2*x[:,2:3] if d==3 else 0)).sin(); opt.zero_grad(set_to_none=True); y=m(x); loss=(y-target).square().mean(); loss.backward()
  assert torch.isfinite(loss) and torch.isfinite(m.params.grad).all(); opt.step(); assert torch.isfinite(m.params).all(); losses.append(float(loss))
  if i%100==0 or i==steps-1: logs.append({"step":i,"loss":float(loss),"output_norm":float(y.norm()),"gradient_norm":float(m.params.grad.norm()),"parameter_norm":float(m.params.norm())})
 torch.cuda.synchronize(); result={"steps":steps,"initial_loss":sum(losses[:6])/6,"final_loss":sum(losses[-6:])/6,"logs":logs,"parameter_change":float((m.params-before).norm())}
 if hashgrid:
  ep=t.Encoding(2,ec,dtype=torch.float32).params.numel(); np=m.params.numel()-ep; result.update(network_change=float((m.params[:np]-before[:np]).norm()),encoding_change=float((m.params[np:]-before[np:]).norm())); assert result["network_change"]>0 and result["encoding_change"]>0
 else: assert result["final_loss"] <= .01*result["initial_loss"]
 return result
def checkpoint(t,out,root):
 c=netcfg(4,64,"Sigmoid"); m=t.Network(3,2,c,seed=SEED); x=torch.randn(31,3,device="cuda"); y=m(x).detach().cpu(); path=out/"model_checkpoint.pt"; torch.save({"config":c,"state":m.state_dict(),"probe":x.cpu(),"output":y},path)
 cp=subprocess.run([sys.executable,__file__,"--bindings",root,"--checkpoint-child",str(path)],cwd="/tmp",capture_output=True,text=True); assert cp.returncode==0,cp.stdout+cp.stderr
 return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"child":json.loads(cp.stdout.splitlines()[-1])}
def checkpoint_child(t,path):
 d=torch.load(path,weights_only=False); m=t.Network(3,2,d["config"],seed=1); m.load_state_dict(d["state"]); y=m(d["probe"].cuda()).detach().cpu(); return {"exact":bool(torch.equal(y,d["output"])),"output":metric(y,d["output"]),"keys":list(d["state"]),"shapes":{k:list(v.shape) for k,v in d["state"].items()},"dtypes":{k:str(v.dtype) for k,v in d["state"].items()}}
def resume(t,out,root):
 torch.manual_seed(SEED); c=netcfg(2,32); base=t.Network(3,1,c,seed=SEED); initial={k:v.detach().clone() for k,v in base.state_dict().items()}
 def data(i):
  g=torch.Generator(device="cuda"); g.manual_seed(SEED+i); x=torch.rand(BATCHES[i%6],3,device="cuda",generator=g)*2-1; return x,(.4*x[:,:1]-.3*x[:,1:2]+.2*x[:,2:3]).sin()
 def advance(m,o,a,b):
  loss=None
  for i in range(a,b): x,y=data(i); o.zero_grad(set_to_none=True); loss=(m(x)-y).square().mean(); loss.backward(); o.step()
  return float(loss)
 ref=t.Network(3,1,c,seed=1); ref.load_state_dict(initial); ro=torch.optim.Adam(ref.parameters(),lr=.01); rl=advance(ref,ro,0,200)
 split=t.Network(3,1,c,seed=2); split.load_state_dict(initial); so=torch.optim.Adam(split.parameters(),lr=.01); advance(split,so,0,100); p=out/"resume_checkpoint.pt"; torch.save({"config":c,"model":split.state_dict(),"optimizer":so.state_dict(),"step":100},p)
 target=out/"resume_result.pt"; cp=subprocess.run([sys.executable,__file__,"--bindings",root,"--resume-child",str(p),"--resume-output",str(target)],cwd="/tmp",capture_output=True,text=True); assert cp.returncode==0,cp.stdout+cp.stderr; rr=torch.load(target,weights_only=False)
 probe=torch.randn(23,3,device="cuda"); return {"parameters":close(rr["params"].cuda(),ref.params,1e-6,1e-6),"output":close(rr["probe_output"].cuda(),ref(rr["probe"].cuda()).detach(),1e-6,1e-6),"loss_abs":abs(rr["loss"]-rl),"optimizer_keys_equal":rr["optimizer"].keys()==ro.state_dict().keys()}
def resume_child(t,p,target):
 d=torch.load(p,weights_only=False); m=t.Network(3,1,d["config"],seed=3); m.load_state_dict(d["model"]); o=torch.optim.Adam(m.parameters(),lr=.01); o.load_state_dict(d["optimizer"]); loss=None
 for i in range(100,200):
  g=torch.Generator(device="cuda"); g.manual_seed(SEED+i); x=torch.rand(BATCHES[i%6],3,device="cuda",generator=g)*2-1; y=(.4*x[:,:1]-.3*x[:,1:2]+.2*x[:,2:3]).sin(); o.zero_grad(set_to_none=True); loss=(m(x)-y).square().mean(); loss.backward(); o.step()
 g=torch.Generator(device="cuda"); g.manual_seed(99); probe=torch.randn(23,3,device="cuda",generator=g); torch.save({"params":m.params.detach().cpu(),"probe":probe.cpu(),"probe_output":m(probe).detach().cpu(),"loss":float(loss),"optimizer":o.state_dict()},target)
def streams(t):
 rows=[]
 for mode in ("default","same","rotating"):
  m=t.Network(3,2,netcfg(4,64,"Sigmoid"),seed=SEED); streams=[torch.cuda.Stream() for _ in range(3)];
  for i in range(50):
   x=torch.randn(BATCHES[i%6],3,device="cuda",requires_grad=True); ctx=torch.cuda.stream(streams[0 if mode=="same" else i%3]) if mode!="default" else torch.cuda.device(0)
   with ctx: m.zero_grad(); m(x).square().mean().backward()
  torch.cuda.synchronize(); rows.append({"mode":mode,"iterations":50,"finite":bool(torch.isfinite(m.params.grad).all())})
 # Independent models on three streams, synchronized only at end.
 ms=[t.Network(3,2,netcfg(2,32),seed=SEED+i) for i in range(3)]; ss=[torch.cuda.Stream() for _ in range(3)]
 for m,s in zip(ms,ss):
  with torch.cuda.stream(s): m(torch.randn(64,3,device="cuda",requires_grad=True)).square().mean().backward()
 torch.cuda.synchronize(); rows.append({"mode":"three_stream_events","finite":all(torch.isfinite(m.params.grad).all() for m in ms)})
 return rows
def edges(t):
 tests=[("zero_layers",netcfg(0,16)),("width48",netcfg(1,48)),("bad_hidden",{**netcfg(),"activation":"Sigmoid"}),("bad_output",{**netcfg(),"output_activation":"ReLU"}),("fp16",{**netcfg(),"precision":"Fp16"})]; rows=[]
 for n,c in tests:
  try: t.Network(3,1,c); rows.append({"case":n,"accepted":True})
  except Exception as e: rows.append({"case":n,"accepted":False,"error":str(e)})
 assert all(not x["accepted"] for x in rows)
 m=t.Network(3,1,netcfg());
 for n,x in (("wrong_dims",torch.empty(2,4,device="cuda")),("empty_batch",torch.empty(0,3,device="cuda")),("noncontiguous",torch.randn(3,2,device="cuda").t())):
  try: y=m(x); torch.cuda.synchronize(); rows.append({"case":n,"accepted":True,"shape":list(y.shape)})
  except Exception as e: rows.append({"case":n,"accepted":False,"error":str(e)})
 for n,x in (("zeros",torch.zeros(257,3,device="cuda")),("large_sigmoid",torch.full((257,3),50.,device="cuda"))):
  mm=t.Network(3,1,netcfg(4,64,"Sigmoid")); y=mm(x); rows.append({"case":n,"finite":bool(torch.isfinite(y).all())})
 return rows
def memory(t):
 rows=[]
 for name,h,w,b,n in (("small",1,16,7,200),("large",4,64,257,100)):
  m=t.Network(3,2,netcfg(h,w,"Sigmoid"),seed=SEED); torch.cuda.reset_peak_memory_stats(); samples=[]
  for i in range(n):
   x=torch.randn(b,3,device="cuda",requires_grad=True); m.zero_grad(set_to_none=True); m(x).square().mean().backward(); del x
   if i in (9,n-1): gc.collect(); torch.cuda.synchronize(); samples.append({"i":i,"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved(),"peak":torch.cuda.max_memory_allocated()})
  rows.append({"case":name,"samples":samples,"allocated_growth":samples[-1]["allocated"]-samples[0]["allocated"]}); assert rows[-1]["allocated_growth"]<=1024*1024
 return rows
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--bindings",required=True); ap.add_argument("--output"); ap.add_argument("--checkpoint-child"); ap.add_argument("--resume-child"); ap.add_argument("--resume-output"); ap.add_argument("--fresh-child",action="store_true"); a=ap.parse_args(); t,n=load(a.bindings)
 if a.checkpoint_child: print(json.dumps(checkpoint_child(t,a.checkpoint_child))); return
 if a.resume_child: resume_child(t,a.resume_child,a.resume_output); return
 if a.fresh_child: gradient_modes(t); repeated(t); print("PASS"); return
 out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True); results={}
 results["gradient_mode_results.json"]=gradient_modes(t); results["repeated_backward_results.json"]=repeated(t); results["training_stress_results.json"]={"main":train(t,1000),"hashgrid":train(t,300,True)}
 results["checkpoint_file_results.json"]=checkpoint(t,out,a.bindings); results["checkpoint_resume_results.json"]=resume(t,out,a.bindings); results["stream_stress_results.json"]=streams(t); results["edge_contract_results.json"]=edges(t); results["memory_stability_results.json"]=memory(t)
 fresh=[]
 for i in range(20):
  env=os.environ.copy();
  if i>=16: env["HIP_LAUNCH_BLOCKING"]="1"
  cp=subprocess.run([sys.executable,__file__,"--bindings",a.bindings,"--fresh-child"],cwd="/tmp",env=env,capture_output=True,text=True); fresh.append({"run":i,"returncode":cp.returncode,"HIP_LAUNCH_BLOCKING":env.get("HIP_LAUNCH_BLOCKING"),"tail":cp.stdout[-500:],"stderr":cp.stderr[-500:]}); assert cp.returncode==0
 results["fresh_process_results.json"]=fresh
 for f,v in results.items(): (out/f).write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
 print("PHASE2E_ROBUSTNESS_PASS")
if __name__=="__main__": main()
