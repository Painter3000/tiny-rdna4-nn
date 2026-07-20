#!/usr/bin/env python3
"""TCNN_RDNA4_P3A1_HIPBLASLT_004: complete Phase 3A1 validation gate."""
import argparse, importlib, json, os, pathlib, subprocess, sys, gc
import torch

SEED=20260720; OA=1e-4; OR=8e-4; PA=7e-4; PR=3e-3
CASES=(
 ("A",1,16,2,1,1,"ReLU","None"),("B",1,32,3,3,7,"None","None"),
 ("C",2,16,8,1,64,"ReLU","Sigmoid"),("D",2,32,3,8,257,"ReLU","None"),
 ("E",2,64,8,3,1024,"None","Sigmoid"),("F",4,32,2,1,64,"ReLU","None"),
 ("G",4,64,3,3,257,"ReLU","Sigmoid"),("H",4,128,8,8,1024,"ReLU","None"))
def cfg(c,b): return {"otype":b,"n_hidden_layers":c[1],"n_neurons":c[2],"activation":c[6],"output_activation":c[7]}
def load(root):
 sys.path.insert(0,str(pathlib.Path(root).resolve())); import tinycudann as t
 return t,importlib.import_module("tinycudann_bindings._120_C")
def metric(a,b):
 d=(a-b).float(); n=float(b.float().norm()); return {"max_abs":float(d.abs().max()) if d.numel() else 0.,"mean_abs":float(d.abs().mean()) if d.numel() else 0.,"relative_l2":float(d.norm())/(n or 1.),"nan":int(torch.isnan(a).sum()),"inf":int(torch.isinf(a).sum()),"exact":bool(torch.equal(a,b))}
def close(a,b,pa=False): torch.testing.assert_close(a,b,atol=PA if pa else OA,rtol=PR if pa else OR); return metric(a,b)
def pair(t,c):
 p=t.Network(c[3],c[4],cfg(c,"PortableMLP"),seed=SEED); h=t.Network(c[3],c[4],cfg(c,"HipBLASLtMLP"),seed=SEED+1)
 assert p.params.shape==h.params.shape and p.state_dict().keys()==h.state_dict().keys(); h.params.data.copy_(p.params); return p,h
def public_case(t,c,explicit=False):
 g=torch.Generator(device="cuda"); g.manual_seed(SEED+ord(c[0])); v=torch.randn(c[5],c[3],device="cuda",generator=g)*.25; go=torch.randn(c[5],c[4],device="cuda",generator=g)*.2; p,h=pair(t,c); xp=v.clone().requires_grad_(); xh=v.clone().requires_grad_(); s=torch.cuda.Stream() if explicit else None
 with (torch.cuda.stream(s) if s else torch.cuda.device(0)): yp=p(xp); yh=h(xh); yp.backward(go); yh.backward(go)
 (s.synchronize() if s else torch.cuda.synchronize()); return {"case":c[0],"stream":"explicit" if explicit else "default","output":close(yh,yp),"input_gradient":close(xh.grad,xp.grad),"parameter_gradient":close(h.params.grad,p.params.grad,True),"params":p.params.numel()}
def raw(m,x,go,start,mode):
 n=x.shape[0]; q=(n+255)//256*256; xx=torch.nn.functional.pad(x,(0,0,0,q-n)).contiguous().requires_grad_(); gg=torch.nn.functional.pad(go,(0,0,0,q-n)).contiguous(); p=m.params.detach().clone().requires_grad_(); ctx,y=m.native_tcnn_module.fwd(xx,p); dx,dg=m.native_tcnn_module.bwd_mode(ctx,xx,p,y,gg,start.clone(),mode); torch.cuda.synchronize(); return y[:n],dx[:n],dg
def modes(t):
 rows=[]
 for name in ("B","G"):
  c=next(x for x in CASES if x[0]==name); p,h=pair(t,c); x=torch.randn(c[5],c[3],device="cuda")*.2; go=torch.randn(c[5],c[4],device="cuda")*.1; z=torch.zeros_like(p.params); s=torch.linspace(-.2,.2,p.params.numel(),device="cuda"); py,pdx,pg=raw(p,x,go,z,"Overwrite"); hy,hdx,hg=raw(h,x,go,z,"Overwrite"); _,_,ha=raw(h,x,go,s,"Accumulate"); _,hix,hi=raw(h,x,go,s,"Ignore"); _,_,h1=raw(h,x,go,z,"Accumulate"); _,_,h2=raw(h,x,go,h1,"Accumulate"); assert torch.equal(hi,s)
  rows.append({"case":name,"output":close(hy,py),"input":close(hdx,pdx),"overwrite":close(hg,pg,True),"accumulate":close(ha,s+hg,True),"double":close(h2,2*hg,True),"ignore_exact":True,"ignore_input":close(hix,pdx)})
 return rows
def ecfg(n):
 if n=="Identity": return 3,{"otype":n}
 if n=="Frequency": return 3,{"otype":n,"n_frequencies":4}
 if n=="OneBlob": return 2,{"otype":n,"n_bins":8}
 return 2,{"otype":"HashGrid","n_levels":4,"n_features_per_level":2,"log2_hashmap_size":10,"base_resolution":4,"per_level_scale":1.5,"interpolation":"Linear"}
def encodings(t):
 rows=[]; c=("ENC",2,32,0,3,257,"ReLU","Sigmoid")
 for name in ("Identity","Frequency","OneBlob","HashGrid"):
  d,e=ecfg(name); p=t.NetworkWithInputEncoding(d,3,e,cfg(c,"PortableMLP"),seed=SEED); h=t.NetworkWithInputEncoding(d,3,e,cfg(c,"HipBLASLtMLP"),seed=1); h.params.data.copy_(p.params); v=torch.rand(257,d,device="cuda"); xp=v.clone().requires_grad_(); xh=v.clone().requires_grad_(); go=torch.randn(257,3,device="cuda")*.1; yp=p(xp); yh=h(xh); yp.backward(go); yh.backward(go); torch.cuda.synchronize(); ep=t.Encoding(d,e,dtype=torch.float32).params.numel(); np=p.params.numel()-ep; row={"encoding":name,"output":close(yh,yp),"input":close(xh.grad,xp.grad),"network_gradient":close(h.params.grad[:np],p.params.grad[:np],True),"encoding_params":ep};
  if ep: row["encoding_gradient"]=close(h.params.grad[np:],p.params.grad[np:],True); assert float(h.params.grad[np:].norm())>0
  rows.append(row)
 return rows
def streams(t):
 c=next(x for x in CASES if x[0]=="G"); p,h=pair(t,c); rows=[]
 for mode in ("default","same","rotating"):
  ss=[torch.cuda.Stream() for _ in range(3)]
  for i in range(50):
   x=torch.randn((1,7,64,257)[i%4],c[3],device="cuda",requires_grad=True); context=torch.cuda.stream(ss[0 if mode=="same" else i%3]) if mode!="default" else torch.cuda.device(0)
   with context: h.zero_grad(); h(x).square().mean().backward()
  torch.cuda.synchronize(); rows.append({"mode":mode,"iterations":50,"finite":bool(torch.isfinite(h.params.grad).all())})
 # Real event-ordered handoff across three independent buffers/models.
 ss=[torch.cuda.Stream() for _ in range(3)]; es=[torch.cuda.Event() for _ in range(3)]; ms=[pair(t,c)[1] for _ in range(3)]
 for i,(m,s,e) in enumerate(zip(ms,ss,es)):
  if i: s.wait_event(es[i-1])
  with torch.cuda.stream(s): m(torch.randn(257,c[3],device="cuda",requires_grad=True)).square().mean().backward(); e.record(s)
 es[-1].synchronize(); rows.append({"mode":"three_stream_events","finite":all(bool(torch.isfinite(m.params.grad).all()) for m in ms)})
 return rows
def repeat_memory(t):
 c=next(x for x in CASES if x[0]=="G"); h=pair(t,c)[1]; batches=(1,7,13,64,129,257,1024); torch.cuda.reset_peak_memory_stats(); samples=[]
 for i in range(200):
  x=torch.randn(batches[i%7],c[3],device="cuda",requires_grad=True); h.zero_grad(set_to_none=True); h(x).square().mean().backward()
  if i in (49,199): gc.collect(); torch.cuda.synchronize(); samples.append({"i":i,"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()})
 assert samples[1]["allocated"]-samples[0]["allocated"]<=1024*1024; return {"iterations":200,"samples":samples,"allocated_growth":samples[1]["allocated"]-samples[0]["allocated"]}
def training_checkpoint(t):
 c=next(x for x in CASES if x[0]=="F"); p,h=pair(t,c); h.load_state_dict(p.state_dict()); opt=torch.optim.Adam(h.parameters(),lr=.01); losses=[]
 for i in range(500):
  g=torch.Generator(device="cuda"); g.manual_seed(SEED+i); x=torch.rand(64,c[3],device="cuda",generator=g)*2-1; target=(.4*x[:,:1]-.3*x[:,1:2]).sin(); opt.zero_grad(set_to_none=True); loss=(h(x)-target).square().mean(); loss.backward(); opt.step(); assert torch.isfinite(loss) and torch.isfinite(h.params).all() and torch.isfinite(h.params.grad).all(); losses.append(float(loss))
 assert losses[-1]<=.05*losses[0]; probe=torch.randn(37,c[3],device="cuda"); p.load_state_dict(h.state_dict()); exact=close(p(probe),h(probe)); return {"steps":500,"initial":losses[0],"final":losses[-1],"ratio":losses[-1]/losses[0],"state_interchange":exact}
def train_range(model,opt,start,stop):
 losses=[]
 for i in range(start,stop):
  g=torch.Generator(device="cuda"); g.manual_seed(SEED+10000+i); x=torch.rand(64,2,device="cuda",generator=g)*2-1; target=(.4*x[:,:1]-.3*x[:,1:2]).sin(); opt.zero_grad(set_to_none=True); loss=(model(x)-target).square().mean(); loss.backward(); opt.step(); losses.append(float(loss))
 return losses
def resume_child(t,checkpoint,result_path):
 c=next(x for x in CASES if x[0]=="F"); model=t.Network(c[3],c[4],cfg(c,"HipBLASLtMLP"),seed=1); opt=torch.optim.Adam(model.parameters(),lr=.01); state=torch.load(checkpoint,map_location="cuda",weights_only=True); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); losses=train_range(model,opt,100,200); probe=torch.linspace(-1,1,74,device="cuda").reshape(37,2); torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"probe":model(probe).detach(),"final_loss":losses[-1]},result_path); print("PHASE3A1_RESUME_CHILD=PASS")
def fresh_checkpoint_resume(t,out):
 c=next(x for x in CASES if x[0]=="F"); reference=t.Network(c[3],c[4],cfg(c,"HipBLASLtMLP"),seed=SEED); split=t.Network(c[3],c[4],cfg(c,"HipBLASLtMLP"),seed=1); split.load_state_dict(reference.state_dict()); ropt=torch.optim.Adam(reference.parameters(),lr=.01); sopt=torch.optim.Adam(split.parameters(),lr=.01); train_range(reference,ropt,0,200); train_range(split,sopt,0,100); checkpoint=out/"checkpoint_step100.pt"; resumed=out/"checkpoint_resumed_step200.pt"; torch.save({"model":split.state_dict(),"optimizer":sopt.state_dict()},checkpoint)
 cp=subprocess.run([sys.executable,__file__,"--bindings",sys.path[0],"--resume-checkpoint",str(checkpoint),"--resume-output",str(resumed)],cwd="/tmp",capture_output=True,text=True); assert cp.returncode==0,(cp.stdout,cp.stderr); state=torch.load(resumed,map_location="cuda",weights_only=True); probe=torch.linspace(-1,1,74,device="cuda").reshape(37,2); params=close(state["model"]["params"],reference.state_dict()["params"],True); outputs=close(state["probe"],reference(probe).detach()); return {"split_steps":[100,100],"fresh_process_returncode":cp.returncode,"parameters":params,"probe_output":outputs,"child_stdout":cp.stdout.strip(),"final_loss":state["final_loss"]}
def contracts(t):
 c=CASES[0]; p,h=pair(t,c); assert p.params.numel()==h.params.numel(); rejected=[]
 for bad in ({**cfg(c,"HipBLASLtMLP"),"precision":"Fp16"},{**cfg(c,"HipBLASLtMLP"),"n_neurons":48},{**cfg(c,"HipBLASLtMLP"),"n_hidden_layers":3}):
  try: t.Network(c[3],c[4],bad); rejected.append(False)
  except Exception: rejected.append(True)
 assert all(rejected); return {"portable":p.native_tcnn_module.hyperparams(),"hipblaslt":h.native_tcnn_module.hyperparams(),"unsupported_rejected":rejected}
def child(t,name):
 if name=="HASH": encodings(t)
 else: public_case(t,next(x for x in CASES if x[0]==name),True)
 print("PHASE3A1_CHILD=PASS")
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--bindings",required=True); ap.add_argument("--output"); ap.add_argument("--child"); ap.add_argument("--resume-checkpoint"); ap.add_argument("--resume-output"); a=ap.parse_args(); t,n=load(a.bindings); prop=torch.cuda.get_device_properties(0); assert torch.version.hip and prop.gcnArchName=="gfx1201"
 if a.resume_checkpoint: resume_child(t,a.resume_checkpoint,a.resume_output); return
 if a.child: child(t,a.child); return
 out=pathlib.Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True); result={"result":"PASS","environment":{"torch":torch.__version__,"hip":torch.version.hip,"device":torch.cuda.get_device_name(0),"arch":prop.gcnArchName,"binding":n.__file__},"contracts":contracts(t),"public_cases":[public_case(t,c) for c in CASES]+[public_case(t,next(x for x in CASES if x[0]==n),True) for n in ("D","G","H")],"gradient_modes":modes(t),"encodings":encodings(t),"streams":streams(t),"repeat_memory":repeat_memory(t),"training_checkpoint":training_checkpoint(t)}; result["fresh_checkpoint_resume"]=fresh_checkpoint_resume(t,out)
 fresh=[]
 for i in range(20):
  name=("A","D","G","H","HASH")[i%5]; env=os.environ.copy();
  if i>=16: env["HIP_LAUNCH_BLOCKING"]="1"
  cp=subprocess.run([sys.executable,__file__,"--bindings",a.bindings,"--child",name],cwd="/tmp",env=env,capture_output=True,text=True); fresh.append({"run":i,"case":name,"returncode":cp.returncode,"blocking":env.get("HIP_LAUNCH_BLOCKING"),"stderr":cp.stderr[-500:]}); assert cp.returncode==0
 result["fresh_processes"]=fresh; (out/"phase3a1_validation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print("PHASE3A1_VALIDATION=PASS")
if __name__=="__main__": main()
