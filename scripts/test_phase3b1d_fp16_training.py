#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1D_FP16_TRAINING_001 training qualification."""
import argparse,copy,hashlib,json,math,os,pathlib,resource,subprocess,sys,time
import torch, tinycudann as tcnn
import test_phase3b1c_fp16_backward as bw
from tinycudann.modules import _C
ROOT=pathlib.Path(__file__).resolve().parents[1];MARKER="TCNN_RDNA4_P3B1D_FP16_TRAINING_001";BASE="09b6a777bda2b4ed4d14bf22e265b5fb8f83fb65"
TRAIN_TOL={"loss_abs":2e-3,"master_max_abs":2e-3,"gradient_max_abs":2e-3,"optimizer_state_max_abs":2e-3}
def cfg(w,l,ha="ReLU",oa="None"):return {"otype":"HipBLASLtMLPFP16","precision":"Fp16","n_neurons":w,"n_hidden_layers":l,"activation":ha,"output_activation":oa}
def counters():
 ns=("cache_misses","cache_size","heuristic_queries","execution_handle_count","execution_handle_creations","descriptor_count","scratch_bytes_live","scratch_bytes_peak")
 return {n:int(getattr(_C,"_hipblaslt_fp16_"+n)()) for n in ns}
def optimizer(name,p):return torch.optim.SGD([p],lr=2e-3,momentum=.9) if name=="SGD" else torch.optim.Adam([p],lr=5e-4)
def state_finite(opt):return all(not torch.is_tensor(v) or (v.dtype==torch.float32 and torch.isfinite(v).all()) for s in opt.state.values() for v in s.values())
def state_clone(opt):return copy.deepcopy(opt.state_dict())
def state_equal(a,b):
 if type(a)!=type(b):return False
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
 return a==b
class DynamicScaler:
 def __init__(self,scale=128.,growth_factor=2.,backoff_factor=.5,growth_interval=8):self.scale=float(scale);self.growth_factor=float(growth_factor);self.backoff_factor=float(backoff_factor);self.growth_interval=int(growth_interval);self.successful_steps=0;self.growth_tracker=0;self.overflow_count=0;self.skip_step_count=0
 def state_dict(self):return dict(self.__dict__)
 def load_state_dict(self,s):self.__dict__.update(s)
 def step(self,model,opt):
  finite=model.params.grad is not None and bool(torch.isfinite(model.params.grad).all())
  if not finite:self.overflow_count+=1;self.skip_step_count+=1;self.growth_tracker=0;self.scale*=self.backoff_factor;opt.zero_grad(set_to_none=True);return False
  opt.step();self.successful_steps+=1;self.growth_tracker+=1
  if self.growth_tracker>=self.growth_interval:self.scale*=self.growth_factor;self.growth_tracker=0
  opt.zero_grad(set_to_none=True);return True
def data(step,b,ni,no,near=False,gen=None):
 x=torch.randn(b,ni,device="cuda",generator=gen)*(.01 if near else .2);target=torch.sin(x[:,:no] if no<=ni else torch.nn.functional.pad(x,(0,no-ni)))
 return x,target
def train_step(m,opt,x,target,scaler=None,force_overflow=False):
 opt.zero_grad(set_to_none=True);m.loss_scale=scaler.scale if scaler else m.loss_scale;y=m(x);loss=(y.float()-target.float()).square().mean()
 if force_overflow:loss=loss*1e12
 loss.backward();grad=m.params.grad.detach().clone() if m.params.grad is not None else None;before=m.params.detach().clone();ostate=state_clone(opt)
 stepped=scaler.step(m,opt) if scaler else (opt.step() or True);return {"loss":float(loss.detach()),"grad":grad,"before":before,"stepped":bool(stepped),"state_before":ostate}
def short_case(spec):
 optn,ha,oa,l,w,b,near,ni,no=spec;m=tcnn.Network(ni,no,cfg(w,l,ha,oa),seed=9000+w+l);opt=optimizer(optn,m.params);g=torch.Generator(device="cuda").manual_seed(1000+w+l+b);losses=[];maxgrad=0.
 for step in range(50):
  x,t=data(step,b,ni,no,near,g);r=train_step(m,opt,x,t);losses.append(r["loss"]);maxgrad=max(maxgrad,float(r["grad"].abs().max()));assert m.params.dtype==torch.float32 and r["grad"].dtype==torch.float32 and state_finite(opt)
 torch.cuda.synchronize();return {"optimizer":optn,"activation":[ha,oa],"layers":l,"width":w,"batch":b,"shape":[ni,w,no],"near_zero":near,"steps":50,"start_loss":losses[0],"end_loss":losses[-1],"min_loss":min(losses),"max_loss":max(losses),"max_gradient":maxgrad,"passed":all(math.isfinite(x) for x in losses) and torch.isfinite(m.params).all().item()}
def static_scaling():
 torch.manual_seed(700);x,t=data(0,128,32,32);results=[];reference=None
 for scale in (1,8,128,1024,8192):
  m=tcnn.Network(32,32,cfg(32,2),seed=77);m.loss_scale=scale;opt=optimizer("Adam",m.params);r=train_step(m,opt,x,t);g=r["grad"].cpu();p=m.params.detach().cpu();reference=(g,p) if reference is None else reference;results.append({"scale":scale,"finite":bool(torch.isfinite(g).all()),"grad_norm":float(g.norm()),"gradient_max_abs_vs_scale1":float((g-reference[0]).abs().max()),"parameter_max_abs_vs_scale1":float((p-reference[1]).abs().max())})
 # Underflow rescue and deliberate finite-input overflow/skip are characterized separately.
 return {"scales":results,"all_finite":all(x["finite"] for x in results),"passed":all(x["finite"] and x["gradient_max_abs_vs_scale1"]<=2e-3 and x["parameter_max_abs_vs_scale1"]<=2e-5 for x in results)}
def accumulation():
 base=tcnn.Network(32,32,cfg(32,2),seed=123);master=base.params.detach().clone();x,t=data(0,128,32,32)
 out=[]
 for chunks in (1,2,4):
  m=tcnn.Network(32,32,cfg(32,2),seed=123);m.params.data.copy_(master);opt=optimizer("SGD",m.params);opt.zero_grad(set_to_none=True);loss=0.
  for xx,tt in zip(x.chunk(chunks),t.chunk(chunks)):
   z=(m(xx).float()-tt).square().mean()/chunks;z.backward();loss+=float(z.detach())
  grad=m.params.grad.clone();opt.step();out.append({"chunks":chunks,"loss":loss,"grad":grad.cpu(),"params":m.params.detach().cpu()})
 # Native Accumulate itself is already a fail-closed C1a gate; microbatches remain FP32 here.
 mg=max(float((x["grad"]-out[0]["grad"]).abs().max()) for x in out);mp=max(float((x["params"]-out[0]["params"]).abs().max()) for x in out)
 return {"variants":[{"chunks":x["chunks"],"loss":x["loss"]} for x in out],"max_gradient_abs":mg,"max_parameter_abs":mp,"native_accumulate_gate":True,"fp32_microbatch_accumulation":True,"passed":mg<=2e-3 and mp<=2e-5}
def quantized_reference():
 ss=bw.shapes(16,16,16,2);p=bw.make_params(ss,2468).cuda().contiguous().requires_grad_();x=torch.randn(256,16);target=torch.randn(256,16).half();acts=bw.quant_forward(x,p.detach().cpu(),ss,"ReLU","None");go=(2*(acts[-1].float()-target.float())/acts[-1].numel()).half();rdx,rdp,*_=bw.quant_backward(x,p.detach().cpu(),ss,acts,go,"ReLU","None");m=tcnn.Network(16,16,cfg(16,2,"ReLU","None"));xx=x.cuda().contiguous().requires_grad_();ctx,y=m.native_tcnn_module.fwd(xx,p);dx,dp=m.native_tcnn_module.bwd_mode(ctx,xx,p,y,go.cuda(),torch.zeros_like(p),"Overwrite");torch.cuda.synchronize();mdx=bw.metric(dx,rdx);mdp=bw.metric(dp,rdp);opts=[]
 for name in ("SGD","Adam"):
  a=torch.nn.Parameter(p.detach().float());b=torch.nn.Parameter(p.detach().float());oa=optimizer(name,a);ob=optimizer(name,b);a.grad=dp.float();b.grad=rdp.cuda().float();oa.step();ob.step();opts.append({"optimizer":name,"master_max_abs":float((a-b).abs().max()),"state_equal":state_equal(oa.state_dict(),ob.state_dict())})
 return {"dx":mdx,"native_fp16_gradient":mdp,"optimizers":opts,"passed":bw.gate(mdx,"dx") and bw.gate(mdp,"dw") and all(z["master_max_abs"]<=TRAIN_TOL["master_max_abs"] for z in opts)}
def dynamic_test():
 m=tcnn.Network(16,16,cfg(16,1,"None","None"),seed=8);opt=optimizer("Adam",m.params);s=DynamicScaler();events=[]
 for step in range(20):
  x,t=data(step,128,16,16);forced=step==5;before=m.params.detach().clone();os=state_clone(opt);r=train_step(m,opt,x,t,s,forced);same=torch.equal(before,m.params) and state_equal(os,opt.state_dict());events.append({"step":step,"forced_overflow":forced,"stepped":r["stepped"],"skip_unchanged":same if forced else None,"scale":s.scale})
 return {"events":events,"state":s.state_dict(),"passed":events[5]["stepped"] is False and events[5]["skip_unchanged"] and events[6]["stepped"] and s.overflow_count==1 and s.skip_step_count==1}
def stream_training():
 streams=[torch.cuda.Stream(),torch.cuda.Stream()];models=[tcnn.Network(32,32,cfg(32,2),seed=810+i) for i in range(2)];opts=[optimizer("SGD",m.params) for m in models]
 def rounds():
  for step in range(64):
   j=step%2
   with torch.cuda.stream(streams[j]):
    x,t=data(step,128,32,32);train_step(models[j],opts[j],x,t)
  for s in streams:s.synchronize()
 rounds();warm=counters();rounds();end=counters();keys=("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count")
 return {"models":2,"streams":2,"rounds_per_pass":64,"terminal_sync_only":True,"warm":warm,"end":end,"passed":all(warm[k]==end[k] for k in keys) and all(torch.isfinite(m.params).all() for m in models)}
def long_run(optn,shape,dynamic=False):
 ni,w,no,l,b=shape;m=tcnn.Network(ni,no,cfg(w,l,"ReLU","None"),seed=333);opt=optimizer(optn,m.params);s=DynamicScaler() if dynamic else None;g=torch.Generator(device="cuda").manual_seed(444);losses=[];gn=[];mem0=None;c0=None
 nonfinite_grad=0
 for step in range(1000):
  x,t=data(step,b,ni,no,False,g);r=train_step(m,opt,x,t,s,False);losses.append(r["loss"]);gn.append(float(r["grad"].norm()))
  if not torch.isfinite(r["grad"]).all():nonfinite_grad+=1
  if step==49:torch.cuda.synchronize();mem0=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved());c0=counters()
 torch.cuda.synchronize();mem1=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved());c1=counters();stable=all(c1[k]==c0[k] for k in ("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count"))
 expected_nonfinite=(s.overflow_count if s else 0);return {"optimizer":optn,"dynamic":dynamic,"shape":shape,"steps":1000,"start_loss":losses[0],"end_loss":losses[-1],"min_loss":min(losses),"max_loss":max(losses),"parameter_norm":float(m.params.norm()),"gradient_norm_max_finite":max((x for x in gn if math.isfinite(x)),default=0.),"gradient_nonfinite_steps":nonfinite_grad,"optimizer_state_fp32":state_finite(opt),"scaler":s.state_dict() if s else None,"loss_nan_inf":sum(not math.isfinite(x) for x in losses),"memory_warm":mem0,"memory_end":mem1,"counters_warm":c0,"counters_end":c1,"host_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"passed":stable and not any(not math.isfinite(x) for x in losses) and nonfinite_grad==expected_nonfinite and state_finite(opt)}
def checkpoint_protocol(optn,scaling,outpath=None,resume=None):
 config={"ni":16,"no":16,"network":cfg(16,2),"optimizer":optn,"scaling":scaling};m=tcnn.Network(16,16,config["network"],seed=555);opt=optimizer(optn,m.params);sc=DynamicScaler() if scaling=="dynamic" else None;gen=torch.Generator(device="cuda").manual_seed(987);start=0;losses=[]
 if resume:
  cp=torch.load(resume,weights_only=False);m.params.data.copy_(cp["master_params"]);opt.load_state_dict(cp["optimizer_state"]);start=cp["step"];gen.set_state(cp["rng_state_cuda"]);losses=cp["losses"];sc.load_state_dict(cp["scaler_state"]) if sc else None
 for step in range(start,200):
  x,t=data(step,128,16,16,False,gen);forced=scaling=="dynamic" and step==50;r=train_step(m,opt,x,t,sc,forced);losses.append(r["loss"])
  if outpath and step==99:
   torch.save({"master_params":m.params.detach(),"model_config":config,"optimizer_type":optn,"optimizer_state":opt.state_dict(),"step":100,"rng_state_cpu":torch.get_rng_state(),"rng_state_cuda":gen.get_state(),"scaler_state":sc.state_dict() if sc else {"mode":scaling,"scale":128 if scaling=="static" else 1},"losses":losses,"backend":"HipBLASLtMLPFP16","dtypes":{"master":"FP32","native":"FP16","gradient":"FP32"}},outpath);break
 return {"params":m.params.detach().cpu(),"optimizer":opt.state_dict(),"scaler":sc.state_dict() if sc else {"mode":scaling},"loss":losses[-1],"step":len(losses),"rng":gen.get_state()}
def resume_worker(args):
 cp=torch.load(args.resume,weights_only=False);r=checkpoint_protocol(cp["optimizer_type"],cp["model_config"]["scaling"],resume=args.resume);torch.save(r,args.worker_output);return 0
def checkpoint_tests(tmp):
 results=[]
 for optn,scaling in (("SGD","none"),("Adam","none"),("Adam","static"),("Adam","dynamic")):
  torch.manual_seed(1);continuous=checkpoint_protocol(optn,scaling);cp=tmp/f"{optn}_{scaling}.pt";checkpoint_protocol(optn,scaling,cp);out=tmp/f"{optn}_{scaling}_resumed.pt";env=os.environ.copy();done=subprocess.run([sys.executable,__file__,"--resume",str(cp),"--worker-output",str(out)],env=env,capture_output=True,text=True);res=torch.load(out,weights_only=False) if done.returncode==0 else None;ok=res is not None and torch.equal(continuous["params"],res["params"]) and state_equal(continuous["optimizer"],res["optimizer"]) and continuous["scaler"]==res["scaler"] and continuous["loss"]==res["loss"] and torch.equal(continuous["rng"],res["rng"]);results.append({"optimizer":optn,"scaling":scaling,"fresh_process_returncode":done.returncode,"bit_identical":ok,"passed":ok,"stderr":done.stderr})
 return {"protocols":results,"passed":all(x["passed"] for x in results)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path);ap.add_argument("--baseline",type=pathlib.Path);ap.add_argument("--resume",type=pathlib.Path);ap.add_argument("--worker-output",type=pathlib.Path);a=ap.parse_args()
 if a.resume:return resume_worker(a)
 baseline=json.loads(a.baseline.read_text());baseline_ok=baseline.get("decision")=="PROCEED_TO_3B1D" and baseline.get("functional_cases")==296 and baseline.get("passed_cases")==296
 specs=[];acts=(("ReLU","None"),("ReLU","ReLU"),("None","None"));widths=(16,32,64,128);layers=(1,2,4);batches=(16,128,1024)
 for oi,optn in enumerate(("SGD","Adam")):
  for i in range(12):specs.append((optn,*acts[i%3],layers[i%3],widths[i%4],batches[i%3],bool(i%2),widths[i%4],widths[(i+1)%4] if i in (10,11) else widths[i%4]))
 short=[short_case(x) for x in specs];reference=quantized_reference();acc=accumulation();static=static_scaling();dynamic=dynamic_test();streams=stream_training();longs=[long_run("SGD",(16,64,32,4,1024)),long_run("Adam",(32,128,64,4,1024)),long_run("Adam",(16,64,32,4,1024),True)];tmp=pathlib.Path("/tmp/phase3b1d_checkpoints");tmp.mkdir(exist_ok=True);checkpoints=checkpoint_tests(tmp)
 # Explicitly exercise both zero_grad contracts.
 zm=tcnn.Network(16,16,cfg(16,1));zo=optimizer("SGD",zm.params);zx,zt=data(0,16,16,16);((zm(zx).float()-zt).square().mean()).backward();zo.zero_grad(set_to_none=False);zero_false=zm.params.grad is not None and bool(torch.count_nonzero(zm.params.grad)==0);zo.zero_grad(set_to_none=True);zero_true=zm.params.grad is None
 doc={"marker":MARKER,"base_commit":BASE,"fresh_gpu_baseline":{"path":str(a.baseline),"passed":baseline_ok},"contract":{"python_parameter":"FP32","native_parameter":"FP16","python_gradient":"FP32","native_gradient":"FP16","optimizer_state":"FP32","zero_grad":{"set_to_none_false":zero_false,"set_to_none_true":zero_true}},"tolerances_frozen_before_final_run":TRAIN_TOL,"quantized_training_reference":reference,"short_training":{"count":len(short),"steps_each":50,"cases":short,"passed":all(x["passed"] for x in short)},"gradient_accumulation":acc,"static_loss_scaling":static,"dynamic_loss_scaling":dynamic,"stream_training":streams,"long_runs":{"count":len(longs),"total_steps":sum(x["steps"] for x in longs),"runs":longs,"passed":all(x["passed"] for x in longs)},"checkpoint_resume":checkpoints,"final_counters":counters()}
 doc["decision"]="PROCEED_TO_3B1E" if baseline_ok and reference["passed"] and doc["short_training"]["passed"] and acc["passed"] and static["passed"] and dynamic["passed"] and streams["passed"] and zero_false and zero_true and doc["long_runs"]["passed"] and checkpoints["passed"] else "PHASE3B1D_BLOCKED";a.output.write_text(json.dumps(doc,indent=2,default=lambda x:x.tolist() if torch.is_tensor(x) else str(x))+"\n");print(doc["decision"]);return 0 if doc["decision"]=="PROCEED_TO_3B1E" else 1
if __name__=="__main__":raise SystemExit(main())
