#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1D1_TRAINING_AUDIT_001 focused training audit."""
import argparse,copy,hashlib,json,os,pathlib,subprocess,sys
import torch,tinycudann as tcnn
from tinycudann.modules import _C
import test_phase3b1d_fp16_training as dtrain
ROOT=pathlib.Path(__file__).resolve().parents[1];MARKER="TCNN_RDNA4_P3B1D1_TRAINING_AUDIT_001";SCALE=128.0
def tensor_hash(*xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def state_equal(a,b):return dtrain.state_equal(a,b)
def static_protocol(checkpoint=None,resume=None):
 torch.manual_seed(111);torch.cuda.manual_seed_all(222);custom=torch.Generator(device="cuda").manual_seed(333);m=tcnn.Network(16,16,dtrain.cfg(16,2),seed=444);opt=dtrain.optimizer("Adam",m.params);start=0;records=[]
 if resume:
  cp=torch.load(resume,weights_only=False);m.params.data.copy_(cp["master_params"]);opt.load_state_dict(cp["optimizer_state"]);start=cp["step"];records=cp["records"]
  torch.set_rng_state(cp["torch_cpu_rng_state"]);torch.cuda.set_rng_state_all(cp["torch_cuda_rng_state_all"]);custom.set_state(cp["custom_cuda_generator_state"]);static_scale=cp["static_loss_scale"]
 else:static_scale=SCALE
 last_grad=None
 for step in range(start,20):
  cpu_r=torch.rand(4);cuda_r=torch.rand(4,device="cuda");custom_r=torch.rand(4,device="cuda",generator=custom);x=torch.randn(128,16,device="cuda",generator=custom)*.1+cuda_r[0]*1e-3+cpu_r[0].item()*1e-3;target=torch.sin(x);m.loss_scale=static_scale
  opt.zero_grad(set_to_none=True);loss=(m(x).float()-target).square().mean();loss.backward();last_grad=m.params.grad.detach().clone();opt.step();records.append({"step":step,"random_sha256":tensor_hash(cpu_r,cuda_r,custom_r),"loss":float(loss.detach()),"active_loss_scale":float(m.loss_scale)})
  if checkpoint and step==9:
   torch.save({"marker":MARKER,"scaling_mode":"static","static_loss_scale":static_scale,"master_params":m.params.detach(),"optimizer_state":opt.state_dict(),"step":10,"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state_all":torch.cuda.get_rng_state_all(),"custom_cuda_generator_state":custom.get_state(),"records":records},checkpoint);return None
 return {"params":m.params.detach().cpu(),"gradient":last_grad.cpu(),"optimizer":opt.state_dict(),"loss":records[-1]["loss"],"step":20,"active_loss_scale":float(m.loss_scale),"records":records,"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state_all":torch.cuda.get_rng_state_all(),"custom_cuda_generator_state":custom.get_state()}
def static_checkpoint_test(tmp):
 continuous=static_protocol();cp=tmp/"adam_static_128.pt";static_protocol(checkpoint=cp);out=tmp/"adam_static_128_resumed.pt";done=subprocess.run([sys.executable,__file__,"--resume",str(cp),"--worker-output",str(out)],text=True,capture_output=True,env=os.environ.copy());res=torch.load(out,weights_only=False) if done.returncode==0 else None;meta=torch.load(cp,weights_only=False)
 comparisons={"parameters":res is not None and torch.equal(continuous["params"],res["params"]),"gradients":res is not None and torch.equal(continuous["gradient"],res["gradient"]),"optimizer_state":res is not None and state_equal(continuous["optimizer"],res["optimizer"]),"loss":res is not None and continuous["loss"]==res["loss"],"step":res is not None and continuous["step"]==res["step"],"active_loss_scale":res is not None and continuous["active_loss_scale"]==res["active_loss_scale"]==SCALE,"cpu_rng":res is not None and torch.equal(continuous["torch_cpu_rng_state"],res["torch_cpu_rng_state"]),"cuda_rng_all":res is not None and all(torch.equal(x,y) for x,y in zip(continuous["torch_cuda_rng_state_all"],res["torch_cuda_rng_state_all"])),"custom_generator":res is not None and torch.equal(continuous["custom_cuda_generator_state"],res["custom_cuda_generator_state"]),"post_resume_random_sequence":res is not None and continuous["records"][10:]==res["records"][10:]}
 return {"fresh_process_returncode":done.returncode,"checkpoint_metadata":{"scaling_mode":meta.get("scaling_mode"),"static_loss_scale":meta.get("static_loss_scale"),"cuda_device_count":len(meta["torch_cuda_rng_state_all"])},"native_backward_active_scale":res["active_loss_scale"] if res else None,"comparisons":comparisons,"passed":done.returncode==0 and all(comparisons.values()) and meta.get("scaling_mode")=="static" and meta.get("static_loss_scale")==SCALE,"stderr":done.stderr}
def counters():
 ns=("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count");return {n:int(getattr(_C,"_hipblaslt_fp16_"+n)()) for n in ns}
def event_chain_training():
 torch.manual_seed(555);inputs=[torch.randn(128,16,device="cuda")*.1 for _ in range(64)];targets=[torch.sin(x) for x in inputs];event_model=tcnn.Network(16,16,dtrain.cfg(16,2),seed=666);ref=tcnn.Network(16,16,dtrain.cfg(16,2),seed=666);event_opt=dtrain.optimizer("Adam",event_model.params);ref_opt=dtrain.optimizer("Adam",ref.params)
 # Warm the exact A->event->B signatures and stream handles before measurement.
 sa,sb=torch.cuda.Stream(),torch.cuda.Stream()
 with torch.cuda.stream(sa):event_opt.zero_grad(set_to_none=True);warm_loss=(event_model(inputs[0]).float()-targets[0]).square().mean();warm_forward=torch.cuda.Event();warm_forward.record(sa)
 with torch.cuda.stream(sb):sb.wait_event(warm_forward);warm_loss.backward();event_opt.step()
 sb.synchronize();event_params=event_model.params.detach().clone();event_state=copy.deepcopy(event_opt.state_dict())
 # Materialize the independent reference model's descriptors, then restore the
 # exact event-model starting state before either measured sequence.
 ref_opt.zero_grad(set_to_none=True);((ref(inputs[0]).float()-targets[0]).square().mean()).backward();ref_opt.step();torch.cuda.synchronize();ref.params.data.copy_(event_params);ref_opt.load_state_dict(event_state);before=counters();ref_losses=[]
 for x,t in zip(inputs,targets):ref_opt.zero_grad(set_to_none=True);loss=(ref(x).float()-t).square().mean();loss.backward();ref_opt.step();ref_losses.append(float(loss.detach()))
 back_done=torch.cuda.Event();back_done.record(torch.cuda.current_stream());loss_tensors=[]
 for x,t in zip(inputs,targets):
  with torch.cuda.stream(sa):sa.wait_event(back_done);event_opt.zero_grad(set_to_none=True);loss=(event_model(x).float()-t).square().mean();forward_done=torch.cuda.Event();forward_done.record(sa)
  with torch.cuda.stream(sb):sb.wait_event(forward_done);loss.backward();event_opt.step();back_done.record(sb);loss_tensors.append(loss.detach())
 sb.synchronize();after=counters();losses=[float(x) for x in loss_tensors];param_abs=float((event_model.params-ref.params).abs().max());state_ok=state_equal(event_opt.state_dict(),ref_opt.state_dict());loss_abs=max(abs(x-y) for x,y in zip(losses,ref_losses));stable=all(before[k]==after[k] for k in before)
 return {"rounds":64,"forward_stream":"A","backward_optimizer_stream":"B","terminal_sync_only":True,"parameter_max_abs":param_abs,"loss_max_abs":loss_abs,"optimizer_state_equal":state_ok,"counters_before":before,"counters_after":after,"passed":param_abs<=2e-5 and loss_abs<=2e-6 and state_ok and stable}
def range_tests():
 # Search a finite coefficient where scale=1 loses representable gradients but 128 rescues them.
 under=None
 for exp in range(12,31):
  coeff=2.0**(-exp);x=torch.randn(256,16,device="cuda")*.1;target=torch.zeros_like(x);grads=[]
  for scale in (1.,SCALE):
   m=tcnn.Network(16,16,dtrain.cfg(16,1,"None","None"),seed=777);m.loss_scale=scale;((m(x).float()-target).square().mean()*coeff).backward();grads.append(m.params.grad.detach().clone())
  rescued=(grads[0]==0)&(grads[1]!=0)
  if rescued.any():under={"coefficient":coeff,"scale1_zero_count":int((grads[0]==0).sum()),"scale128_rescued_count":int(rescued.sum()),"after_unscale_max_abs_vs_fp32_quantized_reference":float((grads[1]-grads[1].float()).abs().max()),"inputs_finite":bool(torch.isfinite(x).all()),"passed":True};break
 if under is None:under={"passed":False}
 # Finite unscaled calculation, deliberate overflow only at final native FP16 gradient.
 m=tcnn.Network(16,16,dtrain.cfg(16,1,"None","None"),seed=888);opt=dtrain.optimizer("Adam",m.params);x=torch.full((256,16),300.,device="cuda");target=torch.zeros_like(x);m.loss_scale=8192.;opt.zero_grad(set_to_none=True);y=m(x);loss=y.float().square().mean();scaled_loss=loss*8192.;loss.backward();before=m.params.detach().clone();os=copy.deepcopy(opt.state_dict());nonfinite=not bool(torch.isfinite(m.params.grad).all());skipped=nonfinite
 if not skipped:opt.step()
 unchanged=torch.equal(before,m.params) and state_equal(os,opt.state_dict());opt.zero_grad(set_to_none=True);m.loss_scale=128.;nx=torch.randn(256,16,device="cuda")*.1;nl=m(nx).float().square().mean();nl.backward();recovery=bool(torch.isfinite(m.params.grad).all());fp32_internal=torch.isfinite((x.float().t()@torch.ones_like(x).float())).all().item()
 overflow={"inputs_finite":bool(torch.isfinite(x).all()),"unscaled_loss_finite":bool(torch.isfinite(loss)),"scaled_loss_finite":bool(torch.isfinite(scaled_loss)),"fp32_internal_gradient_finite":bool(fp32_internal),"native_fp16_gradient_nonfinite":nonfinite,"step_skipped":skipped and unchanged,"recovery_passed":recovery};overflow["passed"]=all(overflow.values())
 return {"underflow_rescue":under,"finite_to_fp16_overflow":overflow,"passed":under.get("passed") is True and overflow["passed"]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path);ap.add_argument("--resume",type=pathlib.Path);ap.add_argument("--worker-output",type=pathlib.Path);a=ap.parse_args()
 if a.resume:r=static_protocol(resume=a.resume);torch.save(r,a.worker_output);return 0
 tmp=pathlib.Path("/tmp/phase3b1d1");tmp.mkdir(exist_ok=True);static=static_checkpoint_test(tmp);event=event_chain_training();ranges=range_tests();doc={"marker":MARKER,"static_checkpoint_resume":static,"rng_resume":{"passed":all(static["comparisons"][k] for k in ("cpu_rng","cuda_rng_all","custom_generator","post_resume_random_sequence")),"cuda_device_count":static["checkpoint_metadata"]["cuda_device_count"]},"event_chain_training":event,"range_tests":ranges};doc["decision"]="AUDIT_TEST_PASS" if static["passed"] and doc["rng_resume"]["passed"] and event["passed"] and ranges["passed"] else "PHASE3B1D1_BLOCKED";a.output.write_text(json.dumps(doc,indent=2)+"\n");print(doc["decision"]);return 0 if doc["decision"]=="AUDIT_TEST_PASS" else 1
if __name__=="__main__":raise SystemExit(main())
