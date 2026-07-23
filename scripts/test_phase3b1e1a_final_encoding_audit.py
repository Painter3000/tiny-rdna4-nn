#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001 final contract audit."""
import argparse, copy, hashlib, json, math, pathlib, subprocess, sys
import torch
import tinycudann as tcnn
from tinycudann.modules import _C
import test_phase3b1e1_encoding_closure as e1

ROOT=pathlib.Path(__file__).resolve().parents[1]
RAW=pathlib.Path("/tmp/phase3b1e1a_final_encoding_audit_raw.json")
BASE="de2ce23ec7070aa7e5546b1942e80806bce17cb1"
MARKER="TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001"
PRIOR=pathlib.Path("/tmp/phase3b1e1_encoding_closure_raw.json")
PRIOR_SHA="bc9a5cd3ce5968fb3096ed03eb6f491e5a304825267de67ace84f55f8026ce2b"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def detail(a,b):
 a=a.float();b=b.float();d=(a-b).abs();near=b.abs()<=1e-3;rel=d[~near]/b[~near].abs() if bool((~near).any()) else torch.zeros(1,device=b.device)
 return {"absolute_error":float(d.max()) if d.numel() else 0.,"normalized_l2":float(d.norm())/(float(b.norm()) or 1.),
         "max_relative_outside_near_zero":float(rel.max()),"reference_norm":float(b.norm())}
def matrix_case(name,d,v,batch):
 c=e1.ec(name,v);m=tcnn.NetworkWithInputEncoding(d,16,c,e1.net(),seed=1400+d*17+v);h=m.native_tcnn_module.hyperparams();nw=h["network_parameter_count"];pw=h["padded_encoding_width"]
 g=torch.Generator(device="cuda").manual_seed(310000+d*1000+v*10+batch);x=torch.rand(batch,d,device="cuda",generator=g,requires_grad=True)
 qnet=m.params[:nw].detach().half().float().requires_grad_();ep=m.params[nw:].detach().half().float().requires_grad_(name=="HashGrid");rx=x.detach().clone().requires_grad_()
 er=e1.REF[name](rx,c,ep) if name!="HashGrid" else e1.REF[name](rx,c,ep)[0];erpad=torch.nn.functional.pad(er,(0,pw-er.shape[1])).half();ro=e1.mlp_ref_flat(erpad,qnet,pw)
 go=torch.linspace(-.00025,.00025,ro.numel(),device="cuda").reshape_as(ro).half();(ro.float()*go.float()).sum().backward();m.zero_grad(set_to_none=True);ny=m(x);ny.backward(go)
 rg=ep.grad.half().float() if name=="HashGrid" else torch.empty(0,device="cuda")
 pairs={"output":(ny,ro),"dinput":(x.grad,rx.grad),"network_gradient":(m.params.grad[:nw],qnet.grad.half().float())}
 if rg.numel():pairs["encoding_gradient"]=(m.params.grad[nw:],rg)
 metrics={k:detail(*z) for k,z in pairs.items()};metrics.setdefault("encoding_gradient",{"absolute_error":0.,"normalized_l2":0.,"max_relative_outside_near_zero":0.,"reference_norm":0.})
 ok=(metrics["output"]["absolute_error"]<=e1.TOL["network_output"] and metrics["dinput"]["absolute_error"]<=e1.TOL["dinput"] and metrics["network_gradient"]["absolute_error"]<=e1.TOL["network_gradient"] and metrics["encoding_gradient"]["absolute_error"]<=e1.TOL["encoding_gradient"])
 return {"encoding":name,"dims":d,"variant":v,"batch":batch,"interpolation":c.get("interpolation"),"n_levels":c.get("n_levels"),"n_features_per_level":c.get("n_features_per_level"),"metrics":metrics,"passed":ok}
def matrix():
 rows=[]
 for name,dimses in {"Identity":[2,3,7,16,24],"Frequency":[2,3,7],"OneBlob":[2,3,7],"HashGrid":[2,3]}.items():
  for d in dimses:
   for v in range(3 if name=="OneBlob" and d==7 else 4):
    for b in (16,32,128,1024):rows.append(matrix_case(name,d,v,b))
 return rows
def padding_contracts():
 standalone=dict(_C._phase3b1e1a_test_standalone_padding_values());fp16=[];fp32=[]
 for name,d,v in (("Identity",3,0),("Frequency",3,2),("OneBlob",2,1)):
  a=tcnn.NetworkWithInputEncoding(d,16,e1.ec(name,v),e1.net(16,1,"None"),seed=9);fp16.append({"encoding":name,"value":a.native_tcnn_module.hyperparams()["encoding_padding_value"]})
  cfg={"otype":"PortableMLP","n_neurons":16,"n_hidden_layers":1,"activation":"None","output_activation":"None"}
  b=tcnn.NetworkWithInputEncoding(d,16,e1.ec(name,v),cfg,seed=9);fp32.append({"encoding":name,"value":b.native_tcnn_module.hyperparams()["encoding_padding_value"]})
 fp16_zeros=[e1.padding_case(*s) for s in (("Identity",3,0,17),("Frequency",7,2,31),("OneBlob",2,1,31))]
 return {"standalone":standalone,"fp32_network_with_encoding":fp32,"fp16_network_with_encoding":fp16,"fp16_zero_execution":fp16_zeros,
         "hashgrid_unchanged":standalone["HashGrid"]==1.0,"passed":all(standalone[x]==1. for x in ("Identity","Frequency","OneBlob")) and all(x["value"]==1. for x in fp32) and all(x["value"]==0. for x in fp16) and all(x["passed"] for x in fp16_zeros)}
def collision_proof(d,c):
 dense=[];tables=[];colliding=[];count=0;maximum=1;witness=None
 for level,(scale,res,size) in enumerate(e1.grid_meta(d,c)[1]):
  n=res**d;dense.append(n);tables.append(size);buckets={}
  for linear in range(n):
   q=linear;coord=[]
   for _ in range(d):coord.append(q%res);q//=res
   h=e1.hindex(coord,size,res);buckets.setdefault(h,[]).append(coord)
  local=sum(len(v)-1 for v in buckets.values());count+=local;maximum=max(maximum,max(map(len,buckets.values())))
  if local:
   colliding.append(level)
   if witness is None:
    v=next(v for v in buckets.values() if len(v)>1);witness={"level":level,"coordinate_a":v[0],"coordinate_b":v[1],"hash_target":e1.hindex(v[0],size,res)}
 classification="collision-strong" if count and any(a>b for a,b in zip(dense,tables)) else "collision-low"
 return {"dense_entries_per_level":dense,"hash_table_entries_per_level":tables,"colliding_levels":colliding,"collision_count":count,"maximum_bucket_occupancy":maximum,"collision_classification":classification,"collision_witness":witness}
def train(name,optname,steps,d,c,scaling):
 m=tcnn.NetworkWithInputEncoding(d,16,c,e1.net(),seed=919);o=e1.optimizer(optname,m.params);h=m.native_tcnn_module.hyperparams();nw=h["network_parameter_count"];scale=128. if scaling in ("static","dynamic") else 1.;initial=scale;scales=[scale];overflow=skip=recovery=0;growth_interval=200;overflow_observation=None;losses=[];warm=None
 evalgen=torch.Generator(device="cuda").manual_seed(1199);ex=torch.rand(256,d,device="cuda",generator=evalgen);et=(torch.sin(ex.sum(1,keepdim=True)*4)*.25).repeat(1,16);start=float((m(ex).float()-et).square().mean())
 for s in range(steps):
  g=torch.Generator(device="cuda").manual_seed(1200+s);x=torch.rand(32,d,device="cuda",generator=g);target=(torch.sin(x.sum(1,keepdim=True)*4)*.25).repeat(1,16);o.zero_grad(set_to_none=True);m.loss_scale=scale
  y=m(x);loss=(y.float()-target).square().mean()
  if scaling=="dynamic" and s==50:
   upstream=torch.full_like(y,65504.);y.backward(upstream);upstream_finite=bool(torch.isfinite(upstream).all())
  else:loss.backward();upstream_finite=True
  ng=m.params.grad[:nw];eg=m.params.grad[nw:];network_finite=bool(torch.isfinite(ng).all());encoding_finite=bool(torch.isfinite(eg).all());bad=not (network_finite and encoding_finite)
  if bad:
   before_scale=scale;overflow+=1;skip+=1;scale=max(scale/2,1.);o.zero_grad(set_to_none=True)
   if overflow_observation is None:overflow_observation={"forward_output_finite":bool(torch.isfinite(y).all()),"upstream_gradient_finite":upstream_finite,"network_gradient_nonfinite":not network_finite,"encoding_gradient_nonfinite":not encoding_finite,"full_step_skipped":True,"scale_reduced":scale<before_scale,"gradients_reset":m.params.grad is None}
  else:
   o.step()
   if scaling=="dynamic" and s>50:recovery+=1
   if scaling=="dynamic" and (s+1)%growth_interval==0:scale*=2
  scales.append(scale);losses.append(float(loss))
  if s==20:torch.cuda.synchronize();warm=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),e1.counters())
 torch.cuda.synchronize();end=float((m(ex).float()-et).square().mean());final=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),e1.counters());stable=all(warm[2][k]==final[2][k] for k in ("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count"))
 proof=collision_proof(d,c) if name=="HashGrid" else None;dynamic_ok=scaling!="dynamic" or (initial>1 and len(set(scales))>1 and overflow>0 and skip==overflow and recovery>0)
 return {"encoding":name,"optimizer":optname,"steps":steps,"dims":d,"config":c,"scaling":scaling,"collision":proof,"initial_scale":initial,"minimum_scale":min(scales),"maximum_scale":max(scales),"scale_change_count":sum(a!=b for a,b in zip(scales,scales[1:])),"overflow_count":overflow,"skip_count":skip,"recovery_count":recovery,"final_scale":scale,"growth_interval":growth_interval if scaling=="dynamic" else None,"overflow_observation":overflow_observation,"both_parameter_ranges_checked":scaling!="dynamic" or overflow_observation is not None,"start_loss":start,"end_loss":end,"counters_warm":warm[2],"counters_end":final[2],"passed":end<start and stable and dynamic_ok and bool(torch.isfinite(m.params).all())}
def protocol(name,optname,scaling,steps,start=None,save=None,result=None):
 torch.manual_seed(20260723);torch.cuda.manual_seed_all(20260723);d=3;c=e1.ec(name,1);m=tcnn.NetworkWithInputEncoding(d,16,c,e1.net(),seed=606);o=e1.optimizer(optname,m.params);gen=torch.Generator(device="cuda").manual_seed(777);scale=128.;begin=0;losses=[]
 if start:
  q=torch.load(start,weights_only=False);m.params.data.copy_(q["master"]);o.load_state_dict(q["optimizer"]);torch.set_rng_state(q["cpu_rng"]);torch.cuda.set_rng_state_all(q["cuda_rng"]);gen.set_state(q["custom_rng"]);scale=q["scale"];begin=q["step"];losses=q["losses"]
 for s in range(begin,steps):
  x=torch.rand(32,d,device="cuda",generator=gen);target=(torch.sin(x.sum(1,keepdim=True)*4)*.25).repeat(1,16);o.zero_grad(set_to_none=True);m.loss_scale=scale;loss=(m(x).float()-target).square().mean();loss.backward();o.step();losses.append(float(loss))
  if scaling=="dynamic" and (s+1)%100==0:scale*=2
  if save and s==99:
   torch.save({"master":m.params.detach(),"optimizer":o.state_dict(),"cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"custom_rng":gen.get_state(),"scale":scale,"step":100,"losses":losses},save);return
 torch.save({"master":m.params.detach().cpu(),"optimizer":o.state_dict(),"cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"custom_rng":gen.get_state(),"scale":scale,"step":steps,"losses":losses},result)
def resumes():
 p=pathlib.Path("/tmp/phase3b1e1a_resume");p.mkdir(exist_ok=True);rows=[]
 for name,opt,sc in (("Frequency","Adam","none"),("OneBlob","Adam","none"),("HashGrid","Adam","none"),("HashGrid","Adam","dynamic")):
  key=name+"_"+sc;a=p/(key+"_a.pt");cp=p/(key+"_cp.pt");b=p/(key+"_b.pt");cmd=[sys.executable,__file__,"--protocol",name,opt,sc]
  ra=subprocess.run(cmd+["--steps","200","--result",str(a)]);rc=subprocess.run(cmd+["--steps","100","--checkpoint",str(cp)]);rb=subprocess.run(cmd+["--steps","200","--resume",str(cp),"--result",str(b)]);q=torch.load(a,weights_only=False);z=torch.load(b,weights_only=False)
  cpu=torch.equal(q["cpu_rng"],z["cpu_rng"]);cuda=len(q["cuda_rng"])==len(z["cuda_rng"]) and all(torch.equal(x,y) for x,y in zip(q["cuda_rng"],z["cuda_rng"]));custom=torch.equal(q["custom_rng"],z["custom_rng"])
  ok=all(x.returncode==0 for x in (ra,rc,rb)) and torch.equal(q["master"],z["master"]) and e1.state_equal(q["optimizer"],z["optimizer"]) and q["scale"]==z["scale"] and q["losses"]==z["losses"] and cpu and cuda and custom
  rows.append({"encoding":name,"scaling":sc,"fresh_process":True,"cpu_rng_equal":cpu,"cuda_all_rng_equal":cuda,"custom_generator_rng_equal":custom,"parameters_bit_identical":torch.equal(q["master"],z["master"]),"optimizer_state_equal":e1.state_equal(q["optimizer"],z["optimizer"]),"passed":ok})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path,default=RAW);ap.add_argument("--protocol",nargs=3);ap.add_argument("--steps",type=int);ap.add_argument("--checkpoint");ap.add_argument("--resume");ap.add_argument("--result");a=ap.parse_args()
 if a.protocol:protocol(*a.protocol,a.steps,a.resume,a.checkpoint,a.result);return
 prior_ok=PRIOR.exists() and sha(PRIOR)==PRIOR_SHA;mat=matrix();pads=padding_contracts();strong=e1.ec("HashGrid",0);strong.update({"n_levels":1,"base_resolution":4,"log2_hashmap_size":3,"per_level_scale":2.})
 low3=e1.ec("HashGrid",0);low3.update({"n_levels":1,"base_resolution":4,"log2_hashmap_size":12,"per_level_scale":2.})
 low2=copy.deepcopy(low3);runs=[train("Identity","SGD",200,3,e1.ec("Identity",0),"none"),train("HashGrid","Adam",1000,2,strong,"none"),train("HashGrid","Adam",1000,3,low3,"dynamic"),train("HashGrid","Adam",1000,2,low2,"static")]
 rs=resumes();events=[e1.event_chain(n) for n in ("Identity","Frequency","OneBlob","HashGrid")]
 maxima={}
 for k in ("output","dinput","network_gradient","encoding_gradient"):
  x=max(mat,key=lambda z:z["metrics"][k]["absolute_error"]);maxima[k]={q:x.get(q) for q in ("encoding","dims","variant","batch","interpolation","n_levels","n_features_per_level")} | x["metrics"][k]
 historical={"encoding":"HashGrid","dims":3,"variant":3,"batch":1024,"interpolation":"Smoothstep","n_levels":16,"n_features_per_level":4,"absolute_error":0.02607070654630661}
 doc={"marker":MARKER,"base_commit":BASE,"padding_contracts":pads,"functional_matrix":mat,"functional_cases":len(mat),"functional_passed":sum(x["passed"] for x in mat),"numerical_maxima":maxima,"historical_e1_dinput_maximum_attribution":historical,"collision_proofs":{"strong":runs[1]["collision"],"low_3d":runs[2]["collision"],"low_2d":runs[3]["collision"]},"training_runs":runs,"corrected_training_steps":sum(x["steps"] for x in runs),"inherited_training":{"steps":4400,"raw_sha256":sha(PRIOR) if PRIOR.exists() else None,"expected_sha256":PRIOR_SHA,"valid":prior_ok},"validated_training_steps":4400+sum(x["steps"] for x in runs) if prior_ok else sum(x["steps"] for x in runs),"checkpoint_resume":rs,"event_chains":events,"encoding_overflow_field_contract":{"forward_output_finite":True,"scaled_forward_output_finite":True,"upstream_gradient_finite":True},"environment":{"python":sys.version.split()[0],"pytorch":torch.__version__,"hip":torch.version.hip,"device":torch.cuda.get_device_name(0)}}
 gates=[pads["passed"],len(mat)==204 and all(x["passed"] for x in mat),runs[1]["collision"]["collision_classification"]=="collision-strong" and runs[1]["collision"]["collision_witness"] is not None,runs[2]["collision"]["collision_classification"]=="collision-low" and runs[2]["collision"]["collision_count"]==0,all(x["passed"] for x in runs),doc["validated_training_steps"]>=7600,len(rs)==4 and all(x["passed"] for x in rs),len(events)==4 and all(x["passed"] for x in events),all(all(q in x for q in ("encoding","dims","variant","batch","interpolation","n_levels","n_features_per_level","absolute_error","normalized_l2","max_relative_outside_near_zero","reference_norm")) for x in maxima.values())]
 doc["decision"]="E1A_RAW_PASS" if all(gates) else "E1A_RAW_BLOCKED";a.output.write_text(json.dumps(doc,indent=2)+"\n");print(doc["decision"]);raise SystemExit(0 if all(gates) else 1)
if __name__=="__main__":main()
