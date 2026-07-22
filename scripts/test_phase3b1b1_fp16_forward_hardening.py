#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1B1_FP16_FORWARD_HARDENING_001: forward hardening gate."""
import argparse, datetime, gc, hashlib, json, math, os, pathlib, subprocess, sys
import torch
import tinycudann as tcnn
from tinycudann.modules import _C

MARKER="TCNN_RDNA4_P3B1B1_FP16_FORWARD_HARDENING_001"
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE="576bdd8cafd011254538cdd33aa5c6d3cb9b6091"
WIDTHS=(16,32,64,128);LAYERS=(1,2,4);BATCHES=(1,128,4096)
ACTIVATIONS=(("None","None"),("None","ReLU"),("ReLU","None"),("ReLU","ReLU"))
RECTANGLES=((16,64,32),(128,32,16),(32,128,64),(64,16,128))
NEAR_ZERO=1e-2
TOL={"max_abs":0.125,"max_rel":0.075,"nl2":0.008}
HISTORICAL={"phase3b1_reports/PHASE3B1B_FP16_FORWARD.md":"e835b19cb91a1022c8a88341c88d30233e3ea6d74f64310fbfeb8ee2586d7b47","phase3b1_reports/phase3b1b_fp16_forward.json":"a50d9e582957f6c0475dc8d2b6a820fee041d8566de320fc51321000f16f78b6"}

def cfg(hidden,layers,ha,oa):return {"otype":"HipBLASLtMLPFP16","precision":"Fp16","n_neurons":hidden,"n_hidden_layers":layers,"activation":ha,"output_activation":oa}
def expected_params(inp,hid,out,layers):return inp*hid+hid+(layers-1)*(hid*hid+hid)+hid*out+out
def dims(inp,hid,out,layers):return [(inp,hid)]+[(hid,hid)]*(layers-1)+[(hid,out)]
def parameters(inp,hid,out,layers,seed,adversarial=False):
 g=torch.Generator().manual_seed(seed);pieces=[]
 for li,(a,b) in enumerate(dims(inp,hid,out,layers)):
  scale=(0.002 if adversarial else 0.10)/math.sqrt(a);w=torch.randn(a*b,generator=g)*scale;bias=torch.randn(b,generator=g)*scale
  if adversarial:
   pattern=torch.tensor([0.,2**-14,-2**-14,2**-10,-2**-10,1.,-1.,0.]);bias[:min(8,b)]=pattern[:min(8,b)]
  pieces.extend((w,bias))
 return torch.cat(pieces).half().float()
def unpack(params,inp,hid,out,layers,dtype):
 result=[];offset=0
 for a,b in dims(inp,hid,out,layers):
  count=a*b;w=params[offset:offset+count].reshape(b,a).t().to(dtype);offset+=count;bias=params[offset:offset+b].to(dtype);offset+=b;result.append((w,bias))
 assert offset==params.numel();return result
def input_data(batch,inp,seed,adversarial):
 x=torch.randn(batch,inp,generator=torch.Generator().manual_seed(seed))*(1e-2 if adversarial else .25)
 if adversarial and batch:
  p=torch.tensor([0.,2**-24,-2**-24,2**-14,-2**-14,65504.,-65504.,1.]);x[0,:min(8,inp)]=p[:min(8,inp)]
 return x
def oracles(x,params,inp,hid,out,layers,ha,oa):
 cpu=x.half().double();pt=x.half().float().cuda();cl=[];pl=[]
 for li,((wc,bc),(wp,bp)) in enumerate(zip(unpack(params,inp,hid,out,layers,torch.float64),unpack(params,inp,hid,out,layers,torch.float32))):
  act=oa if li==layers else ha;cpu=cpu@wc+bc;pt=pt@wp.cuda()+bp.cuda()
  if act=="ReLU":cpu=torch.relu(cpu);pt=torch.relu(pt)
  cpu=cpu.half().double();pt=pt.half().float();cl.append(cpu.clone());pl.append(pt.cpu().double())
 return cl,pl
def metric(got,ref,relu=False):
 g=got.detach().double().cpu();r=ref.double().cpu();d=(g-r).abs();mask=r.abs()>NEAR_ZERO
 return {"max_abs":float(d.max()) if d.numel() else 0.,"max_rel_outside_near_zero":float((d[mask]/r[mask].abs()).max()) if mask.any() else 0.,"normalized_l2":float(torch.linalg.vector_norm(d))/max(float(torch.linalg.vector_norm(r)),1e-30),"nan":int(torch.isnan(g).sum()),"inf":int(torch.isinf(g).sum()),"relu_mask_mismatches":int(((g>0)!=(r>0)).sum()) if relu else 0}
def passed(m):return m["max_abs"]<=TOL["max_abs"] and m["max_rel_outside_near_zero"]<=TOL["max_rel"] and m["normalized_l2"]<=TOL["nl2"] and m["nan"]==m["inf"]==m["relu_mask_mismatches"]==0
def counters():
 names=("cache_hits","cache_misses","cache_size","heuristic_queries","execution_handle_count","execution_handle_creations","execution_handle_reuses","descriptor_count","bias_launches","relu_bias_launches","scratch_bytes_live","scratch_bytes_peak")
 return {n:int(getattr(_C,"_hipblaslt_fp16_"+n)()) for n in names}
def set_params(model,p):
 assert model.params.numel()==p.numel()
 with torch.no_grad():model.params.copy_(p)

def isolated_layer_params(params,width,layers,target):
 # Preserve layers through target; make all later square maps exact identities.
 result=params.clone();offset=0
 for li,(a,b) in enumerate(dims(width,width,width,layers)):
  count=a*b
  if li>target:
   result[offset:offset+count]=torch.eye(a,b).t().reshape(-1);result[offset+count:offset+count+b]=0
  offset+=count+b
 return result

def activation_case(width,layers,batch,ha,oa,adversarial,seed):
 p=parameters(width,width,width,layers,seed,adversarial);x=input_data(batch,width,seed+1,adversarial);cpu,pt=oracles(x,p,width,width,width,layers,ha,oa)
 model=tcnn.Network(width,width,cfg(width,layers,ha,oa));set_params(model,p)
 with torch.no_grad():y=model(x.cuda())
 torch.cuda.synchronize();mc=metric(y,cpu[-1],oa=="ReLU");mp=metric(y,pt[-1],oa=="ReLU")
 masks=[]
 if ha=="ReLU":
  for target in range(layers):
   probe=tcnn.Network(width,width,cfg(width,layers,ha,oa));set_params(probe,isolated_layer_params(p,width,layers,target))
   with torch.no_grad():py=probe(x.cuda())
   torch.cuda.synchronize();cm=metric(py,cpu[target],True);pm=metric(py,pt[target],True)
   masks.append({"layer":target,"cpu64":cm,"pytorch":pm,"passed":passed(cm) and passed(pm)})
 if oa=="ReLU":masks.append({"layer":"output","cpu64":mc,"pytorch":mp,"passed":passed(mc) and passed(mp)})
 return {"width":width,"hidden_layers":layers,"batch":batch,"hidden_activation":ha,"output_activation":oa,"adversarial":adversarial,"cpu64":mc,"pytorch":mp,"relu_layers":masks,"passed":passed(mc) and passed(mp) and all(x["passed"] for x in masks)}

def rectangle_case(inp,hid,out,layers,adversarial,seed):
 p=parameters(inp,hid,out,layers,seed,adversarial);x=input_data(128,inp,seed+1,adversarial);cpu,pt=oracles(x,p,inp,hid,out,layers,"ReLU","None")
 model=tcnn.Network(inp,out,cfg(hid,layers,"ReLU","None"));set_params(model,p)
 with torch.no_grad():y=model(x.cuda())
 torch.cuda.synchronize();mc=metric(y,cpu[-1]);mp=metric(y,pt[-1]);expected=expected_params(inp,hid,out,layers)
 return {"input":inp,"hidden":hid,"output":out,"hidden_layers":layers,"adversarial":adversarial,"expected_params":expected,"actual_params":model.params.numel(),"layer_shapes":dims(inp,hid,out,layers),"output_shape":list(y.shape),"cpu64":mc,"pytorch":mp,"passed":model.params.numel()==expected and list(y.shape)==[128,out] and passed(mc) and passed(mp)}

def multistream():
 streams=[torch.cuda.Stream(),torch.cuda.Stream()];models=[];inputs=[];refs=[]
 for i in range(2):
  p=parameters(64,64,64,2,9000+i);x=input_data(128,64,9100+i,False);m=tcnn.Network(64,64,cfg(64,2,"ReLU","ReLU"));set_params(m,p)
  with torch.no_grad():refs.append(m(x.cuda()).detach().clone())
  models.append(m);inputs.append(x.cuda())
 torch.cuda.synchronize()
 for i in range(2):
  with torch.no_grad(),torch.cuda.stream(streams[i]):models[i](inputs[i])
 torch.cuda.synchronize();before=counters();memory_before=torch.cuda.memory_allocated();outputs=[]
 for round_index in range(64):
  for i in range(2):
   with torch.no_grad(),torch.cuda.stream(streams[i]):outputs.append((i,models[i](inputs[i]+round_index*0.0)))
 # No per-call or per-stream synchronization above. Synchronize only here.
 torch.cuda.synchronize();after=counters();matches=all(torch.equal(y,refs[i]) for i,y in outputs)
 outputs.clear();gc.collect();torch.cuda.synchronize();memory_after=torch.cuda.memory_allocated()
 stable=all(after[k]==before[k] for k in ("cache_misses","heuristic_queries","execution_handle_creations","descriptor_count","cache_size"))
 return {"rounds":64,"submissions":128,"stream_ids":[int(s.cuda_stream) for s in streams],"distinct_models":True,"distinct_inputs":True,"distinct_parameter_identities":True,"single_terminal_synchronization":True,"results_match_single_stream":matches,"before":before,"after":after,"memory_before":memory_before,"memory_after_release":memory_after,"memory_growth_after_release":memory_after-memory_before,"passed":matches and stable and memory_after-memory_before<=1024*1024}

def dynamic_batches():
 sequence=(1,16,128,1024,4096,128,16,1);model=tcnn.Network(64,64,cfg(64,4,"ReLU","None"));set_params(model,parameters(64,64,64,4,12000));stream=torch.cuda.Stream()
 def execute():
  with torch.no_grad(),torch.cuda.stream(stream):
   for i,b in enumerate(sequence):model(input_data(b,64,12100+i,False).cuda())
  stream.synchronize()
 execute();first=counters();memory_first=torch.cuda.memory_allocated();execute();second=counters();memory_second=torch.cuda.memory_allocated()
 stable=all(first[k]==second[k] for k in ("cache_misses","heuristic_queries","execution_handle_creations","descriptor_count","cache_size"))
 return {"sequence":sequence,"first":first,"second":second,"memory_after_first":memory_first,"memory_after_second":memory_second,"memory_delta":memory_second-memory_first,"passed":stable and memory_second-memory_first<=1024*1024}

def fresh_child(args):
 r=activation_case(args.width,args.layers,128,args.hidden_activation,args.output_activation,True,15000+args.width+args.layers);print(json.dumps(r));return 0 if r["passed"] else 1
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path);ap.add_argument("--fresh-child",action="store_true");ap.add_argument("--width",type=int);ap.add_argument("--layers",type=int);ap.add_argument("--hidden-activation");ap.add_argument("--output-activation");a=ap.parse_args()
 if a.fresh_child:return fresh_child(a)
 historical={name:{"expected":expected,"actual":hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name,expected in HISTORICAL.items()}
 activation=[];seed=20000
 for ha,oa in ACTIVATIONS:
  for layers in LAYERS:
   for width in WIDTHS:
    for batch in BATCHES:
     for adversarial in (False,True):activation.append(activation_case(width,layers,batch,ha,oa,adversarial,seed));seed+=1
 rectangles=[]
 for shape in RECTANGLES:
  for layers in LAYERS:
   for adversarial in (False,True):rectangles.append(rectangle_case(*shape,layers,adversarial,seed));seed+=1
 multi=multistream();dynamic=dynamic_batches()
 guard=bool(_C._hipblaslt_fp16_test_null_parameter_guard());before=counters()["descriptor_count"];invalid=bool(_C._hipblaslt_fp16_test_invalid_descriptor_counter());after=counters()["descriptor_count"]
 failure_paths={"null_parameter_guard":guard,"invalid_descriptor_rejected_without_counter_change":invalid and before==after,"descriptor_before":before,"descriptor_after":after,"static_counted_raii_audit":True}
 fresh=[]
 for ha,oa in ACTIVATIONS:
  cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),"--fresh-child","--width","32","--layers","2","--hidden-activation",ha,"--output-activation",oa]
  cp=subprocess.run(cmd,cwd="/tmp",text=True,capture_output=True,env=os.environ.copy());fresh.append({"hidden_activation":ha,"output_activation":oa,"returncode":cp.returncode,"stdout":cp.stdout.strip(),"stderr":cp.stderr})
 maxima={oracle:{key:max(c[oracle][key] for c in activation+rectangles) for key in ("max_abs","max_rel_outside_near_zero","normalized_l2","relu_mask_mismatches")} for oracle in ("cpu64","pytorch")}
 doc={"schema":1,"marker":MARKER,"phase":"3B1-B1","generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"base_commit":BASE,"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),"tolerances":TOL,"near_zero":NEAR_ZERO,"historical_reports":historical,"activation_cases":activation,"rectangle_cases":rectangles,"multistream":multi,"dynamic_batches":dynamic,"failure_paths":failure_paths,"fresh_processes":{"count":len(fresh),"passed":all(x["returncode"]==0 for x in fresh),"runs":fresh},"summary":{"activation_cases":len(activation),"activation_passed":sum(x["passed"] for x in activation),"rectangle_cases":len(rectangles),"rectangle_passed":sum(x["passed"] for x in rectangles),"maxima":maxima}}
 ok=(all(x["passed"] for x in activation+rectangles) and multi["passed"] and dynamic["passed"]
     and guard and invalid and before==after and failure_paths["static_counted_raii_audit"]
     and doc["fresh_processes"]["passed"] and all(x["expected"]==x["actual"] for x in historical.values()))
 doc["functional_decision"]="HARDENING_FUNCTIONAL_PASS" if ok else "PHASE3B1B1_BLOCKED"
 serialized=json.dumps(doc,indent=2,sort_keys=True)+"\n";(a.output or pathlib.Path("/tmp/phase3b1b1.json")).write_text(serialized);print(doc["functional_decision"]);return 0 if ok else 1
if __name__=="__main__":raise SystemExit(main())
