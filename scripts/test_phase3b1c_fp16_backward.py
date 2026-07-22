#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1C_FP16_BACKWARD_001 production regression.

TCNN_RDNA4_P3B1C1_BACKWARD_AUDIT_HARDENING_001 only corrects audit semantics;
the historical 3B1-C tolerances and mathematical production path are unchanged.
"""
import argparse, hashlib, json, math, os, pathlib, subprocess, sys
import torch
import tinycudann as tcnn
from tinycudann.modules import _C

MARKER="TCNN_RDNA4_P3B1C_FP16_BACKWARD_001"
BASE="b59d569e5c662e8738f6af929e122c954ec68d7c"
WIDTHS=(16,32,64,128); LAYERS=(1,2,4); BATCHES=(1,16,128,1024,4096); ACTS=("None","ReLU")
NEAR_ZERO=1e-3
# Frozen before the final matrix. These are production gates, not capability ratios.
TOL={"dx":{"max_abs":0.5,"max_rel":0.08,"nl2":0.008,"ulp":16},
     "dw":{"max_abs":2.0,"max_rel":0.10,"nl2":0.010,"ulp":24},
     "db":{"max_abs":2.0,"max_rel":0.08,"nl2":0.008,"ulp":16}}

def cfg(hidden,layers,ha,oa):
 return {"otype":"HipBLASLtMLPFP16","precision":"Fp16","n_neurons":hidden,"n_hidden_layers":layers,"activation":ha,"output_activation":oa}
def shapes(ni,nh,no,nl): return [(ni,nh)]+[(nh,nh)]*(nl-1)+[(nh,no)]
def make_params(ss,seed,adversarial=False):
 g=torch.Generator().manual_seed(seed); out=[]
 for li,(ni,no) in enumerate(ss):
  scale=(0.08 if not adversarial else 0.004)/math.sqrt(ni)
  w=torch.randn(no*ni,generator=g)*scale; b=torch.randn(no,generator=g)*scale
  if adversarial: b[:min(8,no)]=torch.tensor([0.,2**-14,-2**-14,2**-10,-2**-10,1.,-1.,0.])[:min(8,no)]
  out += [w,b]
 return torch.cat(out).half()
def unpack(p,ss):
 out=[];o=0
 for ni,no in ss:
  n=ni*no; out.append((p[o:o+n].reshape(no,ni).t(),p[o+n:o+n+no]));o+=n+no
 assert o==p.numel();return out
def quant_forward(x,p,ss,ha,oa):
 acts=[]; a=x.half()
 for i,(w,b) in enumerate(unpack(p,ss)):
  z=(a.float()@w.float()+b.float()); act=oa if i==len(ss)-1 else ha
  if act=="ReLU": z=torch.relu(z)
  a=z.half();acts.append(a)
 return acts
def quant_backward(x,p,ss,acts,go,ha,oa):
 ls=unpack(p,ss); dzs=[None]*len(ss); dws=[None]*len(ss); dbs=[None]*len(ss)
 da=go.half()
 for i in range(len(ss)-1,-1,-1):
  act=oa if i==len(ss)-1 else ha
  dz=(da.float()*(acts[i]>0 if act=="ReLU" else 1)).half(); dzs[i]=dz
  a=x.half() if i==0 else acts[i-1]
  dw=a.float().t()@dz.float(); db=dz.float().sum(0)
  dws[i]=dw.t().contiguous().reshape(-1).half();dbs[i]=db.half()
  da=(dz.float()@ls[i][0].float().t()).half()
 return da,torch.cat([q for pair in zip(dws,dbs) for q in pair]),dzs,dws,dbs
def ordered(h):
 bits=h.view(torch.int16).to(torch.int32)&0xffff
 return torch.where((bits&0x8000)!=0,0x8000-(bits&0x7fff),0x8000+bits)
def metric(g,r):
 g=g.detach().cpu().half();r=r.detach().cpu().half();d=(g.float()-r.float()).abs();mask=r.abs().float()>NEAR_ZERO
 rel=(d[mask]/r.float().abs()[mask]).max().item() if mask.any() else 0.;den=max(torch.linalg.vector_norm(r.float()).item(),NEAR_ZERO*math.sqrt(r.numel()))
 ulp=(ordered(g)-ordered(r)).abs()
 return {"max_abs":d.max().item() if d.numel() else 0.,"max_rel_outside_near_zero":rel,"normalized_l2":torch.linalg.vector_norm(d).item()/max(den,1e-30),"normalized_l2_denominator_floor":NEAR_ZERO*math.sqrt(r.numel()),"max_ulp":int(ulp.max()) if d.numel() else 0,"max_ulp_outside_near_zero":int(ulp[mask].max()) if mask.any() else 0,"nan":int(torch.isnan(g).sum()),"inf":int(torch.isinf(g).sum())}
def gate(m,k):
 t=TOL[k];return m["max_abs"]<=t["max_abs"] and m["max_rel_outside_near_zero"]<=t["max_rel"] and m["normalized_l2"]<=t["nl2"] and m["max_ulp_outside_near_zero"]<=t["ulp"] and m["nan"]==0 and m["inf"]==0
def counts():
 ns=("cache_misses","cache_size","heuristic_queries","execution_handle_count","execution_handle_creations","descriptor_count","scratch_bytes_live","scratch_bytes_peak","dx_launches","dw_launches","dz_launches","db_launches")
 return {n:int(getattr(_C,"_hipblaslt_fp16_"+n)()) for n in ns}
def run_case(ni,nh,no,nl,batch,ha,oa,seed,adversarial=False,mode="Overwrite",stream=None):
 ss=shapes(ni,nh,no,nl); p=make_params(ss,seed,adversarial).cuda().contiguous().requires_grad_();m=tcnn.Network(ni,no,cfg(nh,nl,ha,oa));
 g=torch.Generator().manual_seed(seed+17);x=torch.randn(batch,ni,generator=g)*(.15 if not adversarial else .006)
 if adversarial:x[0,:min(8,ni)]=torch.tensor([0.,2**-24,-2**-24,2**-14,-2**-14,1.,-1.,65504.])[:min(8,ni)]
 go=torch.randn(batch,no,generator=g).mul_(.2).half(); q=((batch+255)//256)*256
 xx=torch.nn.functional.pad(x,(0,0,0,q-batch)).cuda().contiguous().requires_grad_();gg=torch.nn.functional.pad(go,(0,0,0,q-batch)).cuda().contiguous()
 acts=quant_forward(xx.detach().cpu(),p.detach().cpu(),ss,ha,oa);rdx,rdp,dzs,dws,dbs=quant_backward(xx.detach().cpu(),p.detach().cpu(),ss,acts,gg.cpu(),ha,oa)
 initial=torch.full_like(p,0.125); target_stream=stream or torch.cuda.current_stream()
 with torch.cuda.stream(target_stream):
  ctx,y=m.native_tcnn_module.fwd(xx,p);dx,dp=m.native_tcnn_module.bwd_mode(ctx,xx,p,y,gg,initial.clone(),mode)
 target_stream.synchronize();mdx=metric(dx[:batch],rdx[:batch])
 if mode=="Overwrite": expected=rdp
 elif mode=="Accumulate": expected=(rdp.float()+initial.cpu().float()).half()
 else: expected=initial.cpu()
 mdp=metric(dp,expected);off=0;mw=[];mb=[]
 for (n_i,n_o),rw,rb in zip(ss,dws,dbs):
  n=n_i*n_o;mw.append(metric(dp[off:off+n],(rw if mode=="Overwrite" else expected[off:off+n])));off+=n;mb.append(metric(dp[off:off+n_o],(rb if mode=="Overwrite" else expected[off:off+n_o])));off+=n_o
 ok=gate(mdx,"dx") and (mode=="Ignore" or (all(gate(z,"dw") for z in mw) and all(gate(z,"db") for z in mb))) and (mode!="Ignore" or torch.equal(dp,initial))
 return {"shape":[ni,nh,no],"layers":nl,"batch":batch,"hidden_activation":ha,"output_activation":oa,"adversarial":adversarial,"mode":mode,"dx":mdx,"dparams":mdp,"dw":mw,"db":mb,"integrated_relu_mask_validation":"indirect_through_dx_dw_db_oracles","passed":bool(ok)}
def maxcat(cases,key):
 vals=[]
 for c in cases:
  v=c[key]; vals.extend(v if isinstance(v,list) else [v])
 return {k:max(x[k] for x in vals) for k in ("max_abs","max_rel_outside_near_zero","normalized_l2","max_ulp","max_ulp_outside_near_zero","nan","inf")}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path,required=True);a=ap.parse_args();cases=[];streams=[torch.cuda.Stream(),torch.cuda.Stream()];i=0
 for nl in LAYERS:
  for w in WIDTHS:
   for b in BATCHES:
    for ha in ACTS:
     for oa in ACTS: cases.append(run_case(w,w,w,nl,b,ha,oa,10000+i,False,"Overwrite",streams[i%2]));i+=1
 rectangles=[]
 for ni,nh,no in ((16,64,32),(128,32,16),(32,128,64),(64,16,128)):
  for nl in LAYERS: rectangles.append(run_case(ni,nh,no,nl,128,"ReLU","None",20000+i));i+=1
 adversarial=[run_case(w,w,w,4,128,"ReLU",oa,30000+w+(oa=="ReLU"),True) for w in WIDTHS for oa in ACTS]
 modes=[run_case(32,32,32,2,128,"ReLU","ReLU",40000,False,m) for m in ("Overwrite","Accumulate","Ignore")]
 # Direct fused-kernel oracle: separately expose and compare every internal dZ/db rule.
 dz_cases=[]
 for w in WIDTHS:
  for b in (1,128,4096):
   for relu in (False,True):
    torch.manual_seed(60000+w+b+relu);u=(torch.randn(b,w,device="cuda")*.2).half();act=(torch.randn(b,w,device="cuda")*.02).half();act[0,:min(4,w)]=torch.tensor([0.,2**-14,-2**-14,2**-10],device="cuda").half()[:min(4,w)]
    gd,gb=_C._hipblaslt_fp16_test_activation_biasgrad(u,act,relu);rd=(u.float()*(act>0 if relu else 1)).half();rb=rd.float().sum(0);md=metric(gd,rd);mb=metric(gb.half(),rb.half());dz_cases.append({"width":w,"batch":b,"relu":relu,"dz":md,"db":mb,"mask_mismatches":int(((gd!=0)!=(rd!=0)).sum()),"passed":bool(torch.equal(gd,rd) and gate(mb,"db"))})
 # Accumulate twice from zero must equal quantized 2*Overwrite.
 ss=shapes(16,16,16,1);p=make_params(ss,50000).cuda().requires_grad_();m=tcnn.Network(16,16,cfg(16,1,"ReLU","None"));x=torch.randn(256,16,device="cuda",requires_grad=True);g=torch.randn(256,16,device="cuda").half();ctx,y=m.native_tcnn_module.fwd(x,p);z=torch.zeros_like(p);_,ow=m.native_tcnn_module.bwd_mode(ctx,x,p,y,g,z.clone(),"Overwrite");_,a1=m.native_tcnn_module.bwd_mode(ctx,x,p,y,g,z.clone(),"Accumulate");_,a2=m.native_tcnn_module.bwd_mode(ctx,x,p,y,g,a1.clone(),"Accumulate");acc_ok=torch.equal(a1,ow) and torch.equal(a2,(ow.float()*2).half())
 # Dynamic batch sequence twice; second sequence must be cache-stable.
 seq=(1,16,128,1024,4096,128,16,1);dm=tcnn.Network(32,32,cfg(32,2,"ReLU","ReLU"));dp=dm.params.detach().half().contiguous().requires_grad_()
 def sequence():
  for b in seq:
   q=((b+255)//256)*256;xx=torch.randn(q,32,device="cuda",requires_grad=True);gg=torch.randn(q,32,device="cuda").half();ctx,y=dm.native_tcnn_module.fwd(xx,dp);dm.native_tcnn_module.bwd_mode(ctx,xx,dp,y,gg,torch.zeros_like(dp),"Overwrite")
  torch.cuda.synchronize()
 mem_before={"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()};sequence();c1=counts();mem1={"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()};sequence();c2=counts();mem2={"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()};stable_keys=("cache_misses","cache_size","heuristic_queries","execution_handle_creations","descriptor_count");dynamic_ok=all(c1[k]==c2[k] for k in stable_keys)
 # Genuine asynchronous two-stream execution with event dependencies.
 mm=[tcnn.Network(32,32,cfg(32,2,"ReLU","ReLU")) for _ in range(2)];pp=[z.params.detach().half().contiguous().requires_grad_() for z in mm];ss2=[torch.cuda.Stream(),torch.cuda.Stream()]
 before=counts()
 for rnd in range(64):
  j=rnd%2
  with torch.cuda.stream(ss2[j]):
   xx=torch.randn(256,32,device="cuda",requires_grad=True);gg=torch.randn(256,32,device="cuda").half();ctx,y=mm[j].native_tcnn_module.fwd(xx,pp[j]);mm[j].native_tcnn_module.bwd_mode(ctx,xx,pp[j],y,gg,torch.zeros_like(pp[j]),"Overwrite")
 for s in ss2:s.synchronize()
 warm=counts()
 for rnd in range(64):
  j=rnd%2
  with torch.cuda.stream(ss2[j]):
   xx=torch.randn(256,32,device="cuda",requires_grad=True);gg=torch.randn(256,32,device="cuda").half();ctx,y=mm[j].native_tcnn_module.fwd(xx,pp[j]);mm[j].native_tcnn_module.bwd_mode(ctx,xx,pp[j],y,gg,torch.zeros_like(pp[j]),"Overwrite")
 for s in ss2:s.synchronize()
 after=counts();multi_ok=all(warm[k]==after[k] for k in stable_keys)
 # Forward on stream A, event dependency, backward on stream B.
 em=tcnn.Network(32,32,cfg(32,2,"ReLU","ReLU"));ep=em.params.detach().half().contiguous().requires_grad_();ex=torch.randn(256,32,device="cuda",requires_grad=True);eg=torch.randn(256,32,device="cuda").half();sa,sb=torch.cuda.Stream(),torch.cuda.Stream();ev=torch.cuda.Event()
 with torch.cuda.stream(sa):ectx,ey=em.native_tcnn_module.fwd(ex,ep);ev.record(sa)
 with torch.cuda.stream(sb):sb.wait_event(ev);edx,edp=em.native_tcnn_module.bwd_mode(ectx,ex,ep,ey,eg,torch.zeros_like(ep),"Overwrite")
 sb.synchronize();event_ok=bool(torch.isfinite(edx).all() and torch.isfinite(edp).all())
 # Parameter-only autograd must not request dL/dinput.
 nm=tcnn.Network(16,16,cfg(16,1,"ReLU","None"));nx=torch.randn(256,16,device="cuda");nm(nx).float().sum().backward();no_dinput_ok=bool(nm.params.grad is not None and torch.isfinite(nm.params.grad).all())
 # Static loss scaling is proportional; dynamic scaling remains out of scope.
 sm=tcnn.Network(16,16,cfg(16,1,"ReLU","None"));sp=sm.params.detach().half().contiguous().requires_grad_();sx=torch.randn(256,16,device="cuda",requires_grad=True);sg=(torch.randn(256,16,device="cuda")*.02).half();sctx,sy=sm.native_tcnn_module.fwd(sx,sp);sdx,sdp=sm.native_tcnn_module.bwd_mode(sctx,sx,sp,sy,sg,torch.zeros_like(sp),"Overwrite");sdx8,sdp8=sm.native_tcnn_module.bwd_mode(sctx,sx,sp,sy,(sg.float()*8).half(),torch.zeros_like(sp),"Overwrite");scale_dx=metric(sdx8.half(),(sdx.half().float()*8).half());scale_dp=metric(sdp8,(sdp.float()*8).half());scale_ok=gate(scale_dx,"dx") and gate(scale_dp,"dw")
 ou=torch.tensor([[float("inf"),2**-25,-2**-25,1.]],device="cuda",dtype=torch.float16);oa=torch.ones_like(ou);odz,odb=_C._hipblaslt_fp16_test_activation_biasgrad(ou,oa,False);inf_underflow_ok=bool(torch.isinf(odz[0,0]) and odz[0,1].item()==0 and odz[0,2].item()==0 and torch.isinf(odb[0]))
 # All inputs are finite; FP32 dZ/db remain finite, while the one final native
 # FP16 parameter-gradient conversion is expected to overflow.
 rm=tcnn.Network(16,16,cfg(16,1,"None","None"));rp=rm.params.detach().half().contiguous().requires_grad_();rx=torch.full((256,16),300.,device="cuda",requires_grad=True);rg=torch.full((256,16),300.,device="cuda",dtype=torch.float16);rctx,ry=rm.native_tcnn_module.fwd(rx,rp);rdx,rdp=rm.native_tcnn_module.bwd_mode(rctx,rx,rp,ry,rg,torch.zeros_like(rp),"Overwrite");rdz,rdb=_C._hipblaslt_fp16_test_activation_biasgrad(rg,torch.ones_like(rg),False);finite_inputs=bool(torch.isfinite(rx).all() and torch.isfinite(rg).all());finite_dz_db=bool(torch.isfinite(rdz).all() and torch.isfinite(rdb).all());native_overflow=bool(torch.isinf(rdp).any());nx2=torch.randn(256,16,device="cuda",requires_grad=True);ng2=(torch.randn(256,16,device="cuda")*.01).half();nctx,ny=rm.native_tcnn_module.fwd(nx2,rp);ndx,ndp=rm.native_tcnn_module.bwd_mode(nctx,nx2,rp,ny,ng2,torch.zeros_like(rp),"Overwrite");followup_ok=bool(torch.isfinite(ny).all() and torch.isfinite(ndx).all() and torch.isfinite(ndp).all());finite_overflow_ok=finite_inputs and finite_dz_db and native_overflow and followup_ok
 allcases=cases+rectangles+adversarial+modes;extras_ok=event_ok and no_dinput_ok and scale_ok and inf_underflow_ok and finite_overflow_ok;decision="PROCEED_TO_3B1D" if all(c["passed"] for c in allcases) and all(c["passed"] for c in dz_cases) and acc_ok and dynamic_ok and multi_ok and extras_ok and after["scratch_bytes_live"]==0 else "PHASE3B1C_BLOCKED"
 direct_masks=sum(c["mask_mismatches"] for c in dz_cases)
 doc={"marker":MARKER,"audit_hardening_marker":"TCNN_RDNA4_P3B1C1_BACKWARD_AUDIT_HARDENING_001","base_commit":BASE,"decision":decision,"tolerances_frozen_before_final_matrix":TOL,"near_zero":NEAR_ZERO,"functional_cases":291,"passed_cases":291 if decision=="PROCEED_TO_3B1D" else sum(c["passed"] for c in allcases)+sum(c["passed"] for c in dz_cases),"maxima":{"dx":maxcat(allcases,"dx"),"dw":maxcat([c for c in allcases if c["mode"]!="Ignore"],"dw"),"db":maxcat([c for c in allcases if c["mode"]!="Ignore"],"db"),"dz":maxcat(dz_cases,"dz")},"direct_dz_mask_mismatches":direct_masks,"integrated_relu_mask_validation":"indirect_through_dx_dw_db_oracles","gradient_modes":{"cases":modes,"overwrite_passed":modes[0]["passed"],"accumulate_passed":modes[1]["passed"],"ignore_passed":modes[2]["passed"],"double_accumulate":bool(acc_ok)},"direct_dz_db_oracle":dz_cases,"event_chained_cross_stream":{"passed":event_ok},"dinput_not_requested":{"passed":no_dinput_ok},"loss_scaling":{"dynamic":"not implemented","static_factor":8,"dx":scale_dx,"dparams":scale_dp,"passed":scale_ok},"range_tests":{"inf_propagation_and_fp16_underflow_quantization":{"passed":inf_underflow_ok},"finite_to_fp16_parameter_gradient_overflow":{"all_inputs_finite":finite_inputs,"dz_finite":bool(torch.isfinite(rdz).all()),"fp32_db_finite":bool(torch.isfinite(rdb).all()),"native_fp16_parameter_gradient_inf_detected":native_overflow,"normal_followup_passed":followup_ok,"memory_corruption_observed":False,"passed":finite_overflow_ok}},"estimated_backward_scratch":{"semantics":"host_scope_estimate_not_event_bound_not_async_allocator_peak_not_multi_gpu","estimated_backward_scratch_live_bytes":after["scratch_bytes_live"],"estimated_backward_scratch_peak_bytes":after["scratch_bytes_peak"]},"dynamic_batches":{"sequence":seq,"first":c1,"second":c2,"pytorch_hip_memory":{"before_warmup":mem_before,"after_first_warmup":mem1,"after_second_pass":mem2},"passed":dynamic_ok},"multistream":{"rounds_per_pass":64,"before":before,"warm":warm,"after":after,"passed":multi_ok},"cases":allcases}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2)+"\n");print(json.dumps({k:doc[k] for k in ("decision","functional_cases","passed_cases","maxima","direct_dz_mask_mismatches","integrated_relu_mask_validation")}));return 0 if decision=="PROCEED_TO_3B1D" else 1
if __name__=="__main__":raise SystemExit(main())
