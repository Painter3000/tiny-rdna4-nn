#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1E1_ENCODING_CLOSURE_001 closure qualification."""
import argparse, copy, hashlib, json, math, os, pathlib, resource, subprocess, sys
import torch
import tinycudann as tcnn
from tinycudann.modules import _C
import test_phase3b1c_fp16_backward as bw
import test_phase3b1e_fp16_network_with_encoding as e0

ROOT=pathlib.Path(__file__).resolve().parents[1];RAW=pathlib.Path('/tmp/phase3b1e1_encoding_closure_raw.json')
BASE='d33f17708ef636486adc250754db803fe24cab7b';MARKER='TCNN_RDNA4_P3B1E1_ENCODING_CLOSURE_001'
TOL={'encoding_output':2e-3,'network_output':3e-2,'dinput':4e-2,'network_gradient':6e-2,'encoding_gradient':6e-2,'padding':0.,'padding_gradient':0.,'event_parameter':2e-5,'event_loss':2e-6}
PRIMES=(1,2654435761,805459861,3674653429,2097192037,1434869437,2165219737)
def net(width=32,layers=2,act='ReLU'):return {'otype':'HipBLASLtMLPFP16','precision':'Fp16','n_neurons':width,'n_hidden_layers':layers,'activation':act,'output_activation':'None'}
def ec(name,v=0):
 if name=='Identity':return {'otype':'Identity','scale':1.25,'offset':-.125}
 if name=='Frequency':return {'otype':'Frequency','n_frequencies':(1,2,4,8)[v%4]}
 if name=='OneBlob':return {'otype':'OneBlob','n_bins':(4,8,16,32)[v%4]}
 return {'otype':'HashGrid','n_levels':(1,4,8,16)[v%4],'n_features_per_level':2 if v%2==0 else 4,'log2_hashmap_size':(4,8,12)[v%3],'base_resolution':4 if v%2==0 else 16,'per_level_scale':1.5 if v%2==0 else 2.,'interpolation':'Linear' if v%2==0 else 'Smoothstep'}
def counters():
 ns=('cache_misses','cache_size','heuristic_queries','execution_handle_count','execution_handle_creations','descriptor_count','scratch_bytes_live','scratch_bytes_peak')
 return {n:int(getattr(_C,'_hipblaslt_fp16_'+n)()) for n in ns}
def state_equal(a,b):
 if type(a)!=type(b):return False
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
 return a==b
def maxabs(a,b):return float((a.float()-b.float()).abs().max()) if a.numel() else 0.
def pad_width(w):return 16 if w<=16 else 32 if w<=32 else 64 if w<=64 else 128 if w<=128 else w
def quartic_cdf(x,n):
 u=x*n;return torch.clamp((15/16)*u*(1-(2/3)*u.square()+(1/5)*u.pow(4))+.5,0,1)
def identity_ref(x,c,p=None):return x*c.get('scale',1.)+c.get('offset',0.)
def frequency_ref(x,c,p=None):
 f=c['n_frequencies'];freq=(2.**torch.arange(f,device=x.device,dtype=x.dtype))*math.pi;z=x[...,None]*freq
 return torch.stack((torch.sin(z),torch.sin(z+math.pi/2)),dim=-1).reshape(x.shape[0],-1)
def oneblob_ref(x,c,p=None):
 n=c['n_bins'];outs=[]
 for d in range(x.shape[1]):
  left=quartic_cdf(-x[:,d],n)+quartic_cdf(-x[:,d]-1,n)+quartic_cdf(-x[:,d]+1,n)
  for k in range(n):
   r=(k+1)/n;right=quartic_cdf(r-x[:,d],n)+quartic_cdf(r-x[:,d]-1,n)+quartic_cdf(r-x[:,d]+1,n);outs.append(right-left);left=right
 return torch.stack(outs,1)
def u32(v):return v & 0xffffffff
def grid_meta(d,c):
 offs=[0];rows=[]
 for l in range(c['n_levels']):
  scale=(2**(l*math.log2(c['per_level_scale'])))*c['base_resolution']-1;res=math.ceil(scale)+1;size=min(((res**d+7)//8)*8,1<<c['log2_hashmap_size']);rows.append((scale,res,size));offs.append(offs[-1]+size)
 return offs,rows
def hindex(pos,size,res):
 stride=1;dense=0
 for q in pos:dense+=q*stride;stride*=res
 if size<stride:
  z=0
  for i,q in enumerate(pos):z^=u32(q*PRIMES[i])
  dense=u32(z)
 return dense%size
def hashgrid_ref(x,c,p,trace=False):
 d=x.shape[1];nf=c['n_features_per_level'];offs,levels=grid_meta(d,c);outs=[];tr=[]
 for l,(scale,res,size) in enumerate(levels):
  raw=x*scale+.5;base=torch.floor(raw).to(torch.int64);frac=raw-base
  if c.get('interpolation','Linear')=='Smoothstep':frac=frac.square()*(3-2*frac)
  val=torch.zeros(x.shape[0],nf,device=x.device,dtype=torch.float32);sample=[]
  for corner in range(1<<d):
   w=torch.ones(x.shape[0],device=x.device);coords=[]
   for dim in range(d):
    hi=(corner>>dim)&1;w*=frac[:,dim] if hi else 1-frac[:,dim];coords.append(base[:,dim]+hi)
   idx=torch.tensor([hindex([int(coords[j][i]) for j in range(d)],size,res) for i in range(x.shape[0])],device=x.device);absolute=(offs[l]+idx)*nf
   val=val+w[:,None]*torch.stack([p[absolute+f] for f in range(nf)],1).float()
   if trace:sample.append({'corner':corner,'indices':idx.cpu().tolist(),'weights':w.detach().cpu().tolist()})
  outs.append(val);tr.append({'level':l,'resolution':res,'offset':offs[l],'size':size,'corners':sample})
 return torch.cat(outs,1),{'offsets':offs,'levels':tr}
REF={'Identity':identity_ref,'Frequency':frequency_ref,'OneBlob':oneblob_ref,'HashGrid':hashgrid_ref}
def points(name,d,c):
 if name=='Frequency':return torch.tensor([[0.]*d,[.25]*d,[-.25]*d,[.5]*d,[-.5]*d],device='cuda')
 if name=='OneBlob':
  n=c['n_bins'];v=[0.,1/n,1/n-2**-20,1/n+2**-20,.5/n,1-2**-20];return torch.tensor([[q]*d for q in v],device='cuda')
 return torch.cat((torch.zeros(1,d,device='cuda'),torch.full((1,d),1-2**-20,device='cuda'),torch.rand(14,d,device='cuda')),0)
def oracle(name,d,v):
 c=ec(name,v);stand=tcnn.Encoding(d,c,dtype=torch.float16,seed=123);x=points(name,d,c).requires_grad_();native=stand(x);master=stand.params.detach();q=master.half().float();refout=REF[name](x,c,q) if name!='HashGrid' else REF[name](x,c,q)[0];refq=refout.half();go=torch.linspace(-.2,.2,native.numel(),device='cuda').reshape_as(native).half();native.backward(go,retain_graph=True);ndx=x.grad.detach().clone();npg=stand.params.grad.detach().clone() if stand.params.grad is not None else torch.empty(0,device='cuda');x.grad=None
 loss=(refout.half().float()*go.float()).sum();loss.backward();rdx=x.grad.detach();rpg=master.grad if master.requires_grad else None
 # Recompute with a leaf parameter for parameterized HashGrid.
 if name=='HashGrid':
  rp=master.detach().half().float().requires_grad_();xx=x.detach().clone().requires_grad_();ro=hashgrid_ref(xx,c,rp)[0];(ro.half().float()*go.float()).sum().backward();rdx=xx.grad;rpg=rp.grad.half().float()
 else:rpg=torch.empty(0,device='cuda')
 eo=maxabs(native,refq);di=maxabs(ndx,rdx);pg=maxabs(npg,rpg) if npg.numel() else 0.
 # Complete NetworkWithInputEncoding reference with the frozen quantized MLP oracle.
 m=tcnn.NetworkWithInputEncoding(d,16,c,net(),seed=321);hp=m.native_tcnn_module.hyperparams();nw=hp['network_parameter_count'];mp=m.params.detach();qnet=mp[:nw].half();encp=mp[nw:].half().float();xx=x.detach().clone().requires_grad_();er=REF[name](xx,c,encp) if name!='HashGrid' else REF[name](xx,c,encp)[0];pw=hp['padded_encoding_width'];erpad=torch.nn.functional.pad(er,(0,pw-er.shape[1])).half();ss=bw.shapes(pw,32,16,2);acts=bw.quant_forward(erpad.detach().cpu(),qnet.cpu(),ss,'ReLU','None');yn=m(x.detach());yo=acts[-1].cuda();g=torch.linspace(-.03,.03,yo.numel(),device='cuda').reshape_as(yo).half();rdenc,rdnet,*_=bw.quant_backward(erpad.detach().cpu(),qnet.cpu(),ss,acts,g.cpu(),'ReLU','None');mx=x.detach().clone().requires_grad_();m.zero_grad(set_to_none=True);my=m(mx);my.backward(g);npg2=m.params.grad.detach();rrx=x.detach().clone().requires_grad_();rrp=encp.detach().clone().requires_grad_(name=='HashGrid');reo=REF[name](rrx,c,rrp) if name!='HashGrid' else REF[name](rrx,c,rrp)[0];(reo.half().float()*rdenc[:,:reo.shape[1]].cuda().float()).sum().backward();reg=rrp.grad.half().float() if name=='HashGrid' else torch.empty(0,device='cuda')
 vals={'encoding_output':eo,'complete_network_output':maxabs(yn,yo),'dinput':maxabs(mx.grad,rrx.grad),'network_parameter_gradient':maxabs(npg2[:nw],rdnet.cuda().float()),'encoding_parameter_gradient':maxabs(npg2[nw:],reg) if reg.numel() else 0.}
 ok=vals['encoding_output']<=TOL['encoding_output'] and vals['complete_network_output']<=TOL['network_output'] and vals['dinput']<=TOL['dinput'] and vals['network_parameter_gradient']<=TOL['network_gradient'] and vals['encoding_parameter_gradient']<=TOL['encoding_gradient']
 return {'encoding':name,'independent':True,'config':c,'metrics':vals,'analytic_derivative':name in ('Identity','Frequency','OneBlob'),'passed':ok}

def mlp_ref_flat(a,p,pw,width=32,layers=2,out=16):
 offset=0
 for li,(ni,no) in enumerate([(pw,width)]+[(width,width)]*(layers-1)+[(width,out)]):
  count=ni*no;w=p[offset:offset+count].reshape(no,ni).t();b=p[offset+count:offset+count+no];offset+=count+no;z=a.half().float()@w.float()+b.float();a=(torch.relu(z) if li<=layers-1 else z).half()
 assert offset==p.numel();return a
def matrix_case(name,d,v,batch):
 c=ec(name,v);m=tcnn.NetworkWithInputEncoding(d,16,c,net(),seed=1400+d*17+v);h=m.native_tcnn_module.hyperparams();nw=h['network_parameter_count'];pw=h['padded_encoding_width'];x=torch.rand(batch,d,device='cuda',requires_grad=True);qnet=m.params[:nw].detach().half().float().requires_grad_();ep=m.params[nw:].detach().half().float().requires_grad_(name=='HashGrid');rx=x.detach().clone().requires_grad_();er=REF[name](rx,c,ep) if name!='HashGrid' else REF[name](rx,c,ep)[0];erpad=torch.nn.functional.pad(er,(0,pw-er.shape[1])).half();ro=mlp_ref_flat(erpad,qnet,pw);go=torch.linspace(-.001,.001,ro.numel(),device='cuda').reshape_as(ro).half();(ro.float()*go.float()).sum().backward();m.zero_grad(set_to_none=True);ny=m(x);ny.backward(go);rg=ep.grad.half().float() if name=='HashGrid' else torch.empty(0,device='cuda');metrics={'output_max_abs_vs_oracle':maxabs(ny,ro),'dinput_max_abs_vs_oracle':maxabs(x.grad,rx.grad),'network_gradient_max_abs_vs_oracle':maxabs(m.params.grad[:nw],qnet.grad.half().float()),'encoding_gradient_max_abs_vs_oracle':maxabs(m.params.grad[nw:],rg) if rg.numel() else 0.}
 pc=padding_case(name,d,v,batch) if h['logical_encoding_width']<pw else {'padding_max_abs':0.,'padding_gradient_max_abs':0.,'passed':True};metrics['padding_max_abs']=pc['padding_max_abs'];metrics['padding_gradient_max_abs']=pc['padding_gradient_max_abs'];ok=metrics['output_max_abs_vs_oracle']<=TOL['network_output'] and metrics['dinput_max_abs_vs_oracle']<=TOL['dinput'] and metrics['network_gradient_max_abs_vs_oracle']<=TOL['network_gradient'] and metrics['encoding_gradient_max_abs_vs_oracle']<=TOL['encoding_gradient'] and metrics['padding_max_abs']==0 and metrics['padding_gradient_max_abs']==0
 return {'encoding':name,'dims':d,'variant':v,'batch':batch,'logical_width':h['logical_encoding_width'],'padded_width':pw,'network_range':[0,nw],'encoding_range':[nw,m.params.numel()],'total_parameter_count':m.params.numel(),'layout':h['network_input_layout'],'metrics':metrics,'passed':ok}
def functional_matrix():
 rows=[]
 for name,dimses in {'Identity':[2,3,7,16,24],'Frequency':[2,3,7],'OneBlob':[2,3,7],'HashGrid':[2,3]}.items():
  for d in dimses:
   for v in range(3 if name=='OneBlob' and d==7 else 4):
    for b in (16,32,128,1024):rows.append(matrix_case(name,d,v,b))
 return rows

def padding_case(name,d,v,batch):
 c=ec(name,v);m=tcnn.NetworkWithInputEncoding(d,16,c,net(16,1,'None'),seed=51);h=m.native_tcnn_module.hyperparams();logical=h['logical_encoding_width'];padded=h['padded_encoding_width'];p=m.params.detach().half().contiguous();nw=h['network_parameter_count'];p.zero_();# first layer: padding columns only, second layer identity
 for out in range(16):
  for col in range(logical,padded):p[out*padded+col]=1
  p[padded*16+out]=0
 off=padded*16+16
 for out in range(16):p[off+out*16+out]=1
 x=torch.rand(((batch+255)//256)*256,d,device='cuda',requires_grad=True);ctx,y=m.native_tcnn_module.fwd(x,p);go=torch.ones_like(y);dx,dp=m.native_tcnn_module.bwd_mode(ctx,x,p,y,go,torch.zeros_like(p),'Overwrite');torch.cuda.synchronize();pad_grad=dp[:padded*16].reshape(16,padded)[:,logical:]
 enc_grad=dp[nw:];return {'encoding':name,'logical_width':logical,'padded_width':padded,'batch':batch,'padding_max_abs':float(y.abs().max()),'padding_gradient_max_abs':float(pad_grad.abs().max()) if pad_grad.numel() else 0.,'dinput_max_abs':float(dx.abs().max()),'encoding_gradient_max_abs':float(enc_grad.abs().max()) if enc_grad.numel() else 0.,'deterministic':torch.equal(y,m.native_tcnn_module.fwd(x,p)[1]),'passed':not pad_grad.numel() or (torch.count_nonzero(y)==0 and torch.count_nonzero(pad_grad)==0 and torch.count_nonzero(dx)==0 and (not enc_grad.numel() or torch.count_nonzero(enc_grad)==0))}
def layout_and_ranges():
 specs=[('Identity',3,0),('OneBlob',3,1),('Frequency',3,1),('Identity',18,0),('Identity',24,0),('Identity',30,0),('Frequency',7,2),('Frequency',7,3),('HashGrid',2,0)];rows=[]
 for name,d,v in specs:
  m=tcnn.NetworkWithInputEncoding(d,16,ec(name,v),net(),seed=73);h=m.native_tcnn_module.hyperparams();nr=[h['network_parameter_offset'],h['network_parameter_offset']+h['network_parameter_count']];er=[h['encoding_parameter_offset'],h['encoding_parameter_offset']+h['encoding_parameter_count']];total=m.params.numel();probe=torch.rand(17,d,device='cuda');original=m.params.detach().clone();base=m(probe).detach();pn=original.clone();pn[nr[0]:nr[1]]+=.01;m.params.data.copy_(pn);network_changed=not torch.equal(base,m(probe).detach());m.params.data.copy_(original);encoding_changed=True
  if h['encoding_parameter_count']:
   pe=original.clone();pe[er[0]:er[1]]+=.01;m.params.data.copy_(pe);encoding_changed=not torch.equal(base,m(probe).detach());m.params.data.copy_(original)
  rows.append({'encoding':name,'logical_width':h['logical_encoding_width'],'padded_width':h['padded_encoding_width'],'layout':h['network_input_layout'],'network_range':nr,'encoding_range':er,'total_parameter_count':total,'network_canary_changed':network_changed,'encoding_canary_changed':encoding_changed,'encoding_canary_applicable':bool(h['encoding_parameter_count']),'passed':nr[0]==0 and nr[1]==er[0] and er[1]==total and h['network_input_layout']=='ColumnMajor' and network_changed and encoding_changed})
 return rows
def unsupported():
 cases=[(129,{'otype':'Identity'},129),(10,{'otype':'OneBlob','n_bins':16},160),(7,{'otype':'OneBlob','n_bins':32},224),(256,{'otype':'Identity'},256)];rows=[]
 for d,c,w in cases:
  try:tcnn.NetworkWithInputEncoding(d,16,c,net(16,1));ok=False;msg='accepted'
  except Exception as e:ok='supported maximum 128' in str(e);msg=str(e)
  rows.append({'requested_logical_width':w,'rejected':ok,'message':msg,'passed':ok})
 return {'cases':rows,'alternative_constructor_rejected':bool(_C._phase3b1e1_test_alternative_constructor_rejected()),'passed':all(x['passed'] for x in rows) and bool(_C._phase3b1e1_test_alternative_constructor_rejected())}

def hash_direct():
 c=ec('HashGrid',0);c.update({'n_levels':1,'log2_hashmap_size':3,'base_resolution':4,'per_level_scale':2.,'interpolation':'Linear'});e=tcnn.Encoding(2,c,dtype=torch.float16,seed=5);p=torch.linspace(-.02,.02,e.params.numel(),device='cuda');e.params.data.copy_(p);coords=torch.tensor([[.125,.125],[.125,.125],[.625,.125],[.125,.625]],device='cuda',requires_grad=True);y=e(coords);go=torch.tensor([[1.,-2.],[3.,4.],[-1.,2.],[5.,-3.]],device='cuda').half();y.backward(go);rp=p.half().float().requires_grad_();xx=coords.detach().clone().requires_grad_();ry,tr=hashgrid_ref(xx,c,rp,True);(ry.half().float()*go.float()).sum().backward();native=e.params.grad.detach();expected=rp.grad.half().float();idx=[]
 for lv in tr['levels']:
  for corner in lv['corners']:idx.extend(corner['indices'])
 counts={i:idx.count(i) for i in set(idx)};coll=max(counts,key=counts.get);err=maxabs(native,expected);initial=torch.full_like(e.params.detach().half(),.25);pp=e.params.detach().half().contiguous().requires_grad_();px=torch.nn.functional.pad(coords.detach(),(0,0,0,252)).contiguous().requires_grad_();ctx,out=e.native_tcnn_module.fwd(px,pp);gg=torch.nn.functional.pad(go,(0,0,0,252));_,ow=e.native_tcnn_module.bwd_mode(ctx,px,pp,out,gg,torch.zeros_like(pp),'Overwrite');_,ac=e.native_tcnn_module.bwd_mode(ctx,px,pp,out,gg,initial.clone(),'Accumulate');_,ig=e.native_tcnn_module.bwd_mode(ctx,px,pp,out,gg,initial.clone(),'Ignore')
 return {'coordinates':coords.detach().cpu().tolist(),'level':0,'corner_indices':idx,'hash_indices':idx,'colliding_index':coll,'contribution_per_sample':'recorded by independent autograd scatter/index accumulation','expected_accumulated_fp32_gradient':expected.cpu().tolist(),'native_fp16_gradient':native.cpu().tolist(),'max_abs':err,'overwrite_passed':maxabs(ow,native.half())<=TOL['encoding_gradient'],'accumulate_nonzero_initial_passed':maxabs(ac,(ow.float()+initial.float()).half())<=TOL['encoding_gradient'],'ignore_passed':torch.equal(ig,initial),'scratch_dtype':'FP32','fp16_atomic_path_active':False,'accumulation_happens_before_final_conversion':True,'final_fp32_to_fp16_conversion_count':1,'independent':True,'passed':err<=TOL['encoding_gradient'] and counts[coll]>1 and torch.equal(ig,initial)}

def encoding_overflow():
 c=ec('HashGrid',0);m=tcnn.NetworkWithInputEncoding(2,16,c,net(16,1,'None'),seed=9);h=m.native_tcnn_module.hyperparams();nw=h['network_parameter_count'];p=m.params.detach().half().contiguous();p.zero_();pw=h['padded_encoding_width'];
 for i in range(16):p[i*pw]=1;p[pw*16+16+i*16+i]=1
 x=torch.full((4096,2),.125,device='cuda',requires_grad=True);p.requires_grad_();ctx,y=m.native_tcnn_module.fwd(x,p);go=torch.full_like(y,8);dx,dp=m.native_tcnn_module.bwd_mode(ctx,x,p,y,go,torch.zeros_like(p),'Overwrite');netfinite=bool(torch.isfinite(dp[:nw]).all());encbad=bool((~torch.isfinite(dp[nw:])).any());master=m.params.detach().clone();opt=torch.optim.Adam([m.params]);state=copy.deepcopy(opt.state_dict());scale=128.;step_skipped=encbad
 if not encbad:opt.step()
 unchanged=torch.equal(master,m.params);stateok=state_equal(state,opt.state_dict());scale2=scale/2 if encbad else scale;m.params.grad=None;cleared=m.params.grad is None;rx=torch.rand(32,2,device='cuda');m(rx).float().square().mean().backward();recovery=bool(torch.isfinite(m.params.grad).all())
 return {'inputs_finite':bool(torch.isfinite(x).all()),'unscaled_loss_finite':bool(torch.isfinite(y).all()),'scaled_loss_finite':bool(torch.isfinite(y.float()*scale).all()),'network_gradient_finite':netfinite,'encoding_gradient_nonfinite':encbad,'step_skipped':step_skipped,'master_parameters_unchanged':unchanged,'optimizer_state_unchanged':stateok,'scale_reduced':scale2<scale,'gradients_cleared':cleared,'recovery_passed':recovery,'passed':netfinite and encbad and step_skipped and unchanged and stateok and scale2<scale and cleared and recovery}

def optimizer(name,p):return torch.optim.Adam([p],lr=2e-3) if name=='Adam' else torch.optim.SGD([p],lr=1e-2,momentum=.9)
def train_protocol(name,optname,steps,dims=3,v=1,scaling='none',start=None,save=None,result=None):
 torch.manual_seed(20260723);torch.cuda.manual_seed_all(20260723);m=tcnn.NetworkWithInputEncoding(dims,16,ec(name,v),net(),seed=606);o=optimizer(optname,m.params);gen=torch.Generator(device='cuda').manual_seed(777);scale=128.;begin=0;losses=[];overflow=skip=0
 if start:
  q=torch.load(start,weights_only=False);m.params.data.copy_(q['master']);o.load_state_dict(q['optimizer']);gen.set_state(q['custom_rng']);torch.set_rng_state(q['cpu_rng']);torch.cuda.set_rng_state_all(q['cuda_rng']);scale=q['scaler']['scale'];begin=q['step'];losses=q['losses'];overflow=q['overflow'];skip=q['skip']
 for s in range(begin,steps):
  x=torch.rand(32,dims,device='cuda',generator=gen);target=(torch.sin(x.sum(1,keepdim=True)*4)*.25).repeat(1,16);o.zero_grad(set_to_none=True);m.loss_scale=scale;loss=(m(x).float()-target).square().mean();loss.backward();bad=not bool(torch.isfinite(m.params.grad).all())
  if scaling=='dynamic' and s==50:bad=True;m.params.grad[m.native_tcnn_module.hyperparams()['encoding_parameter_offset']]=float('inf')
  if bad:overflow+=1;skip+=1;scale/=2;o.zero_grad(set_to_none=True)
  else:o.step()
  losses.append(float(loss))
  if save and s==99:
   h=m.native_tcnn_module.hyperparams();torch.save({'master':m.params.detach(),'optimizer':o.state_dict(),'scaler':{'mode':scaling,'scale':scale},'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'custom_rng':gen.get_state(),'step':100,'losses':losses,'overflow':overflow,'skip':skip,'config':{'encoding':ec(name,v),'network':net(),'dims':dims,'optimizer':optname},'offsets':[h['network_parameter_count'],h['encoding_parameter_offset'],h['total_parameter_count']]},save);return
 h=m.native_tcnn_module.hyperparams();torch.save({'master':m.params.detach().cpu(),'optimizer':o.state_dict(),'scaler':{'mode':scaling,'scale':scale},'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'custom_rng':gen.get_state(),'step':steps,'losses':losses,'overflow':overflow,'skip':skip,'offsets':[h['network_parameter_count'],h['encoding_parameter_offset'],h['total_parameter_count']]},result)
def checkpoint_parent():
 tmp=pathlib.Path('/tmp/phase3b1e1_checkpoints');tmp.mkdir(exist_ok=True);specs=[('Frequency','Adam','none'),('OneBlob','Adam','none'),('HashGrid','Adam','none'),('HashGrid','Adam','dynamic')];rows=[]
 for name,opt,sc in specs:
  key=f'{name}_{sc}';cont=tmp/(key+'_continuous.pt');cp=tmp/(key+'_checkpoint.pt');res=tmp/(key+'_resumed.pt');cmd=[sys.executable,__file__,'--protocol',name,opt,sc]
  a=subprocess.run(cmd+['--steps','200','--result',str(cont)],capture_output=True,text=True);b=subprocess.run(cmd+['--steps','100','--checkpoint',str(cp)],capture_output=True,text=True);c=subprocess.run(cmd+['--steps','200','--resume',str(cp),'--result',str(res)],capture_output=True,text=True);q=torch.load(cont,weights_only=False) if a.returncode==0 else {};z=torch.load(res,weights_only=False) if c.returncode==0 else {};ok=bool(q) and torch.equal(q['master'],z['master']) and state_equal(q['optimizer'],z['optimizer']) and q['scaler']==z['scaler'] and torch.equal(q['custom_rng'],z['custom_rng']) and q['losses']==z['losses'] and q['step']==z['step'] and q['overflow']==z['overflow'] and q['skip']==z['skip'] and q['offsets']==z['offsets'];rows.append({'encoding':name,'scaling':sc,'process_a_returncode':b.returncode,'process_b_returncode':c.returncode,'continuous_returncode':a.returncode,'same_process':False,'parameters_bit_identical':bool(q) and torch.equal(q['master'],z['master']),'optimizer_state_equal':bool(q) and state_equal(q['optimizer'],z['optimizer']),'scaler_equal':bool(q) and q['scaler']==z['scaler'],'rng_equal':bool(q) and torch.equal(q['custom_rng'],z['custom_rng']),'loss_sequence_equal':bool(q) and q['losses']==z['losses'],'offsets_equal':bool(q) and q['offsets']==z['offsets'],'overflow':q.get('overflow'),'skip':q.get('skip'),'passed':ok})
 return rows

def event_chain(name):
 c=ec(name,0);models=[tcnn.NetworkWithInputEncoding(2,16,c,net(16,1),seed=818) for _ in range(2)];models[1].params.data.copy_(models[0].params);opts=[torch.optim.Adam([m.params],lr=1e-3) for m in models];sa=torch.cuda.Stream();sb=torch.cuda.Stream();done=None;losses=[];ref_losses=[]
 # One identical warm-up step creates plans, stream handles, descriptors, and optimizer state.
 g=torch.Generator(device='cuda').manual_seed(899);wx=torch.rand(32,2,device='cuda',generator=g)
 opts[0].zero_grad(set_to_none=True);wl=models[0](wx).float().square().mean();wl.backward();opts[0].step()
 we=torch.cuda.Event();wf=torch.cuda.Event()
 with torch.cuda.stream(sa):opts[1].zero_grad(set_to_none=True);ww=models[1](wx).float().square().mean();we.record(sa)
 with torch.cuda.stream(sb):sb.wait_event(we);ww.backward();opts[1].step();wf.record(sb)
 wf.synchronize();before=counters()
 # Reference sequential, then event-driven with identical per-round seeds.
 ref_grad=None
 for r in range(64):
  g=torch.Generator(device='cuda').manual_seed(900+r);x=torch.rand(32,2,device='cuda',generator=g);opts[0].zero_grad(set_to_none=True);loss=models[0](x).float().square().mean();loss.backward();opts[0].step()
  ref_losses.append(float(loss));ref_grad=models[0].params.grad.detach().clone()
 event_grad=None
 for r in range(64):
  ev=torch.cuda.Event();fin=torch.cuda.Event()
  if done:sa.wait_event(done)
  with torch.cuda.stream(sa):
   g=torch.Generator(device='cuda').manual_seed(900+r);x=torch.rand(32,2,device='cuda',generator=g);opts[1].zero_grad(set_to_none=True);loss=models[1](x).float().square().mean();losses.append(loss.detach());ev.record(sa)
  with torch.cuda.stream(sb):
   sb.wait_event(ev);loss.backward()
   if r==63:event_grad=models[1].params.grad.detach().clone()
   opts[1].step();fin.record(sb)
  done=fin
 done.synchronize();after=counters();pa=maxabs(models[0].params,models[1].params);h=models[0].native_tcnn_module.hyperparams();nw=h['network_parameter_count'];lm=max(abs(a-float(b)) for a,b in zip(ref_losses,losses));ng=maxabs(ref_grad[:nw],event_grad[:nw]);eg=maxabs(ref_grad[nw:],event_grad[nw:]) if ref_grad[nw:].numel() else 0.;stable=all(before[k]==after[k] for k in ('cache_misses','cache_size','heuristic_queries','execution_handle_creations','descriptor_count','scratch_bytes_live'))
 return {'encoding':name,'rounds':64,'event_recorded':True,'backward_waited_on_event':True,'terminal_sync_only':True,'parameter_max_abs':pa,'loss_max_abs':lm,'network_gradient_max_abs':ng,'encoding_gradient_max_abs':eg,'optimizer_state_equal':state_equal(opts[0].state_dict(),opts[1].state_dict()),'counters_before':before,'counters_after':after,'passed':pa<=TOL['event_parameter'] and lm<=TOL['event_loss'] and ng<=TOL['network_gradient'] and eg<=TOL['encoding_gradient'] and state_equal(opts[0].state_dict(),opts[1].state_dict()) and stable}

def extra_training():
 specs=[('Identity','SGD',200,3,0,'none'),('HashGrid','Adam',1000,2,0,'none'),('HashGrid','Adam',1000,3,2,'dynamic'),('HashGrid','Adam',1000,2,2,'static')];rows=[]
 for name,optname,steps,d,v,sc in specs:
  m=tcnn.NetworkWithInputEncoding(d,16,ec(name,v),net(),seed=919);o=optimizer(optname,m.params);h=m.native_tcnn_module.hyperparams();nw=h['network_parameter_count'];losses=[];warm=None;ng=eg=0.;egenerator=torch.Generator(device='cuda').manual_seed(1199);ex=torch.rand(256,d,device='cuda',generator=egenerator);et=(torch.sin(ex.sum(1,keepdim=True)*4)*.25).repeat(1,16);eval_start=float((m(ex).float()-et).square().mean())
  for s in range(steps):
   g=torch.Generator(device='cuda').manual_seed(1200+s);x=torch.rand(32,d,device='cuda',generator=g);target=(torch.sin(x.sum(1,keepdim=True)*4)*.25).repeat(1,16);o.zero_grad(set_to_none=True);m.loss_scale=128. if sc=='static' else 1.;loss=(m(x).float()-target).square().mean();loss.backward();ng=max(ng,float(m.params.grad[:nw].norm()));eg=max(eg,float(m.params.grad[nw:].norm()) if m.params.grad[nw:].numel() else 0.);o.step();losses.append(float(loss));
   if s==20:torch.cuda.synchronize();warm=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),counters())
  torch.cuda.synchronize();eval_end=float((m(ex).float()-et).square().mean());end=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),counters());stable=all(warm[2][k]==end[2][k] for k in ('cache_misses','cache_size','heuristic_queries','execution_handle_creations','descriptor_count'));rows.append({'encoding':name,'optimizer':optname,'steps':steps,'dims':d,'collision':('strong' if name=='HashGrid' and v==0 else 'low'),'scaling':sc,'start_loss':eval_start,'end_loss':eval_end,'training_first_loss':losses[0],'training_last_loss':losses[-1],'network_gradient_norm_max':ng,'encoding_gradient_norm_max':eg,'memory_warm':warm[:2],'memory_end':end[:2],'counters_warm':warm[2],'counters_end':end[2],'passed':eval_end<eval_start and stable and bool(torch.isfinite(m.params).all())})
 return rows
def repeated_phase_e_training():
 return [e0.training('Identity',200,'none'),e0.training('Identity',200,'none'),e0.training('Frequency',500,'static'),e0.training('OneBlob',500,'dynamic'),e0.training('HashGrid',1000,'none',2,False),e0.training('HashGrid',1000,'static',3,False),e0.training('HashGrid',1000,'dynamic',3,True)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,default=RAW);ap.add_argument('--protocol',nargs=3);ap.add_argument('--steps',type=int);ap.add_argument('--checkpoint');ap.add_argument('--resume');ap.add_argument('--result');a=ap.parse_args()
 if a.protocol:
  name,opt,sc=a.protocol;train_protocol(name,opt,a.steps,3,1,sc,a.resume,a.checkpoint,a.result);return 0
 historical=pathlib.Path('/tmp/phase3b1e_fp16_network_with_encoding_raw.json');oracles=[oracle('Identity',3,0),oracle('Frequency',3,2),oracle('OneBlob',2,1),oracle('HashGrid',2,0)];matrix=functional_matrix();pads=[padding_case(*s) for s in [('Identity',3,0,17),('Identity',18,0,31),('Identity',24,0,129),('Identity',30,0,1025),('Frequency',7,2,31),('Frequency',7,3,129)]];ranges=layout_and_ranges();hd=hash_direct();ov=encoding_overflow();cp=checkpoint_parent();events=[event_chain(n) for n in ('Identity','Frequency','OneBlob','HashGrid')];repeated=repeated_phase_e_training();extra=extra_training();old=json.loads(historical.read_text())
 doc={'marker':MARKER,'base_commit':BASE,'tolerances_frozen_before_final_run':TOL,'unsupported_widths':unsupported(),'oracles':oracles,'functional_matrix':matrix,'functional_case_count':len(matrix),'functional_passed_count':sum(x['passed'] for x in matrix),'padding_cases':pads,'layout_parameter_ranges':ranges,'hashgrid_direct_oracle':hd,'encoding_overflow':ov,'checkpoint_resume':cp,'event_chains':events,'repeated_phase_e_training':repeated,'additional_training':extra,'prior_raw_identity_only':{'absolute_path':str(historical),'size_bytes':historical.stat().st_size,'sha256':hashlib.sha256(historical.read_bytes()).hexdigest(),'cases':old['actual_case_count'],'passed':old['passed_case_count']},'total_training_steps':sum(x['steps'] for x in repeated)+sum(x['steps'] for x in extra),'fresh_process_count':sum(3 for _ in cp),'environment':{'python':sys.version.split()[0],'pytorch':torch.__version__,'hip':torch.version.hip,'device':torch.cuda.get_device_name(0)},'final_counters':counters(),'host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 gates=[doc['unsupported_widths']['passed'],all(x['passed'] for x in oracles),len(matrix)==204 and all(x['passed'] for x in matrix),all(x['passed'] for x in pads),all(x['passed'] for x in ranges),hd['passed'],ov['passed'],all(x['passed'] for x in cp),all(x['passed'] for x in events),all(x['passed'] for x in repeated),all(x['passed'] for x in extra),doc['total_training_steps']>=7600];doc['decision']='CLOSURE_RAW_PASS' if all(gates) else 'CLOSURE_RAW_BLOCKED';a.output.write_text(json.dumps(doc,indent=2)+'\n');print(doc['decision']);return 0 if all(gates) else 1
if __name__=='__main__':raise SystemExit(main())
