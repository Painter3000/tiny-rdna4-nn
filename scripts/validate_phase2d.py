#!/usr/bin/env python3
"""TCNN_RDNA4_P2D_FIX_003: deterministic Phase 2D public-API validation."""
import argparse, importlib, json, math, pathlib, subprocess, sys
from dataclasses import dataclass
import torch

SEED = 20260720

@dataclass(frozen=True)
class Case:
    name: str; hidden: int; width: int; inp: int; out: int; batch: int; act: str; out_act: str

CASES = (
    Case("A",1,16,2,1,1,"ReLU","None"), Case("B",1,32,3,3,7,"None","None"),
    Case("C",2,16,8,1,64,"ReLU","Sigmoid"), Case("D",2,32,3,8,257,"ReLU","None"),
    Case("E",2,64,8,3,7,"None","Sigmoid"), Case("F",4,32,2,1,64,"ReLU","None"),
    Case("G",4,64,3,3,257,"ReLU","Sigmoid"), Case("H",4,128,8,8,7,"ReLU","None"),
)

def cfg(c):
    return {"otype":"PortableMLP","n_neurons":c.width,"n_hidden_layers":c.hidden,
            "activation":c.act,"output_activation":c.out_act}

def shapes(c):
    # Public Network is Identity+Network; Identity retains its 8-wide padded output.
    encoded_input=(c.inp+7)//8*8
    return [(encoded_input if i == 0 else c.width, c.out if i == c.hidden else c.width)
            for i in range(c.hidden + 1)]

def count(c): return sum(i*o+o for i,o in shapes(c))

def unpack(c, flat):
    layers=[]; off=0
    for i,o in shapes(c):
        n=i*o; layers.append((flat[off:off+n].view(o,i), flat[off+n:off+n+o])); off += n+o
    assert off == flat.numel(); return layers

def ref(c, x, p, go):
    xr=x.detach().clone().requires_grad_(True); pr=p.detach().clone().requires_grad_(True)
    y=torch.nn.functional.pad(xr,(0,shapes(c)[0][0]-c.inp),value=1.0)
    for idx,(w,b) in enumerate(unpack(c,pr)):
        y=y@w.t()+b
        if idx < c.hidden and c.act == "ReLU": y=torch.relu(y)
        if idx == c.hidden and c.out_act == "Sigmoid": y=torch.sigmoid(y)
    y.backward(go); return y.detach(),xr.grad.detach(),pr.grad.detach()

def metrics(actual, expected):
    d=(actual-expected).float(); denom=float(expected.float().norm().item())
    return {"max_abs":float(d.abs().max().item()) if d.numel() else 0.0,
            "mean_abs":float(d.abs().mean().item()) if d.numel() else 0.0,
            "relative_l2":float(d.norm().item())/(denom if denom else 1.0),
            "nan":int(torch.isnan(actual).sum().item()),"inf":int(torch.isinf(actual).sum().item())}

def check(name, actual, expected, atol=8e-5, rtol=5e-4):
    m=metrics(actual,expected); torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol,msg=lambda s:f"{name}: {s}"); return m

def run_case(tcnn,c,stream_mode="default"):
    g=torch.Generator(device="cuda"); g.manual_seed(SEED+ord(c.name))
    x=(torch.randn(c.batch,c.inp,device="cuda",generator=g)*.25).requires_grad_(True)
    go=torch.randn(c.batch,c.out,device="cuda",generator=g)*.2
    p=torch.randn(count(c),device="cuda",generator=g)*.12
    ey,ex,ep=ref(c,x,p,go); model=tcnn.Network(c.inp,c.out,cfg(c),seed=SEED)
    assert model.dtype == torch.float32 and model.params.numel() == count(c)
    with torch.no_grad(): model.params.copy_(p)
    stream=None
    if stream_mode != "default": stream=torch.cuda.Stream()
    context=torch.cuda.stream(stream) if stream else torch.cuda.device(0)
    with context: y=model(x); y.backward(go)
    (stream.synchronize() if stream else torch.cuda.synchronize())
    result={"case":c.name,"stream":stream_mode,"output":check("output",y,ey),
            "input_gradient":check("input_gradient",x.grad,ex),"parameter_gradient":check("parameter_gradient",model.params.grad,ep),
            "layers":[]}
    off=0
    for idx,(i,o) in enumerate(shapes(c)):
        n=i*o
        result["layers"].append({"layer":idx,"weight":metrics(model.params.grad[off:off+n],ep[off:off+n]),
                                 "bias":metrics(model.params.grad[off+n:off+n+o],ep[off+n:off+n+o])}); off += n+o
    print("PHASE2D_CASE_PASS "+json.dumps(result,sort_keys=True)); return result

def encoding_cfg(name):
    if name=="Identity": return 3,{"otype":"Identity"}
    if name=="Frequency": return 3,{"otype":"Frequency","n_frequencies":4}
    if name=="OneBlob": return 2,{"otype":"OneBlob","n_bins":8}
    return 2,{"otype":"HashGrid","n_levels":4,"n_features_per_level":2,"log2_hashmap_size":10,
              "base_resolution":4,"per_level_scale":1.5,"interpolation":"Linear"}

def encodings(tcnn):
    rows=[]
    for dtype in (torch.float16,torch.float32):
        for name in ("Identity","Frequency","OneBlob","HashGrid"):
            dims,ec=encoding_cfg(name); m=tcnn.Encoding(dims,ec,seed=SEED,dtype=dtype)
            x=torch.rand(17,dims,device="cuda",requires_grad=True); y=m(x); y.square().mean().backward(); torch.cuda.synchronize()
            assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
            if m.params.numel(): assert m.params.grad is not None and torch.isfinite(m.params.grad).all()
            rows.append({"encoding":name,"dtype":str(dtype),"params":m.params.numel()})
    return rows

def composed(tcnn):
    rows=[]
    for name in ("Identity","Frequency","OneBlob","HashGrid"):
        dims,ec=encoding_cfg(name); nc={"otype":"PortableMLP","n_neurons":32,"n_hidden_layers":2,
                                      "activation":"ReLU","output_activation":"Sigmoid"}
        m=tcnn.NetworkWithInputEncoding(dims,3,ec,nc,seed=SEED); x=torch.rand(19,dims,device="cuda",requires_grad=True)
        y=m(x); y.square().mean().backward(); torch.cuda.synchronize()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(m.params.grad).all()
        enc_params=tcnn.Encoding(dims,ec,seed=SEED,dtype=torch.float32).params.numel()
        net_params=m.params.numel()-enc_params
        net_norm=float(m.params.grad[:net_params].norm().item())
        enc_norm=float(m.params.grad[net_params:].norm().item()) if enc_params else 0.0
        assert net_norm > 0 and (not enc_params or enc_norm > 0)
        rows.append({"encoding":name,"params":m.params.numel(),"network_params":net_params,
                     "encoding_params":enc_params,"network_gradient_norm":net_norm,"encoding_gradient_norm":enc_norm})
    return rows

def train_checkpoint(tcnn):
    torch.manual_seed(SEED); g=torch.Generator(device="cuda"); g.manual_seed(SEED)
    x=torch.rand(128,3,device="cuda",generator=g)*2-1; target=(.4*x[:,:1]-.3*x[:,1:2]+.2*x[:,2:3]).sin()
    c=Case("T",2,32,3,1,128,"ReLU","None"); m=tcnn.Network(3,1,cfg(c),seed=SEED)
    before=m.params.detach().clone(); norm_before=float(before.norm().item()); opt=torch.optim.Adam(m.parameters(),lr=1e-2)
    losses=[]
    for _ in range(200):
        opt.zero_grad(set_to_none=True); loss=(m(x)-target).square().mean()
        assert torch.isfinite(loss); losses.append(float(loss.item())); loss.backward(); opt.step()
    torch.cuda.synchronize(); probe=m(x).detach(); state=m.state_dict(); restored=tcnn.Network(3,1,cfg(c),seed=SEED+1)
    restored.load_state_dict(state); roundtrip=check("checkpoint",restored(x).detach(),probe,atol=0,rtol=0)
    result={"steps":200,"initial_loss":losses[0],"final_loss":losses[-1],"loss_ratio":losses[-1]/losses[0],
            "parameter_change":float((m.params-before).norm().item()),"parameter_norm_before":norm_before,
            "parameter_norm_after":float(m.params.norm().item()),"roundtrip":roundtrip,"method":"state_dict"}
    assert result["loss_ratio"] <= .2 and result["parameter_change"] > 0
    return result

def load(bindings):
    sys.path.insert(0,str(pathlib.Path(bindings).resolve())); import tinycudann as tcnn
    native=importlib.import_module("tinycudann_bindings._120_C"); return tcnn,native

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bindings",required=True); ap.add_argument("--output"); ap.add_argument("--child")
    a=ap.parse_args(); tcnn,native=load(a.bindings)
    assert torch.version.hip and torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName=="gfx1201"
    if a.child: run_case(tcnn,next(c for c in CASES if c.name==a.child)); return
    out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in CASES:
        rows.append(run_case(tcnn,c,"default"))
    rows.append(run_case(tcnn,CASES[3],"explicit")); rows.append(run_case(tcnn,CASES[6],"explicit"))
    fresh=[]
    for c in CASES:
        cp=subprocess.run([sys.executable,__file__,"--bindings",a.bindings,"--child",c.name],capture_output=True,text=True)
        fresh.append({"case":c.name,"returncode":cp.returncode,"output":cp.stdout,"stderr":cp.stderr}); assert cp.returncode==0
    summary={"result":"PASS","environment":{"torch":torch.__version__,"hip":torch.version.hip,
             "device":torch.cuda.get_device_name(0),"arch":torch.cuda.get_device_properties(0).gcnArchName,
             "binding":str(pathlib.Path(native.__file__).resolve())},"network_cases":rows,"encodings":encodings(tcnn),
             "composed":composed(tcnn),"training_checkpoint":train_checkpoint(tcnn),"fresh_processes":fresh}
    (out/"phase2d_validation.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("PHASE2D_VALIDATION_PASS "+json.dumps({"cases":len(rows),"fresh":len(fresh),"result":"PASS"}))

if __name__=="__main__": main()
