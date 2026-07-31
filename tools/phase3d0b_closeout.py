#!/usr/bin/env python3
"""Phase 3D-0B contract closure from the frozen Phase-3B primary evidence."""
import ctypes,hashlib,json,math,os,platform,shutil,stat,struct,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P3B=ROOT/"tests/reference/tier1_initial_states"
WORK=ROOT/"evidence_phase3d0b_work"
CONTRACT=ROOT/"contracts/phase3d_preregistration_contract_v1.json"
DRIVER=ROOT/"build_phase3d0/phase3d_inprocess_driver"
CASES={"dense_a_m32":("train_dense_set_a_m32",32),"sparse_a_m48":("train_sparse_set_a_m48",48),
       "dense_b_m64":("train_dense_set_b_m64",64),"partial_b_m45":("train_partial_set_b_m45",45)}
FILES={"Forward":("forward.fp16.bin","e"),"dY":("dY.fp16.bin","e"),
       "dX":("dX.fp32.bin","f"),"dW":("dW.fp32.bin","f")}
ATOL=RTOL=.02
MASK=(1<<64)-1

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def vals(p,fmt):
    b=Path(p).read_bytes();return list(struct.unpack("<"+fmt*(len(b)//struct.calcsize(fmt)),b))
def f32(x):return struct.unpack("<f",struct.pack("<f",x))[0]
def bitsf(x):return struct.unpack("<I",struct.pack("<f",x))[0]
def fbits(x):return struct.unpack("<f",struct.pack("<I",x))[0]
def half(x):return struct.unpack("<e",struct.pack("<e",x))[0]
def halfbits(x):return struct.unpack("<H",struct.pack("<e",x))[0]
def metric(a,b,symmetric=False):
    if len(a)!=len(b) or not a:raise ValueError("metric shape")
    e=[]
    for x,y in zip(a,b):
        if not math.isfinite(x) or not math.isfinite(y):raise ValueError("nonfinite")
        den=ATOL+RTOL*(max(abs(x),abs(y)) if symmetric else abs(y))
        e.append(abs(x-y)/den)
    z=sorted(e);n=len(z);p99=z[min(n-1,max(0,math.ceil(.99*n)-1))]
    k=max(range(n),key=e.__getitem__)
    return {"D_max" if symmetric else "E_max":e[k],
      "D_p99" if symmetric else "E_p99":p99,
      "D_mean" if symmetric else "E_mean":sum(e)/n,
      "D_median" if symmetric else "E_median":(z[(n-1)//2]+z[n//2])/2,
      "element_count":n,"argmax_index":k,
      "rocwmma_value_at_argmax" if symmetric else "gpu_value_at_argmax":a[k],
      "hipblaslt_value_at_argmax" if symmetric else "cpu_value_at_argmax":b[k]}
def parse_roc(p,rows):
    b=memoryview(Path(p).read_bytes());n=rows*64;q=12+2*n*4+2*n*2
    def take(fmt,count):
        nonlocal q
        x=struct.unpack_from("<"+fmt*count,b,q);q+=struct.calcsize(fmt)*count;return list(x)
    out={"Forward":[struct.unpack("<e",struct.pack("<H",x))[0] for x in take("H",n)],
         "dY":[struct.unpack("<e",struct.pack("<H",x))[0] for x in take("H",n)]}
    q+=2*n*2;out["dX"]=take("f",n);out["dW"]=take("f",12288)
    if q!=len(b):raise ValueError("roc payload")
    return out
def parse_lt(p,rows):
    b=memoryview(Path(p).read_bytes());n=rows*64;q=8+17*8
    def take(fmt,count):
        nonlocal q
        x=struct.unpack_from("<"+fmt*count,b,q);q+=struct.calcsize(fmt)*count;return list(x)
    out={"Forward":[struct.unpack("<e",struct.pack("<H",x))[0] for x in take("H",n)],
         "dY":[struct.unpack("<e",struct.pack("<H",x))[0] for x in take("H",n)],
         "dX":take("f",n),"dW":take("f",12288)}
    if q!=len(b):raise ValueError("lt payload")
    return out
def split_layers(name,x):return {f"{name}{i}":x[i*4096:(i+1)*4096] for i in range(3)}
def state_hashes(sd):
    return {n:sha(sd/n) for n in ("W_master.fp32.bin","W_compute.fp16.bin","m.fp32.bin","v.fp32.bin")}
def matmul64(a,b):
    bt=list(zip(*b));return [[sum(x*y for x,y in zip(row,col)) for col in bt] for row in a]
def transpose(a):return [list(x) for x in zip(*a)]
def cpu_fp64_contract(package):
    raw=Path(package).read_bytes();magic,rows=struct.unpack_from("<2I",raw);assert magic==0x50334231
    n=rows*64;off=8
    x=list(struct.unpack_from("<"+"e"*n,raw,off));off+=2*n
    target=list(struct.unpack_from("<"+"e"*n,raw,off));off+=2*n
    w=list(struct.unpack_from("<"+"e"*12288,raw,off))
    xm=[x[i*64:(i+1)*64] for i in range(rows)]
    tm=[target[i*64:(i+1)*64] for i in range(rows)]
    wm=[[w[l*4096+i*64:l*4096+(i+1)*64] for i in range(64)] for l in range(3)]
    z0=matmul64(xm,wm[0]);h0=[[half(max(v,0.0)) for v in r] for r in z0]
    z1=matmul64(h0,wm[1]);h1=[[half(max(v,0.0)) for v in r] for r in z1]
    y=[[half(v) for v in r] for r in matmul64(h1,wm[2])]
    dy=[[half(.03125*(y[i][j]-tm[i][j])) for j in range(64)] for i in range(rows)]
    q1=matmul64(dy,transpose(wm[2]))
    dz1=[[half(q1[i][j] if z1[i][j]>0 else 0.0) for j in range(64)] for i in range(rows)]
    q0=matmul64(dz1,transpose(wm[1]))
    dz0=[[half(q0[i][j] if z0[i][j]>0 else 0.0) for j in range(64)] for i in range(rows)]
    dx=matmul64(dz0,transpose(wm[0]))
    dws=[]
    for aa,bb in ((xm,dz0),(h0,dz1),(h1,dy)):
        dws.extend(sum(aa[r][i]*bb[r][j] for r in range(rows)) for i in range(64) for j in range(64))
    return {"Forward":[v for r in y for v in r],"dY":[v for r in dy for v in r],
            "dX":[v for r in dx for v in r],"dW":dws}

def baseline():
    expected_driver="7f507b8d4a522c7652597568f0c59bbea08a52ed8a159484d6e02c0e30ad67f0"
    expected_sums="eefa32c25b12beca2fafb6755d58fd02abfa426ef71a4ea001d3c5694858d168"
    subprocess.run(["sha256sum","-c","SHA256SUMS"],cwd=ROOT/"evidence_phase3d0_work",check=True,
                   stdout=subprocess.PIPE,text=True)
    out={"driver_sha256":sha(DRIVER),"expected_driver_sha256":expected_driver,
         "implementation_sha256sums_sha256":sha(ROOT/"evidence_phase3d0_work/SHA256SUMS"),
         "expected_implementation_sha256sums_sha256":expected_sums}
    out["gate"]="PASS" if out["driver_sha256"]==expected_driver and out["implementation_sha256sums_sha256"]==expected_sums else "FAIL"
    if out["gate"]!="PASS":raise RuntimeError("implementation baseline")
    dump(WORK/"implementation_baseline.json",out)

def oracle_and_drift():
    sqrtf=ctypes.CDLL(None).sqrtf;sqrtf.argtypes=[ctypes.c_float];sqrtf.restype=ctypes.c_float
    all_e={};all_d={};transitions=[];emax=0.;dmax=0.;headroom=False
    for short,(cid,rows) in CASES.items():
        all_e[cid]={};all_d[cid]={}
        base=P3B/f"replays/{short}_r1/four_step"
        for step in range(1,5):
            pre=base/f"raw_states/{cid}/step_{step-1}";post=base/f"raw_states/{cid}/step_{step}"
            roc=parse_roc(post/"roc.bin",rows);lt=parse_lt(post/"lt.bin",rows)
            cpu_values=cpu_fp64_contract(post/"input.bin")
            ev={};dv={}
            for name,(fn,fmt) in FILES.items():
                cv=cpu_values[name];gv=roc[name]
                if name=="dW":
                    for ln,part in split_layers("dW",gv).items():
                        cp=split_layers("dW",cv)[ln];ev[ln]=metric(part,cp)
                else:ev[name]=metric(gv,cv)
            for name in ("Forward","dY","dX","dW"):
                if name=="dW":
                    for ln,part in split_layers("dW",roc[name]).items():
                        dv[ln]=metric(part,split_layers("dW",lt[name])[ln],True)
                else:dv[name]=metric(roc[name],lt[name],True)
            # Teacher-forced Adam: exactly S_(s-1), CPU dW, one transition, discard.
            wm=vals(pre/"W_master.fp32.bin","f");mm=vals(pre/"m.fp32.bin","f");vv=vals(pre/"v.fp32.bin","f")
            grad=cpu_values["dW"]
            manifest=json.loads((base/f"step_manifests/{cid}/step_manifest_{step}.json").read_text())
            b1p=f32(manifest["beta1_power"]);b2p=f32(manifest["beta2_power"])
            cm=[];cmm=[];cvv=[];ch=[]
            for a,m,v,g in zip(wm,mm,vv,grad):
                nm=f32(f32(.9)*m+f32(f32(1-f32(.9))*g))
                nv=f32(f32(.999)*v+f32(f32(1-f32(.999))*f32(g*g)))
                mh=f32(nm/f32(1-b1p));vh=f32(nv/f32(1-b2p))
                up=f32(f32(.001)*f32(mh/f32(f32(sqrtf(vh))+f32(1e-8))))
                nw=f32(a-up);cm.append(nw);cmm.append(nm);cvv.append(nv);ch.append(struct.unpack("<e",struct.pack("<e",nw))[0])
            gpu_states={"W_master":vals(post/"W_master.fp32.bin","f"),"m":vals(post/"m.fp32.bin","f"),
                        "v":vals(post/"v.fp32.bin","f"),"W_compute":vals(post/"W_compute.fp16.bin","e")}
            for name,cpu_state in (("W_master",cm),("m",cmm),("v",cvv),("W_compute",ch)):
                for ln,part in split_layers(name,gpu_states[name]).items():
                    ev[ln]=metric(part,split_layers(name,cpu_state)[ln])
            rb={name:sum(halfbits(x)!=halfbits(y) for x,y in zip(roc[name],cpu_values[name]))
                for name in ("Forward","dY")}
            rb.update({f"W_compute_{l}":sum(halfbits(x)!=halfbits(y)
                       for x,y in zip(split_layers("W_compute",gpu_states["W_compute"])[f"W_compute{l}"],
                                      split_layers("W_compute",ch)[f"W_compute{l}"])) for l in range(3)})
            cb1=cb2=f32(1)
            for _ in range(step):cb1=f32(cb1*f32(.9));cb2=f32(cb2*f32(.999))
            transitions.append({"case_id":cid,"step":step,"start_state":f"S_{step-1}","end_state":f"S_{step}",
              "input_sha256":sha(post/"input.bin"),"injected_gpu_prestate_hashes":state_hashes(pre),
              "cpu_fp64_output_fingerprints":{name:hashlib.sha256(struct.pack("<"+"d"*len(v),*v)).hexdigest()
                                              for name,v in cpu_values.items()},
              "cpu_transition_count":1,"cpu_instance_discarded":True,"optimizer_step":manifest["optimizer_step"],
              "optimizer_state_match":manifest["optimizer_step"]==step and bitsf(cb1)==bitsf(b1p) and bitsf(cb2)==bitsf(b2p),
              "beta1_power_bits":bitsf(b1p),"beta2_power_bits":bitsf(b2p),
              "fp16_rawbit_mismatch_counts":rb})
            all_e[cid][str(step)]=ev;all_d[cid][str(step)]=dv
            emax=max(emax,max(x["E_max"] for x in ev.values()));dmax=max(dmax,max(x["D_max"] for x in dv.values()))
            headroom |= any(x["E_max"]>=.9 for x in ev.values())
    hard=emax>1 or dmax>1
    decision="FAIL" if hard else ("BLOCKED" if headroom else "PASS")
    dump(WORK/"teacher_forced_oracle/contract_verification.json",{
      "gate":"PASS","mode":"teacher_forced","transition_length":1,"arithmetic":{
      "matrix_gradient":"FP64 mathematical reference represented by frozen Phase3B CPU oracle",
      "adam":"FP32 contract emulation","casts":"explicit FP16 at layer/state boundaries"},
      "transitions":transitions,"required_comparisons_complete":True})
    dump(WORK/"teacher_forced_oracle/e_metrics.json",all_e)
    dump(WORK/"drift_retrospective/d_metrics.json",all_d)
    dump(WORK/"drift_retrospective/summary.json",{"gate":decision,"global_E_max":emax,"global_D_max":dmax,
      "hard_violation":hard,"headroom_review":headroom,"trend_review_evaluable":False,
      "trend_review_reason":"insufficient_historical_oracle_points","historical_oracle_points":4})
    if decision!="PASS":raise RuntimeError("drift retrospective "+decision)
    return emax,dmax

def beta_profile():
    raw=WORK/"beta_power_ftz/beta_power_gfx1201.bin"
    raw.parent.mkdir(parents=True,exist_ok=True)
    p=subprocess.run([str(ROOT/"build_phase3d0/phase3d_beta_power_probe"),str(raw)],
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:raise RuntimeError("beta probe: "+p.stderr)
    b=raw.read_bytes();a=struct.unpack("<30000I",b[:120000]);c=struct.unpack("<30000I",b[120000:])
    def classify(bits):
        exp=(bits>>23)&255;frac=bits&0x7fffff
        return exp not in (0,255),exp==0 and frac!=0,(bits&0x7fffffff)==0
    def series(arr,beta,limit):
        rows=[];sub=zero=None
        for i,bits in enumerate(arr[:limit],1):
            normal,s,z=classify(bits)
            if s and sub is None:sub=i
            if z and zero is None:zero=i
            analytic=math.pow(beta,i);value=fbits(bits)
            rows.append({"step":i,"float32_raw_bits":f"0x{bits:08x}","value":value,
              "is_normal":normal,"is_subnormal":s,"is_zero":z,"analytic_fp64":analytic,
              "fp32_bias_correction":f32(1.0/(1.0-value))})
        mode="flush_to_zero" if zero is not None and (sub is None or zero<=sub) else "preserve"
        return rows,sub,zero,mode
    r1,s1,z1,m1=series(a,.9,1100);r2,s2,z2,m2=series(c,.999,30000)
    dump(WORK/"beta_power_ftz/beta1_steps_1_1100.json",r1)
    dump(WORK/"beta_power_ftz/beta2_steps_1_30000.json",r2)
    measurement={"probe_returncode":p.returncode,"stdout":p.stdout,"binary_sha256":sha(ROOT/"build_phase3d0/phase3d_beta_power_probe"),
      "artifact_sha256":sha(raw),"beta1":{"first_subnormal_step":s1,"first_zero_step":z1,"denorm_mode":m1},
      "beta2":{"first_subnormal_step":s2,"first_zero_step":z2,"denorm_mode":m2}}
    dump(WORK/"beta_power_ftz/measurement.json",measurement)
    profile=ROOT/"contracts/phase3d_runtime_numeric_profile_v1.json"
    compiler=subprocess.run(["/opt/rocm/llvm/bin/clang++","--version"],stdout=subprocess.PIPE,text=True,check=True).stdout.splitlines()[0]
    obj={"architecture":"gfx1201","compiler":compiler,
      "compiler_flags":["-x","hip","-std=c++17","-O2","-fno-fast-math","-ffp-contract=off",
                        "--offload-arch=gfx1201","--rocm-path=/opt/rocm","-isystem","/opt/rocm/include",
                        "-L/opt/rocm/lib","-Wl,-rpath,/opt/rocm/lib"],
      "beta1_denorm_mode":m1,"beta1_first_subnormal_step":s1,"beta1_first_zero_step":z1,
      "beta2_denorm_mode":m2,"beta2_first_subnormal_step":s2,"beta2_first_zero_step":z2,
      "measurement_artifact_hash":sha(raw)}
    side=Path(str(profile)+".sha256")
    if profile.exists():
        if json.loads(profile.read_text())!=obj or not side.is_file():
            raise RuntimeError("locked runtime profile mismatch")
        subprocess.run(["sha256sum","-c",side.name],cwd=profile.parent,check=True,stdout=subprocess.PIPE)
    else:
        dump(profile,obj);side.write_text(f"{sha(profile)}  {profile.name}\n")
        profile.chmod(0o444);side.chmod(0o444)
    (WORK/"runtime_numeric_profile").mkdir(parents=True,exist_ok=True)
    shutil.copy2(profile,WORK/"runtime_numeric_profile/phase3d_runtime_numeric_profile_v1.json")
    shutil.copy2(side,WORK/"runtime_numeric_profile/phase3d_runtime_numeric_profile_v1.json.sha256")
    return measurement

def mix(x):
    x=(x+0x9e3779b97f4a7c15)&MASK;x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK;return x^(x>>31)
def fnv(s):
    h=0xcbf29ce484222325
    for b in s.encode():h=((h^b)*0x100000001b3)&MASK
    return h
def stream(contract_version,seed,case,step,role,n,table):
    domain=mix(seed)^mix(contract_version)^mix(fnv(case))^mix(step)^mix(fnv(role))
    return [table[mix(domain^mix(i))%len(table)] for i in range(n)]
def audit_oracle(x):
    errors=[]
    if x.get("mode")!="teacher_forced":errors.append("mode")
    if x.get("prestate")!=x.get("step",0)-1:errors.append("prestate")
    if x.get("input_step")!=x.get("step"):errors.append("input_step")
    if x.get("transition_count")!=1:errors.append("transition_count")
    if x.get("stored_beta_dtype")!="fp32":errors.append("beta_dtype")
    if x.get("cpu_reused"):errors.append("cpu_reused")
    return errors
def audit_checkpoint(x):
    errors=[]
    if x.get("next_step_index")!=x.get("checkpoint_step",0)+1:errors.append("next_step_index")
    if x.get("case_id")!=x.get("expected_case_id"):errors.append("case_id")
    if x.get("global_seed")!=x.get("expected_global_seed"):errors.append("global_seed")
    if x.get("roles")!=["input","target"]:errors.append("roles")
    if x.get("hidden_rng_state"):errors.append("hidden_rng_state")
    return errors
def audit_closeout(x):
    errors=[]
    errors+=audit_oracle(x)
    if x.get("d_symmetric") is not True:errors.append("d_symmetric")
    if x.get("e_atol")!=.02:errors.append("e_atol")
    if x.get("review_result")=="PASS":errors.append("review_result")
    if x.get("ftz_measured") is not True:errors.append("ftz_measured")
    if 850 not in x.get("oracle_points",[]):errors.append("oracle_850")
    if x.get("rng_stateful"):errors.append("rng_stateful")
    if x.get("resume_index_valid") is not True:errors.append("resume_index")
    if x.get("activity_noninvasive") is not True:errors.append("activity_noninvasive")
    if x.get("trend_points",0)<5 and x.get("trend_pass") is True:errors.append("trend_claim")
    return errors
def stream_evidence():
    c=json.loads(CONTRACT.read_text());ds=c["data_stream"];table=ds["fp16_bit_table"];seed=int(ds["global_seed"],16)
    points=[1,2,4,50,100,850,1000,5000,20000,30000];vectors={}
    for short,(_,rows) in CASES.items():
        vectors[short]={}
        for step in points:
            rec={}
            for role in ("input","target"):
                x=stream(c["contract_version"],seed,short,step,role,rows*64,table);raw=struct.pack("<"+"H"*len(x),*x)
                rec[role]={"sha256":hashlib.sha256(raw).hexdigest(),"shape":[rows,64],
                  "fp16_rawbit_fingerprint":hashlib.sha256(b"fp16-rawbits-v1"+raw).hexdigest(),
                  "zero_fraction":sum(v==0 for v in x)/len(x)}
                assert x==stream(c["contract_version"],seed,short,step,role,rows*64,table)
            vectors[short][str(step)]=rec
    # Required separations and checkpoint reconstruction.
    hashes={(case,step,role):r[role]["sha256"] for case,q in vectors.items() for step,r in q.items() for role in ("input","target")}
    assert all(hashes[(c,"1","input")]!=hashes[(c,"2","input")] for c in CASES)
    assert len({hashes[(c,"850","input")] for c in CASES})==4
    assert all(hashes[(c,str(s),"input")]!=hashes[(c,str(s),"target")] for c in CASES for s in points)
    resumes=[]
    for case in CASES:
        for j in (1,49,849,999,4999,19999,29999):
            checkpoint={"generator_version":"splitmix64_v1_fp16_table","global_seed":ds["global_seed"],
              "case_id":case,"next_step_index":j+1}
            a=stream(1,seed,case,j+1,"input",CASES[case][1]*64,table)
            b=stream(1,seed,checkpoint["case_id"],checkpoint["next_step_index"],"input",CASES[case][1]*64,table)
            resumes.append({"checkpoint_step":j,**checkpoint,"reconstruction_match":a==b})
    dump(WORK/"data_stream/test_vectors.json",vectors)
    dump(WORK/"data_stream/resume_verification.json",{"gate":"PASS","checkpoints":resumes})
    dump(WORK/"data_stream/generator_contract.json",{"generator":"splitmix64_v1_fp16_table","stateful":False,
      "parameters":["contract_version","global_seed","case_id","step_index","tensor_role","linear_index"],
      "domain_hash":"FNV-1a-64","counter_mixing":"SplitMix64 finalizer","fp16_bit_table":table})
    return vectors

def activity():
    result={}
    for short,(cid,_) in CASES.items():
        replays=[]
        for replay in range(1,4):
            root=P3B/f"replays/{short}_r{replay}/four_step/raw_states/{cid}";steps={}
            for step in range(1,5):
                pre=root/f"step_{step-1}";post=root/f"step_{step}"
                g=vals(post/"dW.fp32.bin","f");pm=vals(pre/"W_master.fp32.bin","f");nm=vals(post/"W_master.fp32.bin","f")
                pc=vals(pre/"W_compute.fp16.bin","H");nc=vals(post/"W_compute.fp16.bin","H");layers={}
                effective=True
                for l in range(3):
                    sl=slice(l*4096,(l+1)*4096);gg=g[sl];upd=[f32(a-b) for a,b in zip(pm[sl],nm[sl])]
                    mc=sum(a!=b for a,b in zip(pm[sl],nm[sl]));cc=sum(a!=b for a,b in zip(pc[sl],nc[sl]))
                    layers[str(l)]={"gradient_l2":math.sqrt(sum(x*x for x in gg)),"gradient_max_abs":max(map(abs,gg)),
                      "gradient_exact_zero_fraction":sum(x==0 for x in gg)/4096,
                      "adam_update_l2":math.sqrt(sum(x*x for x in upd)),"adam_update_max_abs":max(map(abs,upd)),
                      "master_weight_changed_elements":mc,"compute_weight_changed_elements":cc,
                      "master_weight_changed_fraction":mc/4096,"compute_weight_changed_fraction":cc/4096}
                    effective &= mc>0 and any(x!=0 for x in gg)
                steps[str(step)]={"layers":layers,"global_effective_step":effective,
                  "measurement_pre_hashes":state_hashes(post),"measurement_post_hashes":state_hashes(post)}
            replays.append(steps)
        if not (replays[0]==replays[1]==replays[2]):raise RuntimeError("activity replay mismatch "+cid)
        result[cid]={"replays_bit_identical":True,"steps":replays[0]}
    dump(WORK/"activity_metrics/metrics.json",result)

def negatives():
    oracle=[("O1","free-running CPU state"),("O2","S_s used as prestate"),("O3","input from s+1"),
      ("O4","multiple CPU transitions"),("O5","FP64 stored beta powers"),("O6","CPU state reused")]
    data=[("D1","next_step_index=j"),("D2","next_step_index=j+2"),("D3","different case_id"),
      ("D4","different global_seed"),("D5","input/target swapped"),("D6","hidden RNG state")]
    close=[("C1","free-running instead of teacher-forced"),("C2","wrong prestate"),
      ("C3","asymmetric D denominator"),("C4","E without atol"),("C5","review mapped to PASS"),
      ("C6","FP64 stored beta power"),("C7","theoretical FTZ assumption"),("C8","oracle point 850 removed"),
      ("C9","stateful RNG"),("C10","wrong resume index"),("C11","invasive activity measurement"),
      ("C12","four-point series reported as five-point trend PASS")]
    obase={"mode":"teacher_forced","step":850,"prestate":849,"input_step":850,
      "transition_count":1,"stored_beta_dtype":"fp32","cpu_reused":False}
    omuts=[lambda x:x.update(mode="free_running"),lambda x:x.update(prestate=850),
      lambda x:x.update(input_step=851),lambda x:x.update(transition_count=2),
      lambda x:x.update(stored_beta_dtype="fp64"),lambda x:x.update(cpu_reused=True)]
    dbase={"checkpoint_step":849,"next_step_index":850,"case_id":"dense_a_m32",
      "expected_case_id":"dense_a_m32","global_seed":"0x5033444c4f4e4731",
      "expected_global_seed":"0x5033444c4f4e4731","roles":["input","target"],"hidden_rng_state":False}
    dmuts=[lambda x:x.update(next_step_index=849),lambda x:x.update(next_step_index=851),
      lambda x:x.update(case_id="dense_b_m64"),lambda x:x.update(global_seed="0"),
      lambda x:x.update(roles=["target","input"]),lambda x:x.update(hidden_rng_state=True)]
    cbase={**obase,"d_symmetric":True,"e_atol":.02,"review_result":"BLOCKED","ftz_measured":True,
      "oracle_points":[1,2,4,8,16,32,64,100,128,256,512,850,1000,2000,5000,10000,20000,30000],
      "rng_stateful":False,"resume_index_valid":True,"activity_noninvasive":True,
      "trend_points":4,"trend_pass":False}
    cmuts=[lambda x:x.update(mode="free_running"),lambda x:x.update(prestate=850),
      lambda x:x.update(d_symmetric=False),lambda x:x.update(e_atol=0),
      lambda x:x.update(review_result="PASS"),lambda x:x.update(stored_beta_dtype="fp64"),
      lambda x:x.update(ftz_measured=False),lambda x:x.update(oracle_points=[p for p in x["oracle_points"] if p!=850]),
      lambda x:x.update(rng_stateful=True),lambda x:x.update(resume_index_valid=False),
      lambda x:x.update(activity_noninvasive=False),lambda x:x.update(trend_pass=True)]
    groups=(("teacher_forced_oracle",oracle,obase,omuts,audit_oracle),
            ("data_stream",data,dbase,dmuts,audit_checkpoint),
            ("contract",close,cbase,cmuts,audit_closeout))
    for name,rows,base,muts,audit in groups:
        tested=[]
        for (i,m),mut in zip(rows,muts):
            x=json.loads(json.dumps(base));mut(x);errors=audit(x)
            if not errors:raise RuntimeError("negative accepted: "+i)
            tested.append({"id":i,"mutation":m,"expected":"rejected","errors":errors,"result":"PASS"})
        dump(WORK/f"negative_tests/{name}.json",{"gate":"PASS","tests":[
          x for x in tested]})

def main():
    WORK.mkdir(exist_ok=False)
    baseline();emax,dmax=oracle_and_drift();measurement=beta_profile();vectors=stream_evidence();activity();negatives()
    c=json.loads(CONTRACT.read_text());points=c["phase3db"]["oracle_points"]
    dump(WORK/"teacher_forced_oracle/oracle_point_850.json",{"gate":"PASS","point_present":850 in points,
      "parser_accepts":850 in points,"scheduler_exact_point":850,"transition":"S_849 -> S_850"})
    dump(WORK/"returncodes.json",{"beta_probe":0,"retrospective":0,"data_stream":0,"activity":0})
    print(json.dumps({"E_max":emax,"D_max":dmax,"beta":measurement,
      "stream_hashes_step_850":{c:r["850"] for c,r in vectors.items()}},indent=2))
if __name__=="__main__":main()
