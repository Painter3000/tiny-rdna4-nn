#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1C_FP16_BACKWARD_001: run dX/dW capability before integration."""
import datetime,hashlib,json,pathlib,subprocess,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1];SRC=ROOT/"scripts/probe_phase3b1c_fp16_backward_gemm.cpp";OUT=ROOT/"phase3b1_reports/phase3b1c_backward_capability.json";MARKER="TCNN_RDNA4_P3B1C_FP16_BACKWARD_001"
TRANSITIONS=((16,16),(32,32),(64,64),(128,128),(16,64),(64,32),(128,32),(32,16),(32,128),(128,64),(64,16),(16,128));BATCHES=(1,16,128,1024,4096)
def ident(run):return{key:run.get("algorithm",{}).get(key) for key in ("index","kernel","solution","workspace")}
def valid(x):
 ls=x.get("launches",[]);return x.get("state")=="EXECUTED" and len(ls)==6 and all(y["status"]==y["sync"]==0 and y["memory"]["prefix"] and y["memory"]["suffix"] and not y["memory"]["unchanged"] for y in ls) and x["workspace_memory"]["prefix"] and x["workspace_memory"]["suffix"] and x["metrics"]["max_abs"]<=5e-4
def main():
 if subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()!="b59d569e5c662e8738f6af929e122c954ec68d7c":raise SystemExit("base precondition")
 with tempfile.TemporaryDirectory(prefix="p3b1c_cap_") as td:
  binary=pathlib.Path(td)/"probe";cmd=["/opt/rocm/bin/hipcc","-std=c++17","-O2","--offload-arch=gfx1201","-I/opt/rocm/include",f"-I{ROOT/'dependencies'}",str(SRC),"-L/opt/rocm/lib","-lhipblaslt","-lamdhip64","-o",str(binary)];cp=subprocess.run(cmd,text=True,capture_output=True)
  if cp.returncode:raise SystemExit(cp.stderr)
  cases=[]
  for inp,out in TRANSITIONS:
   for batch in BATCHES:
    for role,direction,m,n,k,types in (("dX","NN",inp,batch,out,("f16",)),("dW","NT",inp,out,batch,("f32","f16"))):
     for dtype in types:
      runs=[]
      for repeat in (1,2):
       p=subprocess.run([str(binary),direction,str(m),str(n),str(k),dtype,str(repeat),role],cwd=td,text=True,capture_output=True);runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
      cases.append({"input":inp,"output":out,"batch":batch,"role":role,"direction":direction,"d_type":dtype,"fresh_runs":runs,"algorithm_stable":ident(runs[0])==ident(runs[1]),"passed":valid(runs[0]) and valid(runs[1]) and ident(runs[0])==ident(runs[1])})
 failures=[x for x in cases if not x["passed"]];doc={"schema":1,"marker":MARKER,"generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),"contract":{"dX":{"direction":"NN","D":"FP16"},"dW":{"direction":"NT","primary_D":"FP32","characterization_D":"FP16"},"operands":"FP16","compute":"FP32"},"summary":{"cases":len(cases),"passed":len(cases)-len(failures),"failed":len(failures),"fresh_processes":2*len(cases)},"decision":"BACKWARD_GEMM_CAPABILITY_PASS" if not failures else "PHASE3B1C_BLOCKED","cases":cases};OUT.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n");print(doc["decision"]);return 0 if not failures else 1
if __name__=="__main__":raise SystemExit(main())
