#!/usr/bin/env python3
"""Fail-closed finalizer for TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001."""
import copy, hashlib, json, pathlib, subprocess
R=pathlib.Path(__file__).resolve().parents[1];RAW=pathlib.Path("/tmp/phase3b1e1a_final_encoding_audit_raw.json")
OUT=R/"phase3b1_reports/phase3b1e1a_final_encoding_audit.json";MD=R/"phase3b1_reports/PHASE3B1E1A_FINAL_ENCODING_AUDIT.md"
BASE="de2ce23ec7070aa7e5546b1942e80806bce17cb1";MARKER="TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001"
def blocked(x):
 p=x["padding_contracts"];runs=x["training_runs"];strong=x["collision_proofs"]["strong"];low=(x["collision_proofs"]["low_3d"],x["collision_proofs"]["low_2d"]);rs=x["checkpoint_resume"];mx=x["numerical_maxima"]
 checks={"raw":x.get("decision")=="E1A_RAW_PASS","standalone_padding":all(p["standalone"].get(k)==1. for k in ("Identity","Frequency","OneBlob")),
 "fp32_padding":all(q.get("value")==1. for q in p["fp32_network_with_encoding"]),"fp16_padding":all(q.get("value")==0. for q in p["fp16_network_with_encoding"]) and all(q.get("passed") is True for q in p["fp16_zero_execution"]),
 "hashgrid_unchanged":p.get("hashgrid_unchanged") is True,"matrix":x.get("functional_cases")==204 and x.get("functional_passed")==204 and all(q.get("passed") is True for q in x["functional_matrix"]),
 "collision_strong":strong.get("collision_classification")=="collision-strong" and strong.get("collision_count",0)>0 and strong.get("maximum_bucket_occupancy",1)>1 and strong.get("collision_witness") is not None,
 "collision_low":all(q.get("collision_classification")=="collision-low" and q.get("collision_count",1)==0 for q in low),
 "dynamic_scale":runs[2].get("scaling")=="dynamic" and runs[2].get("initial_scale",1)>1 and runs[2].get("scale_change_count",0)>0,
 "dynamic_overflow":runs[2].get("overflow_count",0)>0 and runs[2].get("skip_count")==runs[2].get("overflow_count") and runs[2].get("recovery_count",0)>0,
 "training":len(runs)==4 and sum(q.get("steps",0) for q in runs)==3200 and all(q.get("passed") is True for q in runs) and x.get("validated_training_steps",0)>=7600,
 "resume":len(rs)==4 and all(q.get("passed") is True and q.get("cpu_rng_equal") is True and q.get("cuda_all_rng_equal") is True and q.get("custom_generator_rng_equal") is True for q in rs),
 "events":len(x["event_chains"])==4 and all(q.get("passed") is True for q in x["event_chains"]),
 "maxima":set(mx)=={"output","dinput","network_gradient","encoding_gradient"} and all(all(k in q for k in ("encoding","dims","variant","batch","interpolation","n_levels","n_features_per_level","absolute_error","normalized_l2","max_relative_outside_near_zero","reference_norm")) for q in mx.values()),
 "historical_dinput_max":x.get("historical_e1_dinput_maximum_attribution",{}).get("absolute_error")==0.02607070654630661}
 return checks
d=json.loads(RAW.read_text());checks=blocked(d);mut=[]
def m(name,fn):
 q=copy.deepcopy(d);fn(q);passed=not all(blocked(q).values());mut.append({"name":name,"decision":"PHASE3B1E1A_BLOCKED" if passed else "INVALID_PASS","passed":passed})
m("standalone_padding_globally_zero",lambda x:x["padding_contracts"]["standalone"].update({"Identity":0.,"Frequency":0.,"OneBlob":0.}))
m("fp16_padding_nonzero",lambda x:x["padding_contracts"]["fp16_network_with_encoding"][0].update({"value":1.}))
m("collision_strong_without_collision",lambda x:x["collision_proofs"]["strong"].update({"collision_count":0,"collision_witness":None}))
m("collision_low_rate_invalid",lambda x:x["collision_proofs"]["low_3d"].update({"collision_count":1}))
m("dynamic_without_scale_change",lambda x:x["training_runs"][2].update({"scale_change_count":0}))
m("dynamic_without_overflow_skip_recovery",lambda x:x["training_runs"][2].update({"overflow_count":0,"skip_count":0,"recovery_count":0}))
m("cpu_rng_mismatch",lambda x:x["checkpoint_resume"][0].update({"cpu_rng_equal":False}))
m("cuda_all_rng_mismatch",lambda x:x["checkpoint_resume"][0].update({"cuda_all_rng_equal":False}))
m("custom_generator_rng_mismatch",lambda x:x["checkpoint_resume"][0].update({"custom_generator_rng_equal":False}))
m("maximum_without_attribution",lambda x:x["numerical_maxima"]["dinput"].pop("encoding"))
history={}
for p in subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"phase3b1_reports"],cwd=R,text=True).splitlines():
 if pathlib.Path(p).name.startswith(("PHASE3B1A","PHASE3B1B","PHASE3B1C","PHASE3B1D","PHASE3B1E","phase3b1a","phase3b1b","phase3b1c","phase3b1d","phase3b1e")):
  old=subprocess.check_output(["git","show",f"{BASE}:{p}"],cwd=R);now=(R/p).read_bytes();history[p]={"sha256":hashlib.sha256(now).hexdigest(),"byte_equal":now==old}
allowed={"bindings/torch/tinycudann/bindings.cpp","include/tiny-cuda-nn/encoding.h","include/tiny-cuda-nn/network_with_input_encoding.h","include/tiny-cuda-nn/encodings/identity.h","include/tiny-cuda-nn/encodings/frequency.h","include/tiny-cuda-nn/encodings/oneblob.h"}
changed=set(subprocess.check_output(["git","diff","--name-only",BASE],cwd=R,text=True).splitlines());production={p for p in changed if p.startswith(("src/","include/","bindings/"))}
checks["historical_reports"]=bool(history) and all(q["byte_equal"] for q in history.values());checks["production_scope"]=production<=allowed and not any(p.startswith("src/") for p in production);checks["manipulations"]=all(q["passed"] for q in mut)
regression_files={"phase3a1":pathlib.Path("/tmp/phase3b1e1a_regression_phase3a4/phase3a1_regression/phase3a1_validation.json"),"phase3a4":pathlib.Path("/tmp/phase3b1e1a_regression_phase3a4/phase3a4_validation.json"),"b":pathlib.Path("/tmp/e1a_regression_b.json"),"b1":pathlib.Path("/tmp/e1a_regression_b1.json"),"c":pathlib.Path("/tmp/e1a_regression_c.json"),"d1":pathlib.Path("/tmp/e1a_regression_d1.json"),"e":pathlib.Path("/tmp/e1a_regression_e.json"),"e1":pathlib.Path("/tmp/e1a_regression_e1_seeded.json")}
reg={k:json.loads(p.read_text()) if p.exists() else {} for k,p in regression_files.items()}
regression_status={"phase1_encoding_via_phase3a1":reg["phase3a1"].get("result")=="PASS" and len(reg["phase3a1"].get("encodings",[]))==4,"phase3a1":reg["phase3a1"].get("result")=="PASS","phase3a4":reg["phase3a4"].get("result")=="PASS","phase3b1b_historical_forward_only":reg["b"].get("decision"),"phase3b1b1":reg["b1"].get("functional_decision")=="HARDENING_FUNCTIONAL_PASS","phase3b1c":reg["c"].get("decision")=="PROCEED_TO_3B1D","phase3b1d1":reg["d1"].get("decision")=="AUDIT_TEST_PASS","phase3b1e":reg["e"].get("decision")=="RAW_PASS" and reg["e"].get("actual_case_count")==reg["e"].get("passed_case_count"),"phase3b1e1":reg["e1"].get("decision")=="CLOSURE_RAW_PASS" and reg["e1"].get("functional_case_count")==reg["e1"].get("functional_passed_count")}
checks["regressions"]=all(v is True for k,v in regression_status.items() if k!="phase3b1b_historical_forward_only")
decision="PROCEED_TO_3B1F_FP16_PERFORMANCE" if all(checks.values()) else "PHASE3B1E1A_BLOCKED"
rb=RAW.read_bytes();out={"marker":MARKER,"base_commit":BASE,"decision":decision,"gates":checks,"blocking_gates":[k for k,v in checks.items() if not v],"manipulation_tests":mut,"padding_contracts":d["padding_contracts"],"collision_proofs":d["collision_proofs"],"training_runs":d["training_runs"],"validated_training_steps":d["validated_training_steps"],"checkpoint_resume":d["checkpoint_resume"],"event_chains":d["event_chains"],"functional_cases":d["functional_cases"],"functional_passed":d["functional_passed"],"numerical_maxima":d["numerical_maxima"],"historical_e1_dinput_maximum_attribution":d["historical_e1_dinput_maximum_attribution"],"overflow_field_contract":d["encoding_overflow_field_contract"],"historical_reports":history,"production_files":sorted(production),"raw_data":{"absolute_path":str(RAW),"size_bytes":len(rb),"sha256":hashlib.sha256(rb).hexdigest()},"environment":d["environment"]}
out["regressions"]=regression_status
OUT.write_text(json.dumps(out,indent=2)+"\n");MD.write_text(f"""# Phase 3B1-E1a – Final Encoding Contract Hardening

Marker: `{MARKER}`

- Entscheidung: `{decision}`
- Funktionale Matrix: {d['functional_passed']}/{d['functional_cases']}
- Padding: standalone/FP32 = 1, qualifizierter FP16-Pfad = 0
- Kollisionsbeweise: strong={d['collision_proofs']['strong']['collision_count']}, low-3D={d['collision_proofs']['low_3d']['collision_count']}, low-2D={d['collision_proofs']['low_2d']['collision_count']}
- Training: {d['validated_training_steps']} validierte Schritte ({sum(x['steps'] for x in d['training_runs'])} neu)
- Fresh-Process-Resume: {sum(x['passed'] for x in d['checkpoint_resume'])}/4
- Event-Ketten: {sum(x['passed'] for x in d['event_chains'])}/4
- Manipulationstests: {sum(x['passed'] for x in mut)}/{len(mut)}
- Historische Reports bytegleich: {checks['historical_reports']}
- Regressionen Phase 1/3A1/3A4/B1/C/D1/E/E1: {checks['regressions']}
- Produktionsumfang: ausschließlich backendgebundener Paddingvertrag und testseitiger Binding-Audit-Hook
""")
print(decision);raise SystemExit(0 if decision.startswith("PROCEED") else 1)
