#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1C1A_FINAL_AUDIT_001 fail-closed final audit."""
import argparse,copy,hashlib,json,pathlib,subprocess
from finalize_phase3b1c1_audit import ENVELOPE
R=pathlib.Path(__file__).resolve().parents[1];P=R/"phase3b1_reports";BASE="37c55f52fb1518ef691ca39681184bc80040d6eb";MARKER="TCNN_RDNA4_P3B1C1A_FINAL_AUDIT_001"
def load(p):
 try:return json.loads(p.read_text())
 except Exception:return None
def sha(x):return hashlib.sha256(x).hexdigest()
def evaluate(cap,raw,a1,a4,historical_ok,marker_ok,production_ok):
 checks={};cs=cap.get("summary") if isinstance(cap,dict) else None;cc=cap.get("cases") if isinstance(cap,dict) else None
 checks["capability_decision"]=isinstance(cap,dict) and cap.get("decision")=="BACKWARD_GEMM_CAPABILITY_PASS"
 checks["capability_summary"]=isinstance(cs,dict) and (cs.get("cases"),cs.get("passed"),cs.get("failed"),cs.get("fresh_processes"))==(180,180,0,360)
 checks["capability_cases"]=isinstance(cc,list) and len(cc)==180 and all(c.get("passed") is True and c.get("algorithm_stable") is True for c in cc)
 fresh=[f for c in cc for f in c.get("fresh_runs",[])] if isinstance(cc,list) else [];checks["zero_workspace"]=len(fresh)==360 and all(isinstance(f.get("algorithm"),dict) and f["algorithm"].get("workspace")==0 for f in fresh)
 checks["phase3a1"]=isinstance(a1,dict) and a1.get("result")=="PASS";checks["phase3a4"]=isinstance(a4,dict) and a4.get("result")=="PASS"
 cases=raw.get("cases") if isinstance(raw,dict) else None;direct=raw.get("direct_dz_db_oracle") if isinstance(raw,dict) else None;supp=raw.get("supplemental_checks") if isinstance(raw,dict) else None
 checks["all_functional_case_entries"]=isinstance(cases,list) and len(cases)>0 and all(c.get("passed") is True for c in cases)
 checks["all_direct_dz_db_entries"]=isinstance(direct,list) and len(direct)>0 and all(c.get("passed") is True for c in direct)
 checks["all_supplemental_entries"]=isinstance(supp,list) and len(supp)>0 and all(c.get("passed") is True for c in supp)
 derived_total=sum(len(x) for x in (cases,direct,supp) if isinstance(x,list));derived_passed=sum(c.get("passed") is True for x in (cases,direct,supp) if isinstance(x,list) for c in x)
 checks["derived_functional_counts"]=isinstance(raw,dict) and raw.get("functional_cases")==derived_total and raw.get("passed_cases")==derived_passed and derived_total==derived_passed
 checks["functional_decision"]=isinstance(raw,dict) and raw.get("decision")=="PROCEED_TO_3B1D"
 gm=raw.get("gradient_modes") if isinstance(raw,dict) else None;checks["gradient_modes"]=isinstance(gm,dict) and all(gm.get(k) is True for k in ("overwrite_passed","accumulate_passed","ignore_passed","double_accumulate"))
 checks["multistream_event_dynamic"]=isinstance(raw,dict) and raw.get("multistream",{}).get("passed") is True and raw.get("event_chained_cross_stream",{}).get("passed") is True and raw.get("dynamic_batches",{}).get("passed") is True
 checks["mask_contract"]=isinstance(raw,dict) and raw.get("direct_dz_mask_mismatches")==0 and raw.get("integrated_relu_mask_validation")=="indirect_through_dx_dw_db_oracles"
 rt=raw.get("range_tests") if isinstance(raw,dict) else None;fo=rt.get("finite_to_fp16_parameter_gradient_overflow") if isinstance(rt,dict) else None
 checks["measured_overflow_followup"]=isinstance(fo,dict) and fo.get("post_overflow_followup_passed") is True and fo.get("memory_corruption_detected_by_followup") is False and fo.get("memory_corruption_detected_by_followup") is (not fo.get("post_overflow_followup_passed"))
 maxima=raw.get("maxima") if isinstance(raw,dict) else None;numeric=isinstance(maxima,dict)
 if numeric:
  for kind in ("dx","dw","db","dz"):
   m=maxima.get(kind);numeric=numeric and isinstance(m,dict) and all(m.get(k,float("inf"))<=v for k,v in ENVELOPE[kind].items()) and m.get("nan")==0 and m.get("inf")==0
 checks["prospective_numeric_envelope"]=numeric;checks["historical_artifacts"]=historical_ok is True;checks["marker_audit"]=marker_ok is True;checks["production_unchanged"]=production_ok is True
 return checks,all(checks.values()),{"functional_cases":derived_total,"passed_cases":derived_passed}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--functional",type=pathlib.Path,required=True);ap.add_argument("--capability",type=pathlib.Path,default=P/"phase3b1c_backward_capability.json");ap.add_argument("--phase3a1",type=pathlib.Path,default=P/"phase3b1c_fp32_regression.json/phase3a1_regression/phase3a1_validation.json");ap.add_argument("--phase3a4",type=pathlib.Path,default=P/"phase3b1c_fp32_regression.json/phase3a4_validation.json");a=ap.parse_args();cap,raw,a1,a4=map(load,(a.capability,a.functional,a.phase3a1,a.phase3a4))
 tracked=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"phase3b1_reports"],cwd=R,text=True).splitlines();historical_paths=[]
 for rel in tracked:
  scoped=rel.lower().removeprefix("phase3b1_reports/")
  if scoped.startswith(("phase3b1a","phase3b1b","phase3b1c")) and not scoped.startswith(("phase3b1c1","phase3b1c1a")):historical_paths.append(rel)
 historical={}
 for rel in historical_paths:
  old=subprocess.check_output(["git","show",f"{BASE}:{rel}"],cwd=R);now=(R/rel).read_bytes() if (R/rel).is_file() else b"";historical[rel]={"base_sha256":sha(old),"current_sha256":sha(now),"byte_identical":old==now}
 historical_ok=len(historical)>0 and all(v["byte_identical"] for v in historical.values())
 changed=subprocess.check_output(["git","diff","--name-only",BASE],cwd=R,text=True).splitlines()+subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=R,text=True).splitlines();sources=[x for x in changed if x.endswith((".py",".sh"))];marker_files={x:MARKER in (R/x).read_text(errors="replace") for x in sources};marker_ok=len(marker_files)>0 and all(marker_files.values())
 production_changed=[x for x in changed if x.startswith(("src/","include/","bindings/"))];production_ok=not production_changed
 checks,passed,counts=evaluate(cap,raw,a1,a4,historical_ok,marker_ok,production_ok)
 mutations=[]
 def mutate(name,fn):
  mc,mr,m1,m4=map(copy.deepcopy,(cap,raw,a1,a4));fn(mc,mr,m1,m4);ck,ok,_=evaluate(mc,mr,m1,m4,historical_ok,marker_ok,production_ok);mutations.append({"name":name,"decision":"PROCEED_TO_3B1D_AUDITED_FINAL" if ok else "PHASE3B1C1A_BLOCKED","passed":ok is False,"failed_checks":[k for k,v in ck.items() if not v]})
 mutate("capability_decision_blocked",lambda c,r,x,y:c.__setitem__("decision","BLOCKED"));mutate("phase3a1_result_removed",lambda c,r,x,y:x.pop("result",None));mutate("functional_count_reduced",lambda c,r,x,y:r.__setitem__("functional_cases",r["functional_cases"]-1));mutate("functional_case_passed_false",lambda c,r,x,y:r["cases"][0].__setitem__("passed",False))
 decision="PROCEED_TO_3B1D_AUDITED_FINAL" if passed and all(m["passed"] for m in mutations) else "PHASE3B1C1A_BLOCKED"
 doc={"marker":MARKER,"base_commit":BASE,"decision":decision,"derived_counts":counts,"fail_closed_checks":checks,"manipulation_tests":mutations,"historical_artifacts":historical,"historical_scope":"all Phase-3B1-A/B/B1/C artifacts; C1/C1a excluded","marker_audit":{"files":marker_files,"passed":marker_ok},"production_changed":production_changed,"range_tests":raw.get("range_tests") if isinstance(raw,dict) else None,"source_functional":str(a.functional)}
 jp=P/"phase3b1c1a_final_audit.json";mp=P/"PHASE3B1C1A_FINAL_AUDIT.md";jp.write_text(json.dumps(doc,indent=2)+"\n");mp.write_text("\n".join(["# Phase 3B1-C1a – Finales Audit-Hardening","",f"Marker: `{MARKER}`","",f"**Entscheidung: `{decision}`**","",f"- Abgeleitete Fälle: {counts['passed_cases']}/{counts['functional_cases']}",f"- Fail-closed Gates: {sum(checks.values())}/{len(checks)}",f"- Manipulationstests: {sum(m['passed'] for m in mutations)}/{len(mutations)}",f"- Historische A/B/B1/C-Artefakte bytegleich: `{historical_ok}`",f"- Produktionscode unverändert: `{production_ok}`",f"- Post-overflow-Folgeprüfung: `{raw.get('range_tests',{}).get('finite_to_fp16_parameter_gradient_overflow',{}).get('post_overflow_followup_passed')}`","","C1- und C1a-Ausgaben sind ausdrücklich aus der historischen Hashmenge ausgeschlossen.",""]));print(decision);return 0 if decision=="PROCEED_TO_3B1D_AUDITED_FINAL" else 1
if __name__=="__main__":raise SystemExit(main())
