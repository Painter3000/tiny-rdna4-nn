#!/usr/bin/env python3
"""Fail-closed Phase 3B1-C1 audit finalizer.

TCNN_RDNA4_P3B1C1_BACKWARD_AUDIT_HARDENING_001
"""
import argparse,copy,hashlib,json,pathlib,subprocess
R=pathlib.Path(__file__).resolve().parents[1];P=R/"phase3b1_reports"
BASE="fc2432d09a344624a8ecdf1cc0065d879fb5db31";MARKER="TCNN_RDNA4_P3B1C1_BACKWARD_AUDIT_HARDENING_001"
ENVELOPE={
 "baseline_commit":BASE,"policy":"prospective fixed envelope; historical 3B1-C gates unchanged",
 "dx":{"max_abs":1e-5,"max_rel_outside_near_zero":0.002,"normalized_l2":1e-4,"max_ulp_outside_near_zero":2},
 "dw":{"max_abs":0.001,"max_rel_outside_near_zero":0.025,"normalized_l2":0.0005,"max_ulp_outside_near_zero":28},
 "db":{"max_abs":0.002,"max_rel_outside_near_zero":0.02,"normalized_l2":0.00075,"max_ulp_outside_near_zero":32},
 "dz":{"max_abs":0.0,"max_rel_outside_near_zero":0.0,"normalized_l2":0.0,"max_ulp_outside_near_zero":0},
 "masks":{"direct_dz_mask_mismatches":0,"integrated_validation":"indirect_through_dx_dw_db_oracles"},
 "nonfinite":{"nan":0,"inf":0},
}
def load(path):
 try:return json.loads(path.read_text())
 except Exception:return None
def sha(data):return hashlib.sha256(data).hexdigest()
def gate(cap,raw,a1,a4,historical_ok,marker_ok,frozen_ok):
 checks={}
 checks["capability_decision"]=isinstance(cap,dict) and cap.get("decision")=="BACKWARD_GEMM_CAPABILITY_PASS"
 s=cap.get("summary") if isinstance(cap,dict) else None
 checks["capability_summary"]=isinstance(s,dict) and s.get("cases")==180 and s.get("passed")==180 and s.get("failed")==0 and s.get("fresh_processes")==360
 cc=cap.get("cases") if isinstance(cap,dict) else None
 checks["capability_all_cases"]=isinstance(cc,list) and len(cc)==180 and all(c.get("passed") is True for c in cc)
 checks["algorithms_stable"]=isinstance(cc,list) and len(cc)==180 and all(c.get("algorithm_stable") is True for c in cc)
 fresh=[f for c in cc for f in c.get("fresh_runs",[])] if isinstance(cc,list) else []
 checks["workspace_zero_only"]=len(fresh)==360 and all(isinstance(f.get("algorithm"),dict) and f["algorithm"].get("workspace")==0 for f in fresh)
 checks["phase3a1_exact_pass"]=isinstance(a1,dict) and a1.get("result")=="PASS"
 checks["phase3a4_exact_pass"]=isinstance(a4,dict) and a4.get("result")=="PASS"
 checks["functional_decision"]=isinstance(raw,dict) and raw.get("decision")=="PROCEED_TO_3B1D"
 checks["functional_counts"]=isinstance(raw,dict) and raw.get("functional_cases")==291 and raw.get("passed_cases")==291
 gm=raw.get("gradient_modes") if isinstance(raw,dict) else None
 checks["gradient_modes"]=isinstance(gm,dict) and all(gm.get(k) is True for k in ("overwrite_passed","accumulate_passed","ignore_passed","double_accumulate"))
 checks["multistream"]=isinstance(raw,dict) and isinstance(raw.get("multistream"),dict) and raw["multistream"].get("passed") is True
 checks["event_chain"]=isinstance(raw,dict) and isinstance(raw.get("event_chained_cross_stream"),dict) and raw["event_chained_cross_stream"].get("passed") is True
 checks["dynamic_batches"]=isinstance(raw,dict) and isinstance(raw.get("dynamic_batches"),dict) and raw["dynamic_batches"].get("passed") is True
 checks["direct_masks"]=isinstance(raw,dict) and raw.get("direct_dz_mask_mismatches")==0 and raw.get("integrated_relu_mask_validation")=="indirect_through_dx_dw_db_oracles"
 checks["range_tests"]=isinstance(raw,dict) and isinstance(raw.get("range_tests"),dict) and all(isinstance(v,dict) and v.get("passed") is True for v in raw["range_tests"].values()) and len(raw["range_tests"])==2
 maxima=raw.get("maxima") if isinstance(raw,dict) else None
 numeric=isinstance(maxima,dict)
 if numeric:
  for kind in ("dx","dw","db","dz"):
   metric=maxima.get(kind); numeric=numeric and isinstance(metric,dict) and all(metric.get(k,10**99)<=v for k,v in ENVELOPE[kind].items()) and metric.get("nan") == 0 and metric.get("inf") == 0
 checks["prospective_numeric_envelope"]=numeric
 checks["historical_reports_byte_identical"]=historical_ok is True
 checks["marker_audit"]=marker_ok is True
 checks["frozen_kernels_unchanged"]=frozen_ok is True
 return checks,all(checks.values())
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--capability",type=pathlib.Path,default=P/"phase3b1c_backward_capability.json");ap.add_argument("--functional",type=pathlib.Path,default=P/"phase3b1c1_backward_audit_hardening_raw.json");ap.add_argument("--phase3a1",type=pathlib.Path,default=P/"phase3b1c_fp32_regression.json/phase3a1_regression/phase3a1_validation.json");ap.add_argument("--phase3a4",type=pathlib.Path,default=P/"phase3b1c_fp32_regression.json/phase3a4_validation.json");ap.add_argument("--json-output",type=pathlib.Path,default=P/"phase3b1c1_backward_audit_hardening.json");ap.add_argument("--md-output",type=pathlib.Path,default=P/"PHASE3B1C1_BACKWARD_AUDIT_HARDENING.md");a=ap.parse_args()
 cap,raw,a1,a4=map(load,(a.capability,a.functional,a.phase3a1,a.phase3a4))
 tracked=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"phase3b1_reports"],cwd=R,text=True).splitlines();historic_paths=[x for x in tracked if "PHASE3B1C" in x or "phase3b1c" in x]
 historical={}
 for rel in historic_paths:
  old=subprocess.check_output(["git","show",f"{BASE}:{rel}"],cwd=R);now=(R/rel).read_bytes() if (R/rel).is_file() else b"";historical[rel]={"base_sha256":sha(old),"current_sha256":sha(now),"byte_identical":old==now}
 historical_ok=len(historical)>0 and all(v["byte_identical"] for v in historical.values())
 changed=subprocess.check_output(["git","diff","--name-only",BASE],cwd=R,text=True).splitlines()+subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=R,text=True).splitlines();sources=[x for x in changed if x.endswith((".cu",".cpp",".h",".py",".sh"))]
 marker_files={x:MARKER in (R/x).read_text(errors="replace") for x in sources};marker_ok=len(marker_files)>0 and all(marker_files.values())
 frozen=["src/hipblaslt_mlp.cu","src/hipblaslt_relu_biasgrad.cu","include/tiny-cuda-nn/networks/hipblaslt_mlp.h","src/hipblaslt_mlp_fp16.cu","src/hipblaslt_mlp_fp16_backward.cu"]
 frozen_diff=subprocess.check_output(["git","diff","--name-only",BASE,"--",*frozen],cwd=R,text=True).splitlines();frozen_ok=not frozen_diff
 checks,passed=gate(cap,raw,a1,a4,historical_ok,marker_ok,frozen_ok)
 mutations=[]
 for name,mutate in (("capability_decision_blocked",lambda c,r,x,y:c.__setitem__("decision","BLOCKED")),("phase3a1_result_removed",lambda c,r,x,y:x.pop("result",None)),("functional_case_count_reduced",lambda c,r,x,y:r.__setitem__("functional_cases",290))):
  mc,mr,m1,m4=map(copy.deepcopy,(cap,raw,a1,a4));mutate(mc,mr,m1,m4);mchecks,mpass=gate(mc,mr,m1,m4,historical_ok,marker_ok,frozen_ok);mutations.append({"name":name,"decision":"PROCEED_TO_3B1D_AUDITED" if mpass else "PHASE3B1C1_BLOCKED","passed":mpass is False,"failed_checks":[k for k,v in mchecks.items() if not v]})
 manipulation_ok=all(x["passed"] for x in mutations);decision="PROCEED_TO_3B1D_AUDITED" if passed and manipulation_ok else "PHASE3B1C1_BLOCKED"
 doc={"marker":MARKER,"base_commit":BASE,"decision":decision,"fail_closed_checks":checks,"manipulation_tests":mutations,"prospective_phase3b1d_regression_envelope":ENVELOPE,"relu_mask_statistics":{"direct_dz_mask_mismatches":raw.get("direct_dz_mask_mismatches") if isinstance(raw,dict) else None,"integrated_relu_mask_validation":raw.get("integrated_relu_mask_validation") if isinstance(raw,dict) else None},"scratch_semantics":{"counter_names":{"legacy_live":"estimated_backward_scratch_live_bytes","legacy_peak":"estimated_backward_scratch_peak_bytes"},"host_scope_based":True,"event_bound":False,"actual_async_allocator_peak":False,"multi_gpu_eligible":False,"estimate":raw.get("estimated_backward_scratch") if isinstance(raw,dict) else None,"pytorch_hip_observations":raw.get("dynamic_batches",{}).get("pytorch_hip_memory") if isinstance(raw,dict) else None},"range_tests":raw.get("range_tests") if isinstance(raw,dict) else None,"historical_reports":historical,"marker_audit":{"files":marker_files,"passed":marker_ok},"frozen_kernel_diff":frozen_diff,"source_inputs":{"capability":str(a.capability),"functional":str(a.functional),"phase3a1":str(a.phase3a1),"phase3a4":str(a.phase3a4)}}
 a.json_output.write_text(json.dumps(doc,indent=2)+"\n")
 lines=["# Phase 3B1-C1 – FP16 Backward Audit Hardening","",f"Marker: `{MARKER}`","",f"**Entscheidung: `{decision}`**","","## Fail-closed Abschlussgate","",f"- Pflichtchecks: {sum(checks.values())}/{len(checks)} bestanden.",f"- Manipulationstests: {sum(x['passed'] for x in mutations)}/{len(mutations)} lieferten jeweils `PHASE3B1C1_BLOCKED`.",f"- Historische Phase-3B1-C-Berichte bytegleich: `{historical_ok}`.",f"- Mathematische FP16-Backward-Produktion unverändert: `{frozen_ok}`.","","## Korrigierte Semantik","",f"- Direkte dZ-Maskenabweichungen: `{doc['relu_mask_statistics']['direct_dz_mask_mismatches']}`.","- Integrierte ReLU-Maskenvalidierung: `indirect_through_dx_dw_db_oracles`.","- Scratch-Zähler sind Host-Scope-Schätzungen, nicht eventgebunden, kein asynchrones Allocator-Peak und nicht Multi-GPU-fähig.","- Reale PyTorch/HIP allocated/reserved Beobachtungen vor Warm-up und nach beiden Folgen stehen im JSON.","- Range-Tests trennen Inf-Propagation/FP16-Unterlauf von echtem finite-to-FP16-Overflow; ein normaler Folgeaufruf bestand.","","## Prospektive Regression","",f"Die feste Hülle für Phase 3B1-D ist gegen `{BASE}` vorab im JSON dokumentiert. Historische 3B1-C-Gates wurden nicht verändert.",""]
 a.md_output.write_text("\n".join(lines));print(decision);return 0 if decision=="PROCEED_TO_3B1D_AUDITED" else 1
if __name__=="__main__":raise SystemExit(main())
