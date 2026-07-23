#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001 fail-closed finalizer."""
import copy, hashlib, json, pathlib, subprocess
R=pathlib.Path(__file__).resolve().parents[1];P=R/"phase3b1_reports";RAW=pathlib.Path("/tmp/phase3b1e_fp16_network_with_encoding_raw.json")
BASE="965112dc3df7754d31e645eb7c56c8a2498c80d8";MARKER="TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001"
def sha(b):return hashlib.sha256(b).hexdigest()
def evaluate(d,historical_ok=True):
 t=d.get("tolerances_frozen_before_final_run",{});cases=d.get("cases",[]);hg=d.get("hashgrid_backward",{});cp=d.get("checkpoint_resume",[]);streams=d.get("multistream_event_chain",[])
 checks={
  "raw_pass":d.get("decision")=="RAW_PASS","case_counts":len(cases)>0 and d.get("actual_case_count")==len(cases) and d.get("passed_case_count")==len(cases),
  "individual_cases":all(c.get("passed") is True and c.get("output_dtype")=="torch.float16" and c.get("dinput_dtype")=="torch.float32" and c.get("gradient_dtype")=="torch.float32" for c in cases),
  "padding":all(c.get("padding_zero") is True and c.get("padding_gradient_zero") is True and c.get("alignment")==16 for c in cases),
  "offsets":all(c.get("ranges_disjoint") is True and c.get("network_range",[1,0])[1]<=c.get("encoding_range",[0])[0] for c in cases),
  "gradient_modes":len(d.get("gradient_modes",[]))==4 and all(x.get("passed") is True for x in d.get("gradient_modes",[])),
  "accumulation":len(d.get("gradient_accumulation",[]))==4 and all(x.get("passed") is True and x.get("max_abs",999)<=t.get("accumulation_max_abs",-1) for x in d.get("gradient_accumulation",[])),
  "hashgrid":hg.get("passed") is True and hg.get("collision_low",{}).get("passed") is True and hg.get("collision_strong",{}).get("passed") is True and hg.get("scratch_dtype")=="FP32" and hg.get("fp16_atomic_path_active") is False and hg.get("final_fp32_to_fp16_conversions")==1 and hg.get("scratch_live_after")==0,
  "scaling":d.get("loss_scaling",{}).get("passed") is True and d.get("loss_scaling",{}).get("overflow",{}).get("gradient_nonfinite") is True and d.get("loss_scaling",{}).get("overflow",{}).get("step_skipped") is True,
  "training":len(d.get("training_runs",[]))==7 and d.get("training_steps",0)>=4400 and all(x.get("passed") is True and x.get("end_loss",999)<x.get("start_loss",0)*t.get("training_loss_ratio",0) for x in d.get("training_runs",[])),
  "checkpoint":len(cp)==4 and all(x.get("passed") is True and x.get("encoding_configuration_present") is True and x.get("params_bit_identical") is True and x.get("rng_sequence_bit_identical") is True and x.get("fresh_process") is True for x in cp),
  "streams":len(streams)==4 and all(x.get("passed") is True and x.get("rounds")==64 and x.get("terminal_sync_only") is True and x.get("counters_before")==x.get("counters_after") for x in streams),
  "fresh":d.get("fresh_process_matrix",{}).get("passed") is True and d.get("fresh_process_matrix",{}).get("count")==2,
  "independent_encoding_oracles":len(d.get("independent_encoding_oracles",[]))==4 and all(x.get("passed") is True and x.get("independent") is True for x in d.get("independent_encoding_oracles",[])),
  "direct_hashgrid_parameter_oracle":hg.get("direct_parameter_gradient_oracle",{}).get("passed") is True and hg.get("direct_parameter_gradient_oracle",{}).get("independent") is True,
  "fresh_process_resume":len(cp)==4 and all(x.get("fresh_process_verified") is True for x in cp),
  "true_event_chain":len(streams)==4 and all(x.get("forward_event_backward_chain_verified") is True for x in streams),
  "required_training_matrix":d.get("training_steps",0)>=7600 and {("Identity","SGD",200),("Identity","Adam",200),("Frequency","Adam",500),("OneBlob","Adam",500),("HashGrid","Adam",1000)}.issubset({(x.get("encoding"),x.get("optimizer"),x.get("steps")) for x in d.get("training_runs",[])}),
  "regressions":d.get("regressions",{}).get("baseline")=="BASELINE_PASS" and d.get("regressions",{}).get("phase3b1_c1a",{}).get("passed")==296 and d.get("regressions",{}).get("phase3b1_d1a")=="AUDIT_TEST_PASS",
  "historical_reports":historical_ok,
 }
 return checks,all(checks.values())
def main():
 d=json.loads(RAW.read_text());tracked=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"phase3b1_reports"],cwd=R,text=True).splitlines();historical={}
 for rel in tracked:
  old=subprocess.check_output(["git","show",f"{BASE}:{rel}"],cwd=R);now=(R/rel).read_bytes() if (R/rel).is_file() else b"";historical[rel]={"base_sha256":sha(old),"current_sha256":sha(now),"byte_identical":old==now}
 hist_ok=bool(historical) and all(v["byte_identical"] for v in historical.values());checks,ok=evaluate(d,hist_ok);mut=[]
 def mutation(name,fn,h=hist_ok):
  x=copy.deepcopy(d);fn(x);_,v=evaluate(x,h);mut.append({"name":name,"decision":"PROCEED_TO_3B1F_FP16_PERFORMANCE" if v else "PHASE3B1E_BLOCKED","passed":not v})
 mutation("encoding_output_outside_tolerance",lambda x:x["cases"][0].__setitem__("passed",False));mutation("padding_nonzero",lambda x:x["cases"][0].__setitem__("padding_zero",False));mutation("parameter_ranges_overlap",lambda x:x["cases"][0].__setitem__("encoding_range",[0,1]));mutation("hash_collision_gradient_wrong",lambda x:x["hashgrid_backward"]["collision_strong"].__setitem__("passed",False));mutation("scratch_not_fp32",lambda x:x["hashgrid_backward"].__setitem__("scratch_dtype","FP16"));mutation("fp16_atomic_active",lambda x:x["hashgrid_backward"].__setitem__("fp16_atomic_path_active",True));mutation("overflow_step_executed",lambda x:x["loss_scaling"]["overflow"].__setitem__("step_skipped",False));mutation("checkpoint_missing_encoding",lambda x:x["checkpoint_resume"][0].__setitem__("encoding_configuration_present",False));mutation("resume_not_bitidentical",lambda x:x["checkpoint_resume"][0].__setitem__("params_bit_identical",False));mutation("event_counter_growth",lambda x:x["multistream_event_chain"][0]["counters_after"].__setitem__("cache_misses",999999));mutation("case_removed",lambda x:x["cases"].pop());mutation("historical_report_changed",lambda x:None,False)
 decision="PROCEED_TO_3B1F_FP16_PERFORMANCE" if ok and all(x["passed"] for x in mut) else "PHASE3B1E_BLOCKED";rb=RAW.read_bytes();changed=subprocess.check_output(["git","diff","--name-only",BASE],cwd=R,text=True).splitlines()+subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=R,text=True).splitlines();prod=[x for x in changed if x.startswith(("src/","include/","bindings/"))]
 out={"marker":MARKER,"base_commit":BASE,"decision":decision,"gates":checks,"blocking_gates":[k for k,v in checks.items() if not v],"manipulation_tests":mut,"actual_case_count":d["actual_case_count"],"passed_case_count":d["passed_case_count"],"fresh_process_count":d.get("fresh_process_matrix",{}).get("count",0),"checkpoint_protocols_not_fresh":len(d.get("checkpoint_resume",[])),"training_steps":d["training_steps"],"results_by_encoding":{n:{"cases":sum(c["encoding"]==n for c in d["cases"]),"passed":all(c["passed"] for c in d["cases"] if c["encoding"]==n)} for n in ("Identity","Frequency","OneBlob","HashGrid")},"hashgrid_backward":d["hashgrid_backward"],"loss_scaling":d["loss_scaling"],"checkpoint_resume":d["checkpoint_resume"],"multistream_event_chain":d["multistream_event_chain"],"training_runs":d["training_runs"],"regressions":d["regressions"],"historical_reports":historical,"production_changes":prod,"raw_data":{"absolute_path":str(RAW),"size_bytes":len(rb),"sha256":sha(rb)},"environment":d["environment"],"final_counters":d["final_counters"]}
 P.mkdir(exist_ok=True);(P/"phase3b1e_fp16_network_with_encoding.json").write_text(json.dumps(out,indent=2)+"\n")
 blockers=", ".join(out["blocking_gates"]) or "keine"
 (P/"PHASE3B1E_FP16_NETWORK_WITH_ENCODING.md").write_text(f"# Phase 3B1-E – FP16 NetworkWithInputEncoding\n\nMarker: `{MARKER}`\n\n**Entscheidung: `{decision}`**\n\n- Fälle: {out['passed_case_count']}/{out['actual_case_count']}\n- Fresh Processes: {out['fresh_process_count']}\n- Trainingsschritte: {out['training_steps']}\n- Fail-closed Manipulationen: {sum(x['passed'] for x in mut)}/{len(mut)}\n- Blockierende Gates: `{blockers}`\n- Rohdaten: `{RAW}` (`{out['raw_data']['sha256']}`)\n")
 (P/"PHASE3B1E_INTEGRATION_CONTRACT.md").write_text(f"# Phase 3B1-E – Integrationsvertrag\n\nMarker: `{MARKER}`\n\nFP32 Input → FP16 Encoding-Aktivierung → FP16 HipBLASLtMLP → FP32 externes dInput und FP32 Python-Gradienten/Master/Optimizer-State. Native MLP- und Encodingparameter sind FP16. Netzwerkparameter liegen zuerst, Encodingparameter danach. Encoding-Ausgaben werden auf die kleinste unterstützte Breite 16/32/64/128 gepolstert und für den expliziten FP16-Backendpfad contiguous ColumnMajor verkettet.\n")
 (P/"PHASE3B1E_HASHGRID_BACKWARD_AUDIT.md").write_text(f"# Phase 3B1-E – HashGrid Backward Audit\n\nMarker: `{MARKER}`\n\n- FP16-Atomics aktiv: `{out['hashgrid_backward']['fp16_atomic_path_active']}`\n- Scratch: `{out['hashgrid_backward']['scratch_dtype']}`, {out['hashgrid_backward']['scratch_size_bytes']} Byte pro geprüftem Modell.\n- Lebensdauer: {out['hashgrid_backward']['scratch_lifetime']}.\n- Finale FP32→FP16-Konvertierungen pro Backward: {out['hashgrid_backward']['final_fp32_to_fp16_conversions']}.\n- Entscheidung: `{decision}`.\n")
 print(decision);return 0 if decision.startswith("PROCEED") else 1
if __name__=="__main__":raise SystemExit(main())
