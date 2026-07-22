#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1D1A_FINAL_TRAINING_AUDIT_001 fail-closed audit."""
import copy,hashlib,json,pathlib,subprocess
R=pathlib.Path(__file__).resolve().parents[1];P=R/"phase3b1_reports";BASE="ab2c1f6f28457688c9ba7736aedd156a9db09417";MARKER="TCNN_RDNA4_P3B1D1A_FINAL_TRAINING_AUDIT_001"
def load(x):return json.loads(pathlib.Path(x).read_text())
def sha(x):return hashlib.sha256(x).hexdigest()
rawp=pathlib.Path("/tmp/phase3b1d1_training_audit_raw.json");raw=load(rawp);d1=load(P/"phase3b1d1_training_audit_hardening.json")
tracked=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"phase3b1_reports"],cwd=R,text=True).splitlines();historical={}
for rel in tracked:
 scoped=rel.lower().removeprefix("phase3b1_reports/")
 if not scoped.startswith(("phase3b1a","phase3b1b","phase3b1c","phase3b1d")) or scoped.startswith("phase3b1d1a"):continue
 old=subprocess.check_output(["git","show",f"{BASE}:{rel}"],cwd=R);now=(R/rel).read_bytes() if (R/rel).is_file() else b"";historical[rel]={"base_sha256":sha(old),"current_sha256":sha(now),"byte_identical":old==now}
changed=subprocess.check_output(["git","diff","--name-only",BASE],cwd=R,text=True).splitlines()+subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=R,text=True).splitlines();production=[x for x in changed if x.startswith(("src/","include/","bindings/"))];sources=[x for x in changed if x.endswith((".py",".sh"))];markers={x:MARKER in (R/x).read_text(errors="replace") for x in sources}
def evaluate(x):
 checks={};under=x.get("range_tests",{}).get("underflow_rescue",{});over=x.get("range_tests",{}).get("finite_to_fp16_overflow",{});event=x.get("event_chain_training",{});static=x.get("static_checkpoint_resume",{});rng=x.get("rng_resume",{})
 checks["independent_underflow_reference"]=under.get("passed") is True and under.get("reference_independent") is True and under.get("rescued_positions",0)>0 and under.get("max_abs_vs_independent_quantized_reference",float("inf"))<=2e-3
 checks["honest_fp32_proxy"]=over.get("passed") is True and over.get("actual_internal_fp32_gradient_observed") is False and over.get("external_fp32_finiteness_proxy") is True and all(over.get(k) is True for k in ("inputs_finite","unscaled_loss_finite","scaled_loss_finite","native_fp16_gradient_nonfinite","step_skipped","recovery_passed"))
 checks["event_chain_fail_closed"]=event.get("passed") is True and event.get("parameter_max_abs",float("inf"))<=2e-5 and event.get("loss_max_abs",float("inf"))<=2e-6 and event.get("optimizer_state_equal") is True and event.get("counters_before")==event.get("counters_after") and event.get("rounds")==64 and event.get("terminal_sync_only") is True
 checks["static_resume"]=static.get("passed") is True and static.get("checkpoint_metadata",{}).get("static_loss_scale")==128.0 and static.get("native_backward_active_scale")==128.0
 checks["rng_resume"]=rng.get("passed") is True
 checks["raw_decision"]=x.get("decision")=="AUDIT_TEST_PASS"
 checks["prior_d1_gate"]=d1.get("decision")=="PROCEED_TO_3B1E_AUDITED_FINAL"
 checks["historical_reports"]=len(historical)>0 and all(v["byte_identical"] for v in historical.values());checks["production_unchanged"]=not production;checks["marker_audit"]=len(markers)>0 and all(markers.values())
 return checks,all(checks.values())
checks,passed=evaluate(raw);mutations=[]
def mutate(name,fn):
 x=copy.deepcopy(raw);fn(x);ck,ok=evaluate(x);mutations.append({"name":name,"decision":"PROCEED_TO_3B1E_AUDITED_FINAL" if ok else "PHASE3B1D1A_BLOCKED","passed":ok is False,"failed_checks":[k for k,v in ck.items() if not v]})
mutate("event_parameter_max_abs_over_limit",lambda x:x["event_chain_training"].__setitem__("parameter_max_abs",2.1e-5));mutate("event_optimizer_state_not_equal",lambda x:x["event_chain_training"].__setitem__("optimizer_state_equal",False));mutate("independent_underflow_reference_disabled",lambda x:x["range_tests"]["underflow_rescue"].__setitem__("reference_independent",False))
decision="PROCEED_TO_3B1E_AUDITED_FINAL" if passed and all(m["passed"] for m in mutations) else "PHASE3B1D1A_BLOCKED";rb=rawp.read_bytes();doc={"marker":MARKER,"base_commit":BASE,"decision":decision,"gates":checks,"manipulation_tests":mutations,"underflow_rescue":raw["range_tests"]["underflow_rescue"],"finite_to_fp16_overflow":raw["range_tests"]["finite_to_fp16_overflow"],"event_chain_training":raw["event_chain_training"],"static_checkpoint_resume":raw["static_checkpoint_resume"],"rng_resume":raw["rng_resume"],"raw_data":{"absolute_path":str(rawp),"size_bytes":len(rb),"sha256":sha(rb)},"historical_reports":historical,"production_changed":production,"marker_audit":markers}
(P/"phase3b1d1a_final_training_audit.json").write_text(json.dumps(doc,indent=2)+"\n");lines=["# Phase 3B1-D1a – Finales Training-Audit","",f"Marker: `{MARKER}`","",f"**Entscheidung: `{decision}`**","",f"- Unabhängige quantisierte Unterlaufreferenz: `{checks['independent_underflow_reference']}`; gerettete Positionen: {doc['underflow_rescue'].get('rescued_positions')}; max_abs: {doc['underflow_rescue'].get('max_abs_vs_independent_quantized_reference')}.","- Interner FP32-Scratch wurde nicht direkt beobachtet; externe FP32-Endlichkeit ist ausdrücklich nur Proxy.",f"- Event-Kette mit expliziten Grenzwerten: `{checks['event_chain_fail_closed']}`.",f"- Manipulationstests: {sum(m['passed'] for m in mutations)}/{len(mutations)} blockierten fail-closed.",f"- Historische Reports bytegleich: `{checks['historical_reports']}`; Produktionscode unverändert: `{checks['production_unchanged']}`.",""]
(P/"PHASE3B1D1A_FINAL_TRAINING_AUDIT.md").write_text("\n".join(lines));print(decision)
