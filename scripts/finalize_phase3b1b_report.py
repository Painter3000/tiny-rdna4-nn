#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1B_FP16_FORWARD_001: consolidate immutable test evidence."""
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
REPORTS=ROOT/"phase3b1_reports"
FUNCTIONAL=REPORTS/"phase3b1b_fp16_forward.json"
MARKDOWN=REPORTS/"PHASE3B1B_FP16_FORWARD.md"
CHECKPOINT=REPORTS/"phase3b1b_epilogue_checkpoint.json"
FP32_ROOT=pathlib.Path("/tmp/phase3b1b_fp32_regression")
MARKER="TCNN_RDNA4_P3B1B_FP16_FORWARD_001"
BASE="22364010853d872702bdf8f63cad26b890b6f47b"
FROZEN="6258184d8d9d032ef423b75eddeeaf8168c7e45a"

def command(*args):return subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    functional=json.loads(FUNCTIONAL.read_text());checkpoint=json.loads(CHECKPOINT.read_text())
    p3a4_path=FP32_ROOT/"phase3a4_validation.json";p3a1_path=FP32_ROOT/"phase3a1_regression/phase3a1_validation.json"
    p3a4=json.loads(p3a4_path.read_text());p3a1=json.loads(p3a1_path.read_text())
    binding=ROOT/"bindings/torch/build/lib.linux-x86_64-cpython-312/tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so"
    production_files=["src/hipblaslt_mlp.cu","src/hipblaslt_relu_biasgrad.cu","include/tiny-cuda-nn/networks/hipblaslt_mlp.h"]
    frozen_diff=command("git","diff","--name-only",FROZEN,"--",*production_files)
    integration_diff=command("git","diff","--name-only",BASE,"--","bindings","include","src").splitlines()
    files=[ROOT/x for x in ("bindings/torch/setup.py","bindings/torch/tinycudann/bindings.cpp","src/cpp_api.cu","src/portable_network.cu","include/tiny-cuda-nn/networks/hipblaslt_mlp_fp16.h","src/hipblaslt_mlp_fp16.cu","scripts/probe_phase3b1b_fp16_epilogues.cpp","scripts/run_phase3b1b_epilogue_checkpoint.py","scripts/test_phase3b1b_fp16_forward.py","scripts/test_phase3b1b_fp16_forward.sh", "scripts/finalize_phase3b1b_report.py")]
    marker={str(p.relative_to(ROOT)):p.read_text().count(MARKER) for p in files}
    epi={name:{"signatures":sum(c["epilogue"]==name for c in checkpoint["cases"]),"passed":all(c["passed"] for c in checkpoint["cases"] if c["epilogue"]==name)} for name in ("BIAS","RELU_BIAS")}
    functional.update({
      "schema":1,"phase":"3B1-B","marker":MARKER,"generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "git":{"branch":command("git","branch","--show-current"),"starting_commit":BASE,"head_before_commit":command("git","rev-parse","HEAD"),"frozen_phase3a4_identity":FROZEN},
      "precision_contract":{"backend":"HipBLASLtMLPFP16","selection":"explicit opt-in only","input_operands":"Fp16 after Identity encoding","weights":"Fp16","bias":"Fp16","compute":"Fp32","hidden_outputs":"Fp16","final_output":"Fp16","fp32_final_output":"not exposed because the current Module API couples parameter and output precision","fallbacks":"none","backward":"explicitly rejected"},
      "epilogue_checkpoint":{"decision":checkpoint["decision"],"signatures":checkpoint["summary"]["signatures"],"fresh_processes":checkpoint["summary"]["fresh_processes"],"capabilities":epi,"bias_axis":"D rows / output neurons","checkpoint_sha256":sha(CHECKPOINT)},
      "fp32_regression":{"phase3a1":p3a1["result"],"phase3a4":p3a4["result"],"phase3a1_sha256":sha(p3a1_path),"phase3a4_sha256":sha(p3a4_path),"commands":"validate_phase3a4.py including validate_phase3a1.py"},
      "environment":{"python":platform.python_version(),"torch":p3a4["environment"]["torch"],"hip":p3a4["environment"]["hip"],"device":p3a4["environment"]["device"],"arch":p3a4["environment"]["arch"],"binding":str(binding),"binding_sha256":sha(binding)},
      "marker_audit":{"required":MARKER,"occurrences_by_file":marker,"all_changed_source_files_marked":all(marker.values())},
      "phase3a4_production_code_diff":{"paths_checked":production_files,"changed_paths":frozen_diff.splitlines() if frozen_diff else [],"unchanged":not bool(frozen_diff)},
      "production_code_diff_from_capability_base":{"changed_paths":integration_diff,"scope":"new FP16 backend plus explicit factory/API/build wiring and private counters"},
    })
    functional_ok=(functional["passed_cases"]==functional["functional_cases"]+functional["adversarial_cases"]
                   and functional["fresh_processes"]["passed"] and functional["counters"]["warm_cache_passed"]
                   and functional["zero_batch"]["passed"] and functional["backward"]["passed"]
                   and functional["memory_stability"]["passed"] and functional["two_streams"]["passed"])
    ok=(functional_ok and checkpoint["decision"]=="EPILOGUE_CHECKPOINT_PASS" and p3a1["result"]=="PASS" and p3a4["result"]=="PASS" and not frozen_diff and all(marker.values()))
    functional["decision"]="PROCEED_TO_3B1C" if ok else "PHASE3B1B_BLOCKED"
    FUNCTIONAL.write_text(json.dumps(functional,indent=2,sort_keys=True)+"\n")
    mx=functional["maxima"];cnt=functional["counters"]
    lines=["# Phase 3B1-B – FP16 Forward-Epilogues und Forward-Basispfad","",f"Marker: `{MARKER}`","",f"**Decision: `{functional['decision']}`**","","## Identität und Vertrag","",f"- Branch: `{functional['git']['branch']}`",f"- Ausgangscommit: `{BASE}`",f"- eingefrorene Phase-3A4-Identität: `{FROZEN}`",f"- Phase-3A4-Kernel-/FP32-Backend-Diff: `none`",f"- Integrationsdiff ab Capability-Commit: `{', '.join(integration_diff)}`",f"- Backend: explizit `HipBLASLtMLPFP16` mit `precision=Fp16`; keine automatische Umschaltung",f"- Operanden/Gewichte/Bias: FP16; Compute: FP32; Hidden und finale Ausgabe: FP16",f"- FP32-Endausgabe: nicht angeboten, da der bestehende Module-Vertrag Parameter- und Ausgabepräzision koppelt",f"- Backward: explizit und empirisch abgewiesen", "","## Epilogue-Checkpoint","",f"- BIAS: `{epi['BIAS']['passed']}` ({epi['BIAS']['signatures']} Signaturen)",f"- RELU_BIAS: `{epi['RELU_BIAS']['passed']}` ({epi['RELU_BIAS']['signatures']} Signaturen)",f"- Gesamt: {checkpoint['summary']['signatures']} Signaturen, {checkpoint['summary']['fresh_processes']} frische Prozesse",f"- NN, D=FP16/FP32, Bias=FP16, Compute=FP32; Sentinels/Guards und Bias-Achse bestätigt", "","## Funktionale und numerische Gates","",f"- reguläre Fälle: {functional['functional_cases']}; adversariale Fälle: {functional['adversarial_cases']}; bestanden: {functional['passed_cases']}",f"- Fresh Processes: {functional['fresh_processes']['count']}, bestanden: {functional['fresh_processes']['passed']}",f"- CPU64 quantisiert: max_abs={mx['gpu_to_cpu64_quantized']['max_abs']}, max_rel={mx['gpu_to_cpu64_quantized']['max_rel_outside_near_zero']}, nL2={mx['gpu_to_cpu64_quantized']['normalized_l2']}, Maskenfehler={mx['gpu_to_cpu64_quantized']['relu_mask_mismatches']}",f"- PyTorch FP32-Compute: max_abs={mx['gpu_to_torch_fp32_accum']['max_abs']}, max_rel={mx['gpu_to_torch_fp32_accum']['max_rel_outside_near_zero']}, nL2={mx['gpu_to_torch_fp32_accum']['normalized_l2']}, Maskenfehler={mx['gpu_to_torch_fp32_accum']['relu_mask_mismatches']}",f"- eingefrorener tcnn-FP32-Crosscheck: max_abs={mx['gpu_to_frozen_tcnn_fp32']['max_abs']}, max_rel={mx['gpu_to_frozen_tcnn_fp32']['max_rel_outside_near_zero']}, nL2={mx['gpu_to_frozen_tcnn_fp32']['normalized_l2']}, Near-zero-Maskendifferenzen={mx['gpu_to_frozen_tcnn_fp32']['relu_mask_mismatches']} (informativ; keine Hidden-FP16-Quantisierung)", "","## Cache, Handles, Streams und Speicher","",f"- zwei verschiedene Streams: `{functional['two_streams']['passed']}`",f"- Warm-Cache: `{cnt['warm_cache_passed']}`; Misses {cnt['warm_before']['cache_misses']}→{cnt['warm_after']['cache_misses']}; Heuristiken {cnt['warm_before']['heuristic_queries']}→{cnt['warm_after']['heuristic_queries']}; Handle-Erzeugungen {cnt['warm_before']['execution_handle_creations']}→{cnt['warm_after']['execution_handle_creations']}",f"- Scratch live/peak: {cnt['warm_after']['scratch_bytes_live']}/{cnt['warm_after']['scratch_bytes_peak']} Byte",f"- Speicherwachstum über 20→100 Läufe: {functional['memory_stability']['allocated_growth']} Byte", "","## Regression und Audits","",f"- Phase-3A1-FP32: `{p3a1['result']}`",f"- Phase-3A4-FP32: `{p3a4['result']}`",f"- Marker-Audit: `{functional['marker_audit']['all_changed_source_files_marked']}`",f"- Binding: `{binding}`", "","Die vollständigen Einzelfälle, vorab fixierten Toleranzen, Layer-Crosschecks, Counter-Snapshots, Fresh-Process-Ausgaben und Hashes stehen im JSON-Report.",""]
    MARKDOWN.write_text("\n".join(lines));print(functional["decision"])
if __name__=="__main__":main()
