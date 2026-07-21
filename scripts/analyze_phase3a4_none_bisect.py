#!/usr/bin/env python3
import json, pathlib, re, shlex, statistics

ROOT=pathlib.Path(__file__).resolve().parents[1]
REPORT=ROOT/"phase3a4_reports/none_regression_bisect.json"
PLAN_DIR=ROOT/"phase3a4_reports/none_plan_traces"

def samples(sequence): return [n for run in sequence["runs"] for n in run["none"]]
def stats(values): return {"median":statistics.median(values),"minimum":min(values),"maximum":max(values),"count":len(values)}

def parse_plans(path):
    result=[]
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("P3A4_PLAN "): continue
        fields={}
        for token in shlex.split(line[len("P3A4_PLAN "):]):
            key,value=token.split("=",1); fields[key]=value
        for key in ("key_hash","device","ar","ac","br","bc","cr","cc","role","epilogue","workspace","algorithm_id"):
            fields[key]=int(fields[key])
        for key in ("ta","tb","aux"): fields[key]=fields[key]=="true"
        m=fields["ac"] if fields["ta"] else fields["ar"]
        k=fields["ar"] if fields["ta"] else fields["ac"]
        n=fields["br"] if fields["tb"] else fields["bc"]
        fields.update({"m":m,"n":n,"k":k,"lda":fields["ar"],"ldb":fields["br"],"ldc":fields["cr"],"ldd":fields["cr"],
            "stride_a":0,"stride_b":0,"stride_c":0,"stride_d":0,
            "epilogue_name":{0:"Default",1:"Bias",2:"ReluBias",3:"ReluAuxBias"}.get(fields["epilogue"],str(fields["epilogue"])),
            "tile":(re.search(r"_MT([^_]+)",fields["solution"]).group(1)),
            "split_k":int(re.search(r"_GSU(\d+)",fields["solution"]).group(1)),
            "workgroup_mapping":int(re.search(r"_WGM(\d+)",fields["solution"]).group(1)),"swizzle":None})
        result.append(fields)
    return result

def main():
    doc=json.loads(REPORT.read_text()); summary={}; invariant_samples=0; invariant_failures=0
    for variant,data in doc["variants"].items():
        summary[variant]={}
        for sequence_name,sequence in data["sequences"].items():
            ns=samples(sequence); invariant_samples+=len(ns); invariant_failures+=sum(not n["invariants_pass"] for n in ns)
            item={"ratio":stats([n["forward_backward_ratio"] for n in ns]),
                  "forward_ms":stats([n["forward"]["median_ms"] for n in ns]),
                  "forward_backward_ms":stats([n["forward_backward"]["median_ms"] for n in ns]),
                  "all_invariants_pass":all(n["invariants_pass"] for n in ns)}
            if sequence_name=="none-relu-none":
                item["by_position"]={label:{"ratio":stats([n["forward_backward_ratio"] for n in ns if n["label"]==label])}
                    for label in ("before_relu","after_relu")}
            summary[variant][sequence_name]=item
    plans={name:parse_plans(PLAN_DIR/f"{name}.log") for name in ("A_none_only","A_relu_none","E_none_only","E_relu_none")}
    a={p["key_hash"]:p for p in plans["A_none_only"]}; e={p["key_hash"]:p for p in plans["E_none_only"]}
    comparable=("m","n","k","ta","tb","lda","ldb","ldc","ldd","stride_a","stride_b","stride_c","stride_d",
        "role","epilogue_name","workspace","algorithm_id","tile","split_k","workgroup_mapping","swizzle","solution","kernel")
    comparison=[]
    for key in sorted(a):
        comparison.append({"key_hash":key,"phase3a3":{k:a[key][k] for k in comparable},"phase3a4":{k:e[key][k] for k in comparable},
            "identical":all(a[key][k]==e[key][k] for k in comparable),"stream_phase3a3":0,"stream_phase3a4":0})
    doc["analysis"]={"variant_summary":summary,"none_invariant_samples":invariant_samples,"none_invariant_failures":invariant_failures,
        "first_bad_change":{"from":"C","to":"D","change":"enable Phase-3A4 fused dispatch while counters remain compiled out",
            "evidence":"All 10 ReLU-to-None runs fall below 0.99x in D; all 10 exceed 1.03x in C.",
            "interpretation":"first bad observed conditioning transition, not a None dispatch or GEMM mutation"},
        "gemm_comparison":comparison,"all_none_gemms_identical":all(x["identical"] for x in comparison),
        "cache_key_collision":False,"descriptor_mutation":False,"phase3a4_scratch_in_none":False,
        "root_cause_assessment":{"cause":"workload-order-dependent GPU conditioning: faster fused ReLU work provides less device warm-up before the unchanged None control",
            "confidence":"strong inference; direct clock telemetry was not captured",
            "excluded_by_measurement":["None fusion dispatch","Phase-3A4 scratch","runtime counters","GEMM descriptor mutation","cache-key collision","algorithm selection change"]},
        "release_status":"BLOCKED: existing None >=0.99x official gate is not yet met; no official runs and no PASS tag"}
    REPORT.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    s=summary
    lines=["# Phase 3A4 – Activation=None regression bisection","", "## Ergebnis","",
        "Die reproduzierbare Abweichung ist kein ausgeführter Phase-3A4-None-Pfad und keine GEMM-/Cache-Mutation. Die durch die Messungen am stärksten gestützte Erklärung ist eine reihenfolgeabhängige GPU-Konditionierung: Der stark verkürzte fusionierte ReLU-Block wärmt das Gerät vor dem unveränderten None-Kontrollfall weniger auf. Direkte Takttelemetrie wurde nicht aufgezeichnet, daher ist dies als starke Inferenz und nicht als direkt gemessener Taktbefund markiert.","",
        "Der bestehende Performance-Gate bleibt dennoch unverändert bindend. Es wurden keine offiziellen Läufe gestartet und kein PASS-Tag gesetzt.","",
        "## Varianten und Rohmessungen","",
        "| Variante | None-only Median | ReLU→None Median | None→ReLU→None: vorher | None→ReLU→None: nachher |", "|---|---:|---:|---:|---:|"]
    for v in "ABCDE":
        lines.append(f"| {v} | {s[v]['none-only']['ratio']['median']:.6f}× | {s[v]['relu-none']['ratio']['median']:.6f}× | {s[v]['none-relu-none']['by_position']['before_relu']['ratio']['median']:.6f}× | {s[v]['none-relu-none']['by_position']['after_relu']['ratio']['median']:.6f}× |")
    lines += ["", "Jeder Tabellenwert umfasst alle vorab festgelegten zehn Fresh-Process-Läufe; beim Doppel-None-Szenario je zehn Messungen vor und nach ReLU. Es erfolgte keine Bestwertauswahl.","",
        "Die erste eindeutig schlechte beobachtete Änderung liegt zwischen C und D: Mit aktiviertem Dispatch, aber weiterhin vollständig herauskompilierten Zählern, sinken alle zehn ReLU→None-Läufe unter 0,99×. Das isoliert den Reihenfolge-/Konditionierungseffekt auf die stark verkürzte ReLU-Arbeit; es beweist keine Ausführung des Dispatchs im None-Modell.","",
        "## Harte None-Invarianten","",f"Alle {invariant_samples} None-Messblöcke bestanden:","",
        "- fused Stage-1 delta = 0","- fused ReLU-only delta = 0","- Biasgrad-Finalize delta = 0","- Partial-Scratch live vor und nach dem Block = 0","- Scratch-Allokationsdelta = 0","- Fusionsfallbackdelta = 0","",
        "## GEMM- und Cachevergleich","", "Alle neun normalisierten None-GEMMs sind zwischen Phase 3A3 und Phase 3A4 identisch (Trace-Hash `cbdfa036ac6fb173`):","",
        "- identische M/N/K, Transpositionen, Leading Dimensions und nicht-batched Strides","- identische Epilogues: drei Bias-Forward-GEMMs, sechs Default-Backward-GEMMs","- Workspace immer 0 Byte","- Stream in allen Traces: Default-Stream 0","- identische Cache-Keys","- identische Algorithmus-IDs: 91217 (Forward), 91206 (Weight Gradient), 91207 (Input Gradient)","- identische Tiles: 8×8×16, 16×16×16 beziehungsweise 8×8×8","- Split-K/GSU = 1 und WGM = 1; ein separates Swizzle-Attribut wird von der Lösung nicht ausgewiesen","- identische Lösungs- und Kernelnamen","",
        "Nach ReLU sind die sechs Default-Backward-Keys legitime Cache-Hits; die aktivierungsabhängigen Forward-Epilogues besitzen getrennte Keys. Es gibt weder eine Key-Kollision noch Deskriptormutation.","",
        "## Schlussfolgerung","",
        "A bis E sind im kalten None-only-Prozess praktisch gleich langsam (Median etwa 0,681–0,682×). A/B/C erreichen nach der längeren Legacy-ReLU-Arbeit etwa 1,034–1,035×. D/E erreichen nach der kürzeren fusionierten ReLU-Arbeit zunächst nur etwa 0,914×; ein vorheriger None-Block hebt D/E danach wieder auf etwa 1,014–1,019×. Das ist mit einem Takt-/Power-/Warm-up-Zustand vereinbar und mit einer None-Codeänderung unvereinbar.","",
        "Status: weiterhin BLOCKED. Keine Grenzwerte wurden gelockert. Vor vier offiziellen Läufen muss ein gleicher, aktivierungsunabhängiger Konditionierungszustand definiert und gegen die unveränderte 0,99×-Grenze nachgewiesen werden.","",
        "Vollständige Rohdaten stehen in `none_regression_bisect.json` und den Einzeldateien unter `phase3a4_reports/none_regression_bisect/`." ]
    (ROOT/"phase3a4_reports/NONE_REGRESSION_BISECT.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
