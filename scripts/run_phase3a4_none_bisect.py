#!/usr/bin/env python3
"""Run the fixed Phase-3A4 None bisection matrix using fresh subprocesses."""
import argparse, json, pathlib, statistics, subprocess, sys

SEQUENCES=("none-only","relu-none","none-relu-none")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--baseline",required=True)
    p.add_argument("--output",required=True); p.add_argument("--repetitions",type=int,default=10); a=p.parse_args()
    if a.repetitions < 10: raise SystemExit("At least 10 repetitions are required")
    manifest=json.loads(pathlib.Path(a.manifest).read_text()); root=pathlib.Path(a.output); root.mkdir(parents=True,exist_ok=True)
    child=pathlib.Path(__file__).with_name("benchmark_phase3a4_none_sequence.py")
    doc={"repetitions":a.repetitions,"variants":{},"first_bad_change":None}
    for variant,bindings in manifest["variants"].items():
        vr={"bindings":bindings,"sequences":{}}; doc["variants"][variant]=vr
        for sequence in SEQUENCES:
            runs=[]
            for repetition in range(1,a.repetitions+1):
                out=root/f"{variant}_{sequence}_{repetition:02d}.json"
                command=[sys.executable,str(child),"--bindings",bindings,"--baseline",a.baseline,"--sequence",sequence,"--output",str(out)]
                completed=subprocess.run(command,text=True,capture_output=True)
                if not out.exists(): raise RuntimeError({"command":command,"returncode":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr})
                run=json.loads(out.read_text()); run["returncode"]=completed.returncode; runs.append(run)
            ratios=[sample["forward_backward_ratio"] for run in runs for sample in run["none"]]
            vr["sequences"][sequence]={"runs":runs,"ratio_median":statistics.median(ratios),
                "ratio_minimum":min(ratios),"all_invariants_pass":all(run["pass"] for run in runs)}
        pathlib.Path(a.output+".partial.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    pathlib.Path(a.output+".json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print("PHASE3A4_NONE_BISECT=COMPLETE")
if __name__=="__main__": main()
