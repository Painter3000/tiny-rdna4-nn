#!/usr/bin/env python3
"""Independent fail-closed validator for the additive Phase-3D execution contract."""
import hashlib,json,math,sys
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def get(obj,path):
    for key in path.split("."):
        if not isinstance(obj,dict) or key not in obj:raise ValueError("missing:"+path)
        obj=obj[key]
    return obj
def trend_review(values,e_max_last,epsilon=5.960464477539063e-08):
    if len(values)!=5:raise ValueError("trend_requires_five_points")
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError("invalid_trend_value")
    rises=sum(b>a for a,b in zip(values,values[1:]))
    return rises>=3 and values[-1]/max(values[0],epsilon)>=2 and e_max_last>=.5
def validate(doc,schema,root):
    missing=[x for x in schema["required_top_level"] if x not in doc]
    if missing:raise ValueError("missing_top_level:"+",".join(missing))
    for path,expected in schema["exact"].items():
        actual=get(doc,path)
        if actual!=expected:raise ValueError(f"exact_mismatch:{path}")
    if doc["cases"]!=schema["case_order"] or len(set(doc["cases"]))!=4:
        raise ValueError("case_matrix")
    expected_bound=2.0/(1.0+doc["drift"]["rtol"])
    if doc["drift"]["d_combined_bound"]!=expected_bound:raise ValueError("combined_bound")
    if doc["drift"]["d_max_ge_result"]!="MANUAL_REVIEW_BLOCKED":raise ValueError("d_classification")
    if doc["drift"]["e_max_gt_result"]!="FAIL":raise ValueError("e_classification")
    if doc["phase3da"]["fresh_complete_replays_per_case"]!=3 or doc["phase3db"]["fresh_complete_replays_per_case"]!=3:
        raise ValueError("replay_count")
    parent=root/doc["parent_contract"]["path"]
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=doc["parent_contract"]["sha256"]:
        raise ValueError("parent_contract_hash")
    if 850 not in load(parent)["phase3db"]["oracle_points"]:raise ValueError("parent_oracle_850")
    if doc["phase3da"]["oracle_points"]!=[1,2,4,8,16,32,50,64,100]:
        raise ValueError("phase3da_oracle_points")
def main(argv):
    if len(argv)!=3:
        print(f"usage: {argv[0]} ADDENDUM.json SCHEMA.json",file=sys.stderr);return 2
    try:
        addendum=Path(argv[1]).resolve();validate(load(addendum),load(argv[2]),addendum.parents[1])
    except Exception as exc:
        print("PHASE3D_EXECUTION_CONTRACT_ADDENDUM_V2_VALIDATION: FAIL",file=sys.stderr)
        print(exc,file=sys.stderr);return 1
    print("PHASE3D_EXECUTION_CONTRACT_ADDENDUM_V2_VALIDATION: PASS");return 0
if __name__=="__main__":raise SystemExit(main(sys.argv))
