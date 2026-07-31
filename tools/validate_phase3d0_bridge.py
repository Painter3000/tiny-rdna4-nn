#!/usr/bin/env python3
import hashlib,json,struct,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.compare_phase3b_states import FILES,sha

SEQUENCE=["STEP_READY","FORWARD_COMPLETE","BACKWARD_COMPLETE","REDUCTION_COMPLETE",
          "ADAM_COMPLETE","CAST_COMPLETE","STEP_COMMITTED"]
CASE_SHORT={"train_dense_set_a_m32":"dense_a_m32","train_sparse_set_a_m48":"sparse_a_m48",
            "train_dense_set_b_m64":"dense_b_m64","train_partial_set_b_m45":"partial_b_m45"}
def f32(x):return struct.pack("<f",float(x))
def audit_ledger(x):
    errors=[]
    if x.get("events")!=SEQUENCE:errors.append("illegal_state_sequence")
    if x.get("adam_count")!=1:errors.append("adam_count")
    if not x.get("cast_complete"):errors.append("cast_missing")
    if x.get("state_hash_event_index",99)<SEQUENCE.index("CAST_COMPLETE"):errors.append("hash_before_cast")
    if x.get("optimizer_step")!=x.get("step"):errors.append("optimizer_step")
    if x.get("beta1_power_advanced") is not True:errors.append("beta1_power")
    if x.get("input_hash_match") is not True:errors.append("input_hash")
    if x.get("fresh_replay_initialization_count")!=1:errors.append("replay_initialization")
    if x.get("expected_hash_match") is not True:errors.append("expectation_hash")
    return errors
def validate_files(ref,out):
    errors=[]
    for name in FILES:
        p=out/name
        if not p.is_file():errors.append("missing:"+name)
        elif sha(ref/name)!=sha(p):errors.append("mismatch:"+name)
    return errors
def main():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/"evidence_phase3d0_work/bridge_verification.json").read_text())
    assert len(data["single_step"])==4 and len(data["full_replays"])==4
    for cid,replays in data["full_replays"].items():
        assert len(replays)==3
        for replay in replays:
            assert len(replay["steps"])==4 and replay["status"]=="PASS"
            inv=root/f"evidence_phase3d0_work/bridge_runs/full/{cid}/replay_{replay['replay']}/allocation_inventory.txt"
            lines=inv.read_text().splitlines();assert len(lines)==14 and len({x.split()[0] for x in lines})==14 and len({x.split()[1] for x in lines})==14
            assert all(all(c["match"] for c in s["comparisons"]) for s in replay["steps"])
            for step in range(1,5):
                actual=(inv.parent/f"step_{step}/optimizer_state.txt").read_text().splitlines()
                manifest=root.parent/f"tests/reference/tier1_initial_states/replays/{CASE_SHORT[cid]}_r{replay['replay']}/four_step/step_manifests/{cid}/step_manifest_{step}.json"
                expected=json.loads(manifest.read_text())
                assert int(actual[0])==expected["optimizer_step"]==step
                assert f32(actual[1])==f32(expected["beta1_power"])
                assert f32(actual[2])==f32(expected["beta2_power"])
    print("INPROCESS_LOOP_PERSISTENT_STATE: PASS")
    print("INPROCESS_LOOP_OPTIMIZER_STATE: PASS")
    print("INPROCESS_LOOP_PHASE3B_EQUIVALENCE: PASS")
if __name__=="__main__":main()
