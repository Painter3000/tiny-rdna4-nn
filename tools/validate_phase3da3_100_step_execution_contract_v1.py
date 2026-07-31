#!/usr/bin/env python3
"""Validate the additive Phase 3D-A3 driver authorization contract."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/phase3da3_100_step_execution_contract_v1.json"
SCHEMA_PATH = ROOT / "contracts/phase3da3_100_step_execution_contract_v1.schema.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    return json.loads(CONTRACT_PATH.read_text())


def validate(contract, verify_files=True):
    require(contract["contract_id"] == "PHASE3DA3_100_STEP_EXECUTION_CONTRACT_V1", "id")
    parent = contract["parent_contract"]
    require(parent["sha256"] == "059715dcfc7aee5b4a22cae373a5ed2f3ec28661a2148bf5133e0a1abff2c40c", "parent hash")
    require(parent["commit"] == "cabfff57977e9cae71c1b782cbf16dea4a51d05a", "parent commit")
    require(parent["numeric_and_decision_rules_inherited_unchanged"] is True, "parent rules")
    history = contract["historical_status"]
    require(history["phase3da"]["status"] == "BLOCKED", "A status")
    require(history["phase3da2"]["status"] == "FAIL", "A2 status")
    require(history["phase3da2"]["reason"] == "PRE_EXECUTION_DRIVER_CAPABILITY_MISMATCH", "A2 reason")
    require(history["phase3da2"]["executed_training_steps"] == 0, "A2 steps")
    require(history["A2_results_are_A3_primary_evidence"] is False, "A2 reuse")
    require(history["A2_run_ids_reusable"] is False, "A2 run id reuse")
    require(history["A2_series_continuation_allowed"] is False, "A2 continuation")
    require(history["A3_is_new_complete_qualification_series"] is True, "A3 fresh series")
    recovery = contract["historical_execution_provenance"]
    require(recovery["gate"] in ("PASS", "PARTIAL"), "recovery gate")
    require(recovery["partial_is_permitted"] is True, "partial policy")
    require(recovery["historical_results_changed"] is False, "history mutation")

    roles = contract["artifact_roles"]
    bridge, driver, cross = (roles["phase3d0_bridge_reference_driver"],
                             roles["phase3da3_execution_driver"],
                             roles["crosscheck_harness"])
    require(bridge["sha256"] == "7f507b8d4a522c7652597568f0c59bbea08a52ed8a159484d6e02c0e30ad67f0", "bridge")
    require("A3_primary_execution" in bridge["prohibited_operations"], "bridge role")
    require(driver["sha256"] == "27807462a404e6dd93bc6c0d70c1e337085577427b697c5eb0c08fcc054702f5", "driver")
    require(driver["source_sha256"] == "41ff6b11878ef2b8c21da4000ab526e0573668b676edc6417036b3a5452ec898", "driver source")
    require(driver["purpose"] == "qualification_orchestration", "driver class")
    require(cross["sha256"] == "e9416fdde6394b0944e7d858332dd52a7826140dfad1c4405e6113a32f1a5748", "crosscheck")
    require(roles["production_kernel_artifacts"]["count"] == 14, "sources")
    require(roles["qualified_phase3b_binaries"]["count"] == 5, "binaries")
    require(len({bridge["absolute_path"], driver["absolute_path"], cross["absolute_path"]}) == 3, "role separation")

    provenance = contract["driver_provenance"]
    require(provenance["source_commit"] == "b7b225bbc06d24aecf245cdd3b5695fb4ebc5bff", "source commit")
    for key in ("production_kernel", "optimizer_semantics", "training_state_semantics",
                "counter_data_stream", "state_hash_serialization"):
        require(provenance[key] == "unchanged", key)
    equivalence = contract["driver_equivalence"]
    require(equivalence["single_step_processes"] == 24, "single processes")
    require(equivalence["four_step_processes"] == 12, "four processes")
    require(equivalence["single_step_gate"] == equivalence["four_step_gate"] == "PASS", "equivalence")
    capability = contract["driver_capability"]
    require(capability["mode"] == "validate_arguments_only", "capability mode")
    require(capability["training_steps_executed"] == 0, "capability steps")
    require(capability["S0_must_remain_unchanged"] is True, "S0")
    require(capability["run_id_reserved_during_validation"] is False, "reservation")
    require(capability["reservation_order"] == [
        "CAPABILITY_PREFLIGHT_PASS", "ATOMIC_RUN_ID_RESERVATION", "PRIMARY_PROCESS_START"], "reservation order")
    require(capability["negative_old_driver_100_step_returncode"] == 3, "old driver negative")

    series = contract["series"]
    require(series["cases"] == ["dense_a_m32", "sparse_a_m48", "dense_b_m64", "partial_b_m45"], "cases")
    require((series["replays_per_case"], series["steps"], series["primary_processes"],
             series["resume_processes"]) == (3, 100, 12, 12), "matrix")
    require("phase3da3" in series["primary_run_id_template"], "primary id")
    require("phase3da3" in series["resume_run_id_template"], "resume id")
    require(series["forbidden_identifier"] == "phase3da2", "forbidden id")
    require(not series["replacement_runs_allowed"] and not series["adaptive_extension_allowed"], "run policy")
    inherited = contract["inherited_rules"]
    require(inherited["teacher_forced_points"] == [1, 2, 4, 8, 16, 32, 50, 64, 100], "oracle")
    require(inherited["E_max_greater_than_1"] == "FAIL", "E fail")
    require(inherited["E_max_greater_than_or_equal_0_9"] == "MANUAL_REVIEW_BLOCKED", "E review")
    require(inherited["D_max_greater_than_or_equal_1"] == "MANUAL_REVIEW_BLOCKED", "D review")
    require(inherited["resume_range"] == [51, 66], "resume")
    if not verify_files:
        return
    require(sha(ROOT / parent["path"]) == parent["sha256"], "parent file")
    require(sha(Path(driver["absolute_path"])) == driver["sha256"], "driver file")
    require(sha(ROOT / driver["source_path"]) == driver["source_sha256"], "source file")
    require(sha(Path(bridge["absolute_path"])) == bridge["sha256"], "bridge file")
    require(sha(Path(cross["absolute_path"])) == cross["sha256"], "crosscheck file")
    require(sha(ROOT / equivalence["evidence_path"]) == equivalence["evidence_sha256"], "equivalence file")
    require(sha(ROOT / recovery["path"]) == recovery["sha256"], "recovery file")
    require(sha(ROOT / roles["manifest_path"]) == roles["manifest_sha256"], "roles file")
    require(sha(ROOT / capability["launcher_path"]) == capability["launcher_sha256"], "launcher file")
    require(sha(ROOT / capability["positive_evidence_path"]) == capability["positive_evidence_sha256"], "capability file")
    rows = [json.loads(line) for line in (ROOT / capability["positive_evidence_path"]).read_text().splitlines()]
    require(len(rows) == 12, "capability count")
    require(all(row["training_steps_executed"] == 0 and not row["run_id_reserved"]
                and row["S0_unchanged"] for row in rows), "capability semantics")


def mutations():
    return {
        "reuse_A2": lambda x: x["historical_status"].update(A2_results_are_A3_primary_evidence=True),
        "reuse_A2_id": lambda x: x["historical_status"].update(A2_run_ids_reusable=True),
        "bridge_as_primary": lambda x: x["artifact_roles"]["phase3d0_bridge_reference_driver"].update(
            prohibited_operations=[]),
        "wrong_driver_hash": lambda x: x["artifact_roles"]["phase3da3_execution_driver"].update(sha256="0" * 64),
        "wrong_crosscheck": lambda x: x["artifact_roles"]["crosscheck_harness"].update(sha256="0" * 64),
        "two_replays": lambda x: x["series"].update(replays_per_case=2),
        "A2_run_template": lambda x: x["series"].update(
            primary_run_id_template="{case}_replay_{replay}_phase3da2_100"),
        "replacement": lambda x: x["series"].update(replacement_runs_allowed=True),
        "E_threshold": lambda x: x["inherited_rules"].update(E_max_greater_than_1="MANUAL_REVIEW_BLOCKED"),
        "D_threshold": lambda x: x["inherited_rules"].update(D_max_greater_than_or_equal_1="PASS"),
        "reserve_early": lambda x: x["driver_capability"].update(run_id_reserved_during_validation=True),
        "training_in_preflight": lambda x: x["driver_capability"].update(training_steps_executed=1),
    }


def negative_tests(contract):
    for name, mutate in mutations().items():
        candidate = copy.deepcopy(contract)
        mutate(candidate)
        try:
            validate(candidate, verify_files=False)
        except ValueError:
            continue
        raise ValueError(f"negative mutation accepted: {name}")


def main():
    schema = json.loads(SCHEMA_PATH.read_text())
    require(schema["$schema"] == "https://json-schema.org/draft/2020-12/schema", "schema")
    contract = load()
    validate(contract)
    if "--negative-tests" in sys.argv:
        negative_tests(contract)
    print("PHASE3DA3_EXECUTION_CONTRACT_VALIDATION: PASS")
    print("PHASE3DA3_ARTIFACT_ROLE_SEPARATION: PASS")
    print("PHASE3DA3_RUNTIME_EXECUTION_IDENTITY_BINDING: PASS")
    print("PHASE3DA3_PRODUCTION_OBJECT_IMMUTABILITY: PASS")
    if "--negative-tests" in sys.argv:
        print("PHASE3DA3_EXECUTION_CONTRACT_NEGATIVE_TESTS: PASS")


if __name__ == "__main__":
    main()
