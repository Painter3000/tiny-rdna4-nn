#!/usr/bin/env python3
"""Validate Phase 3D-A3 historical anchors, triage, and metric attestation."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHORS_PATH = ROOT / "contracts/phase3da3_historical_trajectory_anchors_v1.json"
SCHEMA_PATH = ROOT / "contracts/phase3da3_historical_trajectory_anchors_v1.schema.json"
ATTESTATION_PATH = ROOT / "contracts/phase3da3_metric_contract_attestation_v1.json"
VALIDATOR_PATH = Path(__file__).resolve()
CASES = ("dense_a_m32", "sparse_a_m48", "dense_b_m64", "partial_b_m45")
STEPS = (1, 4, 50, 100)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(value, path):
    for token in path.lstrip("/").split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def dotted(value, path):
    for token in path.split("."):
        value = value[token]
    return value


def load():
    return json.loads(ANCHORS_PATH.read_text()), json.loads(ATTESTATION_PATH.read_text())


def validate_anchors(manifest, verify_files=True):
    require(manifest["manifest_id"] == "PHASE3DA3_HISTORICAL_TRAJECTORY_ANCHORS_V1", "anchor id")
    legitimacy = manifest["historical_reference_legitimacy"]
    require(legitimacy["classification"] == "functional_trajectory_reference", "reference class")
    require(legitimacy["qualified_training_path"] is True, "qualified path")
    require(legitimacy["historical_blocked_reason"] == "R1_hipBLASLt_crosscheck_transpose", "blocked reason")
    require(set(legitimacy["training_gates_pass"]) == {
        "INPROCESS_LOOP_100_STEP_INVARIANTS", "INPROCESS_LOOP_100_STEP_CPU_ORACLE",
        "INPROCESS_LOOP_100_STEP_RESUME", "INPROCESS_LOOP_100_STEP_REPLAY_DETERMINISM",
        "INPROCESS_LOOP_100_STEP_SIGNAL_ACTIVITY"}, "historical training gates")
    require(set(legitimacy["not_evidence_for"]) == {
        "historical_crosscheck_correctness", "complete_historical_runtime_provenance"}, "limitations")
    source = manifest["source_integrity"]
    require(source["historical_reference_generated_by_new_gpu_run"] is False, "new GPU reference")
    require(manifest["required_steps"] == list(STEPS), "all anchors required")
    anchors = manifest["anchors"]
    require(len(anchors) == 16, "anchor count")
    require({(a["case_id"], a["step"]) for a in anchors} ==
            {(case, step) for case in CASES for step in STEPS}, "anchor matrix")
    require(all(a["capture_classification"] == "EXACT_RUNTIME_CAPTURE" for a in anchors), "capture class")
    require(all(a["historical_replay_match_count"] == a["historical_replay_total"] == 3
                for a in anchors), "historical replay match")
    require(all(a["source_evidence_path"] == source["source_evidence_path"] and
                a["source_evidence_sha256"] == source["source_evidence_sha256"] for a in anchors), "anchor source")
    mismatch = manifest["mismatch_policy"]
    require(mismatch["qualification_decision"] == mismatch["phase_decision"] == "FAIL", "mismatch fail")
    require(mismatch["reclassification_allowed"] is False, "mismatch reclassification")
    require(mismatch["stop_unstarted_primary_replays"] is True, "stop remaining replays")
    require(set(manifest["localization"]) == {"S1", "S4", "S50", "S100"}, "localization")
    triage = manifest["triage"]
    require(set(triage["paths"]) == {"A", "B", "C"}, "triage paths")
    require(triage["same_input_and_last_common_state_required"] is True, "triage input identity")
    require(set(triage["classifications"]) == {"T1", "T2", "T3", "T4"}, "triage classes")
    require(triage["may_change_a3_fail"] is False, "triage changes fail")
    require(triage["replacement_run"] is False, "replacement")
    require(triage["qualification_run_id_allowed"] is False, "triage run id")
    require(triage["counts_as_primary_evidence"] is False, "triage primary evidence")
    require(set(triage["gate_values"]) == {"PASS_CLASSIFIED", "NOT_TRIGGERED", "INCONCLUSIVE"}, "triage gate")
    if verify_files:
        source_path = ROOT / source["source_evidence_path"]
        require(sha(source_path) == source["source_evidence_sha256"], "source evidence hash")
        require(sha(ROOT / source["source_sha256sums_path"]) == source["source_sha256sums_sha256"],
                "source SHA256SUMS hash")
        checksum_lines = (ROOT / source["source_sha256sums_path"]).read_text().splitlines()
        require(f'{source["source_evidence_sha256"]}  per_step_state_hashes/all.json' in checksum_lines,
                "source absent from SHA256SUMS")
        historical = json.loads(source_path.read_text())
        for anchor in anchors:
            rows = historical[anchor["case_id"]]
            hashes = [replay[anchor["step"] - 1]["post_step_state_hash"] for replay in rows.values()]
            require(len(hashes) == 3 and len(set(hashes)) == 1, "source replay divergence")
            require(hashes[0] == anchor["expected_canonical_state_hash"], "anchor differs from source")
        schema = json.loads(SCHEMA_PATH.read_text())
        require(schema["$schema"] == "https://json-schema.org/draft/2020-12/schema", "schema")


def validate_attestation(attestation, verify_files=True, source_documents=None):
    require(attestation["attestation_id"] == "PHASE3DA3_METRIC_CONTRACT_ATTESTATION_V1", "attestation id")
    paths = attestation["source_contract_paths"]
    require(set(paths) == set(attestation["source_contract_sha256"]), "source hash coverage")
    extraction_map = attestation["extraction_map"]
    required = {
        "E_rocwmma_cpu.fail_above", "E_rocwmma_cpu.review_at_or_above",
        "E_hipblaslt_cpu.fail_above", "E_hipblaslt_cpu.review_at_or_above",
        "D_rocwmma_hipblaslt.review_at_or_above", "D_combined_bound",
        "trend.oracle_points", "trend.minimum_rising_transitions",
        "trend.minimum_last_first_ratio", "trend.minimum_final_E_max", "trend.epsilon",
        "activity.minimum_global_effective_step_fraction",
        "activity.require_update_each_trainable_layer", "activity.long_horizon_noop_steps",
        "activity.long_horizon_noop_rule_evaluable_at_100_steps",
    }
    require(set(extraction_map) == required, "attestation extraction coverage")
    documents = source_documents or {path: json.loads((ROOT / path).read_text()) for path in paths}
    for target, (source_path, source_pointer) in extraction_map.items():
        require(source_path in paths, f"undeclared extraction source: {target}")
        require(dotted(attestation["values"], target) == pointer(documents[source_path], source_pointer),
                f"attestation not extracted from source: {target}")
    values = attestation["values"]
    require(values["E_rocwmma_cpu"] != values.get("E_combined", {}), "backend roles merged")
    require(values["D_rocwmma_hipblaslt"]["review_at_or_above"] == 1.0, "D review threshold")
    require(values["activity"]["minimum_global_effective_step_fraction"] == 0.95, "activity threshold")
    require(values["activity"]["long_horizon_noop_rule_evaluable_at_100_steps"] is False, "256 false pass")
    if verify_files:
        for path in paths:
            require(sha(ROOT / path) == attestation["source_contract_sha256"][path], f"source hash: {path}")
        require(sha(VALIDATOR_PATH) == attestation["extraction_validator_sha256"], "validator hash")


def anchor_mutations():
    return {
        "A1_change_S1_hash": lambda m: m["anchors"][0].update(expected_canonical_state_hash="0" * 64),
        "A2_swap_S4_S50": lambda m: (
            m["anchors"][1].update(step=50), m["anchors"][2].update(step=4)),
        "A3_only_S100": lambda m: m.update(required_steps=[100]),
        "A4_new_GPU_reference": lambda m: m["source_integrity"].update(
            historical_reference_generated_by_new_gpu_run=True),
        "A5_mismatch_review": lambda m: m["mismatch_policy"].update(qualification_decision="REVIEW"),
        "A6_triage_to_pass": lambda m: m["triage"].update(may_change_a3_fail=True),
        "A7_diagnostic_as_primary": lambda m: m["triage"].update(counts_as_primary_evidence=True),
        "A8_different_inputs": lambda m: m["triage"].update(same_input_and_last_common_state_required=False),
        "A9_triage_run_id": lambda m: m["triage"].update(qualification_run_id_allowed=True),
    }


def attestation_mutations():
    return {
        "A10_merge_E_backends": lambda a: a["values"].update(E_combined=a["values"].pop("E_hipblaslt_cpu")),
        "A11_raise_D_review": lambda a: a["values"]["D_rocwmma_hipblaslt"].update(
            review_at_or_above=a["values"]["D_combined_bound"]),
        "A12_accept_below_0_95": lambda a: a["values"]["activity"].update(
            minimum_global_effective_step_fraction=0.94),
        "A13_256_rule_pass": lambda a: a["values"]["activity"].update(
            long_horizon_noop_rule_evaluable_at_100_steps=True),
        "A14_hardcoded_without_extraction": lambda a: a["extraction_map"].pop(
            "E_rocwmma_cpu.fail_above"),
    }


def negative_tests(manifest, attestation):
    for name, mutate in anchor_mutations().items():
        candidate = copy.deepcopy(manifest)
        mutate(candidate)
        try:
            validate_anchors(candidate, verify_files=True)
        except (KeyError, ValueError):
            continue
        raise ValueError(f"negative anchor/triage mutation accepted: {name}")
    for name, mutate in attestation_mutations().items():
        candidate = copy.deepcopy(attestation)
        mutate(candidate)
        try:
            validate_attestation(candidate, verify_files=False)
        except (KeyError, ValueError):
            continue
        raise ValueError(f"negative attestation mutation accepted: {name}")


def main():
    manifest, attestation = load()
    validate_anchors(manifest)
    validate_attestation(attestation)
    print("PHASE3DA3_HISTORICAL_TRAJECTORY_ANCHORS_LOCK: PASS")
    print("PHASE3DA3_HISTORICAL_TRAJECTORY_LOCALIZATION_READY: PASS")
    print("PHASE3DA3_METRIC_CONTRACT_CONTENT_ATTESTATION: PASS")
    print("PHASE3DA3_TRAJECTORY_TRIAGE_ADDENDUM_VALIDATION: PASS")
    if "--negative-tests" in sys.argv:
        negative_tests(manifest, attestation)
        print("PHASE3DA3_TRAJECTORY_TRIAGE_CONTRACT_NEGATIVE_TESTS: PASS")
        print("PHASE3DA3_METRIC_ATTESTATION_NEGATIVE_TESTS: PASS")


if __name__ == "__main__":
    main()
