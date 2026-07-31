#!/usr/bin/env python3
"""Validate the locked Phase 3D-A3 metric and trajectory addendum."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDENDUM_PATH = ROOT / "contracts/phase3da3_metric_trajectory_addendum_v1.json"
SCHEMA_PATH = ROOT / "contracts/phase3da3_metric_trajectory_addendum_v1.schema.json"
MANIFEST_PATH = ROOT / "contracts/phase3da3_historical_s100_manifest_v1.json"
EXPECTED_CASES = {
    "dense_a_m32": "cae85c3691ca39e2c0b76f38a24326d0ab3a1bdf6c56aa37df85d1636850758e",
    "sparse_a_m48": "780f221eb6081f3e8f3014275890cdec7a7978cdaf5aaa98512d7d9ef7547673",
    "dense_b_m64": "8f406072e131875a7b67be811656e4fe6cffebf7dbfb7e7bf02d9cf909ad56b7",
    "partial_b_m45": "fc3c54e613011f2413024c62f56fa782314e478d43c2e1d11244ae3f1d4198ae",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    return json.loads(ADDENDUM_PATH.read_text())


def validate(addendum, manifest=None, verify_files=True):
    manifest = manifest or json.loads(MANIFEST_PATH.read_text())
    require(addendum["addendum_id"] == "PHASE3DA3_METRIC_TRAJECTORY_ADDENDUM_V1", "id")
    parent = addendum["parent_contract"]
    require(parent == {
        "path": "contracts/phase3da3_100_step_execution_contract_v1.json",
        "sha256": "863dc082835adcb144bd87f875af279880a90ac2faf2ff4fde12319854671e03",
        "commit": "59059e5922f57002e44acdb98540410bd8539e17",
        "thresholds_unchanged": True,
    }, "parent contract")
    require(manifest["cases"] == EXPECTED_CASES, "historical S100 values")
    require(manifest["immutable_after_a3_series_start"] is True, "manifest immutability")
    hist = addendum["historical_s100"]
    require(hist["required_matches"] == 12 and hist["replays_per_case"] == 3, "12/12 scope")
    require(hist["expectations_may_be_updated_after_series_start"] is False, "post-start update")
    require(hist["mismatch_decision"] == "FAIL", "fingerprint mismatch")
    require(hist["comparison"] == "A3_S100_SHA256_EQUALS_HISTORICAL_CASE_S100_SHA256", "comparison")

    metrics = addendum["backend_error_metrics"]
    require(metrics["families"] == ["E_rocwmma_cpu", "E_hipblaslt_cpu"], "backend families")
    require(metrics["backend_families_may_be_merged"] is False, "merged E")
    require(metrics["teacher_forced_steps"] == [1, 2, 4, 8, 16, 32, 50, 64, 100], "oracle points")
    fields = {"E_max", "E_p99", "E_mean", "E_median", "argmax_index", "gpu_value",
              "cpu_value", "absolute_difference", "denominator"}
    require(set(metrics["per_family_per_tensor_fields"]) == fields, "metric fields")
    require(metrics["rocwmma_cpu"] == {
        "greater_than_1": "FAIL", "greater_than_or_equal_0_9": "MANUAL_REVIEW_BLOCKED"}, "roc decisions")
    require(metrics["hipblaslt_cpu"] == {
        "greater_than_1": "FAIL_CROSSCHECK_OR_HARNESS",
        "greater_than_or_equal_0_9": "MANUAL_REVIEW_BLOCKED"}, "LT decisions")
    require(set(metrics["global_maximum_required_context"]) ==
            {"backend_role", "case", "replay", "step", "tensor", "index"}, "global E context")

    trend = addendum["trend_evaluation"]
    require(trend["backend_roles"] == ["rocWMMA_cpu", "hipBLASLt_cpu"], "trend backends")
    require(trend["steps"] == [16, 32, 50, 64, 100], "five-point trend")
    require(trend["epsilon"] == 5.960464477539063e-08, "trend epsilon")
    require(trend["trigger"] == {
        "minimum_increasing_E_p99_transitions": 3,
        "minimum_last_first_ratio": 2,
        "minimum_E_max_last": 0.5,
        "all_conditions_required": True,
    }, "trend trigger")
    require(trend["trigger_decision"] == "MANUAL_REVIEW_BLOCKED", "trend decision")

    d = addendum["d_consistency"]
    require(d["review_gate_threshold"] == 1 and
            d["review_gate_comparison"] == "D_max_greater_than_or_equal", "D review gate")
    require(d["review_decision"] == "MANUAL_REVIEW_BLOCKED", "D decision")
    require(d["combined_bound_replaces_review_gate"] is False, "D bound gate replacement")
    require(math.isclose(d["combined_bound"], 2 / (1 + d["rtol"]), rel_tol=0, abs_tol=1e-16), "D bound")
    require(d["pairing_failure"]["decision"] == "FAIL_METRIC_PAIRING_OR_STATE_IDENTITY", "D pairing decision")
    require(set(d["pairing_failure"]["same_data_required"]) ==
            {"step", "tensor", "logical_index", "cpu_reference_value", "prestep_state"}, "D pairing identity")

    activity = addendum["activity_saturation"]
    require(activity["minimum_global_effective_step_fraction"] == 0.95, "activity threshold")
    require(activity["below_minimum_decision"] == "FAIL_DEGENERATE_TRAINING_ACTIVITY", "activity decision")
    require(activity["global_effective_step_count_must_be_positive"] is True, "activity count")
    require(activity["each_trainable_layer_requires_fp32_master_update"] is True, "layer updates")
    require(all(activity[k] is True for k in ("metrics_complete", "metrics_finite",
                                               "measurement_noninvasive")), "activity metrics")
    require(activity["long_horizon_256_noop_rule_evaluable"] is False, "256 evaluability")
    require(activity["long_horizon_reason"] == "qualification_horizon_below_256", "256 reason")
    require(activity["long_horizon_rule_may_be_reported_pass"] is False, "256 false pass")

    noninvasive = addendum["noninvasivity"]
    require(set(noninvasive["protected_state"]) == {
        "W_master", "W_compute", "m", "v", "optimizer_step", "beta1_power",
        "beta2_power", "counter_data_stream_state"}, "protected state")
    require(noninvasive["crosscheck_state_writes_allowed"] is False, "crosscheck write")
    provenance = addendum["historical_provenance_classification"]
    require(provenance["recovery_gate"] == "PARTIAL", "recovery remains partial")
    require(provenance["s100_match_functional_trajectory_fingerprint"] == "PASS", "functional trajectory")
    require(provenance["runtime_capture_completeness"] == "PARTIAL", "runtime completeness")
    require(provenance["s100_match_may_upgrade_recovery_to_pass"] is False, "recovery upgrade")

    if verify_files:
        require(sha(ROOT / parent["path"]) == parent["sha256"], "parent file hash")
        require(sha(MANIFEST_PATH) == hist["manifest_sha256"], "manifest file hash")
        schema = json.loads(SCHEMA_PATH.read_text())
        require(schema["$schema"] == "https://json-schema.org/draft/2020-12/schema", "schema draft")


def mutations():
    return {
        "M1_flip_historical_hash_bit": lambda a, m: m["cases"].update(dense_a_m32="b" + m["cases"]["dense_a_m32"][1:]),
        "M2_only_replay_1": lambda a, m: a["historical_s100"].update(required_matches=4),
        "M3_update_after_run": lambda a, m: a["historical_s100"].update(expectations_may_be_updated_after_series_start=True),
        "M4_blocked_as_primary": lambda a, m: a["historical_s100"].update(comparison="HISTORICAL_BLOCKED_RUN_IS_A3_PRIMARY"),
        "M5_merge_backend_roles": lambda a, m: a["backend_error_metrics"].update(backend_families_may_be_merged=True),
        "M6_omit_global_backend_role": lambda a, m: a["backend_error_metrics"]["global_maximum_required_context"].remove("backend_role"),
        "M7_combined_bound_pass_gate": lambda a, m: a["d_consistency"].update(combined_bound_replaces_review_gate=True),
        "M8_pair_different_indices": lambda a, m: a["d_consistency"]["pairing_failure"]["same_data_required"].remove("logical_index"),
        "M9_one_backend_trend": lambda a, m: a["trend_evaluation"].update(backend_roles=["rocWMMA_cpu"]),
        "M10_four_point_trend": lambda a, m: a["trend_evaluation"].update(steps=[16, 32, 64, 100]),
        "M11_accept_0_94": lambda a, m: a["activity_saturation"].update(minimum_global_effective_step_fraction=0.94),
        "M12_256_pass": lambda a, m: a["activity_saturation"].update(long_horizon_rule_may_be_reported_pass=True),
        "M13_recovery_pass": lambda a, m: a["historical_provenance_classification"].update(recovery_gate="PASS"),
        "M14_crosscheck_write": lambda a, m: a["noninvasivity"].update(crosscheck_state_writes_allowed=True),
    }


def negative_tests(addendum, manifest):
    for name, mutate in mutations().items():
        candidate, candidate_manifest = copy.deepcopy(addendum), copy.deepcopy(manifest)
        mutate(candidate, candidate_manifest)
        try:
            validate(candidate, candidate_manifest, verify_files=False)
        except ValueError:
            continue
        raise ValueError(f"negative mutation accepted: {name}")


def main():
    addendum = load()
    manifest = json.loads(MANIFEST_PATH.read_text())
    validate(addendum, manifest)
    print("PHASE3DA3_METRIC_TRAJECTORY_ADDENDUM_VALIDATION: PASS")
    if "--negative-tests" in sys.argv:
        negative_tests(addendum, manifest)
        print("PHASE3DA3_METRIC_TRAJECTORY_ADDENDUM_NEGATIVE_TESTS: PASS")


if __name__ == "__main__":
    main()
