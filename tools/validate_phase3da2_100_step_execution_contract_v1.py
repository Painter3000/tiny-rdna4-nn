#!/usr/bin/env python3
"""Fail-closed semantic validator for the Phase 3D-A2 execution contract."""
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/phase3da2_100_step_execution_contract_v1.json"
SCHEMA_PATH = ROOT / "contracts/phase3da2_100_step_execution_contract_v1.schema.json"
CASES = ["dense_a_m32", "sparse_a_m48", "dense_b_m64", "partial_b_m45"]
ORACLE_POINTS = [1, 2, 4, 8, 16, 32, 50, 64, 100]
TREND_POINTS = [16, 32, 50, 64, 100]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trend_decision(points, e_p99, e_max_last):
    require(points == TREND_POINTS, "trend points/order")
    require(len(e_p99) == 5 and all(math.isfinite(x) and x >= 0 for x in e_p99), "trend values")
    rises = sum(b > a for a, b in zip(e_p99, e_p99[1:]))
    ratio = e_p99[-1] / max(e_p99[0], 5.960464477539063e-08)
    return "MANUAL_REVIEW_BLOCKED" if rises >= 3 and ratio >= 2 and e_max_last >= 0.5 else "PASS"


def validate(contract):
    require(contract["contract_id"] == "PHASE3DA2_100_STEP_EXECUTION_CONTRACT_V1", "contract id")
    require(contract["contract_version"] == 1, "contract version")
    anchors = contract["anchors"]
    exact_anchors = {
        "phase3d0_freeze_commit": "8632e9cc3cdf4c92f35caae94f90dc577ae627a2",
        "phase3d0_freeze_tag": "phase3d0-inprocess-bridge-gfx1201-pass",
        "phase3d0_archive_sha256": "ed9bb0ad316e589af0eb7bda558036013d0c30ced09e697d0567b8bc2f202323",
        "phase3da_contract_commit": "624f206152ba7f6dcb2c2360f67c43f093e00e56",
        "phase3da_r1_diagnosis_commit": "cf8b4fb14784c737b49eba642d1081f00f5422ad",
        "phase3da_r1_fix_commit": "cd60898957e1e7a4c1092b3b8870ecc6d94c734c",
        "crosscheck_harness_sha256": "e9416fdde6394b0944e7d858332dd52a7826140dfad1c4405e6113a32f1a5748",
        "driver_sha256": "7f507b8d4a522c7652597568f0c59bbea08a52ed8a159484d6e02c0e30ad67f0",
    }
    for key, value in exact_anchors.items():
        require(anchors[key] == value, f"anchor {key}")
    history = contract["historical_phase3da"]
    require(history["status"] == "BLOCKED", "historical status")
    require(history["reason"] == "R1_hipBLASLt_crosscheck_transpose", "historical reason")
    require(history["historical_results_reusable_as_primary_evidence"] is False, "historical primary reuse")
    require(history["historical_results_reclassifiable"] is False, "historical reclassification")
    require(history["allowed_uses"] == ["diagnostic_context", "regression_fixture", "provenance_anchor"],
            "historical allowed uses")
    require("reuse_old_D_metrics" in history["forbidden_uses"], "old D reuse")

    matrix = contract["primary_matrix"]
    require(matrix["cases"] == CASES, "cases")
    require(matrix["fresh_complete_replays_per_case"] == 3, "replay count")
    require(matrix["persistent_steps_per_replay"] == 100, "step count")
    require(matrix["primary_processes"] == 12 and matrix["primary_steps"] == 1200, "primary totals")
    require(matrix["resume_processes"] == 12, "resume processes")
    require((matrix["resume_start_step"], matrix["resume_end_step"]) == (51, 66), "resume range")
    for key in ("replacement_runs_allowed", "adaptive_extension_allowed",
                "best_result_selection_allowed", "primary_process_reuse_allowed",
                "crashed_replay_continuation_allowed"):
        require(matrix[key] is False, key)

    run_ids = contract["run_ids"]
    require(run_ids["required_namespace"] == "phase3da2", "run namespace")
    require("phase3da2" in run_ids["primary_template"], "primary run id")
    require("phase3da2" in run_ids["resume_template"], "resume run id")
    require(run_ids["historical_unnamespaced_pattern_allowed"] is False, "old run id")

    oracle = contract["teacher_forced_cpu_oracle"]
    require(oracle["points"] == ORACLE_POINTS, "oracle points")
    require(len(oracle["required_comparisons"]) == 13, "oracle comparisons")
    require(oracle["e_max_fail_if_greater_than"] == 1.0, "E fail")
    require(oracle["e_max_manual_review_if_greater_than_or_equal"] == 0.9, "E review")
    require(oracle["manual_review_result"] == "MANUAL_REVIEW_BLOCKED", "E review result")

    cross = contract["corrected_gpu_crosscheck"]
    require(cross["logical_equation"] == "Y = X * W", "crosscheck equation")
    require(cross["physical_weight_mapping"] == "row-major W[K,N] as column-major A[N,K]", "layout")
    require(cross["additional_host_transpose"] is False, "host transpose")
    require(cross["marker"] == "P3DA-R1-HIPBLASLT-LAYOUT-FIX-V1", "marker")
    require(cross["d_max_manual_review_if_greater_than_or_equal"] == 1.0, "D review")
    require(cross["d_combined_bound"] == 1.9607843137254901, "combined bound")
    require(cross["combined_bound_replaces_review_gate"] is False, "combined gate semantics")
    require(cross["manual_review_result"] == "MANUAL_REVIEW_BLOCKED", "D review result")
    require(cross["required_every_step"] is True and len(cross["required_tensors"]) == 6, "crosscheck coverage")
    require(len(cross["required_metrics"]) == 9, "crosscheck metrics")

    trend = contract["trend_rule"]
    require(trend["points"] == TREND_POINTS, "trend points")
    require((trend["minimum_rising_transitions"], trend["transition_count"]) == (3, 4), "trend rises")
    require(trend["first_last_ratio_minimum"] == 2.0, "trend ratio")
    require(trend["epsilon"] == 5.960464477539063e-08, "trend epsilon")
    require(trend["last_e_max_minimum"] == 0.5 and trend["all_conditions_required"] is True, "trend conjunction")
    require(len(trend["required_negative_series"]) == 7, "trend negative series")

    resume = contract["resume_contract"]
    require(resume["checkpoint_after_completed_step"] == 50, "checkpoint step")
    require(resume["optimizer_step"] == 50, "checkpoint optimizer step")
    require(resume["next_step_index"] == 51, "next step")
    require(resume["fresh_process_required"] is True, "fresh resume")
    require((resume["start_step"], resume["end_step"]) == (51, 66), "resume execution")
    require(len(resume["required_bit_identical_comparisons"]) == 13, "resume comparisons")

    determinism = contract["replay_determinism"]
    require(determinism["replays"] == [1, 2, 3], "determinism replays")
    require(determinism["comparison"] == "BYTE_IDENTICAL", "determinism mode")
    require(determinism["canonical_serialization_or_raw_bits_required"] is True, "raw determinism")
    require(len(determinism["required_domains"]) == 10, "determinism domains")

    activity = contract["activity"]
    require(activity["each_trainable_layer_minimum_fp32_master_updates"] == 1, "layer activity")
    require(activity["global_effective_step_count_minimum_exclusive"] == 0, "global activity")
    require(activity["metrics_complete_finite_noninvasive"] is True, "activity metrics")
    require(activity["long_horizon_no_op_trigger"] == 256, "no-op trigger")
    require(activity["long_horizon_global_effective_step_fraction_minimum"] == 0.95, "activity fraction")
    require(activity["claim_full_phase3db_gate_at_100_steps"] is False, "long horizon claim")

    paths = contract["evidence_paths"]
    require(paths == {
        "work": "evidence_phase3da2_100_step_work",
        "final": "evidence_phase3da2_100_step_final",
        "fresh_copy": "evidence_phase3da2_100_step_final_fresh_copy",
        "historical_paths_may_be_overwritten": False,
        "historical_paths_may_be_new_evidence_basis": False,
    }, "evidence paths")
    decision = contract["final_decision"]
    require(decision["manual_review_may_be_reclassified_after_full_run"] is False, "review reclassification")
    require(decision["historical_status_remains_visible"] is True, "history visibility")
    require(not decision["phase3db_allowed_before_pass"], "phase3db")
    require(not decision["qualified_horizon_30000_allowed_before_pass"], "horizon")
    require(not decision["pass_tag_allowed_before_pass"], "tag")


def validate_production_immutability(contract):
    anchors = contract["anchors"]
    require(sha256(ROOT / anchors["crosscheck_harness_path"]) == anchors["crosscheck_harness_sha256"],
            "crosscheck harness file hash")
    require(sha256(ROOT / anchors["driver_path"]) == anchors["driver_sha256"], "driver file hash")
    require(sha256(ROOT / "evidence_phase3da_r1_fix_final/SHA256SUMS") ==
            anchors["fix_evidence_sha256sums_sha256"], "fix evidence hash")
    require(sha256(ROOT / "contracts/phase3d_execution_contract_addendum_v2.json") ==
            anchors["historical_addendum_v2_sha256"], "addendum hash")
    require(sha256(ROOT / "contracts/phase3d_runtime_numeric_profile_v1.json") ==
            anchors["runtime_profile_sha256"], "runtime profile hash")
    imm = contract["production_immutability"]
    source_manifest = ROOT / imm["frozen_source_manifest_path"]
    require(sha256(source_manifest) == imm["frozen_source_manifest_sha256"], "source manifest hash")
    sources = json.loads(source_manifest.read_text())["sources"]
    require(len(sources) == imm["frozen_source_count"], "source count")
    for row in sources:
        local = ROOT / row["path"]
        path = local if local.is_file() else ROOT.parent / row["path"]
        require(sha256(path) == row["expected_sha256"], f"source {row['path']}")
    sums = ROOT.parent / "phase3b_binary_provenance_recovery_v1/recovered_exact_artifacts/SHA256SUMS"
    require(sha256(sums) == imm["qualified_phase3b_sha256sums_sha256"], "qualified sums hash")
    lines = sums.read_text().splitlines()
    require(len(lines) == imm["qualified_phase3b_binary_count"], "qualified binary count")
    for line in lines:
        expected, name = line.split("  ", 1)
        require(sha256(sums.parent / name) == expected, f"qualified binary {name}")
    tagged = subprocess.check_output(
        ["git", "rev-parse", anchors["phase3d0_freeze_tag"] + "^{}"], cwd=ROOT, text=True).strip()
    require(tagged == anchors["phase3d0_freeze_commit"], "phase3d0 tag/commit")


def mutations():
    return {
        "N1_historical_primary": lambda x: x["historical_phase3da"].update(
            historical_results_reusable_as_primary_evidence=True),
        "N2_two_replays": lambda x: x["primary_matrix"].update(fresh_complete_replays_per_case=2),
        "N3_old_run_id": lambda x: x["run_ids"].update(primary_template="{case}_replay_{replay}_100"),
        "N4_old_harness_hash": lambda x: x["anchors"].update(
            crosscheck_harness_sha256="ade20fdcda7bf3b8959664e4e21ac21c224c2286b0c04ab0b6e342222fc66cf1"),
        "N5_wrong_transpose": lambda x: x["corrected_gpu_crosscheck"].update(
            logical_equation="Y = X * W_transpose", additional_host_transpose=True),
        "N6_D_as_pass": lambda x: x["corrected_gpu_crosscheck"].update(manual_review_result="PASS"),
        "N7_E_fail_as_review": lambda x: x["teacher_forced_cpu_oracle"].update(
            e_max_fail_if_greater_than=999.0),
        "N8_review_reclassified": lambda x: x["final_decision"].update(
            manual_review_may_be_reclassified_after_full_run=True),
        "N9_remove_oracle_50": lambda x: x["teacher_forced_cpu_oracle"]["points"].remove(50),
        "N10_resume_50": lambda x: x["resume_contract"].update(next_step_index=50),
        "N11_resume_52": lambda x: x["resume_contract"].update(next_step_index=52),
        "N12_four_trend_points": lambda x: x["trend_rule"].update(points=[16, 32, 64, 100]),
        "N13_activity_change": lambda x: x["activity"].update(long_horizon_no_op_trigger=255),
        "N14_replacement_runs": lambda x: x["primary_matrix"].update(replacement_runs_allowed=True),
        "N15_reuse_old_D": lambda x: x["historical_phase3da"]["forbidden_uses"].remove(
            "reuse_old_D_metrics"),
        "N16_production_hash": lambda x: x["anchors"].update(driver_sha256="0" * 64),
    }


def run_negative_tests(contract):
    for name, mutate in mutations().items():
        candidate = copy.deepcopy(contract)
        mutate(candidate)
        try:
            validate(candidate)
        except ValueError:
            continue
        raise ValueError(f"{name} accepted")
    trend_cases = {
        "flat": ([0.1] * 5, 0.6, "PASS"),
        "falling": ([0.5, 0.4, 0.3, 0.2, 0.1], 0.6, "PASS"),
        "monotonic_rising": ([0.1, 0.15, 0.2, 0.25, 0.3], 0.6, "MANUAL_REVIEW_BLOCKED"),
        "noisy_without_trigger": ([0.1, 0.2, 0.15, 0.25, 0.19], 0.6, "PASS"),
        "noisy_with_trigger": ([0.1, 0.2, 0.15, 0.25, 0.3], 0.6, "MANUAL_REVIEW_BLOCKED"),
    }
    for name, (values, emax, expected) in trend_cases.items():
        require(trend_decision(TREND_POINTS, values, emax) == expected, name)
    for points in ([16, 32, 50, 64], [16, 50, 32, 64, 100]):
        try:
            trend_decision(points, [0.1] * len(points), 0.6)
        except ValueError:
            continue
        raise ValueError("invalid trend point set accepted")


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


def main():
    schema = json.loads(SCHEMA_PATH.read_text())
    require(schema["$schema"] == "https://json-schema.org/draft/2020-12/schema", "schema dialect")
    contract = load_contract()
    validate(contract)
    validate_production_immutability(contract)
    if "--negative-tests" in sys.argv:
        run_negative_tests(contract)
    print("PHASE3DA2_EXECUTION_CONTRACT_VALIDATION: PASS")
    print("PHASE3DA2_PRODUCTION_OBJECT_IMMUTABILITY: PASS")
    if "--negative-tests" in sys.argv:
        print("PHASE3DA2_EXECUTION_CONTRACT_NEGATIVE_TESTS: PASS")


if __name__ == "__main__":
    main()
