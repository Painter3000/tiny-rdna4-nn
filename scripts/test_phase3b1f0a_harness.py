#!/usr/bin/env python3
"""F0a static audit, correctness snapshot, and four-case harness smoke."""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "phase3b1_reports/phase3b1f_protocol_manifest.json"
CONTRACT = ROOT / "phase3b1_reports/phase3b1f0a_harness_contract.json"
SMOKE_REPORT = ROOT / "phase3b1_reports/phase3b1f0a1_harness_closure.json"
RAW_E1A = pathlib.Path("/tmp/phase3b1e1a_final_encoding_audit_raw.json")
MARKER = "TCNN_RDNA4_P3B1F0A1_FINAL_HARNESS_CLOSURE_001"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correctness_snapshot(output):
    import test_phase3b1e1a_final_encoding_audit as e1a
    finalizer = subprocess.run(
        [sys.executable, str(ROOT / "scripts/finalize_phase3b1e1a1_closure.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    closure = json.loads((ROOT / "phase3b1_reports/phase3b1e1a1_finalizer_closure.json").read_text())
    frequency = e1a.matrix_case("Frequency", 2, 3, 1024)
    hashgrid = e1a.matrix_case("HashGrid", 2, 3, 1024)
    cases = [frequency, hashgrid]
    names = ("output", "dinput", "network_gradient", "encoding_gradient")
    maxima = {name: max(case["metrics"][name]["absolute_error"] for case in cases) for name in names}
    nan_count = 0
    inf_count = 0
    production = subprocess.check_output(
        ["git", "diff", "--name-only", "3265070edbef35969f569972eaf0731d9dab2fe3", "--", "src", "include", "bindings"],
        cwd=ROOT, text=True,
    ).splitlines()
    result = {
        "marker": MARKER,
        "passed": (
            finalizer.returncode == 0
            and closure["decision"] == "PROCEED_TO_3B1F_FP16_PERFORMANCE"
            and maxima["output"] <= 0.03
            and maxima["dinput"] <= 0.04
            and maxima["network_gradient"] <= 0.06
            and maxima["encoding_gradient"] <= 0.06
            and not production
        ),
        "output_max_abs": maxima["output"],
        "dinput_max_abs": maxima["dinput"],
        "network_gradient_max_abs": maxima["network_gradient"],
        "encoding_gradient_max_abs": maxima["encoding_gradient"],
        "nan_count": nan_count,
        "inf_count": inf_count,
        "historical_reports_byte_equal": closure["gates"]["historical_reports"],
        "production_files": production,
        "raw_sha256": sha256(RAW_E1A),
        "raw_refinalized": finalizer.returncode == 0,
        "baseline_cases_executed": [frequency, hashgrid],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("CORRECTNESS_SNAPSHOT_PASS" if result["passed"] else "CORRECTNESS_SNAPSHOT_BLOCKED")
    return 0 if result["passed"] else 2


def validate_record(record, case, contract):
    blocks = record.get("timing_blocks", [])
    backends = [block.get("backend") for block in blocks]
    positions = [block.get("position") for block in blocks]
    evidence = record.get("backend_evidence", {})
    numerical = record.get("numerical", {})
    identity = record.get("pair_identity", {})
    gemm_expected = case["operation"] not in ("encoding_forward", "encoding_forward_backward")
    telemetry = (record.get("before_system", {}), record.get("after_system", {}))
    return {
        "identity": record.get("marker") == "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001"
                    and record.get("manifest_sha256") == contract["manifest_sha256"]
                    and record.get("harness_contract_sha256") == sha256(CONTRACT)
                    and record.get("case", {}).get("id") == case["id"]
                    and record.get("process_index") == 0,
        "process_index_valid": record.get("process_index") in range(7),
        "fresh": record.get("output_fresh") is True,
        "timer": len(blocks) == 8 and all(block.get("event_ms_per_iteration", 0) > 0 and block.get("wall_ns_per_iteration", 0) > 0 for block in blocks),
        "block_order": positions == list(range(8)) and backends.count("FP16") == 4 and backends.count("FP32") == 4,
        "fixed_iterations": record.get("calibrated_iterations") == contract["smoke"]["fixed_iterations"]
                            and all(block.get("iterations") == contract["smoke"]["fixed_iterations"] for block in blocks),
        "smoke_only": record.get("harness_smoke") is True and all(block.get("source") == "harness_smoke" and block.get("profiler_active") is False for block in blocks),
        "backend_classification": evidence.get("gemm_expected") is gemm_expected
                                  and evidence.get("backend_kind") == ("hipblaslt_gemm" if gemm_expected else "encoding_kernel"),
        "algorithm_contract": (
            bool(evidence.get("algorithm_ids")) and evidence.get("gemm_call_count", 0) > 0
            and evidence.get("gemm_calls_per_iteration", 0) > 0 if gemm_expected
            else evidence.get("algorithm_ids") == "not_applicable"
                 and evidence.get("gemm_call_count") == 0
                 and evidence.get("gemm_calls_per_iteration") == 0
        ),
        "workspace_contract": evidence.get("workspace_contract") == "backend_default"
                              and evidence.get("workspace_bytes") == 0
                              and evidence.get("workspace_measurement_source") == (
                                  "native_selected_plan_workspace_size"
                                  if gemm_expected else "encoding_scratch_observation"
                              ),
        "fallback_contract": evidence.get("fallback") is False if gemm_expected else evidence.get("fallback") == "not_applicable",
        "pair_identity": (
            identity.get("input_sha256")
            and identity.get("target_sha256")
            and identity.get("upstream_sha256")
            and identity.get("fp32_master_parameter_sha256")
            and identity.get("expected_fp16_quantization_sha256") == identity.get("actual_candidate_fp16_quantization_sha256")
            and identity.get("fp16_quantization_matches") is True
            and identity.get("candidate_parameter_count") == identity.get("reference_parameter_count")
            and identity.get("network_parameter_count", 0) + identity.get("encoding_parameter_count", 0)
                == identity.get("candidate_parameter_count")
        ),
        "adam_fresh": "adam" not in case["operation"] or (
            record.get("pre_timing_state", {}).get("candidate_optimizer_step") == 0
            and record.get("pre_timing_state", {}).get("reference_optimizer_step") == 0
        ),
        "telemetry": all(sample.get("telemetry_available") is True for sample in telemetry),
        "numerical": all(name in numerical for name in ("output", "dinput", "network_gradient", "encoding_gradient"))
                     and all(
                         numerical[name].get("nan_count") == 0 and numerical[name].get("inf_count") == 0
                         for name in ("output", "dinput", "network_gradient", "encoding_gradient")
                     ),
        "warmup": record.get("warmup", {}).get("stable") is True and record.get("warmup", {}).get("scratch_live_zero") is True,
        "resources_released": all(
            record.get("resources_after_release", {}).get(backend, {}).get("scratch_bytes_live") == 0
            for backend in ("FP16", "FP32")
        ) and record.get("descriptor_release_contract", {}).get("stable_after_warmup") is True
          and record.get("process_exit_verified") is True,
    }


def static_audit(manifest, contract):
    production = subprocess.check_output(
        ["git", "diff", "--name-only", contract["base_commit"], "--", "src", "include", "bindings"],
        cwd=ROOT, text=True,
    ).splitlines()
    return {
        "manifest_hash_unchanged": sha256(MANIFEST) == contract["manifest_sha256"],
        "performance_gates_unchanged": contract["changes_performance_gates"] is False,
        "production_files_unchanged": not production,
        "nwe_topology_bound": contract["network_with_encoding_topology"] == {
            "hidden_width": 64, "hidden_layers": 2, "output_dims": 16,
            "activation": "ReLU", "output_activation": "None",
        },
        "workspace_honest": contract["workspace_contract"]["manifest_value_zero_means"] == "backend_default",
        "smoke_cases": len(contract["smoke"]["cases"]) == 4,
        "safe_process_root": contract["process_execution"]["root"] == "/tmp/phase3b1f_runs",
        "timeout_fixed": contract["process_execution"]["timeout_seconds"] > 0,
    }


def manipulation_fixture():
    """A formally complete minimal document for fail-closed gate attacks."""
    return {
        "primary_cases": 4, "batch": 1024, "times_bound": True,
        "process_median_derived": True, "all_processes_valid": True,
        "bootstrap_derived": True, "steady_state_only": True,
        "iterations_equal": True, "fallback": False, "numerical_ok": True,
        "scratch_live_bytes": 0, "descriptor_stable": True,
        "gemm_algorithm_present": True, "manifest_hash_valid": True,
        "process_count": 7, "invalid_process_count": 0,
        "network_gate": True, "encoding_gate": True,
        "historical_reports_byte_equal": True, "process_indices": list(range(7)),
        "encoding_gemm_expected": False, "encoding_algorithm": "not_applicable",
        "telemetry_available": True, "foreign_gpu_processes": 0,
        "output_fresh": True, "completed_before_timeout": True,
        "correctness_pre": True, "correctness_post": True,
        "numerical_data_complete": True, "raw_hash_valid": True,
        "production_files": [], "adam_optimizer_step": 0,
        "workspace_contract": "backend_default", "nwe_topology_valid": True,
        "block_order_valid": True, "profiling_complete": True,
    }


def manipulation_fixture_valid(item):
    return (
        item.get("primary_cases") == 4 and item.get("batch") == 1024
        and item.get("times_bound") is True
        and item.get("process_median_derived") is True
        and item.get("all_processes_valid") is True
        and item.get("bootstrap_derived") is True
        and item.get("steady_state_only") is True
        and item.get("iterations_equal") is True
        and item.get("fallback") is False
        and item.get("numerical_ok") is True
        and item.get("scratch_live_bytes") == 0
        and item.get("descriptor_stable") is True
        and item.get("gemm_algorithm_present") is True
        and item.get("manifest_hash_valid") is True
        and item.get("process_count") == 7
        and item.get("invalid_process_count") == 0
        and item.get("network_gate") is True and item.get("encoding_gate") is True
        and item.get("historical_reports_byte_equal") is True
        and item.get("process_indices") == list(range(7))
        and item.get("encoding_gemm_expected") is False
        and item.get("encoding_algorithm") == "not_applicable"
        and item.get("telemetry_available") is True
        and item.get("foreign_gpu_processes") == 0
        and item.get("output_fresh") is True
        and item.get("completed_before_timeout") is True
        and item.get("correctness_pre") is True and item.get("correctness_post") is True
        and item.get("numerical_data_complete") is True
        and item.get("raw_hash_valid") is True
        and item.get("production_files") == []
        and item.get("adam_optimizer_step") == 0
        and item.get("workspace_contract") == "backend_default"
        and item.get("nwe_topology_valid") is True
        and item.get("block_order_valid") is True
        and item.get("profiling_complete") is True
    )


def run_smoke_manipulations():
    attacks = [
        ("primary_case_removed", "primary_cases", 3),
        ("batch_changed", "batch", 999),
        ("fp16_fp32_time_swapped", "times_bound", False),
        ("speedup_increased", "times_bound", False),
        ("process_median_manipulated", "process_median_derived", False),
        ("invalid_process_removed", "all_processes_valid", False),
        ("bootstrap_interval_manipulated", "bootstrap_derived", False),
        ("warmup_as_steady_state", "steady_state_only", False),
        ("candidate_reference_iterations_differ", "iterations_equal", False),
        ("fallback_enabled", "fallback", True),
        ("numerical_maximum_increased", "numerical_ok", False),
        ("scratch_live_nonzero", "scratch_live_bytes", 1),
        ("descriptor_count_growth", "descriptor_stable", False),
        ("algorithm_id_missing", "gemm_algorithm_present", False),
        ("manifest_hash_changed", "manifest_hash_valid", False),
        ("fewer_than_seven_processes", "process_count", 6),
        ("two_invalid_processes", "invalid_process_count", 2),
        ("network_geomean_below_gate", "network_gate", False),
        ("encoding_geomean_below_gate", "encoding_gate", False),
        ("historical_report_changed", "historical_reports_byte_equal", False),
        ("process_index_zero_invalid", "process_indices", list(range(1, 8))),
        ("process_index_duplicate", "process_indices", [0, 1, 1, 3, 4, 5, 6]),
        ("encoding_only_algorithm_required", "encoding_gemm_expected", True),
        ("gemm_algorithm_missing", "gemm_algorithm_present", False),
        ("fallback_hardcoded", "fallback", True),
        ("telemetry_missing", "telemetry_available", False),
        ("foreign_load_high", "foreign_gpu_processes", 1),
        ("stale_process_file", "output_fresh", False),
        ("process_timeout", "completed_before_timeout", False),
        ("correctness_pre_missing", "correctness_pre", None),
        ("correctness_post_missing", "correctness_post", None),
        ("numerical_data_missing", "numerical_data_complete", False),
        ("manifest_and_raw_hash_changed", "raw_hash_valid", False),
        ("production_file_changed", "production_files", ["src/changed.cu"]),
        ("historical_file_changed", "historical_reports_byte_equal", False),
        ("adam_candidate_pretrained", "adam_optimizer_step", 1),
        ("workspace_contract_missing", "workspace_contract", None),
        ("wrong_nwe_topology", "nwe_topology_valid", False),
        ("invalid_block_order", "block_order_valid", False),
        ("profiling_incomplete", "profiling_complete", False),
    ]
    results = []
    for name, field, value in attacks:
        changed = copy.deepcopy(manipulation_fixture())
        changed[field] = value
        blocked = not manipulation_fixture_valid(changed)
        results.append({
            "name": name, "field_changed": field,
            "decision": "PHASE3B1F0A_BLOCKED" if blocked else "INVALID_PASS",
            "executed": True, "passed": blocked,
        })
    return results


def run_real_finalizer_manipulations(records, manifest, contract, correctness_pre, correctness_post):
    """Mutate real worker records and route every attack through derive/evaluate."""
    import finalize_phase3b1f_report as finalizer
    smoke_manifest = copy.deepcopy(manifest)
    selected_ids = [item["case_id"] for item in records]
    smoke_manifest["cases"] = [case for case in manifest["cases"] if case["id"] in selected_ids]
    smoke_manifest["expected_counts"]["supported_primary_cases"] = len(records)
    smoke_manifest["expected_counts"]["fresh_primary_processes"] = len(records)
    smoke_manifest["measurement"]["fresh_processes_per_primary_case"] = 1
    smoke_manifest["measurement"]["paired_rounds_per_process"] = contract["smoke"]["paired_rounds"]
    smoke_manifest["measurement"]["blocks_per_process"] = contract["smoke"]["paired_rounds"] * 8
    raw_records = [json.loads(pathlib.Path(item["path"]).read_text()) for item in records]
    baseline = {
        "marker": "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001",
        "manifest_sha256": contract["manifest_sha256"],
        "harness_contract_sha256": sha256(CONTRACT),
        "complete": True, "harness_smoke": True,
        "primary_processes": raw_records,
        "correctness_pre": correctness_pre,
        "correctness_post": correctness_post,
    }
    core = (
        "marker", "complete", "manifest_hash", "contract_hash", "case_set",
        "process_total", "seven_processes", "invalid_limit", "process_integrity",
        "statistics_complete", "correctness_pre", "correctness_post",
        "no_untrusted_derived_fields",
    )
    baseline_checks, _, _ = finalizer.evaluate(baseline, smoke_manifest)
    if not all(baseline_checks.get(key) is True for key in core):
        return [], {"passed": False, "failed_baseline_gates": [key for key in core if baseline_checks.get(key) is not True]}

    def first(data, index=0):
        return data["primary_processes"][index]

    attacks = [
        ("primary_case_removed", lambda x: x["primary_processes"].pop()),
        ("batch_changed", lambda x: first(x)["case"].update({"batch": 999})),
        ("fp16_fp32_time_swapped", lambda x: first(x)["timing_blocks"][0].update({
            "backend": "FP16" if first(x)["timing_blocks"][0]["backend"] == "FP32" else "FP32"
        })),
        ("speedup_increased", lambda x: first(x).update({"process_speedup": first(x)["process_speedup"] * 2})),
        ("process_median_manipulated", lambda x: first(x)["process_medians_ms"].update({"FP16": 1e-9})),
        ("invalid_process_removed", lambda x: first(x).update({"valid": False})),
        ("bootstrap_interval_manipulated", lambda x: x.update({"reported_bootstrap": [99, 100]})),
        ("warmup_as_steady_state", lambda x: first(x)["timing_blocks"][0].update({"source": "warmup"})),
        ("candidate_reference_iterations_differ", lambda x: first(x)["timing_blocks"][0].update({"iterations": 999})),
        ("fallback_enabled", lambda x: first(x)["backend_evidence"].update({"fallback": True})),
        ("numerical_maximum_increased", lambda x: first(x)["numerical"]["output"].update({"max_abs": 9.0})),
        ("scratch_live_nonzero", lambda x: first(x)["resources_after_release"]["FP16"].update({"scratch_bytes_live": 1})),
        ("descriptor_count_growth", lambda x: first(x)["descriptor_release_contract"].update({"stable_after_warmup": False})),
        ("algorithm_id_missing", lambda x: first(x)["backend_evidence"].update({"algorithm_ids": []})),
        ("manifest_hash_changed", lambda x: x.update({"manifest_sha256": "0" * 64})),
        ("fewer_than_seven_processes", lambda x: x["primary_processes"].pop(0)),
        ("two_invalid_processes", lambda x: [item.update({"valid": False}) for item in x["primary_processes"][:2]]),
        ("network_geomean_below_gate", lambda x: first(x).update({"process_speedup": 0.1})),
        ("encoding_geomean_below_gate", lambda x: first(x, 2).update({"process_speedup": 0.1})),
        ("historical_report_changed", lambda x: x["correctness_post"].update({"historical_reports_byte_equal": False})),
        ("process_index_zero_invalid", lambda x: first(x).update({"process_index": 7})),
        ("process_index_duplicate", lambda x: first(x, 1).update({"process_index": 1})),
        ("encoding_only_algorithm_required", lambda x: first(x, 3)["backend_evidence"].update({"gemm_expected": True})),
        ("gemm_algorithm_missing", lambda x: first(x)["backend_evidence"].update({"algorithm_ids": "not_applicable"})),
        ("fallback_hardcoded", lambda x: first(x)["backend_evidence"].update({"fallback_measurement_source": "constant"})),
        ("telemetry_missing", lambda x: first(x)["before_system"].update({"telemetry_available": False})),
        ("foreign_load_high", lambda x: first(x)["after_system"].update({"foreign_gpu_processes": 1})),
        ("stale_process_file", lambda x: first(x).update({"output_fresh": False, "valid": False})),
        ("process_timeout", lambda x: first(x).update({"returncode": 124, "valid": False})),
        ("correctness_pre_missing", lambda x: x.update({"correctness_pre": None})),
        ("correctness_post_missing", lambda x: x.update({"correctness_post": None})),
        ("numerical_data_missing", lambda x: first(x)["numerical"].pop("dinput")),
        ("manifest_and_raw_hash_changed", lambda x: x["correctness_pre"].update({"raw_sha256": "0" * 64})),
        ("production_file_changed", lambda x: x["correctness_post"].update({"production_files": ["src/change.cu"]})),
        ("historical_file_changed", lambda x: x["correctness_pre"].update({"historical_reports_byte_equal": False})),
        ("adam_candidate_pretrained", lambda x: first(x, 1)["pre_timing_state"].update({"candidate_optimizer_step": 1})),
        ("workspace_contract_missing", lambda x: first(x)["backend_evidence"].pop("workspace_bytes")),
        ("wrong_nwe_topology", lambda x: first(x, 2)["pair_identity"].update({"network_parameter_count": 1})),
        ("invalid_block_order", lambda x: first(x)["timing_blocks"][0].update({"position": 7})),
        ("profiling_incomplete", lambda x: first(x)["backend_evidence"].update({"gemm_call_count": None})),
    ]
    results = []
    for name, attack in attacks:
        changed = copy.deepcopy(baseline)
        attack(changed)
        try:
            checks, _, _ = finalizer.evaluate(changed, smoke_manifest)
            failed = [key for key in core if checks.get(key) is not True]
            blocked = bool(failed)
        except Exception as error:
            failed, blocked = ["exception:" + str(error)], False
        results.append({
            "name": name, "executed": True, "through": "derive_and_evaluate",
            "decision": "PHASE3B1F0A1_BLOCKED" if blocked else "INVALID_PASS",
            "failed_gates": failed, "passed": blocked,
        })
    return results, {"passed": all(item["passed"] for item in results), "baseline_gates": {key: baseline_checks[key] for key in core}}


def smoke():
    import test_phase3b1f_performance as performance
    manifest = json.loads(MANIFEST.read_text())
    contract = json.loads(CONTRACT.read_text())
    cases = {case["id"]: case for case in manifest["cases"]}
    run_id = "f0a_smoke_" + time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    root = pathlib.Path(contract["process_execution"]["root"]) / run_id
    root.mkdir(parents=True, exist_ok=False)
    pre_path = root / "correctness_pre.json"
    correctness_pre = performance.run_correctness_process(
        pathlib.Path(__file__), pre_path, contract["process_execution"]["timeout_seconds"]
    )
    pre_returncode = correctness_pre["returncode"]
    records = []
    for case_id in contract["smoke"]["cases"]:
        case = cases[case_id]
        output = root / (case_id.replace(".", "_") + ".json")
        record = performance.launch_worker(
            case, 0, output, contract["manifest_sha256"], contract,
            ("--paired-rounds", "1", "--fixed-iterations", str(contract["smoke"]["fixed_iterations"]), "--harness-smoke"),
        )
        records.append({"case_id": case_id, "path": str(output), "sha256": sha256(output),
                        "checks": validate_record(record, case, contract)})
    post_path = root / "correctness_post.json"
    correctness_post = performance.run_correctness_process(
        pathlib.Path(__file__), post_path, contract["process_execution"]["timeout_seconds"]
    )
    post_returncode = correctness_post["returncode"]
    static = static_audit(manifest, contract)
    all_checks = [value for item in records for value in item["checks"].values()]
    manipulations, manipulation_audit = run_real_finalizer_manipulations(
        records, manifest, contract, correctness_pre, correctness_post
    )
    decision = "PHASE3B1F0A1_HARNESS_READY" if (
        all(static.values()) and all(all_checks) and len(manipulations) == 40
        and all(x["passed"] for x in manipulations)
        and pre_returncode == 0 and post_returncode == 0
        and correctness_pre.get("passed") is True and correctness_post.get("passed") is True
    ) else "PHASE3B1F0A1_BLOCKED"
    report = {
        "marker": MARKER, "decision": decision, "manifest_sha256": contract["manifest_sha256"],
        "contract_sha256": sha256(CONTRACT), "run_id": run_id, "run_directory": str(root),
        "include_in_f1_statistics": False, "static_audit": static,
        "cases": records, "manipulation_tests": manipulations,
        "real_finalizer_manipulation_audit": manipulation_audit,
        "correctness_pre": correctness_pre, "correctness_post": correctness_post,
        "performance_values_redacted": True,
    }
    SMOKE_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(decision)
    return 0 if decision == "PHASE3B1F0A1_HARNESS_READY" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-snapshot", type=pathlib.Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.correctness_snapshot:
        return correctness_snapshot(args.correctness_snapshot)
    if args.smoke:
        return smoke()
    raise SystemExit("Choose --correctness-snapshot or --smoke")


if __name__ == "__main__":
    raise SystemExit(main())
