#!/usr/bin/env python3
"""Fail-closed F1 finalizer; not executed during the F0 protocol freeze."""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import random
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "phase3b1_reports/phase3b1f_protocol_manifest.json"
CONTRACT = ROOT / "phase3b1_reports/phase3b1f0a_harness_contract.json"
OUT = ROOT / "phase3b1_reports/phase3b1f_fp16_performance.json"
MD = ROOT / "phase3b1_reports/PHASE3B1F_FP16_PERFORMANCE.md"
BASE = "3265070edbef35969f569972eaf0731d9dab2fe3"
MARKER = "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def bootstrap(values, seed, resamples):
    generator = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [values[generator.randrange(len(values))] for _ in values]
        samples.append(statistics.median(draw))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def geomean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def paired_matrix_bootstrap(items, seed, resamples):
    """Bootstrap case and fresh-process dimensions from paired speedups."""
    usable = [item["speedups"] for item in items if item.get("speedups")]
    if not usable:
        return [None, None]
    generator = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled_cases = [usable[generator.randrange(len(usable))] for _ in usable]
        case_estimates = []
        for values in sampled_cases:
            sampled_processes = [values[generator.randrange(len(values))] for _ in values]
            case_estimates.append(statistics.median(sampled_processes))
        draws.append(geomean(case_estimates))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def correctness_snapshot_valid(snapshot, manifest):
    if not isinstance(snapshot, dict):
        return False
    limits = manifest["numerical_gates"]["frozen_absolute"]
    required = (
        "passed", "output_max_abs", "dinput_max_abs", "network_gradient_max_abs",
        "encoding_gradient_max_abs", "nan_count", "inf_count",
        "historical_reports_byte_equal", "production_files", "raw_sha256",
    )
    if not all(key in snapshot for key in required):
        return False
    return (
        snapshot["passed"] is True
        and snapshot["output_max_abs"] <= limits["network_output"]
        and snapshot["dinput_max_abs"] <= limits["dinput"]
        and snapshot["network_gradient_max_abs"] <= limits["network_gradient"]
        and snapshot["encoding_gradient_max_abs"] <= limits["encoding_gradient"]
        and snapshot["nan_count"] == 0
        and snapshot["inf_count"] == 0
        and snapshot["historical_reports_byte_equal"] is True
        and snapshot["production_files"] == []
        and snapshot["raw_sha256"] == "049d977ee88024017592c4649ff3f16ad21b2f24cbdcbfc90d7873eb1583bf9e"
    )


def backend_evidence_valid(evidence, operation):
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        return False
    common = (
        isinstance(evidence.get("workspace_bytes"), int)
        and evidence["workspace_bytes"] >= 0
        and isinstance(evidence.get("gemm_call_count"), int)
        and isinstance(evidence.get("gemm_calls_per_iteration"), (int, float))
        and math.isfinite(evidence["gemm_calls_per_iteration"])
    )
    if evidence.get("gemm_expected") is False:
        return common and all((
            evidence.get("backend_kind") == "encoding_kernel",
            evidence.get("algorithm_ids") == "not_applicable",
            evidence.get("workspace_measurement_source") == "encoding_scratch_observation",
            evidence.get("fallback") == "not_applicable",
            evidence.get("fallback_measurement_source") == "not_applicable",
            evidence.get("gemm_call_count") == 0,
            evidence.get("gemm_calls_per_iteration") == 0,
            evidence.get("custom_relu_backward_biasgrad") == "not_applicable",
        ))
    backward = "backward" in operation or "adam" in operation
    custom_valid = (
        evidence.get("custom_relu_backward_biasgrad") is True
        and evidence.get("custom_relu_backward_biasgrad_source") == "native_dx_dw_dz_db_launch_counter_deltas"
        if backward else evidence.get("custom_relu_backward_biasgrad") == "not_applicable"
    )
    return common and all((
        evidence.get("backend_kind") == "hipblaslt_gemm",
        evidence.get("gemm_expected") is True,
        isinstance(evidence.get("algorithm_ids"), list),
        bool(evidence.get("algorithm_ids")),
        all(isinstance(value, int) for value in evidence.get("algorithm_ids", [])),
        evidence.get("workspace_measurement_source") == "native_selected_plan_workspace_size",
        evidence.get("fallback") is False,
        evidence.get("fallback_measurement_source") == "orchestrator_hipblaslt_log",
        evidence.get("gemm_call_count") > 0,
        evidence.get("gemm_calls_per_iteration") > 0,
        custom_valid,
    ))


def numerical_valid(numerical, manifest):
    if not isinstance(numerical, dict):
        return False
    limits = manifest["numerical_gates"]["frozen_absolute"]
    mapping = {
        "output": limits["network_output"], "dinput": limits["dinput"],
        "network_gradient": limits["network_gradient"],
        "encoding_gradient": limits["encoding_gradient"],
    }
    return all(
        isinstance(numerical.get(name), dict)
        and isinstance(numerical[name].get("max_abs"), (int, float))
        and math.isfinite(numerical[name]["max_abs"])
        and numerical[name]["max_abs"] <= limit
        and isinstance(numerical[name].get("normalized_l2"), (int, float))
        and math.isfinite(numerical[name]["normalized_l2"])
        and numerical[name].get("nan_count") == 0
        and numerical[name].get("inf_count") == 0
        for name, limit in mapping.items()
    )


def derive(data, manifest):
    cases = {case["id"]: case for case in manifest["cases"] if case["supported"]}
    grouped = {case_id: [] for case_id in cases}
    unknown = []
    for process in data.get("primary_processes", []):
        case_id = process.get("case", {}).get("id")
        if case_id not in grouped:
            unknown.append(process)
        else:
            grouped[case_id].append(process)
    results = {}
    for case_id, processes in grouped.items():
        valid = [item for item in processes if item.get("valid") is True and item.get("returncode") == 0]
        speedups = []
        integrity = []
        for item in processes:
            blocks = item.get("timing_blocks", [])
            iterations = {block.get("iterations") for block in blocks}
            fp16 = [block["event_ms_per_iteration"] for block in blocks if block.get("backend") == "FP16"]
            fp32 = [block["event_ms_per_iteration"] for block in blocks if block.get("backend") == "FP32"]
            calculated = statistics.median(fp32) / statistics.median(fp16) if fp16 and fp32 else None
            speedup_equal = calculated is not None and abs(calculated - item.get("process_speedup", -1)) <= 1e-12
            process_index = item.get("process_index")
            warmup = item.get("warmup", {})
            operation = cases[case_id]["operation"]
            warmup_minimum = manifest["warmup"]["training_iterations_min"] if "adam" in operation else manifest["warmup"]["default_iterations_min"]
            allowed_orders = [
                order + list(reversed(order))
                for order in manifest["measurement"]["orders"]
            ]
            rounds = {}
            for block in blocks:
                rounds.setdefault(block.get("round"), []).append(block)
            expected_rounds = manifest["measurement"]["paired_rounds_per_process"]
            expected_blocks = manifest["measurement"]["blocks_per_process"]
            timing_integrity = (
                len(blocks) == expected_blocks
                and sum(block.get("backend") == "FP16" for block in blocks) == expected_blocks // 2
                and sum(block.get("backend") == "FP32" for block in blocks) == expected_blocks // 2
                and set(rounds) == set(range(expected_rounds))
                and all(
                    [block.get("position") for block in sorted(round_blocks, key=lambda x: x.get("position", -1))] == list(range(8))
                    and [block.get("backend") for block in sorted(round_blocks, key=lambda x: x.get("position", -1))] in allowed_orders
                    for round_blocks in rounds.values()
                )
                and len(iterations) == 1
                and iterations
                and (
                    data.get("harness_smoke") is True
                    or manifest["calibration"]["min_iterations"] <= next(iter(iterations)) <= manifest["calibration"]["max_iterations"]
                )
                and item.get("calibrated_iterations") == next(iter(iterations))
                and all(
                    block.get("event_ms_per_iteration", 0) > 0
                    and block.get("wall_ns_per_iteration", 0) > 0
                    and block.get("source") == ("harness_smoke" if data.get("harness_smoke") is True else "steady_state")
                    and block.get("profiler_active") is False
                    for block in blocks
                )
            )
            stable_fields = ("cache_size", "heuristic_queries", "execution_handle_count", "execution_handle_creations", "descriptor_count")
            warmup_derived = (
                warmup.get("iterations", 0) >= warmup_minimum
                and all(
                    warmup.get("stable_sample", {}).get(backend, {}).get(field)
                    == warmup.get("after", {}).get(backend, {}).get(field)
                    for backend in ("FP16", "FP32") for field in stable_fields
                )
                and all(
                    warmup.get("after", {}).get(backend, {}).get("scratch_bytes_live") == 0
                    for backend in ("FP16", "FP32")
                )
            )
            telemetry = (item.get("before_system", {}), item.get("after_system", {}))
            telemetry_valid = all(
                sample.get("telemetry_available") is True
                and sample.get("foreign_gpu_processes") == 0
                and sample.get("temperature_c") is not None
                and manifest["system_stability"]["temperature_c_min"] <= sample["temperature_c"] <= manifest["system_stability"]["temperature_c_max"]
                and sample.get("clock_mhz") is not None
                for sample in telemetry
            )
            identity = item.get("pair_identity", {})
            ranges_valid = (
                identity.get("fp16_quantization_matches") is True
                and identity.get("expected_fp16_quantization_sha256") == identity.get("actual_candidate_fp16_quantization_sha256")
                and identity.get("candidate_parameter_count") == identity.get("reference_parameter_count")
                and identity.get("network_parameter_count", 0) + identity.get("encoding_parameter_count", 0)
                    == identity.get("candidate_parameter_count")
            )
            adam = "adam" in operation
            pre = item.get("pre_timing_state", {})
            adam_valid = (
                pre.get("candidate_optimizer_step") == 0
                and pre.get("reference_optimizer_step") == 0
                and pre.get("adam_state_initialized_outside_timing") is True
                and isinstance(pre.get("candidate_optimizer_state_sha256"), str)
                and isinstance(pre.get("reference_optimizer_state_sha256"), str)
            ) if adam else True
            calibration = item.get("calibration", {})
            calibration_valid = (
                calibration.get("feasible") is True
                and (
                    data.get("harness_smoke") is True
                    or (
                        calibration.get("both_reach_target_min") is True
                        and calibration.get("slower_within_target_max") is True
                    )
                )
            )
            integrity.append({
                "process_index": item.get("process_index"),
                "process_index_valid": process_index in range(7) if isinstance(process_index, int) else False,
                "timing_integrity": bool(timing_integrity),
                "speedup_derived": speedup_equal,
                "process_medians_derived": (
                    item.get("process_medians_ms", {}).get("FP16") == (statistics.median(fp16) if fp16 else None)
                    and item.get("process_medians_ms", {}).get("FP32") == (statistics.median(fp32) if fp32 else None)
                ),
                "case_config_exact": item.get("case") == cases[case_id],
                "warmup_derived": warmup_derived,
                "telemetry_valid": telemetry_valid,
                "numerical_valid": numerical_valid(item.get("numerical"), manifest),
                "backend_evidence_valid": backend_evidence_valid(item.get("backend_evidence"), operation),
                "pair_identity_valid": ranges_valid,
                "adam_state_valid": adam_valid,
                "calibration_valid": calibration_valid,
                "contract_hash_valid": item.get("harness_contract_sha256") == sha256(CONTRACT),
                "scratch_zero": all(
                    counters.get("scratch_bytes_live") == 0
                    for counters in item.get("resources_after_release", {}).values()
                ),
                "descriptor_released": all(
                    item.get("resources_after_release", {}).get(backend, {}).get("scratch_bytes_live") == 0
                    for backend in ("FP16", "FP32")
                ) and item.get("descriptor_release_contract", {}).get("stable_after_warmup") is True
                  and item.get("process_exit_verified") is True,
            })
            if item in valid and calculated is not None:
                speedups.append(calculated)
        results[case_id] = {
            "case": cases[case_id],
            "process_count": len(processes),
            "valid_process_count": len(valid),
            "invalid_processes": [item for item in processes if item not in valid],
            "integrity": integrity,
            "process_indices": sorted(item.get("process_index") for item in processes if isinstance(item.get("process_index"), int)),
            "speedups": speedups,
        }
        if speedups:
            median = statistics.median(speedups)
            results[case_id]["statistics"] = {
                "median": median,
                "mad": statistics.median(abs(x - median) for x in speedups),
                "p10": percentile(speedups, 0.10),
                "p90": percentile(speedups, 0.90),
                "minimum": min(speedups),
                "maximum": max(speedups),
                "bootstrap_95": bootstrap(
                    speedups,
                    manifest["statistics"]["bootstrap"]["seed"] + cases[case_id]["seed"],
                    manifest["statistics"]["bootstrap"]["resamples"],
                ),
            }
    return results, unknown


def evaluate(data, manifest):
    results, unknown = derive(data, manifest)
    expected = manifest["expected_counts"]
    checks = {
        "marker": data.get("marker") == MARKER,
        "complete": data.get("complete") is True,
        "manifest_hash": data.get("manifest_sha256") == sha256(MANIFEST),
        "contract_hash": data.get("harness_contract_sha256") == sha256(CONTRACT),
        "case_set": set(results) == {case["id"] for case in manifest["cases"] if case["supported"]} and not unknown,
        "process_total": len(data.get("primary_processes", [])) == expected["fresh_primary_processes"],
        "seven_processes": all(
            item["process_count"] == manifest["measurement"]["fresh_processes_per_primary_case"]
            and item["process_indices"] == list(range(manifest["measurement"]["fresh_processes_per_primary_case"]))
            for item in results.values()
        ),
        "invalid_limit": all(len(item["invalid_processes"]) <= 1 for item in results.values()),
        "process_integrity": all(
            all(all(value for key, value in row.items() if key != "process_index") for row in item["integrity"])
            for item in results.values()
        ),
        "statistics_complete": all("statistics" in item for item in results.values()),
        "cold_start": (
            data.get("cold_start", {}).get("status") == "complete"
            and len(data.get("cold_start", {}).get("results", [])) >= expected["cold_start_processes_min"]
            and all(item.get("valid") is True for item in data["cold_start"]["results"])
        ),
        "profiling": (
            data.get("profiling", {}).get("status") == "complete"
            and len(data.get("profiling", {}).get("results", [])) == expected["profiling_cases"]
            and all(
                item.get("valid") is True
                and item.get("parsed", {}).get("parsed_kernel_rows", 0) > 0
                and all(key in item.get("parsed", {}) for key in (
                    "kernels", "gemm_share", "encoding_share", "gradient_scratch_share",
                    "optimizer_share", "unexpected_copies", "host_synchronizations",
                ))
                for item in data["profiling"]["results"]
            )
        ),
        "correctness_pre": correctness_snapshot_valid(data.get("correctness_pre"), manifest),
        "correctness_post": correctness_snapshot_valid(data.get("correctness_post"), manifest),
        "no_untrusted_derived_fields": not any(key in data for key in ("reported_statistics", "reported_bootstrap", "reported_geomeans")),
    }
    network_gates = manifest["performance_gates"]["network_large_batch_geomean"]
    network_ci = manifest["performance_gates"]["network_large_batch_bootstrap_lower"]
    network_summary = {}
    for operation in ("forward", "forward_backward", "adam_training_step"):
        selected = [item for item in results.values() if item["case"]["family"] == "network_only"
                    and item["case"]["operation"] == operation and item["case"]["batch"] >= network_gates["batch_min"]]
        medians = [item["statistics"]["median"] for item in selected if "statistics" in item]
        network_summary[operation] = {
            "geomean": geomean(medians) if medians else None,
            "matrix_bootstrap_95": paired_matrix_bootstrap(
                selected, manifest["statistics"]["bootstrap"]["seed"] + sum(map(ord, operation)),
                manifest["statistics"]["bootstrap"]["resamples"],
            ),
            "required_geomean": network_gates[operation],
            "required_bootstrap_lower": network_ci[operation],
        }
    encoding_gates = manifest["performance_gates"]["encoding_large_batch_geomean"]
    encoding_summary = {}
    for operation in ("network_with_encoding_forward_backward", "network_with_encoding_adam_step"):
        selected = [item for item in results.values() if item["case"]["family"] == "encoding"
                    and item["case"]["operation"] == operation and item["case"]["batch"] >= encoding_gates["batch_min"]]
        medians = [item["statistics"]["median"] for item in selected if "statistics" in item]
        encoding_summary[operation] = {
            "geomean": geomean(medians) if medians else None,
            "matrix_bootstrap_95": paired_matrix_bootstrap(
                selected, manifest["statistics"]["bootstrap"]["seed"] + sum(map(ord, operation)),
                manifest["statistics"]["bootstrap"]["resamples"],
            ),
            "required_geomean": encoding_gates[operation],
        }
    checks["network_performance"] = all(
        item["geomean"] is not None and item["geomean"] >= item["required_geomean"]
        and item["matrix_bootstrap_95"][0] is not None
        and item["matrix_bootstrap_95"][0] >= item["required_bootstrap_lower"]
        for item in network_summary.values()
    )
    checks["encoding_performance"] = all(
        item["geomean"] is not None and item["geomean"] >= item["required_geomean"]
        for item in encoding_summary.values()
    )
    regression = manifest["performance_gates"]["large_case_regression"]
    checks["no_large_regression"] = all(
        not (
            item["case"]["batch"] >= 1024
            and item.get("statistics", {}).get("median", 1) < regression["median_below"]
            and item.get("statistics", {}).get("bootstrap_95", [1, 1])[1] < regression["bootstrap_upper_below"]
        )
        for item in results.values()
    )
    return checks, results, {"network": network_summary, "encoding": encoding_summary}


def mutation_names():
    # These names are audited in F0 and each mutation is executed against F1 raw data.
    return (
        "primary_case_removed", "batch_changed", "fp16_fp32_time_swapped", "speedup_increased",
        "process_median_manipulated", "invalid_process_removed", "bootstrap_interval_manipulated",
        "warmup_as_steady_state", "candidate_reference_iterations_differ", "fallback_enabled",
        "numerical_maximum_increased", "scratch_live_nonzero", "descriptor_count_growth",
        "algorithm_id_missing", "manifest_hash_changed", "fewer_than_seven_processes",
        "two_invalid_processes", "network_geomean_below_gate", "encoding_geomean_below_gate",
        "historical_report_changed", "process_index_zero_invalid", "process_index_duplicate",
        "encoding_only_algorithm_required", "gemm_algorithm_missing", "fallback_hardcoded",
        "telemetry_missing", "foreign_load_high", "stale_process_file", "process_timeout",
        "correctness_pre_missing", "correctness_post_missing", "numerical_data_missing",
        "manifest_and_raw_hash_changed", "production_file_changed", "historical_file_changed",
        "adam_candidate_pretrained", "workspace_contract_missing", "wrong_nwe_topology",
        "invalid_block_order", "profiling_incomplete",
    )


def run_manipulations(data, manifest):
    tests = []
    first_case = manifest["cases"][0]["id"]

    def run(name, callback):
        changed = copy.deepcopy(data)
        callback(changed)
        changed_checks, _, _ = evaluate(changed, manifest)
        blocked = not all(changed_checks.values())
        tests.append({"name": name, "decision": "PHASE3B1F_BLOCKED" if blocked else "INVALID_PASS", "executed": True, "passed": blocked})

    def process(case_id=first_case, index=0):
        return [x for x in data["primary_processes"] if x["case"]["id"] == case_id][index]

    run("primary_case_removed", lambda x: x["primary_processes"].__setitem__(slice(None), [p for p in x["primary_processes"] if p["case"]["id"] != first_case]))
    run("batch_changed", lambda x: x["primary_processes"][0]["case"].update({"batch": 999}))
    run("fp16_fp32_time_swapped", lambda x: [b.update({"backend": "FP32" if b["backend"] == "FP16" else "FP16"}) for b in x["primary_processes"][0]["timing_blocks"]])
    run("speedup_increased", lambda x: x["primary_processes"][0].update({"process_speedup": x["primary_processes"][0]["process_speedup"] * 2}))
    run("process_median_manipulated", lambda x: x["primary_processes"][0]["process_medians_ms"].update({"FP16": 0.000001}))
    run("invalid_process_removed", lambda x: x["primary_processes"].pop(0))
    run("bootstrap_interval_manipulated", lambda x: x.update({"reported_bootstrap": [99, 100]}))
    run("warmup_as_steady_state", lambda x: x["primary_processes"][0]["timing_blocks"][0].update({"source": "warmup"}))
    run("candidate_reference_iterations_differ", lambda x: x["primary_processes"][0]["timing_blocks"][0].update({"iterations": 1}))
    run("fallback_enabled", lambda x: x["primary_processes"][0]["backend_evidence"].update({"fallback": True}))
    run("numerical_maximum_increased", lambda x: x["primary_processes"][0]["numerical"]["output"].update({"max_abs": 9.0}))
    run("scratch_live_nonzero", lambda x: x["primary_processes"][0]["resources_after_release"]["FP16"].update({"scratch_bytes_live": 1}))
    run("descriptor_count_growth", lambda x: x["primary_processes"][0]["descriptor_release_contract"].update({"stable_after_warmup": False}))
    run("algorithm_id_missing", lambda x: x["primary_processes"][0]["backend_evidence"].update({"algorithm_ids": []}))
    run("manifest_hash_changed", lambda x: x.update({"manifest_sha256": "0" * 64}))
    run("fewer_than_seven_processes", lambda x: x["primary_processes"].pop(0))
    def two_invalid(x):
        selected = [p for p in x["primary_processes"] if p["case"]["id"] == first_case][:2]
        for item in selected:
            item.update({"valid": False, "invalid_reasons": ["synthetic"]})
    run("two_invalid_processes", two_invalid)
    def slow_family(x, family, operation):
        for item in x["primary_processes"]:
            if item["case"]["family"] == family and item["case"]["operation"] == operation and item["case"]["batch"] >= 1024:
                for block in item["timing_blocks"]:
                    if block["backend"] == "FP16":
                        block["event_ms_per_iteration"] *= 4
                fp16 = [b["event_ms_per_iteration"] for b in item["timing_blocks"] if b["backend"] == "FP16"]
                fp32 = [b["event_ms_per_iteration"] for b in item["timing_blocks"] if b["backend"] == "FP32"]
                item["process_medians_ms"] = {"FP16": statistics.median(fp16), "FP32": statistics.median(fp32)}
                item["process_speedup"] = item["process_medians_ms"]["FP32"] / item["process_medians_ms"]["FP16"]
    run("network_geomean_below_gate", lambda x: slow_family(x, "network_only", "forward"))
    run("encoding_geomean_below_gate", lambda x: slow_family(x, "encoding", "network_with_encoding_forward_backward"))
    run("historical_report_changed", lambda x: x["correctness_post"].update({"passed": False, "historical_reports_byte_equal": False}))
    run("process_index_zero_invalid", lambda x: x["primary_processes"][0].update({"process_index": 7}))
    run("process_index_duplicate", lambda x: x["primary_processes"][1].update({"process_index": x["primary_processes"][0]["process_index"]}))
    encoding_only = next(i for i, item in enumerate(data["primary_processes"]) if item["backend_evidence"]["gemm_expected"] is False)
    gemm = next(i for i, item in enumerate(data["primary_processes"]) if item["backend_evidence"]["gemm_expected"] is True)
    run("encoding_only_algorithm_required", lambda x: x["primary_processes"][encoding_only]["backend_evidence"].update({"gemm_expected": True}))
    run("gemm_algorithm_missing", lambda x: x["primary_processes"][gemm]["backend_evidence"].update({"algorithm_ids": []}))
    run("fallback_hardcoded", lambda x: x["primary_processes"][gemm]["backend_evidence"].update({"fallback_measurement_source": "constant"}))
    run("telemetry_missing", lambda x: x["primary_processes"][0]["before_system"].update({"telemetry_available": False}))
    run("foreign_load_high", lambda x: x["primary_processes"][0]["after_system"].update({"foreign_gpu_processes": 1}))
    run("stale_process_file", lambda x: x["primary_processes"][0].update({"valid": False, "output_fresh": False}))
    run("process_timeout", lambda x: x["primary_processes"][0].update({"valid": False, "returncode": 124}))
    run("correctness_pre_missing", lambda x: x.update({"correctness_pre": None}))
    run("correctness_post_missing", lambda x: x.update({"correctness_post": None}))
    run("numerical_data_missing", lambda x: x["primary_processes"][0]["numerical"].pop("dinput"))
    run("manifest_and_raw_hash_changed", lambda x: x["correctness_pre"].update({"raw_sha256": "0" * 64}))
    run("production_file_changed", lambda x: x["correctness_post"].update({"production_files": ["src/change.cu"]}))
    run("historical_file_changed", lambda x: x["correctness_pre"].update({"historical_reports_byte_equal": False}))
    adam_index = next(i for i, item in enumerate(data["primary_processes"]) if "adam" in item["case"]["operation"])
    run("adam_candidate_pretrained", lambda x: x["primary_processes"][adam_index]["pre_timing_state"].update({"candidate_optimizer_step": 1}))
    run("workspace_contract_missing", lambda x: x["primary_processes"][gemm]["backend_evidence"].pop("workspace_bytes"))
    nwe_index = next(i for i, item in enumerate(data["primary_processes"]) if item["pair_identity"].get("model_kind") == "network_with_input_encoding")
    run("wrong_nwe_topology", lambda x: x["primary_processes"][nwe_index]["pair_identity"].update({"network_parameter_count": 1}))
    run("invalid_block_order", lambda x: x["primary_processes"][0]["timing_blocks"][0].update({"position": 99}))
    run("profiling_incomplete", lambda x: x["profiling"].update({"status": "incomplete"}))
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text())
        data = json.loads(args.raw.read_text())
    except (OSError, json.JSONDecodeError) as error:
        data = {}
        manifest = {}
        checks, cases, summaries = {"formal_input": False, "input_error": False}, {}, {"error": str(error)}
    else:
        try:
            checks, cases, summaries = evaluate(data, manifest)
            checks["formal_input"] = all((
                data.get("complete") is True,
                isinstance(data.get("primary_processes"), list),
                isinstance(data.get("correctness_pre"), dict),
                isinstance(data.get("correctness_post"), dict),
            ))
        except (KeyError, TypeError, ValueError, IndexError, statistics.StatisticsError, ZeroDivisionError) as error:
            checks, cases, summaries = {"formal_input": False}, {}, {"error": str(error)}
    performance_only = ("network_performance", "encoding_performance", "no_large_regression")
    correctness_stability = checks.get("formal_input") is True and all(
        value for key, value in checks.items() if key not in performance_only
    )
    performance = all(checks.get(key) is True for key in performance_only)
    if not correctness_stability:
        decision = "PHASE3B1F_BLOCKED"
    elif performance:
        decision = "PHASE3B1_FP16_PERFORMANCE_PASS"
    else:
        decision = "PHASE3B1F_CORRECT_BUT_NOT_PERFORMANT"
    manipulations = run_manipulations(data, manifest) if checks.get("formal_input") and checks.get("case_set") and checks.get("process_total") else []
    if [item["name"] for item in manipulations] != list(mutation_names()) or not all(item["passed"] for item in manipulations):
        decision = "PHASE3B1F_BLOCKED"
        checks["manipulations"] = False
    result = {
        "marker": MARKER, "base_commit": BASE, "decision": decision, "gates": checks,
        "manifest_sha256": sha256(MANIFEST), "cases": cases, "summaries": summaries,
        "manipulation_tests": manipulations,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    failed = [name for name, passed in checks.items() if passed is not True]
    MD.write_text(
        "# Phase 3B1-F – FP16 Performance Qualification\n\n"
        f"Decision: `{decision}`\n\n"
        "## Integrity and correctness\n\n"
        f"- Failed gates: `{failed}`\n"
        f"- Correctness pre/post: `{checks.get('correctness_pre')}` / `{checks.get('correctness_post')}`\n"
        f"- Process integrity: `{checks.get('process_integrity')}`\n"
        f"- Profiling: `{checks.get('profiling')}`\n\n"
        "## Network matrix\n\n"
        f"```json\n{json.dumps(summaries.get('network', {}), indent=2)}\n```\n\n"
        "## Encoding matrix\n\n"
        f"```json\n{json.dumps(summaries.get('encoding', {}), indent=2)}\n```\n\n"
        "## Manipulation audit\n\n"
        f"Executed and blocked: `{sum(item.get('passed') is True for item in manipulations)}/{len(manipulations)}`.\n\n"
        "Timing values are paired FP32/FP16 fresh-process observations. Profiling runs are excluded "
        "from timing statistics; the matrix confidence interval is bootstrapped jointly across cases "
        "and fresh-process speedups.\n"
    )
    print(decision)
    return 0 if decision == "PHASE3B1_FP16_PERFORMANCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
