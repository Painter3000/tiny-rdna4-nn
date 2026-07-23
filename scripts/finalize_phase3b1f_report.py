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
            integrity.append({
                "process_index": item.get("process_index"),
                "iterations_equal": len(iterations) == 1,
                "block_count": len(blocks) == manifest["measurement"]["blocks_per_process"],
                "speedup_derived": speedup_equal,
                "process_medians_derived": (
                    item.get("process_medians_ms", {}).get("FP16") == (statistics.median(fp16) if fp16 else None)
                    and item.get("process_medians_ms", {}).get("FP32") == (statistics.median(fp32) if fp32 else None)
                ),
                "case_config_exact": item.get("case") == cases[case_id],
                "warmup_excluded": all(block.get("source", "steady_state") == "steady_state" for block in blocks),
                "fallback_clear": all(
                    evidence.get("fallback") is False
                    for name, evidence in item.get("backend_evidence", {}).items()
                    if name in ("FP16", "FP32")
                ),
                "algorithm_id_present": item.get("backend_evidence", {}).get("algorithm_log", {}).get("algorithm_id_present") is True,
                "scratch_zero": all(
                    counters.get("scratch_bytes_live") == 0
                    for counters in item.get("resources_after_release", {}).values()
                ),
                "descriptor_released": all(
                    item.get("resources_after_release", {}).get(backend, {}).get("descriptor_count")
                    == item.get("warmup", {}).get("before", {}).get(backend, {}).get("descriptor_count")
                    for backend in ("FP16", "FP32")
                ),
            })
            if item in valid and calculated is not None:
                speedups.append(calculated)
        results[case_id] = {
            "case": cases[case_id],
            "process_count": len(processes),
            "valid_process_count": len(valid),
            "invalid_processes": [item for item in processes if item not in valid],
            "integrity": integrity,
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
        "case_set": set(results) == {case["id"] for case in manifest["cases"] if case["supported"]} and not unknown,
        "process_total": len(data.get("primary_processes", [])) == expected["fresh_primary_processes"],
        "seven_processes": all(item["process_count"] == 7 for item in results.values()),
        "invalid_limit": all(len(item["invalid_processes"]) <= 1 for item in results.values()),
        "process_integrity": all(all(all(row.values()) for row in item["integrity"]) for item in results.values()),
        "statistics_complete": all("statistics" in item for item in results.values()),
        "cold_start": (
            data.get("cold_start", {}).get("status") == "complete"
            and len(data.get("cold_start", {}).get("results", [])) >= expected["cold_start_processes_min"]
            and all(item.get("valid") is True for item in data["cold_start"]["results"])
        ),
        "profiling": (
            data.get("profiling", {}).get("status") == "complete"
            and len(data.get("profiling", {}).get("results", [])) == expected["profiling_cases"]
            and all(item.get("valid") is True for item in data["profiling"]["results"])
        ),
        "correctness_pre": data.get("correctness_pre", {}).get("passed") is True,
        "correctness_post": data.get("correctness_post", {}).get("passed") is True,
        "no_untrusted_derived_fields": not any(key in data for key in ("reported_statistics", "reported_bootstrap", "reported_geomeans")),
    }
    network_gates = manifest["performance_gates"]["network_large_batch_geomean"]
    network_ci = manifest["performance_gates"]["network_large_batch_bootstrap_lower"]
    network_summary = {}
    for operation in ("forward", "forward_backward", "adam_training_step"):
        selected = [item for item in results.values() if item["case"]["family"] == "network_only"
                    and item["case"]["operation"] == operation and item["case"]["batch"] >= network_gates["batch_min"]]
        medians = [item["statistics"]["median"] for item in selected if "statistics" in item]
        lowers = [item["statistics"]["bootstrap_95"][0] for item in selected if "statistics" in item]
        network_summary[operation] = {
            "geomean": geomean(medians) if medians else None,
            "bootstrap_lower_geomean": geomean(lowers) if lowers else None,
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
            "required_geomean": encoding_gates[operation],
        }
    checks["network_performance"] = all(
        item["geomean"] is not None and item["geomean"] >= item["required_geomean"]
        and item["bootstrap_lower_geomean"] >= item["required_bootstrap_lower"]
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
        "historical_report_changed",
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
    run("fallback_enabled", lambda x: x["primary_processes"][0]["backend_evidence"]["FP16"].update({"fallback": True}))
    run("numerical_maximum_increased", lambda x: x["correctness_post"].update({"passed": False, "numerical_maximum_increased": True}))
    run("scratch_live_nonzero", lambda x: x["primary_processes"][0]["resources_after_release"]["FP16"].update({"scratch_bytes_live": 1}))
    run("descriptor_count_growth", lambda x: x["primary_processes"][0]["resources_after_release"]["FP16"].update({"descriptor_count": x["primary_processes"][0]["resources_after_release"]["FP16"]["descriptor_count"] + 1}))
    run("algorithm_id_missing", lambda x: x["primary_processes"][0]["backend_evidence"]["algorithm_log"].update({"algorithm_id_present": False}))
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
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    data = json.loads(args.raw.read_text())
    checks, cases, summaries = evaluate(data, manifest)
    performance_only = ("network_performance", "encoding_performance", "no_large_regression")
    correctness_stability = all(value for key, value in checks.items() if key not in performance_only)
    performance = all(checks[key] for key in performance_only)
    if not correctness_stability:
        decision = "PHASE3B1F_BLOCKED"
    elif performance:
        decision = "PHASE3B1_FP16_PERFORMANCE_PASS"
    else:
        decision = "PHASE3B1F_CORRECT_BUT_NOT_PERFORMANT"
    manipulations = run_manipulations(data, manifest)
    if [item["name"] for item in manipulations] != list(mutation_names()) or not all(item["passed"] for item in manipulations):
        decision = "PHASE3B1F_BLOCKED"
        checks["manipulations"] = False
    result = {
        "marker": MARKER, "base_commit": BASE, "decision": decision, "gates": checks,
        "manifest_sha256": sha256(MANIFEST), "cases": cases, "summaries": summaries,
        "manipulation_tests": manipulations,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    MD.write_text(f"# Phase 3B1-F – FP16 Performance Qualification\\n\\nDecision: `{decision}`\\n")
    print(decision)
    return 0 if decision == "PHASE3B1_FP16_PERFORMANCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
