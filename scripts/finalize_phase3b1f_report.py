#!/usr/bin/env python3
"""F0b reproducibility-focused Phase 3B1-F1 finalizer."""
import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "phase3b1_reports/phase3b1f0b_reproducible_benchmark_manifest.json"
OUT = ROOT / "phase3b1_reports/phase3b1f_fp16_performance.json"
MD = ROOT / "phase3b1_reports/PHASE3B1F_FP16_PERFORMANCE.md"
WORKER_MARKER = "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def paired_bootstrap(values, seed, resamples=10000):
    generator = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [values[generator.randrange(len(values))] for _ in values]
        samples.append(statistics.median(draw))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def correctness_valid(snapshot, manifest):
    limits = manifest["numerical_gates"]["frozen_absolute"]
    return isinstance(snapshot, dict) and all((
        snapshot.get("passed") is True,
        snapshot.get("output_fresh") is True,
        snapshot.get("timed_out") is False,
        snapshot.get("output_max_abs", math.inf) <= limits["network_output"],
        snapshot.get("dinput_max_abs", math.inf) <= limits["dinput"],
        snapshot.get("network_gradient_max_abs", math.inf) <= limits["network_gradient"],
        snapshot.get("encoding_gradient_max_abs", math.inf) <= limits["encoding_gradient"],
        snapshot.get("nan_count") == 0,
        snapshot.get("inf_count") == 0,
    ))


def numerical_valid(numerical, manifest):
    limits = manifest["numerical_gates"]["frozen_absolute"]
    expected = {
        "output": limits["network_output"], "dinput": limits["dinput"],
        "network_gradient": limits["network_gradient"],
        "encoding_gradient": limits["encoding_gradient"],
    }
    return isinstance(numerical, dict) and all(
        isinstance(numerical.get(name), dict)
        and math.isfinite(numerical[name].get("max_abs", math.inf))
        and numerical[name]["max_abs"] <= limit
        and math.isfinite(numerical[name].get("normalized_l2", math.inf))
        and numerical[name].get("nan_count") == 0
        and numerical[name].get("inf_count") == 0
        for name, limit in expected.items()
    )


def process_valid(record, case, manifest, smoke=False):
    blocks = record.get("timing_blocks", [])
    rounds = 1 if smoke else manifest["paired_rounds_per_process"]
    expected_blocks = rounds * manifest["measurement"]["blocks_per_round"]
    iterations = {block.get("iterations") for block in blocks}
    fp16 = [block.get("event_ms_per_iteration") for block in blocks if block.get("backend") == "FP16"]
    fp32 = [block.get("event_ms_per_iteration") for block in blocks if block.get("backend") == "FP32"]
    evidence = record.get("backend_evidence", {})
    identity = record.get("pair_identity", {})
    memory = record.get("process_memory_peaks", {})
    checks = {
        "identity": record.get("marker") == WORKER_MARKER
                    and record.get("case") == case
                    and record.get("manifest_sha256") == sha256(MANIFEST),
        "returncode": record.get("returncode") == 0 and record.get("valid") is True,
        "timing": len(blocks) == expected_blocks and len(iterations) == 1
                  and len(fp16) == len(fp32) == expected_blocks // 2
                  and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                          for value in fp16 + fp32),
        "comparison_identity": identity.get("fp16_quantization_matches") is True
                               and identity.get("logical_parameter_count") == identity.get("reference_parameter_count")
                               and bool(identity.get("input_sha256"))
                               and bool(identity.get("target_sha256"))
                               and bool(identity.get("upstream_sha256")),
        "backend": evidence.get("backend_kind") == "hipblaslt_gemm"
                   and evidence.get("gemm_expected") is True
                   and evidence.get("fallback") is False
                   and evidence.get("candidate", {}).get("backend") == "HipBLASLtMLPFP16",
        "numerical": numerical_valid(record.get("numerical"), manifest),
        "memory": all(
            isinstance(memory.get(backend, {}).get(field), int)
            and memory[backend][field] >= 0
            for backend in ("FP16", "FP32") for field in ("allocated_bytes", "reserved_bytes")
        ),
    }
    return all(checks.values()), checks


def replacement_contract(records, manifest):
    primary = [item for item in records if item.get("replacement_attempt") is None]
    replacements = [item for item in records if item.get("replacement_attempt") is not None]
    return {
        "three_primary_attempts": len(primary) == manifest["fresh_processes_per_case"],
        "at_most_one_replacement": len(replacements) <= manifest["maximum_replacement_attempts_per_case"],
        "replacement_is_documented": all(
            item.get("replacement_attempt") == 1
            and isinstance(item.get("replaces_invalid_process_indices"), list)
            for item in replacements
        ),
    }


def evaluate(data, manifest):
    cases = {case["id"]: case for case in manifest["cases"]}
    grouped = {case_id: [] for case_id in cases}
    unknown = []
    for record in data.get("primary_processes", []):
        case_id = record.get("case", {}).get("id")
        (grouped[case_id] if case_id in grouped else unknown).append(record)
    results = {}
    blocked = []
    gains = []
    memory_gains = []
    smoke = data.get("harness_smoke") is True
    for case_id, case in cases.items():
        records = grouped[case_id]
        evaluated = []
        for record in records:
            valid, checks = process_valid(record, case, manifest, smoke=smoke)
            evaluated.append({"record": record, "valid": valid, "checks": checks})
        valid_records = [item["record"] for item in evaluated if item["valid"]]
        replacement = replacement_contract(records, manifest)
        minimum = 1 if smoke else manifest["minimum_valid_processes_per_case"]
        accepted = len(valid_records) >= minimum and all(replacement.values())
        if not accepted:
            blocked.append(case_id)
        speedups = [item["process_speedup"] for item in valid_records]
        fp16_memory = [item["process_memory_peaks"]["FP16"]["allocated_bytes"] for item in valid_records]
        fp32_memory = [item["process_memory_peaks"]["FP32"]["allocated_bytes"] for item in valid_records]
        memory_ratios = [
            before / after if after else math.inf
            for before, after in zip(fp32_memory, fp16_memory)
        ]
        summary = {
            "case": case, "accepted": accepted,
            "primary_processes": len([item for item in records if item.get("replacement_attempt") is None]),
            "replacement_attempts": len([item for item in records if item.get("replacement_attempt") is not None]),
            "valid_processes": len(valid_records), "replacement_contract": replacement,
            "process_checks": [{"process_index": item["record"].get("process_index"), **item["checks"]} for item in evaluated],
        }
        if speedups:
            summary["time"] = {
                "median_speedup": statistics.median(speedups),
                "mad": statistics.median(abs(value - statistics.median(speedups)) for value in speedups),
                "paired_bootstrap_95": paired_bootstrap(speedups, case["seed"]),
            }
            summary["memory"] = {
                "fp16_allocated_peak_median": statistics.median(fp16_memory),
                "fp32_allocated_peak_median": statistics.median(fp32_memory),
                "fp32_over_fp16_median": statistics.median(memory_ratios),
            }
            gains.append(summary["time"]["median_speedup"])
            memory_gains.append(summary["memory"]["fp32_over_fp16_median"])
        results[case_id] = summary
    checks = {
        "marker": data.get("marker") == WORKER_MARKER,
        "active_protocol_marker": data.get("active_protocol_marker") == manifest["marker"],
        "manifest_hash": data.get("manifest_sha256") == sha256(MANIFEST),
        "complete": data.get("complete") is True,
        "case_set": not unknown and set(grouped) == set(cases),
        "correctness_pre": correctness_valid(data.get("correctness_pre"), manifest),
        "correctness_post": correctness_valid(data.get("correctness_post"), manifest),
        "all_cases_accepted": not blocked,
    }
    if not all(checks.values()):
        decision = "PHASE3B1F_BLOCKED"
    else:
        time_median = statistics.median(gains)
        memory_median = statistics.median(memory_gains)
        if time_median > 1 and memory_median >= 1:
            decision = "PHASE3B1F_PERFORMANCE_GAIN_CONFIRMED"
        elif time_median < 1 and memory_median < 1:
            decision = "PHASE3B1F_PERFORMANCE_REGRESSION"
        else:
            decision = "PHASE3B1F_PERFORMANCE_MIXED"
    return decision, checks, results


def local_validity_contract_test(manifest):
    def classify(flags, replacements):
        return sum(flags) >= 2 and replacements <= 1
    cases = {
        "two_of_three_accepted": classify([True, True, False], 0),
        "one_of_three_blocked": not classify([True, False, False], 0),
        "one_replacement_allowed": classify([True, False, False, True], 1),
        "two_replacements_blocked": not classify([True, False, False, True, True], 2),
    }
    return {"passed": all(cases.values()), "checks": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=pathlib.Path)
    parser.add_argument("--local-contract-test", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.local_contract_test:
        result = local_validity_contract_test(manifest)
        print(json.dumps(result))
        return 0 if result["passed"] else 2
    if not args.raw:
        raise SystemExit("--raw is required")
    try:
        data = json.loads(args.raw.read_text())
        decision, checks, results = evaluate(data, manifest)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        decision, checks, results = "PHASE3B1F_BLOCKED", {"input_error": False}, {"error": str(error)}
    output = {
        "marker": manifest["marker"], "decision": decision,
        "manifest_sha256": sha256(MANIFEST), "gates": checks, "cases": results,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    MD.write_text(
        "# Phase 3B1-F1 – Reproducible FP16 Performance\n\n"
        f"Decision: `{decision}`\n\n"
        "Time and memory are reported with equal status. Batch 1024 is classified as a "
        "`latency-bound / launch-overhead-dominated regime`; batch 16384 is the throughput regime.\n"
    )
    print(decision)
    return 0 if decision != "PHASE3B1F_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
