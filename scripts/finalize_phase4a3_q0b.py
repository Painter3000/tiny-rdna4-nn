#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

MARKER = "TCNN_RDNA4_P4A3_Q0B_APPARATUS_REDESIGN_001"


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def student_t_975(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)


def log_ci(values: list[float]) -> dict[str, Any]:
    logs = [math.log(value) for value in values]
    if not logs:
        return {"n": 0, "geomean": None, "ci95": None}
    mean = statistics.fmean(logs)
    if len(logs) < 2:
        return {"n": 1, "geomean": math.exp(mean), "ci95": None}
    stdev = statistics.stdev(logs)
    half = student_t_975(len(logs) - 1) * stdev / math.sqrt(len(logs))
    return {
        "n": len(logs),
        "log_mean": mean,
        "log_stdev": stdev,
        "geomean": math.exp(mean),
        "ci95": [math.exp(mean - half), math.exp(mean + half)],
    }


def process_valid(record: dict[str, Any], entry: dict[str, Any]) -> bool:
    return entry["returncode"] == 0 and record.get("status") == "PASS" and all(record.get("gates", {}).values())


def lookup_gap(mapping: dict[str, Any], gap: float) -> dict[str, Any]:
    for key in (str(gap), str(float(gap))):
        if key in mapping:
            return mapping[key]
    for item in mapping.values():
        if abs(float(item["gap_ms"]) - gap) < 1e-6:
            return item
    raise KeyError(gap)


def gap_mechanism(records: list[dict[str, Any]], batch: int, contract: dict[str, Any]):
    points = []
    for record in records:
        if record["batch"] != batch:
            continue
        candidate = record["native_gap_sweep"]["rocwmma"]
        reference = record["native_gap_sweep"]["hipblaslt"]
        base_c = lookup_gap(candidate, 0.0)["summary_ns"]["median"]
        base_r = lookup_gap(reference, 0.0)["summary_ns"]["median"]
        for gap in contract["gap_mechanism_diagnostic"]["evaluated_gaps_ms"]:
            dc = lookup_gap(candidate, float(gap))["summary_ns"]["median"] - base_c
            dr = lookup_gap(reference, float(gap))["summary_ns"]["median"] - base_r
            relative = abs(dr - dc) / max(abs(dr), abs(dc), 1.0)
            points.append({
                "process_index": record["process_index"],
                "mode": record["schedule"]["mode"],
                "gap_ms": gap,
                "candidate_increment_ns": dc,
                "reference_increment_ns": dr,
                "relative_increment_difference": relative,
            })
    by_gap = defaultdict(list)
    for point in points:
        by_gap[point["gap_ms"]].append(point["relative_increment_difference"])
    medians = {str(gap): float(statistics.median(values)) for gap, values in by_gap.items()}
    supported = sum(
        value <= float(contract["gap_mechanism_diagnostic"]["support_threshold"])
        for value in medians.values()
    )
    minimum = int(contract["gap_mechanism_diagnostic"]["minimum_supported_gap_points_per_batch"])
    return {
        "points": points,
        "median_relative_difference_by_gap": medians,
        "supported_gap_points": supported,
        "minimum_supported": minimum,
        "mechanism_supported": supported >= minimum,
        "diagnostic_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--index", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-report", type=pathlib.Path, required=True)
    args = parser.parse_args()

    contract = read_json(args.contract)
    context = read_json(args.context)
    index = read_json(args.index)
    if contract["marker"] != MARKER:
        raise RuntimeError("Contract marker mismatch")

    entries = index["workers"]
    expected = (
        len(contract["environment"]["schedule_modes"]) *
        len(contract["matrix"]["batches"]) *
        int(contract["matrix"]["processes_per_batch_and_schedule"])
    )
    records, missing = [], []
    for entry in entries:
        path = pathlib.Path(entry["output"])
        if not path.is_file():
            missing.append(str(path))
            continue
        record = read_json(path)
        record["_entry"] = entry
        records.append(record)

    groups = defaultdict(list)
    for record in records:
        groups[(record["schedule"]["mode"], record["batch"])].append(record)

    group_results = {}
    for mode in contract["environment"]["schedule_modes"]:
        for batch in contract["matrix"]["batches"]:
            group = groups.get((mode, batch), [])
            valid = [record for record in group if process_valid(record, record["_entry"])]
            metrics = {}
            for metric in ("native_single_shot", "public_single_shot", "native_queued"):
                metrics[metric] = log_ci([record["process_ratios_diagnostic"][metric] for record in valid])
            ab = [
                math.log(record["process_ratios_diagnostic"]["public_single_shot"])
                for record in valid if record["start_order"] == "AB"
            ]
            ba = [
                math.log(record["process_ratios_diagnostic"]["public_single_shot"])
                for record in valid if record["start_order"] == "BA"
            ]
            order_effect = abs(statistics.median(ab) - statistics.median(ba)) if ab and ba else None
            passed = (
                len(valid) >= int(contract["analysis_pipeline_frozen"]["minimum_valid_processes_per_group"]) and
                order_effect is not None and
                order_effect <= float(contract["analysis_pipeline_frozen"]["order_effect_abs_log_max"])
            )
            group_results[f"{mode}_b{batch}"] = {
                "records": len(group),
                "valid": len(valid),
                "metrics_diagnostic": metrics,
                "public_order_effect_abs_log": order_effect,
                "passed": passed,
            }

    context_gates = {
        "real_tty": bool(context["real_tty"]),
        "display_manager_inactive": bool(context["display_manager_inactive"]),
        "forbidden_environment_clear": bool(context["forbidden_environment_clear"]),
        "release_identity": bool(context["release_identity_pass"]),
        "production_identity": bool(context["production_identity_pass"]),
        "binding_restored": bool(context["binding_restored"]),
        "fresh_test_binding": bool(context["fresh_test_binding"]),
    }
    spin_pass = all(group_results[f"spin_b{batch}"]["passed"] for batch in contract["matrix"]["batches"])
    auto_recorded = all(
        group_results[f"auto_b{batch}"]["records"] == contract["matrix"]["processes_per_batch_and_schedule"]
        for batch in contract["matrix"]["batches"]
    )
    valid_records = [record for record in records if process_valid(record, record["_entry"])]
    mechanism = {f"b{batch}": gap_mechanism(valid_records, batch, contract) for batch in contract["matrix"]["batches"]}

    gates = {
        "expected_worker_count": len(entries) == expected,
        "all_outputs_present": not missing,
        "context": all(context_gates.values()),
        "spin_apparatus_all_batches": spin_pass,
        "auto_sensitivity_complete": auto_recorded,
        "no_performance_claim": True,
    }
    decision = contract["decisions"]["pass"] if all(gates.values()) else contract["decisions"]["blocked"]

    result = {
        "marker": MARKER,
        "decision": decision,
        "contract_sha256": sha256(args.contract),
        "context": context,
        "context_gates": context_gates,
        "expected_workers": expected,
        "worker_entries": len(entries),
        "worker_records": len(records),
        "missing_outputs": missing,
        "groups": group_results,
        "gap_mechanism_diagnostic": mechanism,
        "gates": gates,
        "analysis_pipeline_was_preregistered": True,
        "performance_claim_allowed": False,
        "known_q0_ratio_used_as_gate": False,
        "floor_subtracted": False,
        "common_additive_S_assumed": False,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Phase 4A3-Q0b – Apparatus Redesign",
        "",
        f"**Decision: `{decision}`**",
        "",
        "No performance PASS/FAIL is authorized by Q0b.",
        "",
        "## Context",
    ]
    for key, value in context_gates.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Groups",
        "",
        "| Group | Valid | Native ratio* | Public ratio* | Queued ratio* | Public order effect | PASS |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, group in group_results.items():
        metrics = group["metrics_diagnostic"]
        lines.append(
            f"| {name} | {group['valid']}/{group['records']} | "
            f"{metrics['native_single_shot']['geomean']} | "
            f"{metrics['public_single_shot']['geomean']} | "
            f"{metrics['native_queued']['geomean']} | "
            f"{group['public_order_effect_abs_log']} | {group['passed']} |"
        )
    lines += [
        "",
        "*Ratios are diagnostic only: reference time / candidate time.",
        "",
        "## Gap mechanism",
    ]
    for name, item in mechanism.items():
        lines.append(f"- {name}: common-increment diagnostic supported = `{item['mechanism_supported']}`")
    lines += [
        "",
        "No floor was subtracted. No common additive S was assumed. The known Q0 ratio was not a gate.",
    ]
    args.output_report.write_text("\n".join(lines) + "\n")

    for key, value in gates.items():
        print(f"WIDTH64_Q0B_{key.upper()}: {'PASS' if value else 'FAIL'}")
    print("PHASE4A3_Q0B_DECISION:", decision)
    print("Evidence JSON:", args.output_json)
    print("Report:", args.output_report)
    return 0 if decision == contract["decisions"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
