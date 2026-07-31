#!/usr/bin/env python3
"""Fail-closed Q0c finalizer with independent per-group decisions."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any

from phase4a3_q0c_common import MARKER, geometric_mean, load_contract

T975 = {2: 4.3026527297, 3: 3.1824463053}


def process_ratio(worker: dict[str, Any]) -> float:
    if len(worker["rounds"]) != 4:
        raise ValueError("worker does not contain exactly four paired rounds")
    return geometric_mean([float(value["ratio"]) for value in worker["rounds"]])


def group_stats(workers: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    valid = []
    for worker in workers:
        if not worker["pre_correctness"]["passed"] or not worker["post_correctness"]["passed"] or ("gates" in worker and not all(worker["gates"].values())):
            continue
        if worker["phase"] in ("LN", "LP"):
            if not all(item["backends"][name]["convergence"]["passed"] and item["backends"][name]["score"]["passed"] for item in worker["rounds"] for name in ("candidate", "reference")):
                continue
        try:
            valid.append({"start_order": worker["start_order"], "ratio": process_ratio(worker)})
        except (KeyError, ValueError):
            continue
    logs = [math.log(item["ratio"]) for item in valid]
    result: dict[str, Any] = {"workers_seen": len(workers), "valid_processes": len(valid), "process_ratios": valid}
    if len(valid) < contract["analysis"]["minimum_valid_processes"]:
        result.update({"passed": False, "reason": "fewer than three valid fresh-process clusters"})
        return result
    mean, stdev = statistics.fmean(logs), statistics.stdev(logs)
    critical = T975.get(len(logs) - 1)
    if critical is None:
        raise ValueError("Q0c groups permit only three or four valid processes")
    half = critical * stdev / math.sqrt(len(logs))
    ab = [math.log(item["ratio"]) for item in valid if item["start_order"] == "AB"]
    ba = [math.log(item["ratio"]) for item in valid if item["start_order"] == "BA"]
    order_effect = abs(statistics.median(ab) - statistics.median(ba)) if ab and ba else math.inf
    result.update({"geometric_mean_ratio_diagnostic": math.exp(mean), "student_t_95_ci": [math.exp(mean - half), math.exp(mean + half)], "order_effect_abs_log": order_effect, "passed": order_effect <= contract["analysis"]["order_effect_abs_log_max"]})
    return result


def throughput_group(workers: list[dict[str, Any]], contract: dict[str, Any], phase: str) -> dict[str, Any]:
    valid = [worker for worker in workers if worker["pre_correctness"]["passed"] and worker["post_correctness"]["passed"] and ("gates" not in worker or all(worker["gates"].values())) and len(worker["rounds"]) == 4]
    result: dict[str, Any] = {"workers_seen": len(workers), "valid_processes": len(valid)}
    if len(valid) < contract["analysis"]["minimum_valid_processes"]:
        return result | {"passed": False, "reason": "fewer than three valid fresh-process clusters"}
    ratios, ordered_ratios, headroom = [], [], True
    for worker in valid:
        round_ratios = []
        for round_value in worker["rounds"]:
            medians = {}
            for name in ("candidate", "reference"):
                blocks = round_value["backends"][name]["blocks"]
                metric = [float(block["host_total_ns"]) / int(block["iterations"]) for block in blocks] if phase == "TP" else [float(block["event_ms"]) * 1e6 / int(block["iterations"]) for block in blocks]
                medians[name] = statistics.median(metric)
                if phase == "TD":
                    headroom &= all(float(block["submission_over_gpu"]) <= contract["throughput"]["TD"]["submission_over_gpu_max"] for block in blocks)
            round_ratios.append(medians["reference"] / medians["candidate"])
        ratio = geometric_mean(round_ratios)
        ratios.append(ratio)
        ordered_ratios.append({"start_order": worker["start_order"], "ratio": ratio})
    logs = [math.log(value) for value in ratios]
    mean, stdev = statistics.fmean(logs), statistics.stdev(logs)
    critical = T975[len(logs) - 1]
    half = critical * stdev / math.sqrt(len(logs))
    ab = [math.log(item["ratio"]) for item in ordered_ratios if item["start_order"] == "AB"]
    ba = [math.log(item["ratio"]) for item in ordered_ratios if item["start_order"] == "BA"]
    order_effect = abs(statistics.median(ab) - statistics.median(ba)) if ab and ba else math.inf
    statistically_valid = order_effect <= contract["analysis"]["order_effect_abs_log_max"]
    result.update({"process_ratios": ordered_ratios, "geometric_mean_ratio_diagnostic": math.exp(mean), "student_t_95_ci": [math.exp(mean-half), math.exp(mean+half)], "order_effect_abs_log": order_effect, "headroom_all_blocks_both_backends": headroom, "passed": statistically_valid and (True if phase == "TP" else headroom)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=pathlib.Path, required=True)
    parser.add_argument("--provenance", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    provenance = json.loads(args.provenance.read_text())
    provenance_pass = provenance.get("marker") == MARKER and provenance.get("decision") == contract["decisions"]["P_pass"]
    values = []
    for path in sorted(args.workers.glob("*.json")):
        value = json.loads(path.read_text())
        if value.get("marker") == MARKER:
            values.append(value)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        key = f'{value["phase"]}/{value["schedule"]}/' + (str(value["metric"]) if value["phase"] == "G" else f'batch{value["batch"]}')
        grouped.setdefault(key, []).append(value)
    groups = {}
    for key, workers in grouped.items():
        phase = workers[0]["phase"]
        if phase in ("LN", "LP"):
            groups[key] = group_stats(workers, contract)
        elif phase in ("TP", "TD"):
            groups[key] = throughput_group(workers, contract, phase)
        else:
            groups[key] = {"workers_seen": len(workers), "passed": len(workers) == 8, "diagnostic_only": True}
    ln = groups.get("LN/spin/batch256", {}).get("passed", False)
    lp = all(groups.get(f"LP/spin/batch{batch}", {}).get("passed", False) for batch in (1, 31, 128))
    tp = all(groups.get(f"TP/spin/batch{batch}", {}).get("passed", False) for batch in contract["matrix"]["TP"]["batches"])
    td_batches = contract["matrix"]["TD"]["batches"]
    qualified = [batch for batch in td_batches if groups.get(f"TD/spin/batch{batch}", {}).get("passed", False)]
    large = [512, 1024, 4096, 16384]
    consecutive = any(a in qualified and b in qualified for a, b in zip(large, large[1:]))
    gap_complete = all(groups.get(f"G/spin/{metric}", {}).get("passed", False) for metric in contract["matrix"]["G"]["metrics"])
    subphases = {
        "P": {"passed": provenance_pass, "decision": contract["decisions"]["P_pass"] if provenance_pass else contract["decisions"]["P_fail"]},
        "LN": {"passed": ln, "decision": contract["decisions"]["LN_pass"] if ln else "PHASE4A3_Q0C_LN_APPARATUS_BLOCKED"},
        "LP": {"passed": lp, "decision": contract["decisions"]["LP_pass"] if lp else "PHASE4A3_Q0C_LP_APPARATUS_BLOCKED"},
        "TP": {"passed": tp, "decision": contract["decisions"]["TP_pass"] if tp else "PHASE4A3_Q0C_TP_APPARATUS_BLOCKED"},
        "TD": {"passed": consecutive, "qualified_batches": qualified, "decision": contract["decisions"]["TD_pass"] if consecutive else "PHASE4A3_Q0C_TD_REGION_NOT_QUALIFIED"},
        "G": {"passed": gap_complete, "is_gate": False, "decision": contract["decisions"]["G_complete"] if gap_complete else "PHASE4A3_Q0C_G_DIAGNOSTIC_INCOMPLETE"}
    }
    proceed = provenance_pass and ln and lp and tp and consecutive
    result = {"marker": MARKER, "phase": "4A3-Q0c", "groups": groups, "subphases": subphases, "overall_decision": contract["decisions"]["overall_proceed"] if proceed else contract["decisions"]["overall_blocked"], "performance_pass_fail": None, "performance_claim_allowed": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["overall_decision"])
    return 0 if proceed else 1


if __name__ == "__main__":
    raise SystemExit(main())
