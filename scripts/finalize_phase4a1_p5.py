#!/usr/bin/env python3
"""Finalize the P5 ISA audit and functional replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P5_ISA_RESOURCE_GLOBAL_TRAFFIC_001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--p4-json", type=Path, required=True)
    parser.add_argument("--p4-reference-json", type=Path, required=True)
    parser.add_argument("--p4-reference-csv", type=Path, required=True)
    parser.add_argument("--process-1-json", type=Path, required=True)
    parser.add_argument("--process-1-csv", type=Path, required=True)
    parser.add_argument("--process-2-json", type=Path, required=True)
    parser.add_argument("--process-2-csv", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    preparation = load(args.preparation)
    audit = load(args.audit_json)
    p4 = load(args.p4_json)

    assert preparation["decision"] == "PHASE4A1_P5_PREPARATION_PASS"
    assert audit["decision"] == (
        "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT_PASS"
    )
    assert all(bool(value) for value in audit["gates"].values())
    assert p4["decision"] == (
        "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_CONSOLIDATED_PASS"
    )
    assert all(bool(value) for value in p4["gates"].values())

    process_json_identical = (
        args.process_1_json.read_bytes()
        == args.process_2_json.read_bytes()
    )
    process_csv_identical = (
        args.process_1_csv.read_bytes()
        == args.process_2_csv.read_bytes()
    )
    p4_json_replayed_exactly = (
        args.process_1_json.read_bytes()
        == args.p4_reference_json.read_bytes()
    )
    p4_csv_replayed_exactly = (
        args.process_1_csv.read_bytes()
        == args.p4_reference_csv.read_bytes()
    )

    passed = (
        process_json_identical
        and process_csv_identical
        and p4_json_replayed_exactly
        and p4_csv_replayed_exactly
    )

    consolidated = {
        "marker": MARKER,
        "decision": (
            "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT_PASS"
            if passed
            else "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT_FAIL"
        ),
        "preparation": {
            "path": str(args.preparation.resolve()),
            "sha256": sha256(args.preparation),
            "data": preparation,
        },
        "isa_audit": {
            "path": str(args.audit_json.resolve()),
            "sha256": sha256(args.audit_json),
            "data": audit,
        },
        "p4": {
            "path": str(args.p4_json.resolve()),
            "sha256": sha256(args.p4_json),
        },
        "functional_replay": {
            "process_json_identical": process_json_identical,
            "process_csv_identical": process_csv_identical,
            "p4_reference_json_exact": p4_json_replayed_exactly,
            "p4_reference_csv_exact": p4_csv_replayed_exactly,
            "processes": [
                {
                    "json": str(args.process_1_json.resolve()),
                    "json_sha256": sha256(args.process_1_json),
                    "csv": str(args.process_1_csv.resolve()),
                    "csv_sha256": sha256(args.process_1_csv),
                },
                {
                    "json": str(args.process_2_json.resolve()),
                    "json_sha256": sha256(args.process_2_json),
                    "csv": str(args.process_2_csv.resolve()),
                    "csv_sha256": sha256(args.process_2_csv),
                },
            ],
        },
        "gates": {
            "isa_resource_audit_pass": True,
            "functional_replay_fresh_process_identical": (
                process_json_identical and process_csv_identical
            ),
            "functional_replay_exactly_matches_p4": (
                p4_json_replayed_exactly and p4_csv_replayed_exactly
            ),
        },
    }

    args.evidence.mkdir(parents=True, exist_ok=True)
    output = (
        args.evidence
        / "phase4a1_p5_width64_isa_resource_global_traffic.json"
    )
    output.write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n"
    )

    shutil.copy2(
        args.audit_report,
        args.evidence / "PHASE4A1_P5_REPORT.md",
    )

    if not passed:
        print("WIDTH64_AUDITED_BINARY_FUNCTIONAL_REPLAY: FAIL")
        print("PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT: FAIL")
        return 1

    print("WIDTH64_AUDITED_BINARY_FUNCTIONAL_REPLAY: PASS")
    print("WIDTH64_AUDITED_BINARY_EXACT_P4_REPLAY: PASS")
    print("PHASE4A1_P5_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
