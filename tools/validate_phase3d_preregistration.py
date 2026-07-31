#!/usr/bin/env python3
"""Independent fail-closed validator for the Phase-3D preregistration contract."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(dotted)
        cur = cur[key]
    return cur


def fail(msg: str) -> None:
    raise ValueError(msg)


def validate(contract: dict[str, Any], schema: dict[str, Any]) -> None:
    missing = [k for k in schema["required_top_level"] if k not in contract]
    if missing:
        fail(f"missing_top_level:{','.join(missing)}")

    for dotted, expected in schema["exact"].items():
        try:
            actual = get_path(contract, dotted)
        except KeyError:
            fail(f"missing_required:{dotted}")
        if actual != expected:
            fail(f"exact_mismatch:{dotted}:expected={expected!r}:actual={actual!r}")

    cases = contract.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        fail("case_count_mismatch")
    ids = [c.get("id") for c in cases]
    if ids != schema["case_ids"]:
        fail(f"case_order_mismatch:{ids!r}")
    if len(set(ids)) != len(ids):
        fail("duplicate_case_id")
    for case in cases:
        cid = case["id"]
        if case.get("m") != schema["case_m"][cid]:
            fail(f"case_m_mismatch:{cid}")

    for dotted in schema["sha256_fields"]:
        value = get_path(contract, dotted)
        if dotted == "base.commit":
            if not isinstance(value, str) or not HEX40.fullmatch(value):
                fail(f"invalid_git_commit:{dotted}")
        elif not isinstance(value, str) or not HEX64.fullmatch(value):
            fail(f"invalid_sha256:{dotted}")

    points = contract["phase3db"]["oracle_points"]
    if sorted(points) != points or len(set(points)) != len(points):
        fail("oracle_points_not_strictly_increasing")
    if 850 not in points or points[-1] != 30000:
        fail("oracle_points_missing_required_boundary")

    resume = contract["phase3db"]["resume_points"]
    if any(p + contract["phase3db"]["resume_window"] > 30000 for p in resume):
        fail("resume_window_exceeds_horizon")

    table = contract["data_stream"]["fp16_bit_table"]
    if not isinstance(table, list) or len(table) != 11:
        fail("fp16_table_mismatch")
    if any(not isinstance(v, int) or not (0 <= v <= 0xFFFF) for v in table):
        fail("fp16_table_invalid_value")

    if contract["drift"]["review_headroom_e_max_ge"] >= contract["drift"]["hard_fail_e_max_gt"]:
        fail("review_threshold_not_below_fail_threshold")

    if not (0.0 < contract["activity"]["global_effective_step_fraction_blocked_lt"] <= 1.0):
        fail("invalid_effective_fraction")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} CONTRACT.json SCHEMA.json", file=sys.stderr)
        return 2
    try:
        contract = load(argv[1])
        schema = load(argv[2])
        validate(contract, schema)
    except Exception as exc:
        print(f"PHASE3D_PREREGISTRATION_VALIDATION: FAIL\n{exc}", file=sys.stderr)
        return 1
    print("PHASE3D_PREREGISTRATION_VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
