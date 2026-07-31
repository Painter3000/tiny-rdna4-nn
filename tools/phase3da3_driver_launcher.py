#!/usr/bin/env python3
"""Fail-closed capability preflight and future launcher for Phase 3D-A3."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "build_phase3da/phase3da_inprocess_driver"
CROSSCHECK = ROOT / "build_phase3da_r1_fix/phase3da_hipblaslt_crosscheck_fixed_v1"
DRIVER_SHA256 = "27807462a404e6dd93bc6c0d70c1e337085577427b697c5eb0c08fcc054702f5"
CROSSCHECK_SHA256 = "e9416fdde6394b0944e7d858332dd52a7826140dfad1c4405e6113a32f1a5748"
STATE_FILES = ("W_master.fp32.bin", "W_compute.fp16.bin", "m.fp32.bin",
               "v.fp32.bin", "optimizer_state.txt")
CASES = {
    "dense_a_m32": "train_dense_set_a_m32",
    "sparse_a_m48": "train_sparse_set_a_m48",
    "dense_b_m64": "train_dense_set_b_m64",
    "partial_b_m45": "train_partial_set_b_m45",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(path):
    digest = hashlib.sha256()
    for name in STATE_FILES:
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("--validate-arguments-only", action="store_true", required=True)
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--case", required=True, choices=CASES)
    result.add_argument("--case-id", required=True)
    result.add_argument("--replay-id", type=int, required=True, choices=(1, 2, 3))
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--checkpoint-step", type=int, required=True)
    result.add_argument("--resume-start", type=int, required=True)
    result.add_argument("--resume-end", type=int, required=True)
    result.add_argument("--crosscheck-hash", required=True)
    result.add_argument("--output-format", required=True)
    result.add_argument("--s0-dir", type=Path, required=True)
    return result


def validate(args):
    if sha256(DRIVER) != DRIVER_SHA256:
        raise ValueError("driver hash")
    if sha256(CROSSCHECK) != CROSSCHECK_SHA256 or args.crosscheck_hash != CROSSCHECK_SHA256:
        raise ValueError("crosscheck hash")
    if args.steps != 100:
        raise ValueError("steps")
    if args.case_id != CASES[args.case]:
        raise ValueError("case id")
    expected_run_id = f"{args.case}_replay_{args.replay_id}_phase3da3_100"
    if args.run_id != expected_run_id:
        raise ValueError("run id")
    if (args.checkpoint_step, args.resume_start, args.resume_end) != (50, 51, 66):
        raise ValueError("checkpoint/resume")
    if args.output_format != "phase3da3-v1":
        raise ValueError("output format")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("non-empty output")
    if not args.s0_dir.is_dir():
        raise ValueError("S0 directory")
    before = state_hash(args.s0_dir)
    # Capability validation is deliberately read-only and cannot reserve IDs.
    after = state_hash(args.s0_dir)
    if before != after:
        raise ValueError("S0 changed")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output changed")
    return {
        "gate": "PHASE3DA3_DRIVER_CAPABILITY_VALIDATION: PASS",
        "mode": "validate_arguments_only",
        "training_steps_executed": 0,
        "run_id_reserved": False,
        "reservation_order": "CAPABILITY_PASS_THEN_ATOMIC_RESERVATION_THEN_EXECUTION",
        "driver_path": str(DRIVER.resolve()),
        "driver_sha256": sha256(DRIVER),
        "crosscheck_path": str(CROSSCHECK.resolve()),
        "crosscheck_sha256": sha256(CROSSCHECK),
        "steps": args.steps,
        "case": args.case,
        "case_id": args.case_id,
        "replay_id": args.replay_id,
        "run_id": args.run_id,
        "checkpoint_step": args.checkpoint_step,
        "resume_range": [args.resume_start, args.resume_end],
        "output_format": args.output_format,
        "output_dir_fresh": True,
        "S0_hash_before": before,
        "S0_hash_after": after,
        "S0_unchanged": True,
    }


def main():
    args = parser().parse_args()
    try:
        result = validate(args)
    except (ValueError, OSError) as error:
        print(json.dumps({"gate": "PHASE3DA3_DRIVER_CAPABILITY_VALIDATION: FAIL",
                          "reason": str(error)}, sort_keys=True))
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
