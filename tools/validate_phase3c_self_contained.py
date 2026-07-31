#!/usr/bin/env python3
"""Fail-closed audit of the public Phase-3C build/runtime dependency closure."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_FILES = (
    "scripts/fresh_clone_user_smoke.sh",
    "tools/phase3c_portable_smoke.py",
    "tools/phase3d0b_closeout.py",
    "tools/phase3da_hipblaslt_crosscheck_fixed_v1.hip",
    "src/impl/phase3d_inprocess_driver.cpp",
    "src/impl/phase3d_inprocess_loop.cpp",
    "src/impl/phase3d_inprocess_loop.hpp",
    "src/impl/qualified/phase3a_fused_backward.hip",
    "src/impl/qualified/phase3b_adam_update.hip",
)
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
HOME_PATH = re.compile(r"/home/oem(?:/|$)")
SIBLING_RUNTIME = re.compile(r"\bROOT\.parent\b|workspace[-_]sibling|\.\./\.\./workspace")
QUALIFIED = "src/impl/qualified/phase3a_fused_backward.hip"
QUALIFIED_SHA256 = "7ad0cc174c25918448b7936bfdca63bf2fdf5aab441063ca3618aefdee135a85"


class AuditFailure(RuntimeError):
    pass


def inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit(root: Path, files=DEFAULT_FILES):
    root = root.resolve()
    findings = []
    qualified = root / QUALIFIED
    if not qualified.is_file():
        raise AuditFailure("missing vendored qualified Phase-3A source")
    actual = hashlib.sha256(qualified.read_bytes()).hexdigest()
    if actual != QUALIFIED_SHA256:
        raise AuditFailure("qualified Phase-3A source hash mismatch")
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise AuditFailure(f"missing dependency: {relative}")
        text = path.read_text(errors="replace")
        if HOME_PATH.search(text) and relative != "qualified_sources/phase3a_fused_backward/PROVENANCE.json":
            raise AuditFailure(f"hard-coded /home/oem path: {relative}")
        if SIBLING_RUNTIME.search(text):
            raise AuditFailure(f"workspace sibling runtime dependency: {relative}")
        for include in INCLUDE.findall(text):
            include_path = Path(include)
            if include_path.is_absolute():
                raise AuditFailure(f"absolute include: {relative}: {include}")
            resolved = (path.parent / include_path).resolve()
            if not inside(root, resolved):
                raise AuditFailure(f"include leaves repository root: {relative}: {include}")
            if not resolved.is_file():
                raise AuditFailure(f"missing in-tree include: {relative}: {include}")
            findings.append({"source": relative, "include": include,
                             "resolved": str(resolved.relative_to(root))})
    for path in root.rglob("*"):
        if path.is_symlink() and not inside(root, path):
            raise AuditFailure(f"external symlink: {path.relative_to(root)}")
    return {
        "gate": "PHASE3C_SELF_CONTAINED_SOURCE_AUDIT: PASS",
        "repo_root": str(root),
        "audited_files": list(files),
        "qualified_source_sha256": actual,
        "resolved_local_includes": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except (AuditFailure, OSError) as error:
        print(json.dumps({"gate": "PHASE3C_SELF_CONTAINED_SOURCE_AUDIT: FAIL",
                          "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
