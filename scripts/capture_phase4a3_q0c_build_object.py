#!/usr/bin/env python3
"""Capture the one exact Q0c build object before provenance auditing."""
from __future__ import annotations

import argparse
import pathlib

from phase4a3_q0c_common import load_contract, sha256


def capture(evidence_dir: pathlib.Path, contract_path: pathlib.Path) -> pathlib.Path:
    cfg = load_contract(contract_path)["provenance"]
    basename = cfg["build_object_basename"]
    objects = sorted(
        path.resolve()
        for path in evidence_dir.rglob(basename)
        if path.is_file()
    )
    if len(objects) != cfg["exact_build_objects"]:
        listing = "\n".join(str(path) for path in objects)
        detail = f"\n{listing}" if listing else ""
        raise RuntimeError(
            f"expected exactly one Q0c build object, found {len(objects)}{detail}"
        )

    obj = objects[0]
    output_dir = evidence_dir / "provenance"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_object_path.txt").write_text(str(obj) + "\n")
    (output_dir / "build_object_sha256.txt").write_text(
        f"{sha256(obj)}  {obj}\n"
    )
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    args = parser.parse_args()
    obj = capture(args.evidence_dir.resolve(), args.contract.resolve())
    print(obj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
