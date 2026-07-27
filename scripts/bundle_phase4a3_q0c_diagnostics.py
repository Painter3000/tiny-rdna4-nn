#!/usr/bin/env python3
"""Create a deterministic Q0c diagnostic tarball and non-self checksum list."""
from __future__ import annotations
import argparse
import hashlib
import pathlib
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--evidence", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
checksum = args.evidence / "SHA256SUMS"
members = sorted(path for path in args.evidence.rglob("*") if path.is_file() and path != checksum and path != args.output)
# run.log must already be closed by the runner before this program starts.
checksum.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.evidence)}\n" for path in members))
with tarfile.open(args.output, "w:gz") as archive:
    for path in members + [checksum]:
        archive.add(path, arcname=path.relative_to(args.evidence))
