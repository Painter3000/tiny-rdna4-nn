#!/usr/bin/env python3
"""Atomic, fail-closed worker-process manifest for Phase 4A3 Q0c."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

MANIFEST_MARKER = "TCNN_RDNA4_P4A3_Q0C_WORKER_MANIFEST_001"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def worker_id(item: dict[str, Any]) -> str:
    phase = item.get("phase")
    label = item.get("metric") if phase == "G" else item.get("batch")
    if label is None:
        raise ValueError("worker item has no phase-appropriate batch/metric label")
    return (
        f"{phase}_{item['schedule']}_{label}_"
        f"p{int(item['process_index'])}_{item['start_order']}"
    )


def parse_matrix(path: pathlib.Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) != 6:
            raise RuntimeError(
                f"matrix line {line_number}: expected 6 TSV fields, got {len(fields)}"
            )
        phase, schedule, batch, metric, process_index, start_order = fields
        if (batch == "-") == (metric == "-"):
            raise RuntimeError(
                f"matrix line {line_number}: exactly one of batch/metric is required"
            )
        item: dict[str, Any] = {
            "phase": phase,
            "schedule": schedule,
            "process_index": int(process_index),
            "start_order": start_order,
        }
        if batch != "-":
            item["batch"] = int(batch)
        else:
            item["metric"] = metric
        items.append(item)
    if not items:
        raise RuntimeError("worker matrix is empty")
    ids = [worker_id(item) for item in items]
    if len(set(ids)) != len(ids):
        raise RuntimeError("worker matrix contains duplicate worker IDs")
    return items


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(matrix_path: pathlib.Path, output: pathlib.Path) -> None:
    if output.exists():
        raise RuntimeError(f"manifest already exists: {output}")
    items = parse_matrix(matrix_path)
    workers = []
    for item in items:
        ident = worker_id(item)
        workers.append(
            {
                **item,
                "worker_id": ident,
                "result_json": f"{ident}.json",
                "log": f"{ident}.log",
                "state": "pending",
                "pid": None,
                "returncode": None,
                "result_json_exists": None,
                "result_json_sha256": None,
                "log_exists": None,
                "completed_utc": None,
            }
        )
    atomic_write_json(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "marker": MANIFEST_MARKER,
            "phase": "4A3-Q0c",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "expected_count": len(workers),
            "workers": workers,
        },
    )


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("worker manifest schema mismatch")
    if value.get("marker") != MANIFEST_MARKER:
        raise RuntimeError("worker manifest marker mismatch")
    if value.get("phase") != "4A3-Q0c":
        raise RuntimeError("worker manifest phase mismatch")
    workers = value.get("workers")
    if not isinstance(workers, list):
        raise RuntimeError("worker manifest workers must be a list")
    if value.get("expected_count") != len(workers):
        raise RuntimeError("worker manifest expected_count mismatch")
    ids = [item.get("worker_id") for item in workers]
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("worker manifest contains invalid or duplicate worker IDs")
    return value


def record(
    manifest_path: pathlib.Path,
    workers_dir: pathlib.Path,
    ident: str,
    pid: int,
    returncode: int,
) -> None:
    value = load_manifest(manifest_path)
    matches = [item for item in value["workers"] if item["worker_id"] == ident]
    if len(matches) != 1:
        raise RuntimeError(f"manifest does not contain exactly one worker: {ident}")
    entry = matches[0]
    if entry.get("state") != "pending":
        raise RuntimeError(f"worker was already recorded: {ident}")
    result_path = workers_dir / entry["result_json"]
    log_path = workers_dir / entry["log"]
    entry.update(
        {
            "state": "completed",
            "pid": int(pid),
            "returncode": int(returncode),
            "result_json_exists": result_path.is_file(),
            "result_json_sha256": sha256(result_path) if result_path.is_file() else None,
            "log_exists": log_path.is_file(),
            "completed_utc": utc_now(),
        }
    )
    value["updated_utc"] = utc_now()
    atomic_write_json(manifest_path, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--matrix", type=pathlib.Path, required=True)
    init_parser.add_argument("--output", type=pathlib.Path, required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    record_parser.add_argument("--workers", type=pathlib.Path, required=True)
    record_parser.add_argument("--worker-id", required=True)
    record_parser.add_argument("--pid", type=int, required=True)
    record_parser.add_argument("--returncode", type=int, required=True)

    args = parser.parse_args()
    if args.command == "init":
        initialize(args.matrix, args.output)
    else:
        record(
            args.manifest,
            args.workers,
            args.worker_id,
            args.pid,
            args.returncode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
