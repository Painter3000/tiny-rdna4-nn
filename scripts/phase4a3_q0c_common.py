#!/usr/bin/env python3
"""Shared, side-effect-free Q0c contract and apparatus helpers."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import struct
from typing import Any, Callable

MARKER = "TCNN_RDNA4_P4A3_Q0C_SPLIT_APPARATUS_001"
MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
PHASES = ("LN", "LP", "TP", "TD", "G")


def load_contract(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("phase") != "4A3-Q0c":
        raise RuntimeError("Q0c contract schema/phase mismatch")
    if value.get("marker") != MARKER:
        raise RuntimeError("Q0c contract marker mismatch")
    frozen = value.get("provenance", {}).get("p4_reference_kernel_isa_sha256")
    if not isinstance(frozen, str) or len(frozen) != 64:
        raise RuntimeError("Q0c P4 reference kernel hash is not frozen")
    return value


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def padded_batch(batch: int, granularity: int = 256) -> int:
    if batch <= 0 or granularity <= 0:
        raise ValueError("batch and granularity must be positive")
    return ((batch + granularity - 1) // granularity) * granularity


def calibrate_count(one_sample_ms: float, target_ms: float, lower: int, upper: int) -> int:
    if one_sample_ms <= 0:
        raise ValueError("calibration duration must be positive")
    raw = int(math.ceil(target_ms / one_sample_ms))
    return max(lower, min(upper, raw))


def calibrate_by_probe(probe: Callable[[int], float], target_ms: float, lower: int, upper: int) -> tuple[int, list[dict[str, float]]]:
    """Calibrate outside scoring; the returned count is then immutable."""
    count, trace = lower, []
    while True:
        elapsed = float(probe(count))
        trace.append({"count": count, "elapsed_ms": elapsed})
        if elapsed >= target_ms or count >= upper:
            return count, trace
        estimate = calibrate_count(elapsed / count, target_ms, lower, upper)
        count = min(upper, max(count + 1, estimate))


def matrix(contract: dict[str, Any], phases: tuple[str, ...] = PHASES) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cfg = contract["matrix"]
    for phase in phases:
        section = cfg[phase]
        if phase == "G":
            for metric in section["metrics"]:
                for index in range(section["processes_per_metric"]):
                    result.append({"phase": phase, "schedule": "spin", "metric": metric, "process_index": index, "start_order": cfg["start_orders"][index % 4]})
            continue
        for schedule in section["schedules"]:
            for batch in section["batches"]:
                for index in range(cfg["processes_per_group"]):
                    result.append({"phase": phase, "schedule": schedule, "batch": batch, "process_index": index, "start_order": cfg["start_orders"][index]})
    return result


def latin_gap_order(contract: dict[str, Any], process_index: int) -> list[float]:
    base = contract["gap"]["latin_base"]
    gaps = contract["gap"]["gaps_ms"]
    row = base[process_index % len(base):] + base[:process_index % len(base)]
    return [gaps[index] for index in row]


def enumerate_bundles(data: bytes) -> list[dict[str, Any]]:
    """Enumerate every valid simple clang bundle at every magic occurrence."""
    found, start = [], 0
    while True:
        magic_at = data.find(MAGIC, start)
        if magic_at < 0:
            break
        start = magic_at + 1
        cursor = magic_at + len(MAGIC)
        if cursor + 8 > len(data):
            continue
        count = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8
        if not 0 < count <= 128:
            continue
        entries = []
        valid = True
        for _ in range(count):
            if cursor + 24 > len(data):
                valid = False
                break
            offset, size, id_size = struct.unpack_from("<QQQ", data, cursor)
            cursor += 24
            if id_size > 4096 or cursor + id_size > len(data):
                valid = False
                break
            ident = data[cursor:cursor + id_size].decode(errors="replace")
            cursor += id_size
            actual_offset = magic_at + offset
            if actual_offset + size > len(data) and offset + size <= len(data):
                actual_offset = offset
            if actual_offset + size > len(data):
                valid = False
                break
            entries.append({"id": ident, "offset": actual_offset, "bundle_relative_offset": offset, "size": size, "payload": data[actual_offset:actual_offset + size]})
        if valid:
            found.append({"magic_offset": magic_at, "entries": entries})
    return found


def symbol_lines(text: str, token: str) -> list[str]:
    """Accept LOCAL or GLOBAL defined executable-symbol fixture lines."""
    result = []
    for line in text.splitlines():
        if token not in line or "UND" in line:
            continue
        fields = line.split()
        if "LOCAL" in fields or "GLOBAL" in fields:
            name = fields[-1] if fields else ""
            if "." not in name:
                result.append(line)
    return result


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive nonempty values required")
    return math.exp(statistics.fmean(math.log(value) for value in values))
