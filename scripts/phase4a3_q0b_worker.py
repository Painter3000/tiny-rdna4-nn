#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import statistics
import subprocess
import time
from typing import Any

MARKER = "TCNN_RDNA4_P4A3_Q0B_APPARATUS_REDESIGN_001"


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - position) + values[hi] * (position - lo))


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None}
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def block_assessment(block_medians: list[float], spread_max: float, deviation_max: float) -> dict[str, Any]:
    center = float(statistics.median(block_medians))
    spread = (max(block_medians) - min(block_medians)) / center if center else 0.0
    deviations = [abs(x - center) / center if center else 0.0 for x in block_medians]
    return {
        "block_medians": block_medians,
        "round_median": center,
        "spread_fraction": spread,
        "max_deviation_fraction": max(deviations, default=0.0),
        "passed": spread <= spread_max and max(deviations, default=0.0) <= deviation_max,
    }


class Bridge:
    def __init__(self, path: pathlib.Path):
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        for name in ("q0b_schedule_auto", "q0b_schedule_spin", "q0b_schedule_mask"):
            getattr(self.lib, name).restype = ctypes.c_uint
        self.lib.q0b_set_schedule.argtypes = [ctypes.c_uint]
        self.lib.q0b_set_schedule.restype = ctypes.c_int
        self.lib.q0b_get_flags.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.lib.q0b_get_flags.restype = ctypes.c_int
        self.lib.q0b_error_name.argtypes = [ctypes.c_int]
        self.lib.q0b_error_name.restype = ctypes.c_char_p
        self.lib.q0b_error_string.argtypes = [ctypes.c_int]
        self.lib.q0b_error_string.restype = ctypes.c_char_p
        self.lib.q0b_measure_floors.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.q0b_measure_floors.restype = ctypes.c_int

    def error(self, code: int) -> dict[str, Any]:
        name = self.lib.q0b_error_name(code)
        text = self.lib.q0b_error_string(code)
        return {
            "code": code,
            "name": name.decode() if name else None,
            "string": text.decode() if text else None,
        }

    def set_mode(self, mode: str) -> dict[str, Any]:
        constants = {
            "auto": int(self.lib.q0b_schedule_auto()),
            "spin": int(self.lib.q0b_schedule_spin()),
            "mask": int(self.lib.q0b_schedule_mask()),
        }
        requested = constants[mode]
        set_code = int(self.lib.q0b_set_schedule(requested))
        flags = ctypes.c_uint(0)
        get_code = int(self.lib.q0b_get_flags(ctypes.byref(flags)))
        effective = int(flags.value) & constants["mask"]
        return {
            "mode": mode,
            "constants": constants,
            "requested": requested,
            "set": self.error(set_code),
            "get": self.error(get_code),
            "effective_flags": int(flags.value),
            "effective_schedule": effective,
            "passed": set_code == 0 and get_code == 0 and effective == requested,
        }

    def floors(self, count: int) -> dict[str, Any]:
        array_type = ctypes.c_uint64 * count
        timer, empty, minimal = array_type(), array_type(), array_type()
        code = int(self.lib.q0b_measure_floors(count, timer, empty, minimal))
        result = {"result": self.error(code), "count": count}
        if code == 0:
            result.update({
                "timer_ns": summary([float(x) for x in timer]),
                "empty_sync_ns": summary([float(x) for x in empty]),
                "minimal_gpu_ns": summary([float(x) for x in minimal]),
            })
        return result


def rocm_smi() -> dict[str, Any]:
    command = ["rocm-smi", "--showtemp", "--showclocks", "--showuse", "--showpids", "--json"]
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    raw = run.stdout
    pids = {int(value) for value in re.findall(r"PID[^0-9]*([0-9]+)", raw, re.I)}
    return {
        "available": run.returncode == 0,
        "returncode": run.returncode,
        "stdout": raw,
        "stderr": run.stderr,
        "gpu_process_ids": sorted(pids),
        "foreign_gpu_process_ids": sorted(pid for pid in pids if pid != os.getpid()),
    }


def counters(binding: Any) -> dict[str, int] | None:
    mapping = {
        "cache_misses": "_hipblaslt_fp16_cache_misses",
        "heuristic_queries": "_hipblaslt_fp16_heuristic_queries",
        "handle_creations": "_hipblaslt_fp16_execution_handle_creations",
        "descriptor_count": "_hipblaslt_fp16_descriptor_count",
        "scratch_live": "_hipblaslt_fp16_scratch_bytes_live",
        "scratch_peak": "_hipblaslt_fp16_scratch_bytes_peak",
    }
    if not all(hasattr(binding, symbol) for symbol in mapping.values()):
        return None
    return {key: int(getattr(binding, symbol)()) for key, symbol in mapping.items()}


def counter_delta(before: dict[str, int] | None, after: dict[str, int] | None):
    if before is None or after is None:
        return None
    return {key: after[key] - before[key] for key in before}


def make_model(tcnn: Any, otype: str, seed: int):
    return tcnn.Network(
        64,
        64,
        {
            "otype": otype,
            "precision": "Fp16",
            "n_neurons": 64,
            "n_hidden_layers": 2,
            "activation": "ReLU",
            "output_activation": "None",
        },
        seed=seed,
    ).cuda().eval()


def initialize_pair(torch: Any, candidate: Any, reference: Any, seed: int, count: int):
    if candidate.params.numel() != count or reference.params.numel() != count:
        raise RuntimeError("Unexpected parameter count")
    generator = torch.Generator(device="cuda").manual_seed(seed + 17)
    master = torch.randn(count, device="cuda", dtype=torch.float32, generator=generator) * 0.03
    with torch.no_grad():
        candidate.params.copy_(master.to(candidate.params.dtype))
        reference.params.copy_(master.to(reference.params.dtype))
    return {
        "master_sha256": tensor_sha256(master),
        "candidate_sha256": tensor_sha256(candidate.params),
        "reference_sha256": tensor_sha256(reference.params),
        "count": count,
    }


def make_input(torch: Any, batch: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed + 31)
    return torch.randn(batch, 64, device="cuda", dtype=torch.float16, generator=generator)


def correctness(torch: Any, candidate: Any, reference: Any, x: Any, cfg: dict[str, Any]):
    with torch.inference_mode():
        a1, b1 = candidate(x), reference(x)
        a2, b2 = candidate(x), reference(x)
    torch.cuda.current_stream().synchronize()
    diff = a1.float() - b1.float()
    ref_norm = float(b1.float().norm())
    return {
        "candidate_finite": bool(torch.isfinite(a1).all()),
        "reference_finite": bool(torch.isfinite(b1).all()),
        "candidate_repeat": bool(torch.equal(a1, a2)),
        "reference_repeat": bool(torch.equal(b1, b2)),
        "max_abs": float(diff.abs().max()),
        "normalized_l2": float(diff.norm()) / (ref_norm or 1.0),
        "allclose": bool(torch.allclose(a1.float(), b1.float(), atol=float(cfg["atol"]), rtol=float(cfg["rtol"]))),
        "candidate_sha256": tensor_sha256(a1),
        "reference_sha256": tensor_sha256(b1),
    }


def native_call(module: Any, x32: Any, params: Any, iterations: int, sync_each: bool, gap_ns: int):
    record, output = module.phase4a3_q0b_benchmark_inference(x32, params, iterations, sync_each, gap_ns)
    return dict(record), output


def convergence(module: Any, x32: Any, params: Any, cfg: dict[str, Any]):
    medians: list[float] = []
    windows = []
    for index in range(int(cfg["maximum_windows"])):
        record, output = native_call(module, x32, params, int(cfg["native_samples_per_window"]), True, 0)
        values = [float(value) for value in record["host_ns"]]
        med = float(statistics.median(values))
        medians.append(med)
        windows.append({"window": index + 1, "summary_ns": summary(values), "output_sha256": tensor_sha256(output)})
        if len(medians) < int(cfg["minimum_windows"]):
            continue
        last = medians[-int(cfg["last_windows"]):]
        center = float(statistics.median(last))
        spread = (max(last) - min(last)) / center if center else 0.0
        endpoint = abs(last[-1] - last[0]) / center if center else 0.0
        if spread <= float(cfg["last_window_spread_max"]) and endpoint <= float(cfg["endpoint_drift_max"]):
            return {
                "converged": True,
                "windows": windows,
                "last_medians_ns": last,
                "spread_fraction": spread,
                "endpoint_drift_fraction": endpoint,
            }
    last = medians[-int(cfg["last_windows"]):]
    center = float(statistics.median(last))
    return {
        "converged": False,
        "windows": windows,
        "last_medians_ns": last,
        "spread_fraction": (max(last) - min(last)) / center if center else 0.0,
        "endpoint_drift_fraction": abs(last[-1] - last[0]) / center if center else 0.0,
    }


def native_measure(module: Any, x32: Any, params: Any, cfg: dict[str, Any]):
    blocks = []
    for block in range(int(cfg["blocks_per_round"])):
        record, output = native_call(module, x32, params, int(cfg["samples_per_block"]), True, 0)
        values = [float(value) for value in record["host_ns"]]
        blocks.append({"block": block + 1, "summary_ns": summary(values), "output_sha256": tensor_sha256(output)})
    assessment = block_assessment(
        [item["summary_ns"]["median"] for item in blocks],
        float(cfg["block_spread_max"]),
        float(cfg["block_deviation_from_round_median_max"]),
    )
    return {"blocks": blocks, "assessment": assessment}


def public_measure(torch: Any, model: Any, x: Any, cfg: dict[str, Any]):
    stream = torch.cuda.current_stream()
    blocks = []
    with torch.inference_mode():
        for block in range(int(cfg["blocks_per_round"])):
            values = []
            last = None
            for _ in range(int(cfg["samples_per_block"])):
                begin = time.perf_counter_ns()
                last = model(x)
                stream.synchronize()
                values.append(float(time.perf_counter_ns() - begin))
            blocks.append({"block": block + 1, "summary_ns": summary(values), "output_sha256": tensor_sha256(last)})
    assessment = block_assessment(
        [item["summary_ns"]["median"] for item in blocks],
        float(cfg["block_spread_max"]),
        float(cfg["block_deviation_from_round_median_max"]),
    )
    return {"blocks": blocks, "assessment": assessment}


def queued_once(module: Any, x32: Any, params: Any, iterations: int):
    record, output = native_call(module, x32, params, iterations, False, 0)
    event_ms = float(record["event_ms"])
    submission_ms = float(record["host_submission_ns"]) / 1e6
    total_ms = float(record["host_total_ns"]) / 1e6
    return {
        "iterations": iterations,
        "event_ms": event_ms,
        "submission_ms": submission_ms,
        "total_ms": total_ms,
        "event_us_per_forward": event_ms * 1000.0 / iterations,
        "host_us_per_forward": total_ms * 1000.0 / iterations,
        "submission_over_gpu": submission_ms / event_ms if event_ms > 0 else float("inf"),
        "output_sha256": tensor_sha256(output),
    }


def queued_measure(module: Any, x32: Any, params: Any, cfg: dict[str, Any]):
    iterations = int(cfg["minimum_iterations"])
    calibration = []
    while True:
        item = queued_once(module, x32, params, iterations)
        calibration.append(item)
        if item["event_ms"] >= float(cfg["target_gpu_ms_per_block"]) or iterations >= int(cfg["maximum_iterations"]):
            break
        iterations = min(int(cfg["maximum_iterations"]), iterations * 2)
    blocks = [queued_once(module, x32, params, iterations) for _ in range(int(cfg["blocks_per_round"]))]
    event_values = [item["event_us_per_forward"] for item in blocks]
    headroom_values = [item["submission_over_gpu"] for item in blocks]
    return {
        "calibration": calibration,
        "iterations": iterations,
        "blocks": blocks,
        "event_us_per_forward": summary(event_values),
        "submission_over_gpu": summary(headroom_values),
        "passed": max(headroom_values) <= float(cfg["submission_over_gpu_max"]),
    }


def python_noop_floor(model: Any, x: Any, count: int):
    def no_op(_model, _x):
        return None
    values = []
    for _ in range(count):
        begin = time.perf_counter_ns()
        no_op(model, x)
        values.append(float(time.perf_counter_ns() - begin))
    return summary(values)


def gap_sweep_native(module: Any, x32: Any, params: Any, gaps_ms: list[float], samples: int):
    result = {}
    for gap_ms in gaps_ms:
        gap_ns = int(round(gap_ms * 1_000_000.0))
        native_call(module, x32, params, 1, True, gap_ns)
        record, output = native_call(module, x32, params, samples, True, gap_ns)
        values = [float(value) for value in record["host_ns"]]
        result[str(gap_ms)] = {"gap_ms": gap_ms, "summary_ns": summary(values), "output_sha256": tensor_sha256(output)}
    return result


def gap_sweep_public(torch: Any, model: Any, x: Any, gaps_ms: list[float], samples: int):
    stream = torch.cuda.current_stream()
    result = {}
    with torch.inference_mode():
        for gap_ms in gaps_ms:
            gap_s = gap_ms / 1000.0
            if gap_s:
                time.sleep(gap_s)
            model(x)
            stream.synchronize()
            values = []
            last = None
            for _ in range(samples):
                if gap_s:
                    time.sleep(gap_s)
                begin = time.perf_counter_ns()
                last = model(x)
                stream.synchronize()
                values.append(float(time.perf_counter_ns() - begin))
            result[str(gap_ms)] = {"gap_ms": gap_ms, "summary_ns": summary(values), "output_sha256": tensor_sha256(last)}
    return result


def correctness_pass(item: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return (
        item["candidate_finite"] and item["reference_finite"] and
        item["candidate_repeat"] and item["reference_repeat"] and
        item["allclose"] and item["normalized_l2"] <= float(cfg["normalized_l2_max"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--bridge", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("spin", "auto"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--start-order", choices=("AB", "BA"), required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    contract = read_json(args.contract)
    if contract["marker"] != MARKER:
        raise RuntimeError("Contract marker mismatch")
    os.sched_setaffinity(0, {args.cpu})

    bridge = Bridge(args.bridge)
    schedule = bridge.set_mode(args.mode)
    if not schedule["passed"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"marker": MARKER, "status": "FAIL", "reason": "schedule", "schedule": schedule}, indent=2) + "\n")
        return 2

    import torch
    import tinycudann as tcnn
    from tinycudann.modules import _C

    if not hasattr(_C.Module, "phase4a3_q0b_benchmark_inference"):
        raise RuntimeError("Q0b test-only binding hook is missing")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName != contract["environment"]["required_arch"]:
        raise RuntimeError(f"Unexpected arch: {props.gcnArchName}")

    seed = 2026072700 + args.batch * 10 + args.process_index
    candidate = make_model(tcnn, contract["comparison"]["candidate"], seed)
    reference = make_model(tcnn, contract["comparison"]["reference"], seed)
    identity = initialize_pair(torch, candidate, reference, seed, int(contract["comparison"]["parameter_elements"]))
    x = make_input(torch, args.batch, seed)

    # TCNN_RDNA4_P4A3_Q0B_TOOL_REPAIR_003:
    # The direct cpp_api hook is governed by TCNN's public API batch-size
    # granularity (256 in the frozen Phase-4A2 release), not by the rocWMMA
    # kernel's internal 16-row tile multiple. Mirror Module.forward() exactly.
    api_batch_granularity = int(_C.batch_size_granularity())
    required_granularity = int(
        contract["comparison"]["required_api_batch_size_granularity"]
    )
    if api_batch_granularity != required_granularity:
        raise RuntimeError(
            f"Q0b expected API batch granularity {required_granularity}, "
            f"got {api_batch_granularity}"
        )
    padded = (
        (args.batch + api_batch_granularity - 1)
        // api_batch_granularity
        * api_batch_granularity
    )
    x32 = torch.nn.functional.pad(
        x, [0, 0, 0, padded - args.batch]
    ).to(torch.float32).contiguous()
    if x32.shape != (padded, 64) or padded % api_batch_granularity != 0:
        raise RuntimeError(
            f"Q0b native input shape/granularity mismatch: "
            f"shape={tuple(x32.shape)}, padded={padded}, "
            f"granularity={api_batch_granularity}"
        )

    # TCNN_RDNA4_P4A3_Q0B_TOOL_REPAIR_002:
    # Module.params is the public FP32 master tensor. Module.forward() converts
    # it to the native backend precision before calling Module::fwd. The
    # test-only native hook bypasses Module.forward(), so it must receive the
    # same preconverted FP16 parameter tensor explicitly.
    native_params = {
        "rocwmma": candidate.params.detach().to(dtype=candidate.dtype).contiguous(),
        "hipblaslt": reference.params.detach().to(dtype=reference.dtype).contiguous(),
    }
    if candidate.params.dtype != torch.float32 or reference.params.dtype != torch.float32:
        raise RuntimeError("Q0b expected public FP32 master parameters")
    if native_params["rocwmma"].dtype != torch.float16 or native_params["hipblaslt"].dtype != torch.float16:
        raise RuntimeError("Q0b expected native FP16 parameter tensors")
    native_parameter_identity = {
        "rocwmma_dtype": str(native_params["rocwmma"].dtype),
        "hipblaslt_dtype": str(native_params["hipblaslt"].dtype),
        "rocwmma_sha256": tensor_sha256(native_params["rocwmma"]),
        "hipblaslt_sha256": tensor_sha256(native_params["hipblaslt"]),
        "equal_bytes": bool(torch.equal(native_params["rocwmma"], native_params["hipblaslt"])),
    }

    floors = bridge.floors(int(contract["floors"]["samples"]))
    noop_floor = python_noop_floor(candidate, x, int(contract["floors"]["samples"]))
    telemetry_before = rocm_smi()

    with torch.inference_mode():
        for model in (candidate, reference):
            for _ in range(256):
                model(x)
    torch.cuda.current_stream().synchronize()

    pre = correctness(torch, candidate, reference, x, contract["correctness"])
    before = counters(_C)

    models = {"A": ("rocwmma", candidate), "B": ("hipblaslt", reference)}
    round_orders = contract["matrix"]["round_orders_for_AB_process" if args.start_order == "AB" else "round_orders_for_BA_process"]
    rounds = []
    for round_index, order in enumerate(round_orders, 1):
        round_record = {"round": round_index, "order": order, "backends": {}}
        for token in order:
            name, model = models[token]
            native_module = model.native_tcnn_module
            params = native_params[name]
            conv = convergence(native_module, x32, params, contract["convergence"])
            native = native_measure(native_module, x32, params, contract["measurement"]["native_single_shot"])
            public = public_measure(torch, model, x, contract["measurement"]["public_python_single_shot"])
            queued = queued_measure(native_module, x32, params, contract["measurement"]["native_queued"])
            round_record["backends"][name] = {
                "convergence": conv,
                "native_single_shot": native,
                "public_single_shot": public,
                "native_queued": queued,
            }

        round_record["native_single_shot_ratio_reference_over_candidate"] = (
            round_record["backends"]["hipblaslt"]["native_single_shot"]["assessment"]["round_median"] /
            round_record["backends"]["rocwmma"]["native_single_shot"]["assessment"]["round_median"]
        )
        round_record["public_single_shot_ratio_reference_over_candidate"] = (
            round_record["backends"]["hipblaslt"]["public_single_shot"]["assessment"]["round_median"] /
            round_record["backends"]["rocwmma"]["public_single_shot"]["assessment"]["round_median"]
        )
        round_record["native_queued_ratio_reference_over_candidate"] = (
            round_record["backends"]["hipblaslt"]["native_queued"]["event_us_per_forward"]["median"] /
            round_record["backends"]["rocwmma"]["native_queued"]["event_us_per_forward"]["median"]
        )
        rounds.append(round_record)

    gaps = list(contract["measurement"]["gap_sweep_native_ms"])
    public_gaps = list(contract["measurement"]["gap_sweep_public_ms"])
    if args.process_index % 2:
        gaps.reverse()
        public_gaps.reverse()
    gap_backend_order = ["rocwmma", "hipblaslt"] if args.start_order == "AB" else ["hipblaslt", "rocwmma"]
    native_gap, public_gap = {}, {}
    for name in gap_backend_order:
        model = candidate if name == "rocwmma" else reference
        params = native_params[name]
        native_gap[name] = gap_sweep_native(model.native_tcnn_module, x32, params, gaps, int(contract["measurement"]["gap_sweep_samples"]))
        public_gap[name] = gap_sweep_public(torch, model, x, public_gaps, int(contract["measurement"]["gap_sweep_samples"]))

    after = counters(_C)
    post = correctness(torch, candidate, reference, x, contract["correctness"])
    telemetry_after = rocm_smi()
    cdelta = counter_delta(before, after)

    all_converged = all(backend["convergence"]["converged"] for rr in rounds for backend in rr["backends"].values())
    native_stationary = all(backend["native_single_shot"]["assessment"]["passed"] for rr in rounds for backend in rr["backends"].values())
    public_stationary = all(backend["public_single_shot"]["assessment"]["passed"] for rr in rounds for backend in rr["backends"].values())
    queued_headroom = all(backend["native_queued"]["passed"] for rr in rounds for backend in rr["backends"].values())
    cache_stable = (
        cdelta is not None and
        cdelta["cache_misses"] == 0 and cdelta["heuristic_queries"] == 0 and
        cdelta["handle_creations"] == 0 and cdelta["descriptor_count"] == 0 and
        cdelta["scratch_peak"] == 0 and after["scratch_live"] == 0
    )
    faster_native = min(
        backend["native_single_shot"]["assessment"]["round_median"]
        for rr in rounds for backend in rr["backends"].values()
    )
    empty_sync_median = floors.get("empty_sync_ns", {}).get("median")
    empty_sync_fraction = float(empty_sync_median) / float(faster_native) if empty_sync_median is not None and faster_native > 0 else None
    foreign = telemetry_before.get("foreign_gpu_process_ids", []) + telemetry_after.get("foreign_gpu_process_ids", [])

    gates = {
        "schedule": schedule["passed"],
        "arch": props.gcnArchName == "gfx1201",
        "pre_correctness": correctness_pass(pre, contract["correctness"]),
        "post_correctness": correctness_pass(post, contract["correctness"]),
        "all_convergence": all_converged,
        "native_stationarity": native_stationary,
        "public_stationarity": public_stationary,
        "native_queued_headroom": queued_headroom,
        "cache_stable": cache_stable,
        "floors_measured": floors.get("result", {}).get("code") == 0,
        "empty_sync_fraction": empty_sync_fraction is not None and empty_sync_fraction <= float(contract["floors"]["empty_sync_fraction_max"]),
        "no_foreign_gpu_processes": not foreign,
    }

    def process_ratio(key: str) -> float:
        values = [float(rr[f"{key}_ratio_reference_over_candidate"]) for rr in rounds]
        return math.exp(statistics.fmean(math.log(value) for value in values))

    result = {
        "marker": MARKER,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "schedule": schedule,
        "batch": args.batch,
        "start_order": args.start_order,
        "process_index": args.process_index,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu_name": torch.cuda.get_device_name(0),
            "gcnArchName": props.gcnArchName,
            "binding": str(pathlib.Path(_C.__file__).resolve()),
        },
        "identity": identity,
        "native_parameter_identity": native_parameter_identity,
        "input": {
            "public_batch": args.batch,
            "api_batch_size_granularity": api_batch_granularity,
            "api_padded_batch": padded,
            "padding_factor": padded / args.batch,
            "public_dtype": str(x.dtype),
            "native_dtype": str(x32.dtype),
            "public_sha256": tensor_sha256(x),
            "native_sha256": tensor_sha256(x32),
        },
        "floors": floors,
        "python_noop_call_ns": noop_floor,
        "empty_sync_fraction_of_faster_native": empty_sync_fraction,
        "telemetry_before_all_warmup": telemetry_before,
        "telemetry_after_all_measurement": telemetry_after,
        "pre_correctness": pre,
        "post_correctness": post,
        "counters_before": before,
        "counters_after": after,
        "counter_delta": cdelta,
        "rounds": rounds,
        "native_gap_sweep": native_gap,
        "public_gap_sweep": public_gap,
        "process_ratios_diagnostic": {
            "native_single_shot": process_ratio("native_single_shot"),
            "public_single_shot": process_ratio("public_single_shot"),
            "native_queued": process_ratio("native_queued"),
        },
        "gates": gates,
        "no_performance_claim": True,
        "floor_subtracted": False,
        "common_additive_S_assumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for name, passed in gates.items():
        print(f"WIDTH64_Q0B_WORKER_{name.upper()}: {'PASS' if passed else 'FAIL'}")
    print("PHASE4A3_Q0B_WORKER:", "PASS" if all(gates.values()) else "FAIL")
    print("output:", args.output)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
