#!/usr/bin/env python3
"""Fresh-process Q0c worker.

The worker deliberately owns one subphase/group/process only. It emits raw
round/block data; only the finalizer makes apparatus decisions.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import pathlib
import statistics
import time
from typing import Any, Callable

from phase4a3_q0c_common import MARKER, calibrate_by_probe, latin_gap_order, load_contract, padded_batch


def tensor_hash(value: Any) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def model(tcnn: Any, otype: str, seed: int) -> Any:
    return tcnn.Network(64, 64, {"otype": otype, "precision": "Fp16", "n_neurons": 64, "n_hidden_layers": 2, "activation": "ReLU", "output_activation": "None"}, seed=seed).cuda().eval()


def public_once(torch: Any, instance: Any, x: Any) -> tuple[float, Any]:
    start = time.perf_counter_ns()
    output = instance(x)
    torch.cuda.current_stream().synchronize()
    return float(time.perf_counter_ns() - start), output


def public_queued(torch: Any, instance: Any, x: Any, iterations: int) -> tuple[dict[str, Any], Any]:
    stream = torch.cuda.current_stream()
    begin_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_start = time.perf_counter_ns()
    begin_event.record(stream)
    last = None
    submit_start = time.perf_counter_ns()
    with torch.inference_mode():
        for _ in range(iterations):
            last = instance(x)
    submission_ns = time.perf_counter_ns() - submit_start
    end_event.record(stream)
    stream.synchronize()
    total_ns = time.perf_counter_ns() - total_start
    return {"iterations": iterations, "host_submission_ns": submission_ns, "host_total_ns": total_ns, "event_ms": begin_event.elapsed_time(end_event)}, last


def convergence(call: Callable[[], tuple[float, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    count, trace = calibrate_by_probe(lambda n: sum(call()[0] for _ in range(n)) / 1e6, cfg["target_ms"], cfg["min_samples"], cfg["max_samples"])
    medians, windows = [], []
    for index in range(cfg["max_windows"]):
        values = [call()[0] for _ in range(count)]
        medians.append(float(statistics.median(values)))
        windows.append({"index": index, "samples": count, "median_ns": medians[-1]})
        if len(medians) >= cfg["min_windows"]:
            last = medians[-cfg["last_windows"]:]
            center = statistics.median(last)
            spread = (max(last) - min(last)) / center
            drift = abs(last[-1] - last[0]) / center
            if spread <= cfg["spread_max"] and drift <= cfg["drift_max"]:
                return {"passed": True, "calibration": trace, "frozen_samples": count, "windows": windows}
    return {"passed": False, "calibration": trace, "frozen_samples": count, "windows": windows}


def latency_score(call: Callable[[], tuple[float, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    count, trace = calibrate_by_probe(lambda n: sum(call()[0] for _ in range(n)) / 1e6, cfg["target_ms"], cfg["min_samples"], cfg["max_samples"])
    blocks, outputs = [], []
    for index in range(cfg["blocks"]):
        values, last = [], None
        for _ in range(count):
            elapsed, last = call()
            values.append(elapsed)
        blocks.append({"index": index, "samples": count, "median_ns": statistics.median(values), "raw_ns": values})
        outputs.append(tensor_hash(last))
    medians = [item["median_ns"] for item in blocks]
    center = statistics.median(medians)
    passed = (max(medians) - min(medians)) / center <= cfg["spread_max"] and max(abs(value - center) / center for value in medians) <= cfg["deviation_max"]
    return {"passed": passed, "calibration": trace, "frozen_samples": count, "blocks": blocks, "round_median_ns": center, "output_hashes": outputs}


def correctness(torch: Any, a: Any, b: Any, x: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    a1, b1, a2, b2 = a(x), b(x), a(x), b(x)
    torch.cuda.current_stream().synchronize()
    diff = a1.float() - b1.float()
    value = {"finite": bool(torch.isfinite(a1).all() and torch.isfinite(b1).all()), "repeat": bool(torch.equal(a1, a2) and torch.equal(b1, b2)), "allclose": bool(torch.allclose(a1.float(), b1.float(), atol=cfg["atol"], rtol=cfg["rtol"])), "normalized_l2": float(diff.norm()) / (float(b1.float().norm()) or 1.0)}
    value["passed"] = value["finite"] and value["repeat"] and value["allclose"] and value["normalized_l2"] <= cfg["normalized_l2_max"]
    return value


def counters(binding: Any) -> dict[str, int] | None:
    names = {"cache_misses":"_hipblaslt_fp16_cache_misses", "heuristic_queries":"_hipblaslt_fp16_heuristic_queries", "handle_creations":"_hipblaslt_fp16_execution_handle_creations", "descriptor_count":"_hipblaslt_fp16_descriptor_count", "scratch_live":"_hipblaslt_fp16_scratch_bytes_live", "scratch_peak":"_hipblaslt_fp16_scratch_bytes_peak"}
    if not all(hasattr(binding, value) for value in names.values()):
        return None
    return {key:int(getattr(binding, value)()) for key, value in names.items()}


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=pathlib.Path, required=True)
    p.add_argument("--bridge", type=pathlib.Path, required=True)
    p.add_argument("--phase", choices=("LN", "LP", "TP", "TD", "G"), required=True)
    p.add_argument("--schedule", choices=("spin", "auto"), required=True)
    p.add_argument("--batch", type=int)
    p.add_argument("--metric", choices=("native256", "public1"))
    p.add_argument("--process-index", type=int, required=True)
    p.add_argument("--start-order", choices=("AB", "BA"), required=True)
    p.add_argument("--cpu", type=int, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    return p


def main() -> int:
    args = argument_parser().parse_args()
    contract = load_contract(args.contract)
    os.sched_setaffinity(0, {args.cpu})
    bridge = ctypes.CDLL(str(args.bridge), mode=ctypes.RTLD_GLOBAL)
    getter = getattr(bridge, "q0c_schedule_" + args.schedule)
    getter.restype = ctypes.c_uint
    bridge.q0c_set_schedule.argtypes = [ctypes.c_uint]
    bridge.q0c_set_schedule.restype = ctypes.c_int
    if bridge.q0c_set_schedule(getter()) != 0:
        raise RuntimeError("failed to set HIP wait schedule before HIP initialization")
    floor_count = int(contract["floors"]["samples"])
    floor_array = ctypes.c_uint64 * floor_count
    timer_floor, empty_floor, minimal_floor = floor_array(), floor_array(), floor_array()
    bridge.q0c_measure_floors.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    bridge.q0c_measure_floors.restype = ctypes.c_int
    import torch
    import tinycudann as tcnn
    from tinycudann.modules import _C
    if not hasattr(_C.Module, "phase4a3_q0c_benchmark_inference"):
        raise RuntimeError("Q0c test-only hook missing from loaded extension")
    if torch.cuda.get_device_properties(0).gcnArchName != contract["baseline"]["required_arch"]:
        raise RuntimeError("not gfx1201")
    if int(_C.batch_size_granularity()) != 256:
        raise RuntimeError("public batch granularity is not 256")
    batch = args.batch or (256 if args.metric == "native256" else 1)
    seed = 2026072700 + batch * 10 + args.process_index
    candidate, reference = model(tcnn, contract["baseline"]["candidate"], seed), model(tcnn, contract["baseline"]["reference"], seed)
    master = torch.randn(contract["baseline"]["parameter_elements"], device="cuda", dtype=torch.float32) * 0.03
    with torch.no_grad():
        candidate.params.copy_(master); reference.params.copy_(master)
    x = torch.randn(batch, 64, device="cuda", dtype=torch.float16)
    pre = correctness(torch, candidate, reference, x, contract["correctness"])
    floor_code = bridge.q0c_measure_floors(floor_count, timer_floor, empty_floor, minimal_floor)
    if floor_code != 0:
        raise RuntimeError("HIP floor measurement failed")
    floors = {"timer_pair_ns":list(timer_floor), "empty_sync_ns":list(empty_floor), "minimal_gpu_completion_ns":list(minimal_floor)}
    counter_before = counters(_C)
    native_params = {"candidate": candidate.params.to(torch.float16).contiguous(), "reference": reference.params.to(torch.float16).contiguous()}
    if tensor_hash(native_params["candidate"]) != tensor_hash(native_params["reference"]):
        raise RuntimeError("native parameter hashes differ")
    padded = padded_batch(batch)
    x32 = torch.nn.functional.pad(x, [0, 0, 0, padded - batch]).to(torch.float32).contiguous()

    def native_call(instance: Any, params: Any, iterations: int = 1, sync_each: bool = True, gap_ns: int = 0) -> tuple[dict[str, Any], Any]:
        record, output = instance.native_tcnn_module.phase4a3_q0c_benchmark_inference(x32, params, iterations, sync_each, gap_ns)
        return dict(record), output

    names = {"A": ("candidate", candidate), "B": ("reference", reference)}
    orders = contract["matrix"]["round_orders_" + args.start_order]
    rounds = []
    if args.phase in ("LN", "LP"):
        cfg = contract["latency"][args.phase]
        for index, order in enumerate(orders):
            record = {"index": index, "order": order, "backends": {}}
            for token in order:
                name, instance = names[token]
                if args.phase == "LP":
                    call = lambda i=instance: public_once(torch, i, x)
                else:
                    def call(i=instance, n=name):
                        sample, output = native_call(i, native_params[n])
                        return float(sample["host_ns"][0]), output
                conv = convergence(call, cfg["convergence"])
                score = latency_score(call, cfg["scoring"])
                record["backends"][name] = {"convergence": conv, "score": score}
            record["ratio"] = record["backends"]["reference"]["score"]["round_median_ns"] / record["backends"]["candidate"]["score"]["round_median_ns"]
            rounds.append(record)
    elif args.phase in ("TP", "TD"):
        cfg = contract["throughput"][args.phase]
        for index, order in enumerate(orders):
            record = {"index": index, "order": order, "backends": {}}
            for token in order:
                name, instance = names[token]
                iterations = cfg["min_iterations"]
                blocks = []
                # Hook supplies host submission, host total and whole-loop HIP event.
                while True:
                    sample, _ = (public_queued(torch, instance, x, iterations) if args.phase == "TP" else native_call(instance, native_params[name], iterations, False))
                    clock = float(sample["host_total_ns"]) / 1e6 if args.phase == "TP" else float(sample["event_ms"])
                    if clock >= cfg["target_ms"] or iterations >= cfg["max_iterations"]: break
                    iterations = min(cfg["max_iterations"], iterations * 2)
                frozen_iterations = iterations
                for _ in range(cfg["blocks"]):
                    sample, output = (public_queued(torch, instance, x, frozen_iterations) if args.phase == "TP" else native_call(instance, native_params[name], frozen_iterations, False))
                    sample = dict(sample); sample["output_sha256"] = tensor_hash(output)
                    sample["submission_over_gpu"] = float(sample["host_submission_ns"]) / 1e6 / float(sample["event_ms"])
                    blocks.append(sample)
                record["backends"][name] = {"frozen_iterations": frozen_iterations, "blocks": blocks}
            rounds.append(record)
    else:
        order = latin_gap_order(contract, args.process_index)
        raw = {}
        backend_order = (("candidate", candidate), ("reference", reference)) if args.start_order == "AB" else (("reference", reference), ("candidate", candidate))
        for name, instance in backend_order:
            raw[name] = {}
            for gap in order:
                gap_ns = round(gap * 1_000_000)
                is_public = args.metric == "public1"
                for _ in range(contract["gap"]["sacrificial_samples_per_gap"]):
                    if is_public:
                        if gap_ns: time.sleep(gap_ns / 1e9)
                        public_once(torch, instance, x)
                    else:
                        native_call(instance, native_params[name], 1, True, gap_ns)
                blocks = []
                for _ in range(contract["gap"]["blocks"]):
                    values = []
                    for _ in range(contract["gap"]["samples_per_block"]):
                        if is_public:
                            if gap_ns: time.sleep(gap_ns / 1e9)
                            elapsed, _ = public_once(torch, instance, x)
                        else:
                            sample, _ = native_call(instance, native_params[name], 1, True, gap_ns)
                            elapsed = float(sample["host_ns"][0])
                        values.append(elapsed)
                    blocks.append(values)
                raw[name][str(gap)] = blocks
        rounds = [{"gap_order_ms": order, "raw_samples_ns": raw}]
    post = correctness(torch, candidate, reference, x, contract["correctness"])
    counter_after = counters(_C)
    counter_delta = ({key:counter_after[key]-counter_before[key] for key in counter_before} if counter_before is not None and counter_after is not None else None)
    counter_stable = counter_delta is not None and all(counter_delta[key] == 0 for key in ("cache_misses","heuristic_queries","handle_creations","descriptor_count","scratch_peak")) and counter_after["scratch_live"] == 0
    empty_fraction = None
    if args.phase in ("LN", "LP") and rounds:
        fastest = min(item["backends"][name]["score"]["round_median_ns"] for item in rounds for name in ("candidate","reference"))
        empty_fraction = statistics.median(floors["empty_sync_ns"]) / fastest
    gates = {"pre_correctness":pre["passed"], "post_correctness":post["passed"], "counters_stable":counter_stable, "floors_measured":True, "empty_sync_fraction":empty_fraction is None or empty_fraction <= contract["floors"]["empty_sync_fraction_max"]}
    result = {"marker": MARKER, "phase": args.phase, "schedule": args.schedule, "batch": batch, "metric": args.metric, "process_index": args.process_index, "start_order": args.start_order, "padded_batch": padded, "pre_correctness": pre, "post_correctness": post, "floors":floors, "empty_sync_fraction":empty_fraction, "counters_before":counter_before, "counters_after":counter_after, "counter_delta":counter_delta, "gates":gates, "rounds": rounds, "no_performance_claim": True, "floor_subtracted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
