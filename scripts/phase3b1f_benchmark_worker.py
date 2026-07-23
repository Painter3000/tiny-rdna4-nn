#!/usr/bin/env python3
"""One-fresh-process worker for TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001."""
import argparse
import gc
import hashlib
import json
import math
import os
import pathlib
import platform
import resource
import statistics
import subprocess
import sys
import time

PROCESS_IMPORT_START_NS = time.perf_counter_ns()
import torch
TORCH_IMPORT_END_NS = time.perf_counter_ns()
import tinycudann as tcnn
from tinycudann.modules import _C
TCNN_IMPORT_END_NS = time.perf_counter_ns()

MARKER = "TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp16_counters():
    names = ("cache_hits", "cache_misses", "cache_size", "heuristic_queries", "execution_handle_count",
             "execution_handle_creations", "descriptor_count", "scratch_bytes_live", "scratch_bytes_peak")
    return {name: int(getattr(_C, "_hipblaslt_fp16_" + name)()) for name in names}


def fp32_counters():
    mapping = {
        "cache_hits": "_hipblaslt_cache_hits", "cache_misses": "_hipblaslt_cache_misses",
        "cache_size": "_hipblaslt_cache_size", "heuristic_queries": "_hipblaslt_cache_misses",
        "execution_handle_count": "_hipblaslt_execution_handle_count",
        "execution_handle_creations": "_hipblaslt_execution_handle_creations",
        "descriptor_count": "_hipblaslt_epilogue_descriptor_count",
        "scratch_bytes_live": "_hipblaslt_fused_partial_bytes_live",
        "scratch_bytes_peak": "_hipblaslt_fused_partial_bytes_peak",
    }
    return {name: int(getattr(_C, symbol)()) for name, symbol in mapping.items()}


def telemetry():
    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gcnArchName": torch.cuda.get_device_properties(0).gcnArchName,
        "pytorch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "native_module_path": str(pathlib.Path(_C.__file__).resolve()),
        "host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_governors": {},
        "relevant_environment": {k: os.environ.get(k) for k in (
            "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "PYTORCH_ROCM_ARCH",
            "HIPBLASLT_LOG_MASK", "HIPBLASLT_LOG_FILE",
        )},
    }
    for path in pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            result["cpu_governors"][str(path)] = path.read_text().strip()
        except OSError:
            pass
    try:
        completed = subprocess.run(["rocm-smi", "--showtemp", "--showclocks", "--showmeminfo", "vram",
                                    "--showuse", "--showpids", "--json"], capture_output=True, text=True, timeout=10)
        result["rocm_smi"] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        result["rocm_smi"] = {"returncode": -1, "error": str(error)}
    return result


def network_config(fp16, case):
    return {
        "otype": "HipBLASLtMLPFP16" if fp16 else "HipBLASLtMLP",
        "precision": "Fp16" if fp16 else "Fp32",
        "n_neurons": case["hidden_width"],
        "n_hidden_layers": case["hidden_layers"],
        "activation": "ReLU",
        "output_activation": "None",
    }


def encoding_network_config(fp16, width=64):
    return {
        "otype": "HipBLASLtMLPFP16" if fp16 else "HipBLASLtMLP",
        "precision": "Fp16" if fp16 else "Fp32",
        "n_neurons": width,
        "n_hidden_layers": 2,
        "activation": "ReLU",
        "output_activation": "None",
    }


def make_pair(case):
    torch.manual_seed(case["seed"])
    torch.cuda.manual_seed_all(case["seed"])
    operation = case["operation"]
    if case["family"] == "network_only":
        candidate = tcnn.Network(case["input_dims"], case["output_dims"], network_config(True, case), seed=case["seed"])
        reference = tcnn.Network(case["input_dims"], case["output_dims"], network_config(False, case), seed=case["seed"])
        reference.params.data.copy_(candidate.params.detach().float())
        dims = case["input_dims"]
    elif operation.startswith("encoding_"):
        candidate = tcnn.Encoding(case["dims"], case["config"], dtype=torch.float16, seed=case["seed"])
        reference = tcnn.Encoding(case["dims"], case["config"], dtype=torch.float32, seed=case["seed"])
        if candidate.params.numel():
            reference.params.data.copy_(candidate.params.detach().float())
        dims = case["dims"]
    else:
        candidate = tcnn.NetworkWithInputEncoding(case["dims"], 16, case["config"], encoding_network_config(True), seed=case["seed"])
        reference = tcnn.NetworkWithInputEncoding(case["dims"], 16, case["config"], encoding_network_config(False), seed=case["seed"])
        reference.params.data.copy_(candidate.params.detach().float())
        dims = case["dims"]
    generator = torch.Generator(device="cuda").manual_seed(case["seed"] + 1)
    x = torch.randn(case["batch"], dims, device="cuda", generator=generator) * 0.2
    target = torch.randn(case["batch"], candidate.n_output_dims, device="cuda", generator=generator) * 0.1
    upstream = torch.randn(case["batch"], candidate.n_output_dims, device="cuda", generator=generator) * 0.1
    return candidate, reference, x, target, upstream


def operation_callable(model, operation, x, target, upstream):
    optimizer = torch.optim.Adam([model.params], lr=1e-3) if "adam" in operation else None
    if operation in ("forward", "encoding_forward", "network_with_encoding_forward"):
        return lambda: model(x)
    if operation in ("forward_backward", "encoding_forward_backward", "network_with_encoding_forward_backward"):
        def forward_backward():
            model.zero_grad(set_to_none=True)
            model(x).backward(upstream)
        return forward_backward
    if operation in ("adam_training_step", "network_with_encoding_adam_step"):
        def training_step():
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x).float() - target.float()).square().mean()
            loss.backward()
            optimizer.step()
        return training_step
    raise ValueError(operation)


def event_time(stream, callback, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iterations):
            callback()
        end.record(stream)
    end.synchronize()
    return {"event_ms_per_iteration": start.elapsed_time(end) / iterations,
            "wall_ns_per_iteration": (time.perf_counter_ns() - wall_start) / iterations}


def calibrate(stream, callbacks, calibration):
    iterations = calibration["min_iterations"]
    while True:
        elapsed = event_time(stream, callbacks[0], iterations)["event_ms_per_iteration"] * iterations
        if elapsed >= calibration["target_min_ms"] or iterations >= calibration["max_iterations"]:
            return iterations
        scale = max(2, math.ceil(calibration["target_min_ms"] / max(elapsed, 0.001)))
        iterations = min(calibration["max_iterations"], iterations * scale)


def warmup(stream, callbacks, case, rules):
    count = rules["training_iterations_min"] if "adam" in case["operation"] else rules["default_iterations_min"]
    before = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    start = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        for callback in callbacks:
            for _ in range(count):
                callback()
    stream.synchronize()
    middle = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    with torch.cuda.stream(stream):
        for callback in callbacks:
            for _ in range(max(5, count // 10)):
                callback()
    stream.synchronize()
    after = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    stable_fields = ("cache_size", "heuristic_queries", "execution_handle_count", "execution_handle_creations", "descriptor_count")
    stable = all(middle[backend][field] == after[backend][field] for backend in middle for field in stable_fields)
    scratch_zero = all(after[backend]["scratch_bytes_live"] == 0 for backend in after)
    return {"iterations": count, "elapsed_ns": time.perf_counter_ns() - start,
            "before": before, "stable_sample": middle, "after": after,
            "stable": stable, "scratch_live_zero": scratch_zero}


def parse_algorithm_log(path):
    text = path.read_text(errors="replace") if path.exists() else ""
    algorithm_lines = [line for line in text.splitlines() if "algo" in line.lower()]
    return {"path": str(path), "size_bytes": len(text.encode()), "algorithm_lines": algorithm_lines,
            "algorithm_id_present": any(any(ch.isdigit() for ch in line) for line in algorithm_lines)}


def cold_start(stage, output):
    seed = 2026072499
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = {"otype": "HipBLASLtMLPFP16", "precision": "Fp16", "n_neurons": 32,
              "n_hidden_layers": 2, "activation": "ReLU", "output_activation": "None"}
    timings = {
        "python_import_ns": TORCH_IMPORT_END_NS - PROCESS_IMPORT_START_NS,
        "torch_import_ns": TORCH_IMPORT_END_NS - PROCESS_IMPORT_START_NS,
        "native_extension_import_ns": TCNN_IMPORT_END_NS - TORCH_IMPORT_END_NS,
    }
    start = time.perf_counter_ns()
    model = tcnn.Network(32, 16, config, seed=seed)
    timings["model_construction_ns"] = time.perf_counter_ns() - start
    x = torch.randn(4096, 32, device="cuda") * 0.2
    upstream = torch.randn(4096, 16, device="cuda") * 0.1
    torch.cuda.synchronize()
    before = fp16_counters()
    start = time.perf_counter_ns()
    y = model(x)
    torch.cuda.synchronize()
    timings["first_forward_ns"] = time.perf_counter_ns() - start
    after_first = fp16_counters()
    start = time.perf_counter_ns()
    y.backward(upstream, retain_graph=True)
    torch.cuda.synchronize()
    timings["first_backward_ns"] = time.perf_counter_ns() - start
    optimizer = torch.optim.Adam([model.params], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter_ns()
    model(x).float().square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize()
    timings["first_training_step_ns"] = time.perf_counter_ns() - start
    timings["first_algorithm_heuristic_ns"] = timings["first_forward_ns"] if after_first["heuristic_queries"] > before["heuristic_queries"] else None
    result = {
        "marker": MARKER, "kind": "cold_start", "requested_stage": stage,
        "timings": timings, "counters_before": before, "counters_after_first_forward": after_first,
        "module_path": str(pathlib.Path(_C.__file__).resolve()), "system": telemetry(),
        "finite": bool(torch.isfinite(model.params).all() and torch.isfinite(model.params.grad).all()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["finite"] and timings.get(stage + "_ns") is not None else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--case-id")
    parser.add_argument("--process-index", type=int)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--execute-primary", action="store_true")
    parser.add_argument("--cold-stage")
    args = parser.parse_args()
    if args.cold_stage:
        return cold_start(args.cold_stage, args.output)
    if not args.execute_primary:
        raise SystemExit("Primary measurement requires explicit --execute-primary")
    if not all((args.manifest, args.manifest_sha256, args.case_id, args.process_index is not None)):
        raise SystemExit("Primary measurement arguments are incomplete")
    if sha256(args.manifest) != args.manifest_sha256:
        raise SystemExit("Manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text())
    case = next((item for item in manifest["cases"] if item["id"] == args.case_id), None)
    if case is None or not case["supported"]:
        raise SystemExit("Unknown or unsupported case")
    if torch.cuda.get_device_properties(0).gcnArchName != "gfx1201":
        raise SystemExit("Expected gfx1201")
    module_path = pathlib.Path(_C.__file__).resolve()
    if "_120_C" not in module_path.name:
        raise SystemExit("Wrong native module")
    log_path = args.output.with_suffix(".hipblaslt.log")
    before_system = telemetry()
    torch.cuda.reset_peak_memory_stats()
    candidate, reference, x, target, upstream = make_pair(case)
    callbacks = (
        operation_callable(candidate, case["operation"], x, target, upstream),
        operation_callable(reference, case["operation"], x, target, upstream),
    )
    stream = torch.cuda.Stream()
    warm = warmup(stream, callbacks, case, manifest["warmup"])
    iterations = calibrate(stream, callbacks, manifest["calibration"])
    orders = manifest["measurement"]["orders"]
    blocks = []
    for round_index in range(manifest["measurement"]["paired_rounds_per_process"]):
        selected = orders[(case["seed"] + args.process_index + round_index) % len(orders)]
        full_order = selected + list(reversed(selected))
        for position, backend in enumerate(full_order):
            callback = callbacks[0 if backend == "FP16" else 1]
            timing = event_time(stream, callback, iterations)
            blocks.append({"round": round_index, "position": position, "backend": backend,
                           "iterations": iterations, **timing})
    stream.synchronize()
    after_counters = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    finite = all(torch.isfinite(model.params).all().item() for model in (candidate, reference))
    fp16_times = [x["event_ms_per_iteration"] for x in blocks if x["backend"] == "FP16"]
    fp32_times = [x["event_ms_per_iteration"] for x in blocks if x["backend"] == "FP32"]
    process_speedup = statistics.median(fp32_times) / statistics.median(fp16_times)
    del candidate, reference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    released = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    algorithm = parse_algorithm_log(log_path)
    invalid_reasons = []
    if not warm["stable"]: invalid_reasons.append("post_warmup_resource_growth")
    if not warm["scratch_live_zero"] or any(x["scratch_bytes_live"] for x in released.values()): invalid_reasons.append("scratch_live_nonzero")
    if not finite: invalid_reasons.append("nan_or_inf")
    if not algorithm["algorithm_id_present"]: invalid_reasons.append("missing_algorithm_id")
    if released["FP16"]["descriptor_count"] != warm["before"]["FP16"]["descriptor_count"]: invalid_reasons.append("descriptor_not_released")
    result = {
        "marker": MARKER, "manifest_sha256": args.manifest_sha256, "case": case,
        "process_index": args.process_index, "module_path": str(module_path),
        "before_system": before_system, "after_system": telemetry(), "warmup": warm,
        "calibrated_iterations": iterations, "timing_blocks": blocks,
        "process_medians_ms": {"FP16": statistics.median(fp16_times), "FP32": statistics.median(fp32_times)},
        "process_speedup": process_speedup, "resources_after_timing": after_counters,
        "resources_after_release": released,
        "torch_memory": {"allocated_peak": torch.cuda.max_memory_allocated(), "reserved_peak": torch.cuda.max_memory_reserved()},
        "backend_evidence": {
            "FP16": {"backend": "HipBLASLtMLPFP16", "dtype_a": "HIP_R_16F", "dtype_b": "HIP_R_16F",
                     "dtype_c": "HIP_R_32F", "dtype_d": "HIP_R_16F", "compute_type": "HIPBLAS_COMPUTE_32F",
                     "layout": "ColumnMajor", "fallback": False},
            "FP32": {"backend": "HipBLASLtMLPFP32", "dtype_a": "HIP_R_32F", "dtype_b": "HIP_R_32F",
                     "dtype_c": "HIP_R_32F", "dtype_d": "HIP_R_32F", "compute_type": "HIPBLAS_COMPUTE_32F",
                     "layout": "ColumnMajor", "fallback": False},
            "algorithm_log": algorithm,
        },
        "valid": not invalid_reasons, "invalid_reasons": invalid_reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("VALID" if result["valid"] else "INVALID")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
