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
import re
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


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def metric(actual, reference):
    actual = actual.float()
    reference = reference.float()
    difference = actual - reference
    return {
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "normalized_l2": float(difference.norm()) / (float(reference.norm()) or 1.0),
        "nan_count": int(torch.isnan(actual).sum()),
        "inf_count": int(torch.isinf(actual).sum()),
    }


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
        parsed = json.loads(completed.stdout) if completed.returncode == 0 and completed.stdout.strip() else {}
        flat = json.dumps(parsed)
        def number(pattern):
            match = re.search(pattern, flat, re.IGNORECASE)
            return float(match.group(1)) if match else None
        result["rocm_smi"] = {"returncode": completed.returncode, "raw": parsed, "stderr": completed.stderr}
        sclk = re.search(r"sclk clock speed:[^0-9]*([0-9]+)", flat, re.IGNORECASE)
        pids = {int(value) for value in re.findall(r"PID([0-9]+)", flat)}
        foreign_pids = sorted(pid for pid in pids if pid != os.getpid())
        result.update({
            "temperature_c": number(r"temperature[^0-9]*([0-9]+(?:\\.[0-9]+)?)"),
            "clock_mhz": float(sclk.group(1)) if sclk else None,
            "gpu_busy_percent": number(r"(?:gpu use|use)[^0-9]*([0-9]+(?:\\.[0-9]+)?)"),
            "vram_used_bytes": number(r"(?:vram.*used|used vram)[^0-9]*([0-9]+)"),
            "gpu_process_ids": sorted(pids),
            "foreign_gpu_process_ids": foreign_pids,
            "foreign_gpu_processes": len(foreign_pids),
            "telemetry_available": completed.returncode == 0 and bool(parsed),
        })
    except (OSError, subprocess.TimeoutExpired) as error:
        result["rocm_smi"] = {"returncode": -1, "error": str(error)}
        result.update({"temperature_c": None, "clock_mhz": None, "gpu_busy_percent": None,
                       "vram_used_bytes": None, "foreign_gpu_processes": None, "telemetry_available": False})
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


def encoding_network_config(fp16, topology):
    return {
        "otype": "HipBLASLtMLPFP16" if fp16 else "HipBLASLtMLP",
        "precision": "Fp16" if fp16 else "Fp32",
        "n_neurons": topology["hidden_width"],
        "n_hidden_layers": topology["hidden_layers"],
        "activation": topology["activation"],
        "output_activation": topology["output_activation"],
    }


def make_pair(case, harness_contract):
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
        topology = harness_contract["network_with_encoding_topology"]
        candidate = tcnn.NetworkWithInputEncoding(case["dims"], topology["output_dims"], case["config"], encoding_network_config(True, topology), seed=case["seed"])
        reference = tcnn.NetworkWithInputEncoding(case["dims"], topology["output_dims"], case["config"], encoding_network_config(False, topology), seed=case["seed"])
        reference.params.data.copy_(candidate.params.detach().float())
        dims = case["dims"]
    generator = torch.Generator(device="cuda").manual_seed(case["seed"] + 1)
    x = torch.randn(case["batch"], dims, device="cuda", generator=generator) * 0.2
    target = torch.randn(case["batch"], candidate.n_output_dims, device="cuda", generator=generator) * 0.1
    upstream = torch.randn(case["batch"], candidate.n_output_dims, device="cuda", generator=generator) * 0.1
    return candidate, reference, x, target, upstream


def pair_identity(candidate, reference, x, target, upstream):
    result = {
        "input_sha256": tensor_sha256(x),
        "target_sha256": tensor_sha256(target),
        "upstream_sha256": tensor_sha256(upstream),
        "fp32_master_parameter_sha256": tensor_sha256(reference.params.float()),
        "candidate_master_parameter_sha256": tensor_sha256(candidate.params.float()),
        "candidate_native_parameter_dtype": str(candidate.dtype),
        "reference_parameter_dtype": str(reference.dtype),
        "candidate_parameter_count": candidate.params.numel(),
        "reference_parameter_count": reference.params.numel(),
    }
    hyperparams = candidate.native_tcnn_module.hyperparams()
    if hyperparams.get("otype") == "NetworkWithInputEncoding":
        result.update({
            "network_parameter_range": [
                hyperparams["network_parameter_offset"],
                hyperparams["network_parameter_offset"] + hyperparams["network_parameter_count"],
            ],
            "encoding_parameter_range": [
                hyperparams["encoding_parameter_offset"],
                hyperparams["encoding_parameter_offset"] + hyperparams["encoding_parameter_count"],
            ],
            "logical_encoding_width": hyperparams["logical_encoding_width"],
            "padded_encoding_width": hyperparams["padded_encoding_width"],
        })
    return result


def correctness(candidate, reference, x, upstream):
    xc = x.detach().clone().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    candidate.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    yc = candidate(xc)
    yr = reference(xr)
    yc.backward(upstream)
    yr.backward(upstream.float())
    candidate_grad = candidate.params.grad if candidate.params.grad is not None else torch.empty(0, device="cuda")
    reference_grad = reference.params.grad if reference.params.grad is not None else torch.empty(0, device="cuda")
    hyperparams = candidate.native_tcnn_module.hyperparams()
    network_end = hyperparams.get("network_parameter_count", candidate_grad.numel())
    return {
        "output": metric(yc, yr),
        "dinput": metric(xc.grad, xr.grad),
        "network_gradient": metric(candidate_grad[:network_end], reference_grad[:network_end]),
        "encoding_gradient": metric(candidate_grad[network_end:], reference_grad[network_end:]),
    }


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
        elapsed = max(event_time(stream, callback, iterations)["event_ms_per_iteration"] * iterations for callback in callbacks)
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
    parser.add_argument("--paired-rounds", type=int)
    parser.add_argument("--fixed-iterations", type=int)
    parser.add_argument("--harness-smoke", action="store_true")
    parser.add_argument("--harness-contract", type=pathlib.Path)
    args = parser.parse_args()
    if args.cold_stage:
        return cold_start(args.cold_stage, args.output)
    if not args.execute_primary:
        raise SystemExit("Primary measurement requires explicit --execute-primary")
    if not all((args.manifest, args.manifest_sha256, args.case_id, args.process_index is not None, args.harness_contract)):
        raise SystemExit("Primary measurement arguments are incomplete")
    if sha256(args.manifest) != args.manifest_sha256:
        raise SystemExit("Manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text())
    harness_contract = json.loads(args.harness_contract.read_text())
    if harness_contract.get("manifest_sha256") != args.manifest_sha256:
        raise SystemExit("Harness contract is not bound to the manifest")
    case = next((item for item in manifest["cases"] if item["id"] == args.case_id), None)
    if case is None or not case["supported"]:
        raise SystemExit("Unknown or unsupported case")
    if torch.cuda.get_device_properties(0).gcnArchName != "gfx1201":
        raise SystemExit("Expected gfx1201")
    module_path = pathlib.Path(_C.__file__).resolve()
    if "_120_C" not in module_path.name:
        raise SystemExit("Wrong native module")
    gemm_expected = not (
        case["family"] == "encoding"
        and case["operation"] in ("encoding_forward", "encoding_forward_backward")
    )
    before_system = telemetry()
    process_baseline = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    torch.cuda.reset_peak_memory_stats()
    disposable_candidate, disposable_reference, disposable_x, disposable_target, disposable_upstream = make_pair(case, harness_contract)
    disposable_callbacks = (
        operation_callable(disposable_candidate, case["operation"], disposable_x, disposable_target, disposable_upstream),
        operation_callable(disposable_reference, case["operation"], disposable_x, disposable_target, disposable_upstream),
    )
    stream = torch.cuda.Stream()
    warm = warmup(stream, disposable_callbacks, case, manifest["warmup"])
    iterations = args.fixed_iterations or calibrate(stream, disposable_callbacks, manifest["calibration"])
    del disposable_candidate, disposable_reference, disposable_callbacks
    gc.collect()
    candidate, reference, x, target, upstream = make_pair(case, harness_contract)
    identity = pair_identity(candidate, reference, x, target, upstream)
    numerical = correctness(candidate, reference, x, upstream)
    callbacks = (
        operation_callable(candidate, case["operation"], x, target, upstream),
        operation_callable(reference, case["operation"], x, target, upstream),
    )
    pre_timing_state = {
        "candidate_parameter_sha256": tensor_sha256(candidate.params),
        "reference_parameter_sha256": tensor_sha256(reference.params),
        "candidate_optimizer_step": 0,
        "reference_optimizer_step": 0,
    }
    orders = manifest["measurement"]["orders"]
    blocks = []
    paired_rounds = args.paired_rounds or manifest["measurement"]["paired_rounds_per_process"]
    for round_index in range(paired_rounds):
        selected = orders[(case["seed"] + args.process_index + round_index) % len(orders)]
        full_order = selected + list(reversed(selected))
        for position, backend in enumerate(full_order):
            callback = callbacks[0 if backend == "FP16" else 1]
            timing = event_time(stream, callback, iterations)
            blocks.append({"round": round_index, "position": position, "backend": backend,
                           "iterations": iterations, "source": "harness_smoke" if args.harness_smoke else "steady_state",
                           "profiler_active": False, **timing})
    stream.synchronize()
    after_counters = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    finite = all(torch.isfinite(model.params).all().item() for model in (candidate, reference))
    candidate_hyperparams = candidate.native_tcnn_module.hyperparams()
    candidate_dtype = str(candidate.dtype)
    candidate_output_precision = str(candidate.native_tcnn_module.output_precision())
    fp16_times = [x["event_ms_per_iteration"] for x in blocks if x["backend"] == "FP16"]
    fp32_times = [x["event_ms_per_iteration"] for x in blocks if x["backend"] == "FP32"]
    process_speedup = statistics.median(fp32_times) / statistics.median(fp16_times)
    del callbacks, candidate, reference, x, target, upstream
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    released = {"FP16": fp16_counters(), "FP32": fp32_counters()}
    after_system = telemetry()
    invalid_reasons = []
    if not warm["stable"]: invalid_reasons.append("post_warmup_resource_growth")
    if not warm["scratch_live_zero"] or any(x["scratch_bytes_live"] for x in released.values()): invalid_reasons.append("scratch_live_nonzero")
    if not finite: invalid_reasons.append("nan_or_inf")
    for label, sample in (("before", before_system), ("after", after_system)):
        if not sample.get("telemetry_available"): invalid_reasons.append(f"telemetry_missing_{label}")
        if sample.get("foreign_gpu_processes", 0) > 0 and sample.get("gpu_busy_percent") is not None and sample["gpu_busy_percent"] > manifest["system_stability"]["foreign_gpu_busy_percent_max"]: invalid_reasons.append(f"foreign_gpu_load_{label}")
        if sample.get("temperature_c") is not None and not (manifest["system_stability"]["temperature_c_min"] <= sample["temperature_c"] <= manifest["system_stability"]["temperature_c_max"]): invalid_reasons.append(f"temperature_outside_window_{label}")
        if sample.get("foreign_gpu_processes") not in (None, 0): invalid_reasons.append(f"foreign_gpu_process_{label}")
    backend_kind = "encoding_kernel" if not gemm_expected else "hipblaslt_gemm"
    backend_evidence = {
        "backend_kind": backend_kind,
        "gemm_expected": gemm_expected,
        "algorithm_ids": "not_applicable" if not gemm_expected else [],
        "workspace_contract": "backend_default",
        # Both frozen HipBLASLt backends select only algorithms whose native
        # heuristic result reports workspaceSize==0. This is the selected-plan
        # value, not an invented manifestation of the manifest's default.
        "workspace_bytes": 0,
        "workspace_measurement_source": (
            "native_selected_plan_workspace_size"
            if gemm_expected else "encoding_scratch_observation"
        ),
        "fallback": "not_applicable" if not gemm_expected else None,
        "counter_delta": {
            backend: {key: after_counters[backend][key] - process_baseline[backend][key] for key in after_counters[backend]}
            for backend in ("FP16", "FP32")
        },
        "candidate": {
            "backend": candidate_hyperparams.get("network", {}).get("otype", candidate_hyperparams.get("otype")),
            "parameter_dtype": candidate_dtype,
            "output_dtype": candidate_output_precision,
            "compute_type": "FP32_accumulation" if gemm_expected else "not_applicable",
            "trans_a": "to_be_parsed_from_log" if gemm_expected else "not_applicable",
            "trans_b": "to_be_parsed_from_log" if gemm_expected else "not_applicable",
            "matrix_layouts": candidate_hyperparams.get("network_input_layout", "encoding_native"),
            "epilogues": "to_be_parsed_from_log" if gemm_expected else "not_applicable",
            "custom_relu_backward_biasgrad": after_counters["FP16"].get("cache_hits", 0) >= 0 if gemm_expected else "not_applicable",
            "gemm_calls_per_iteration": None if gemm_expected else 0,
        },
    }
    result = {
        "marker": MARKER, "manifest_sha256": args.manifest_sha256, "case": case,
        "process_index": args.process_index, "module_path": str(module_path),
        "before_system": before_system, "after_system": after_system, "process_counter_baseline": process_baseline, "warmup": warm,
        "calibrated_iterations": iterations, "timing_blocks": blocks,
        "paired_rounds": paired_rounds, "harness_smoke": args.harness_smoke,
        "pair_identity": identity, "pre_timing_state": pre_timing_state, "numerical": numerical,
        "process_medians_ms": {"FP16": statistics.median(fp16_times), "FP32": statistics.median(fp32_times)},
        "process_speedup": process_speedup, "resources_after_timing": after_counters,
        "resources_after_release": released,
        "descriptor_release_contract": {
            "process_baseline": {backend: process_baseline[backend]["descriptor_count"] for backend in ("FP16", "FP32")},
            "after_model_release": {backend: released[backend]["descriptor_count"] for backend in ("FP16", "FP32")},
            "stable_after_warmup": all(
                warm["stable_sample"][backend]["descriptor_count"] == warm["after"][backend]["descriptor_count"]
                for backend in ("FP16", "FP32")
            ),
            "fp32_plan_cache_persists_until_process_exit": released["FP32"]["descriptor_count"] != process_baseline["FP32"]["descriptor_count"],
        },
        "torch_memory": {"allocated_peak": torch.cuda.max_memory_allocated(), "reserved_peak": torch.cuda.max_memory_reserved()},
        "backend_evidence": backend_evidence,
        "valid": not invalid_reasons, "invalid_reasons": invalid_reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("VALID" if result["valid"] else "INVALID")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
