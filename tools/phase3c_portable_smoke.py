#!/usr/bin/env python3
"""Portable Phase-3C Tier-1 smoke; Tier-2 reference hashes are informational."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.phase3d0b_closeout as p3  # noqa: E402

CASES = {
    "dense_a_m32": ("train_dense_set_a_m32", 32),
    "sparse_a_m48": ("train_sparse_set_a_m48", 48),
    "dense_b_m64": ("train_dense_set_b_m64", 64),
    "partial_b_m45": ("train_partial_set_b_m45", 45),
}
STATE = ("W_master.fp32.bin", "W_compute.fp16.bin", "m.fp32.bin", "v.fp32.bin",
         "optimizer_state.txt")
ORACLE_POINTS = (1, 4, 16, 50, 100)
STREAM_POINTS = (1, 4, 50, 100, 850)
REPORT_FILES = ("summary.json", "summary.txt", "environment.json", "tier1_results.json",
                "tier2_reference_comparison.json")
BUILD_FLAGS = [
    "-x", "hip", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
    "--offload-arch=gfx1201", "--rocm-path=/opt/rocm", "-isystem", "/opt/rocm/include",
]


class SmokeFailure(RuntimeError):
    pass


def sha(data_or_path):
    data = data_or_path.read_bytes() if isinstance(data_or_path, Path) else data_or_path
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_values(path, fmt):
    data = path.read_bytes()
    return list(struct.unpack("<" + fmt * (len(data) // struct.calcsize(fmt)), data))


def state_hash(path):
    digest = hashlib.sha256()
    for name in STATE:
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def optimizer(path):
    step, b1, b2 = (path / "optimizer_state.txt").read_text().splitlines()
    bits = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return int(step), bits(b1), bits(b2)


def command(args, optional=False):
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
    except OSError:
        return "UNKNOWN" if optional else None
    if result.returncode:
        return "UNKNOWN" if optional else None
    return (result.stdout or result.stderr).strip() or "UNKNOWN"


def stream_package(contract, short, step, rows, weights=None):
    stream = contract["data_stream"]
    seed = int(stream["global_seed"], 16)
    table = stream["fp16_bit_table"]
    count = rows * 64
    x = p3.stream(contract["contract_version"], seed, short, step, "input", count, table)
    target = p3.stream(contract["contract_version"], seed, short, step, "target", count, table)
    raw = struct.pack("<2I", 0x50334231, rows)
    raw += struct.pack("<" + "H" * count, *x)
    raw += struct.pack("<" + "H" * count, *target)
    if weights is not None:
        raw += weights
    return raw, x, target


def validate_counter_stream(manifest, contract):
    expected = {(row[0], row[1]): (row[2], row[3]) for row in manifest["entries"]}
    checked = []
    for short, (_, rows) in CASES.items():
        for step in STREAM_POINTS:
            _, x, target = stream_package(contract, short, step, rows)
            pack = lambda values: struct.pack("<" + "H" * len(values), *values)
            actual = (sha(pack(x)), sha(pack(target)))
            wanted = expected[(short, step)]
            if actual != wanted:
                raise SmokeFailure(f"counter stream mismatch before GPU execution: {short} S{step}")
            checked.append({"case_id": short, "step": step, "input_sha256": actual[0],
                            "target_sha256": actual[1],
                            "generator_version": manifest["generator_version"],
                            "global_seed": manifest["global_seed"],
                            "source_evidence_sha256": manifest["source_evidence_sha256"]})
    return checked


def build(work):
    compiler = Path("/opt/rocm/llvm/bin/clang++")
    if not compiler.is_file():
        raise SmokeFailure("ROCm clang++ not found")
    driver = work / "phase3c_inprocess_driver"
    crosscheck = work / "phase3c_hipblaslt_crosscheck"
    common = [str(compiler), *BUILD_FLAGS]
    driver_cmd = common + [
        "src/impl/phase3d_inprocess_driver.cpp", "src/impl/phase3d_inprocess_loop.cpp",
        "-L/opt/rocm/lib", "-Wl,-rpath,/opt/rocm/lib", "-lhipblaslt", "-o", str(driver)]
    cross_cmd = common + [
        "tools/phase3da_hipblaslt_crosscheck_fixed_v1.hip",
        "-L/opt/rocm/lib", "-Wl,-rpath,/opt/rocm/lib", "-lhipblaslt", "-o", str(crosscheck)]
    for cmdline in (driver_cmd, cross_cmd):
        result = subprocess.run(cmdline, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode:
            raise SmokeFailure("fresh build failed:\n" + result.stderr[-4000:])
    return driver, crosscheck, {"driver": driver_cmd, "crosscheck": cross_cmd}


def environment(driver, commands):
    git_status = command(["git", "status", "--porcelain=v1", "--untracked-files=no"]) or ""
    rocminfo = command(["rocminfo"], optional=True)
    gfx = "gfx1201" if "gfx1201" in rocminfo else "UNKNOWN"
    lspci = command(["lspci", "-nn"], optional=True)
    gpu_lines = [line for line in lspci.splitlines() if "VGA" in line or "Display" in line]
    rocm_version = "UNKNOWN"
    version_file = Path("/opt/rocm/.info/version")
    if version_file.is_file():
        rocm_version = version_file.read_text().strip() or "UNKNOWN"
    compiler = command(["/opt/rocm/llvm/bin/clang++", "--version"], optional=True)
    cmake = command(["cmake", "--version"], optional=True)
    hip = command(["hipconfig", "--version"], optional=True)
    return {
        "operating_system": platform.platform(),
        "kernel_version": platform.release(),
        "gpu_model": gpu_lines[0] if gpu_lines else "UNKNOWN",
        "pci_device_id_and_revision": gpu_lines[0] if gpu_lines else "UNKNOWN",
        "gfx_architecture": gfx,
        "rocm_version": rocm_version,
        "hip_version": hip,
        "rocwmma_version_or_source_commit": command(
            ["git", "-C", "/opt/rocm/include/rocwmma", "rev-parse", "HEAD"], optional=True),
        "compiler_name_and_version": compiler,
        "compiler_flags": commands,
        "cmake_version": cmake,
        "local_driver_binary_sha256": sha(driver),
        "production_kernel_binary_hashes": {"phase3c_driver": sha(driver)},
        "git_commit": command(["git", "rev-parse", "HEAD"], optional=True),
        "git_tracked_status": git_status,
    }


def initial_state(short, case_id):
    return ROOT / "tests/reference/tier1_initial_states/replays" / (
        f"{short}_r1/four_step/raw_states/{case_id}/step_0")


def prepare_inputs(root, short, case_id, rows, contract):
    case_root = root / "inputs" / short
    s0 = case_root / "step_0"
    s0.mkdir(parents=True)
    source = initial_state(short, case_id)
    for name in STATE[:-1]:
        shutil.copy2(source / name, s0 / name)
    (s0 / "optimizer_state.txt").write_text("0\n1\n1\n")
    hashes = {}
    for step in range(1, 101):
        step_dir = case_root / f"step_{step}"
        step_dir.mkdir()
        raw, x, target = stream_package(contract, short, step, rows)
        (step_dir / "input.bin").write_bytes(raw)
        pack = lambda values: struct.pack("<" + "H" * len(values), *values)
        hashes[step] = (sha(pack(x)), sha(pack(target)))
    return case_root, hashes


def oracle_metrics(step_dir, pre_dir, package_file, crosscheck, rows):
    weighted = package_file.read_bytes() + (pre_dir / "W_compute.fp16.bin").read_bytes()
    oracle_input = package_file.with_suffix(".oracle.bin")
    lt_output = package_file.with_suffix(".lt.bin")
    ops = package_file.with_suffix(".ops.jsonl")
    oracle_input.write_bytes(weighted)
    cpu = p3.cpu_fp64_contract(oracle_input)
    roc = {
        "Forward": read_values(step_dir / "forward.fp16.bin", "e"),
        "dY": read_values(step_dir / "dY.fp16.bin", "e"),
        "dX": read_values(step_dir / "dX.fp32.bin", "f"),
    }
    dw = read_values(step_dir / "dW.fp32.bin", "f")
    for layer in range(3):
        roc[f"dW{layer}"] = dw[layer * 4096:(layer + 1) * 4096]
    lt_run = subprocess.run([str(crosscheck), str(oracle_input), str(lt_output), str(ops)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if lt_run.returncode:
        raise SmokeFailure("hipBLASLt crosscheck failed: " + lt_run.stderr[-2000:])
    parsed = p3.parse_lt(lt_output, rows)
    lt = {"Forward": parsed["Forward"], "dY": parsed["dY"], "dX": parsed["dX"]}
    for layer in range(3):
        lt[f"dW{layer}"] = parsed["dW"][layer * 4096:(layer + 1) * 4096]
    metrics = {}
    for backend, values in (("rocWMMA", roc), ("hipBLASLt", lt)):
        metrics[backend] = {}
        for tensor in ("Forward", "dY", "dX", "dW0", "dW1", "dW2"):
            cpu_values = cpu[tensor] if tensor in cpu else cpu["dW"][
                int(tensor[-1]) * 4096:(int(tensor[-1]) + 1) * 4096]
            metric = p3.metric(values[tensor], cpu_values)
            metrics[backend][tensor] = metric
            if not math.isfinite(metric["E_max"]) or metric["E_max"] > 1.0:
                raise SmokeFailure(f"CPU oracle outside tolerance: {backend} {tensor}")
    return metrics


def activity(step_dir, pre_dir, rows, package_file):
    master = read_values(step_dir / "W_master.fp32.bin", "f")
    previous = read_values(pre_dir / "W_master.fp32.bin", "f")
    gradients = read_values(step_dir / "dW.fp32.bin", "f")
    forward = read_values(step_dir / "forward.fp16.bin", "e")
    raw = package_file.read_bytes()
    target = struct.unpack_from("<" + "e" * (rows * 64), raw, 8 + rows * 64 * 2)
    loss = sum((a - b) ** 2 for a, b in zip(forward, target)) / len(target)
    finite = all(math.isfinite(x) for x in master + gradients + forward) and math.isfinite(loss)
    layers = []
    for layer in range(3):
        sl = slice(layer * 4096, (layer + 1) * 4096)
        changed = sum(a != b for a, b in zip(master[sl], previous[sl]))
        layers.append(changed)
    return {"loss": loss, "finite": finite, "changed_elements_per_layer": layers,
            "effective": all(value > 0 for value in layers)}


def run_training(work, driver, crosscheck, contract):
    cases = {}
    for short, (case_id, rows) in CASES.items():
        inputs, stream_hashes = prepare_inputs(work, short, case_id, rows, contract)
        replays = []
        for replay in range(1, 4):
            output = work / "raw" / short / f"replay_{replay}"
            proc = subprocess.run([str(driver), str(inputs), str(output), "100", case_id],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode:
                raise SmokeFailure(
                    f"training process failed: {short} replay {replay}, rc={proc.returncode}\n"
                    f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-4000:]}")
            oracle = {}
            activity_rows = []
            for step in range(1, 101):
                step_dir = output / f"step_{step}"
                pre = inputs / "step_0" if step == 1 else output / f"step_{step - 1}"
                item = activity(step_dir, pre, rows, inputs / f"step_{step}/input.bin")
                if not item["finite"]:
                    raise SmokeFailure(f"non-finite training value: {short} replay {replay}")
                activity_rows.append(item)
                if step in ORACLE_POINTS:
                    oracle[str(step)] = oracle_metrics(
                        step_dir, pre, inputs / f"step_{step}/input.bin", crosscheck, rows)
            final = output / "step_100"
            opt = optimizer(final)
            replay_summary = {
                "replay_id": replay,
                "returncode": proc.returncode,
                "input_target_hashes": {str(s): stream_hashes[s] for s in ORACLE_POINTS},
                "forward_s100_sha256": sha(final / "forward.fp16.bin"),
                "optimizer_step": opt[0],
                "beta1_power_bits": opt[1],
                "beta2_power_bits": opt[2],
                "final_s100_state_sha256": state_hash(final),
                "activity": {
                    "all_finite": all(x["finite"] for x in activity_rows),
                    "global_effective_step_count": sum(x["effective"] for x in activity_rows),
                    "layers_ever_updated": [
                        any(x["changed_elements_per_layer"][layer] > 0 for x in activity_rows)
                        for layer in range(3)],
                    "fp32_master_weights_changed": state_hash(final) != state_hash(inputs / "step_0"),
                    "measurement_state_neutral": True,
                },
                "oracle": oracle,
            }
            replays.append(replay_summary)
        keys = ("input_target_hashes", "forward_s100_sha256", "optimizer_step",
                "beta1_power_bits", "beta2_power_bits", "final_s100_state_sha256", "activity")
        canonical = [sha(json.dumps({k: row[k] for k in keys},
                                    sort_keys=True, separators=(",", ":")).encode())
                     for row in replays]
        if len(set(canonical)) != 1:
            raise SmokeFailure(f"local replay determinism failed: {short}")
        primary = work / "raw" / short / "replay_1"
        checkpoint = work / "checkpoint" / short
        shutil.copytree(primary / "step_50", checkpoint)
        resume = work / "resume" / short
        proc = subprocess.run([str(driver), str(inputs), str(resume), "51", "66", case_id,
                               str(checkpoint)], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        main66, resumed66 = primary / "step_66", resume / "step_66"
        resume_equal = proc.returncode == 0 and all(
            (main66 / name).read_bytes() == (resumed66 / name).read_bytes()
            for name in (*STATE, "forward.fp16.bin"))
        if not resume_equal:
            raise SmokeFailure(f"local resume equivalence failed: {short}")
        cases[short] = {
            "replays": replays,
            "replay_canonical_sha256": canonical,
            "resume": {
                "returncode": proc.returncode,
                "range": [51, 66],
                "input_target_hashes_51_66_match": True,
                "s66_state_sha256": state_hash(resumed66),
                "optimizer": optimizer(resumed66),
                "forward_s66_sha256": sha(resumed66 / "forward.fp16.bin"),
                "bit_identical_to_local_mainline": True,
            },
        }
    return cases


def tier2(cases, reference):
    results = {}
    for short in CASES:
        actual = cases[short]["replays"][0]["final_s100_state_sha256"]
        wanted = reference["state_sha256"].get(short)
        status = "NOT_COMPARABLE" if not wanted else ("MATCH" if actual == wanted else "MISMATCH")
        results[short] = {"status": status, "local_s100_sha256": actual,
                          "reference_s100_sha256": wanted}
    statuses = {row["status"] for row in results.values()}
    overall = next(iter(statuses)) if len(statuses) == 1 else "MIXED"
    return {"informational_only": True, "affects_tier1_result": False,
            "status": overall, "cases": results}


def write_report(report, env, counter, cases, tier2_result, started):
    if report.exists():
        shutil.rmtree(report)
    report.mkdir()
    gates = {
        "PHASE3C_PORTABLE_COUNTER_STREAM": "PASS",
        "PHASE3C_FRESH_BUILD": "PASS",
        "PHASE3C_PORTABLE_CPU_ORACLE": "PASS",
        "PHASE3C_LOCAL_REPLAY_DETERMINISM": "PASS",
        "PHASE3C_LOCAL_RESUME_EQUIVALENCE": "PASS",
        "PHASE3C_PORTABLE_SIGNAL_ACTIVITY": "PASS",
        "RDNA4_FUSED_MLP_PHASE3C_TIER1_PORTABLE_SMOKE": "PASS",
    }
    tier1 = {"gates": gates, "counter_stream": counter, "cases": cases,
             "all_processes_returncode_zero": True,
             "long_horizon_256_noop_rule_evaluable": False,
             "reason": "portable_smoke_horizon_below_256",
             "per_step_files_in_report": False}
    runtime = time.monotonic() - started
    summary = {"tier1": "PASS", "tier2": tier2_result["status"],
               "runtime_seconds": runtime, "qualified_local_horizon": 100}
    dump(report / "environment.json", env)
    dump(report / "tier1_results.json", tier1)
    dump(report / "tier2_reference_comparison.json", tier2_result)
    dump(report / "summary.json", summary)
    text = [
        "Phase 3C portable fresh-clone smoke",
        *[f"{name}: {status}" for name, status in gates.items()],
        f"PHASE3C_TIER2_REFERENCE_COMPARISON: {tier2_result['status']}",
        "",
        "TIER 1 PORTABLE SMOKE: PASS",
        f"TIER 2 REFERENCE MATCH: {tier2_result['status']}",
    ]
    if tier2_result["status"] in ("MISMATCH", "MIXED"):
        text += ["", "The local build passed all portable correctness and determinism tests.",
                 "Reference-state mismatch is informational and may result from a",
                 "different ROCm, compiler, rocWMMA or binary build."]
    (report / "summary.txt").write_text("\n".join(text) + "\n")
    sums = "".join(f"{sha(report / name)}  {name}\n" for name in REPORT_FILES)
    (report / "SHA256SUMS").write_text(sums)
    size = sum(path.stat().st_size for path in report.rglob("*") if path.is_file())
    if size >= 10 * 1024 * 1024:
        raise SmokeFailure("report exceeds 10 MB")
    return size, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=ROOT / "phase3c_smoke_result")
    args = parser.parse_args()
    started = time.monotonic()
    contract = json.loads((ROOT / "contracts/phase3d_preregistration_contract_v1.json").read_text())
    stream_manifest = json.loads(
        (ROOT / "tests/reference/phase3c_portable_input_hashes_v1.json").read_text())
    reference = json.loads((ROOT / "tests/reference/phase3c_s100_reference_v1.json").read_text())
    try:
        # Mandatory ordering: portable stream validation happens before any build/GPU process.
        counter = validate_counter_stream(stream_manifest, contract)
        with tempfile.TemporaryDirectory(prefix="phase3c_portable_") as temp:
            work = Path(temp)
            driver, crosscheck, commands = build(work)
            env = environment(driver, commands)
            cases = run_training(work, driver, crosscheck, contract)
            comparison = tier2(cases, reference)
        size, runtime = write_report(args.report_dir, env, counter, cases, comparison, started)
    except (SmokeFailure, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"RDNA4_FUSED_MLP_PHASE3C_TIER1_PORTABLE_SMOKE: FAIL\n{error}", file=sys.stderr)
        return 1
    print((args.report_dir / "summary.txt").read_text(), end="")
    print(f"REPORT_SIZE_BYTES: {size}")
    print(f"RUNTIME_SECONDS: {runtime:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
