#!/usr/bin/env python3
"""Protocol v5: deterministic-clock controller around the unchanged v4 child."""
import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

AMD_SMI = "/opt/rocm/bin/amd-smi"
VALID_PERF_LEVELS = {"AUTO", "LOW", "HIGH", "MANUAL", "STABLE_STD", "STABLE_PEAK",
    "STABLE_MIN_MCLK", "STABLE_MIN_SCLK", "DETERMINISM"}


def command(arguments):
    completed = subprocess.run(arguments, text=True, capture_output=True)
    return {"command": arguments, "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr}


def parse_json(capture):
    if capture["returncode"]:
        raise RuntimeError("command failed")
    return json.loads(capture["stdout"])


def flatten(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, path + (str(index),))
    else:
        yield path, value


def perf_level(document):
    for path, value in flatten(document):
        if "PERF_LEVEL" in "_".join(path).upper():
            normalized = str(value).upper().strip()
            if normalized in VALID_PERF_LEVELS:
                return normalized
    raise RuntimeError("PERF_LEVEL unavailable")


def supported_gfx_clocks(document):
    values = set()
    for path, value in flatten(document):
        joined = "_".join(path).upper()
        if not any(token in joined for token in ("GFX", "SCLK", "SYS")):
            continue
        for match in re.findall(r"(?:^|\D)(\d{2,5})(?:\.\d+)?\s*MHZ", str(value).upper()):
            values.add(int(match))
        if isinstance(value, (int, float)) and "CLOCK" in joined:
            if 100 <= value <= 10000:
                values.add(int(value))
            elif 100_000_000 <= value <= 10_000_000_000:
                values.add(int(round(value / 1_000_000)))
    return sorted(values)


def overdrive_is_zero(document):
    found = []
    for path, value in flatten(document):
        if "OVERDRIVE" in "_".join(path).upper():
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if match:
                found.append(float(match.group()))
    return bool(found) and all(value == 0 for value in found)


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, root = pathlib.Path(args.output), pathlib.Path(args.output_dir)
    if output.exists() or root.exists() or os.geteuid() != 0:
        print("PHASE3A4_WIDTH128_FORWARD_V5=INVALID_ENVIRONMENT")
        return 3
    environment = {"gpu": 0, "captures": {}, "original_perf_level": None,
        "selected_gfxclk_mhz": None, "determinism_verified": False, "restore": None}
    pairs = []
    status = "INVALID_ENVIRONMENT"
    try:
        queries = {
            "asic": [AMD_SMI, "static", "--asic", "--gpu", "0", "--json"],
            "clocks": [AMD_SMI, "static", "--clock", "ALL", "--gpu", "0", "--json"],
            "overdrive": [AMD_SMI, "metric", "--overdrive", "--gpu", "0", "--json"],
            "perf_before": [AMD_SMI, "metric", "--perf-level", "--gpu", "0", "--json"],
        }
        documents = {}
        for name, arguments in queries.items():
            environment["captures"][name] = command(arguments)
            documents[name] = parse_json(environment["captures"][name])
        asic_text = json.dumps(documents["asic"]).upper()
        if "R9700" not in asic_text and "R 9700" not in asic_text:
            raise RuntimeError("GPU 0 is not identified as R9700")
        if not overdrive_is_zero(documents["overdrive"]):
            raise RuntimeError("overdrive is nonzero or unavailable")
        original = perf_level(documents["perf_before"])
        if original == "DETERMINISM":
            raise RuntimeError("pre-existing determinism state cannot be restored exactly")
        clocks = supported_gfx_clocks(documents["clocks"])
        if not clocks:
            raise RuntimeError("no driver-advertised GFX clock capability")
        selected = max(clocks)
        environment.update({"original_perf_level": original, "supported_gfxclk_mhz": clocks,
            "selected_gfxclk_mhz": selected,
            "selection_rule": "maximum driver-advertised stock GFX/SCLK DPM value; overdrive required to be zero"})
        environment["captures"]["enable_determinism"] = command(
            [AMD_SMI, "set", "--perf-determinism", str(selected), "--gpu", "0", "--json"])
        parse_json(environment["captures"]["enable_determinism"])
        environment["captures"]["perf_verified"] = command(
            [AMD_SMI, "metric", "--perf-level", "--gpu", "0", "--json"])
        verified = perf_level(parse_json(environment["captures"]["perf_verified"]))
        environment["verified_perf_level"] = verified
        environment["determinism_verified"] = verified == "DETERMINISM"
        if not environment["determinism_verified"]:
            raise RuntimeError("PERF_LEVEL is not DETERMINISM")

        bindings = json.loads(pathlib.Path(args.manifest).read_text())["bindings"]
        child = pathlib.Path(__file__).with_name("benchmark_phase3a4_conditioned_metric_v4.py")
        for number in range(1, 21):
            order = ("phase3a3", "phase3a4") if number % 2 else ("phase3a4", "phase3a3")
            runs = {}
            for variant in order:
                raw = root / f"pair_{number:02d}_large_1024_w128_relu_forward_{variant}.json"
                completed = subprocess.run([sys.executable, str(child), "--bindings", bindings[variant],
                    "--variant", variant, "--output", str(raw)], text=True, capture_output=True)
                run = json.loads(raw.read_text()) if raw.exists() else {"status": "INFRASTRUCTURE_FAIL", "valid": False}
                run.update({"process_returncode": completed.returncode, "process_stdout": completed.stdout,
                    "process_stderr": completed.stderr})
                runs[variant] = run
            valid = all(run.get("valid", False) for run in runs.values())
            values = {}
            for variant in ("phase3a3", "phase3a4"):
                stationarity = runs[variant].get("measurement", {}).get("stationarity")
                values[variant] = stationarity.get("overall_median_per_operation_ms") if stationarity else None
            pairs.append({"pair": number, "order": list(order), "runs": runs, "valid": valid,
                "phase3a3_per_operation_ms": values["phase3a3"], "phase3a4_per_operation_ms": values["phase3a4"],
                "ratio": values["phase3a3"] / values["phase3a4"] if all(v is not None for v in values.values()) else None})
        valid_pairs = [pair for pair in pairs if pair["valid"]]
        if len(valid_pairs) >= 16:
            base = [pair["phase3a3_per_operation_ms"] for pair in valid_pairs]
            candidate = [pair["phase3a4_per_operation_ms"] for pair in valid_pairs]
            ratio = statistics.median(base) / statistics.median(candidate)
            status = "PASS" if ratio >= 0.99 else "PERFORMANCE_FAIL"
        else:
            ratio = None
            status = "INFRASTRUCTURE_FAIL"
    except Exception as error:
        environment["environment_error"] = f"{type(error).__name__}: {error}"
        ratio = None
        status = "INVALID_ENVIRONMENT"
    finally:
        original = environment.get("original_perf_level")
        if original:
            restore_capture = command([AMD_SMI, "set", "--perf-level", original, "--gpu", "0", "--json"])
            environment["restore"] = restore_capture
            verify_capture = command([AMD_SMI, "metric", "--perf-level", "--gpu", "0", "--json"])
            environment["restore_verification"] = verify_capture
            try:
                restored = perf_level(parse_json(verify_capture))
                environment["restored_perf_level"] = restored
                environment["restore_verified"] = restore_capture["returncode"] == 0 and restored == original
            except Exception:
                environment["restore_verified"] = False
            if not environment["restore_verified"]:
                status = "INVALID_ENVIRONMENT"
                ratio = None
        document = {"schema": 5, "protocol": "conditioning_v5", "status": status,
            "case": "large_1024_w128_relu", "metric": "forward", "declared_pair_count": 20,
            "completed_pair_count": len(pairs), "pairs": pairs, "environment": environment,
            "valid_pair_count": sum(pair["valid"] for pair in pairs), "minimum_valid_pairs": 16,
            "conditioned_ratio": ratio, "gate": {"threshold": 0.99,
                "uses_only_complete_valid_pairs": True,
                "pass": status == "PASS"}, "official_series_started": False,
            "official_series_planning_unlocked": status == "PASS"}
        write(output, document)
    print("PHASE3A4_WIDTH128_FORWARD_V5=" + status)
    return 0 if status == "PASS" else (3 if status == "INVALID_ENVIRONMENT" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
