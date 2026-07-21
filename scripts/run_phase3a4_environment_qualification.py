#!/usr/bin/env python3
"""Predeclared quiet-TTY environment qualification using the unchanged v4 child."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys

from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
AMD_SMI = "/opt/rocm/bin/amd-smi"
DISPLAY_MANAGER = "lightdm.service"
EXPECTED = {
    "scripts/benchmark_phase3a4_conditioned_metric_v4.py": "dde22b68e643f77080977bcca28d547df2eee0cd3595e6ab8dfbffa658817250",
    "scripts/phase3a4_v4_native_window.patch": "77172047e889b0d56bfabda3e475684d2e8bb2883552a699ea9d4bffe974acdd",
    "phase3a4_reports/conditioned_bindings_v4.json": "a6e07f76a222ed1de2cbaac48e5d42a87e489b168b7e2368e9ab9acf8de2c838",
}
FORBIDDEN_NAMES = {"amd-smi", "rocm-smi", "nvtop", "radeontop", "watch", "firefox", "chrome",
    "chromium", "brave", "vlc", "mpv", "totem", "celluloid", "obs", "obs-studio", "lightdm",
    "Xorg", "Xwayland", "gnome-shell", "kwin_wayland", "kwin_x11", "plasmashell", "picom", "compton"}


def capture(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def processes():
    found = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = pathlib.Path(cmdline.split()[0]).name if cmdline else comm
        lowered = {comm.lower(), executable.lower()}
        forbidden_lower = {name.lower() for name in FORBIDDEN_NAMES}
        if any(name in forbidden_lower or any(name.startswith(prefix) for prefix in ("firefox", "chrome", "chromium", "brave", "vlc", "mpv")) for name in lowered):
            found.append({"pid": int(entry.name), "comm": comm, "cmdline": cmdline})
    return sorted(found, key=lambda item: item["pid"])


def flatten(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, path + (str(index),))
    else:
        yield path, value


def gpu_processes(document):
    found = []
    for path, value in flatten(document):
        key = path[-1].upper() if path else ""
        if key in {"PID", "PROCESS_ID"}:
            try:
                pid = int(value)
            except (TypeError, ValueError):
                continue
            if pid > 0:
                found.append({"pid": pid, "path": list(path)})
    return found


def log_summary(pairs):
    ratios = [pair["ratio"] for pair in pairs]
    logs = [math.log(value) for value in ratios]
    n = len(logs)
    mean = statistics.mean(logs)
    sd = statistics.stdev(logs) if n > 1 else None
    if n > 1:
        se = sd / math.sqrt(n)
        critical = float(stats.t.ppf(0.975, n - 1))
        interval = [mean - critical * se, mean + critical * se]
        ratio_interval = [math.exp(value) for value in interval]
    else:
        se = critical = None
        interval = ratio_interval = None
    return {"count": n, "ratios": ratios, "geometric_mean": math.exp(mean), "log_mean": mean,
        "log_sample_standard_deviation": sd, "log_standard_error": se, "t_critical": critical,
        "log_confidence_interval_95": interval, "ratio_confidence_interval_95": ratio_interval}


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, raw_root = pathlib.Path(args.output), pathlib.Path(args.output_dir)
    preflight = {"tty": os.isatty(0), "tty_name": os.ttyname(0) if os.isatty(0) else None,
        "display": os.environ.get("DISPLAY"), "wayland_display": os.environ.get("WAYLAND_DISPLAY")}
    status = "INVALID_ENVIRONMENT"
    pairs = []
    try:
        if output.exists() or raw_root.exists():
            raise RuntimeError("qualification outputs already exist")
        if not preflight["tty"] or preflight["display"] or preflight["wayland_display"]:
            raise RuntimeError("a Linux TTY without display variables is required")
        dm = capture(["systemctl", "is-active", DISPLAY_MANAGER])
        preflight["display_manager"] = dm
        if dm["stdout"].strip() != "inactive":
            raise RuntimeError("lightdm.service is not inactive")
        forbidden = processes()
        preflight["forbidden_processes"] = forbidden
        if forbidden:
            raise RuntimeError("desktop, media, browser, monitor, or display processes remain")
        git_status = capture(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
            "status", "--porcelain=v1", "--untracked-files=all"])
        preflight["git_status"] = git_status
        if git_status["returncode"] or git_status["stdout"]:
            raise RuntimeError("worktree is not clean")
        hashes = {relative: sha256(ROOT / relative) for relative in EXPECTED}
        preflight["harness_sha256"] = hashes
        if hashes != EXPECTED:
            raise RuntimeError("v4 harness or binding manifest hash mismatch")
        manifest = json.loads(pathlib.Path(args.manifest).read_text())
        binding_checks = {}
        for variant in ("phase3a3", "phase3a4"):
            libraries = list((pathlib.Path(manifest["bindings"][variant]) / "tinycudann_bindings").glob("_120_C*.so"))
            if len(libraries) != 1:
                raise RuntimeError(f"unexpected {variant} binding layout")
            digest = sha256(libraries[0])
            binding_checks[variant] = {"path": str(libraries[0]), "sha256": digest,
                "expected_sha256": manifest["binding_sha256"][variant], "pass": digest == manifest["binding_sha256"][variant]}
        preflight["bindings"] = binding_checks
        if not all(item["pass"] for item in binding_checks.values()):
            raise RuntimeError("binding identity mismatch")
        amd_process = capture([AMD_SMI, "process", "--general", "--gpu", "0", "--json"])
        preflight["amd_smi_process"] = amd_process
        if amd_process["returncode"]:
            raise RuntimeError("AMD-SMI process query failed")
        foreign_gpu = gpu_processes(json.loads(amd_process["stdout"]))
        preflight["foreign_gpu_compute_processes"] = foreign_gpu
        if foreign_gpu:
            raise RuntimeError("foreign GPU compute processes are active")

        bindings = manifest["bindings"]
        child = ROOT / "scripts/benchmark_phase3a4_conditioned_metric_v4.py"
        for number in range(1, 11):
            order = ("phase3a3", "phase3a4") if number % 2 else ("phase3a4", "phase3a3")
            runs = {}
            for variant in order:
                raw = raw_root / f"pair_{number:02d}_large_1024_w128_relu_forward_{variant}.json"
                completed = subprocess.run([sys.executable, str(child), "--bindings", bindings[variant],
                    "--variant", variant, "--output", str(raw)], text=True, capture_output=True)
                run = json.loads(raw.read_text()) if raw.exists() else {"status": "INFRASTRUCTURE_FAIL", "valid": False}
                run.update({"process_returncode": completed.returncode, "process_stdout": completed.stdout,
                    "process_stderr": completed.stderr})
                runs[variant] = run
            valid = all(run.get("valid", False) for run in runs.values())
            values = {}
            for variant in ("phase3a3", "phase3a4"):
                stationary = runs[variant].get("measurement", {}).get("stationarity")
                values[variant] = stationary.get("overall_median_per_operation_ms") if stationary else None
            pairs.append({"pair": number, "order": list(order), "runs": runs, "valid": valid,
                "phase3a3_per_operation_ms": values["phase3a3"], "phase3a4_per_operation_ms": values["phase3a4"],
                "ratio": values["phase3a3"] / values["phase3a4"] if valid else None})

        all_runs = [pair["runs"][variant] for pair in pairs for variant in ("phase3a3", "phase3a4")]
        valid_pairs = [pair for pair in pairs if pair["valid"]]
        phase = {variant: {"valid_processes": sum(pair["runs"][variant].get("valid", False) for pair in pairs),
            "stationary_processes": sum(pair["runs"][variant].get("measurement", {}).get("stationarity_pass", False) for pair in pairs)}
            for variant in ("phase3a3", "phase3a4")}
        counts = {"valid_processes": sum(run.get("valid", False) for run in all_runs),
            "valid_pairs": len(valid_pairs), "warmups_converged": sum(run.get("steady_state_warmup", {}).get("convergence", {}).get("converged", False) for run in all_runs),
            "stationary_processes": sum(run.get("measurement", {}).get("stationarity_pass", False) for run in all_runs),
            "headroom_passes": sum(run.get("queue_headroom", {}).get("pass", False) for run in all_runs),
            "invariant_passes": sum(run.get("measurement", {}).get("invariants_pass", False) for run in all_runs)}
        qualification_gates = {"valid_processes_at_least_19_of_20": counts["valid_processes"] >= 19,
            "valid_pairs_at_least_9_of_10": counts["valid_pairs"] >= 9,
            "phase3a3_stationary_at_least_9_of_10": phase["phase3a3"]["stationary_processes"] >= 9,
            "phase3a4_stationary_at_least_9_of_10": phase["phase3a4"]["stationary_processes"] >= 9,
            "all_warmups_converged": counts["warmups_converged"] == 20,
            "all_headroom_pass": counts["headroom_passes"] == 20,
            "all_invariants_pass": counts["invariant_passes"] == 20}
        status = "ENVIRONMENT_QUALIFICATION_PASS" if all(qualification_gates.values()) else "ENVIRONMENT_QUALIFICATION_FAIL"
        statistics_doc = log_summary(valid_pairs) if valid_pairs else None
        order = {}
        for label, first in (("A3->A4", "phase3a3"), ("A4->A3", "phase3a4")):
            subset = [pair for pair in valid_pairs if pair["order"][0] == first]
            order[label] = log_summary(subset) if subset else None
        document = {"schema": 1, "protocol": "phase3a4_environment_qualification",
            "status": status, "qualification_only": True, "phase3a4_pass_authorized": False,
            "declared_pairs": 10, "completed_pairs": len(pairs), "preflight": preflight,
            "pairs": pairs, "counts": counts, "phase_counts": phase, "qualification_gates": qualification_gates,
            "valid_pair_statistics": statistics_doc, "order_description": order,
            "performance_gate_evaluated": False, "pass_tag_authorized": False}
    except Exception as error:
        preflight["error"] = f"{type(error).__name__}: {error}"
        document = {"schema": 1, "protocol": "phase3a4_environment_qualification",
            "status": "INVALID_ENVIRONMENT", "qualification_only": True,
            "phase3a4_pass_authorized": False, "preflight": preflight, "pairs": pairs,
            "performance_gate_evaluated": False, "pass_tag_authorized": False}
        status = "INVALID_ENVIRONMENT"
    write(output, document)
    print(status)
    return 0 if status == "ENVIRONMENT_QUALIFICATION_PASS" else (3 if status == "INVALID_ENVIRONMENT" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
