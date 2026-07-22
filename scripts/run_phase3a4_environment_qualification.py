#!/usr/bin/env python3
"""Predeclared quiet-TTY scatter qualification using the unchanged v4 child."""

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
DECLARED_PAIRS = 28
MINIMUM_VALID_PAIRS = 18
HASH_INVENTORY = ROOT / "phase3a4_reports/environment_qualification_preflight_hashes.json"
FORBIDDEN_NAMES = {
    "amd-smi", "rocm-smi", "nvtop", "radeontop", "watch", "firefox", "chrome",
    "chromium", "brave", "vlc", "mpv", "totem", "celluloid", "obs", "obs-studio",
    "Xorg", "Xwayland", "gnome-shell", "kwin_wayland", "kwin_x11", "plasmashell",
    "picom", "compton",
}


def capture(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def processes():
    found = []
    forbidden_lower = {name.lower() for name in FORBIDDEN_NAMES}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = pathlib.Path(cmdline.split()[0]).name if cmdline else comm
        names = {comm.lower(), executable.lower()}
        if any(
            name in forbidden_lower
            or any(name.startswith(prefix) for prefix in ("firefox", "chrome", "chromium", "brave", "vlc", "mpv"))
            for name in names
        ):
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
        if key not in {"PID", "PROCESS_ID"}:
            continue
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
        degrees = n - 1
        se = sd / math.sqrt(n)
        t_critical = float(stats.t.ppf(0.975, degrees))
        mean_interval = [mean - t_critical * se, mean + t_critical * se]
        ratio_interval = [math.exp(value) for value in mean_interval]
        chi2_lower = float(stats.chi2.ppf(0.025, degrees))
        chi2_upper = float(stats.chi2.ppf(0.975, degrees))
        sd_interval = [
            math.sqrt(degrees * sd * sd / chi2_upper),
            math.sqrt(degrees * sd * sd / chi2_lower),
        ]
    else:
        degrees = None
        se = t_critical = None
        mean_interval = ratio_interval = None
        chi2_lower = chi2_upper = None
        sd_interval = None
    return {
        "count": n,
        "ratios_a3_time_over_a4_time": ratios,
        "log_ratios": logs,
        "log_sample_standard_deviation": sd,
        "log_standard_deviation_confidence_interval_95_chi_square": sd_interval,
        "chi_square_degrees_of_freedom": degrees,
        "chi_square_quantiles_0_025_0_975": [chi2_lower, chi2_upper] if n > 1 else None,
        "geometric_mean_side_product": math.exp(mean),
        "log_mean": mean,
        "log_standard_error": se,
        "t_critical": t_critical,
        "log_mean_confidence_interval_95": mean_interval,
        "geometric_mean_confidence_interval_95_side_product": ratio_interval,
    }


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def fmt(value, digits=9):
    return "n/a" if value is None else f"{value:.{digits}g}"


def write_markdown(path, document):
    status = document["status"]
    counts = document.get("counts", {})
    phase = document.get("phase_counts", {})
    summary = document.get("valid_pair_statistics")
    order = document.get("order_description", {})
    lines = [
        "# Phase 3A4 environment qualification",
        "",
        f"**{status}**",
        "",
        "This is a qualification-only scatter estimate. It is not a performance gate, does not",
        "create Protocol v5, and cannot authorize a Phase-3A4 PASS tag.",
        "",
        "## Fixed v4 context (not re-evaluated)",
        "",
        "- geometric mean: `0.997361`",
        "- Student-t 95% interval: `[0.990639, 1.004129]`",
        "- bootstrap 95% interval: `[0.991771, 1.000526]`",
        "- no Phase-3A4 release because n=7 was selected, leave-one-out was fragile, and pair 6 was influential",
        "",
        "## Qualification yield",
        "",
        f"- declared/completed pairs: {document.get('declared_pairs', 28)}/{document.get('completed_pairs', 0)}",
        f"- valid pairs: {counts.get('valid_pairs', 0)} (minimum 18)",
        f"- valid processes: {counts.get('valid_processes', 0)}/56",
        f"- A3 valid/stationary: {phase.get('phase3a3', {}).get('valid_processes', 0)}/{phase.get('phase3a3', {}).get('stationary_processes', 0)}",
        f"- A4 valid/stationary: {phase.get('phase3a4', {}).get('valid_processes', 0)}/{phase.get('phase3a4', {}).get('stationary_processes', 0)}",
        f"- converged warm-ups: {counts.get('warmups_converged', 0)}/56",
        f"- queue-headroom passes: {counts.get('headroom_passes', 0)}/56",
        f"- handle/heuristic/scratch invariant passes: {counts.get('invariant_passes', 0)}/56",
        "",
        "## Target quantity: log-pair-ratio scatter",
        "",
    ]
    if summary:
        sd_ci = summary["log_standard_deviation_confidence_interval_95_chi_square"]
        gm_ci = summary["geometric_mean_confidence_interval_95_side_product"]
        lines.extend([
            f"- valid ratios (A3 time / A4 time): `{json.dumps(summary['ratios_a3_time_over_a4_time'])}`",
            f"- sample standard deviation of log ratios: `{fmt(summary['log_sample_standard_deviation'])}`",
            f"- chi-square 95% interval for log-ratio standard deviation: `[{fmt(sd_ci[0])}, {fmt(sd_ci[1])}]`",
            f"- geometric mean (side product only): `{fmt(summary['geometric_mean_side_product'])}`",
            f"- Student-t 95% interval for geometric mean (side product only): `[{fmt(gm_ci[0])}, {fmt(gm_ci[1])}]`",
        ])
    else:
        lines.append("No valid pair statistics are available.")
    lines.extend([
        "",
        "A standard deviation estimated from roughly 20 observations remains substantially",
        "uncertain: its 95% interval is approximately 0.76 to 1.46 times the point estimate.",
        "This scatter estimate is therefore a planning quantity with an error band, not an exact",
        "constant. It is intended only to dimension a later Protocol v5.",
        "",
        "## Descriptive order effect",
        "",
    ])
    for label in ("A3->A4", "A4->A3"):
        item = order.get(label)
        lines.append(
            f"- {label}: n={item['count']}, geometric mean={fmt(item['geometric_mean_side_product'])}"
            if item else f"- {label}: no valid pairs"
        )
    lines.extend([
        "",
        "The order split is descriptive only.",
        "",
        "## Interpretation constraint",
        "",
        "Protocol v4 ran with a compositor; this qualification runs without one. Lower scatter is",
        "the desired signal. If scatter is unchanged, that does not prove that the environment was",
        "not causal: the apparent null effect may be due to the small sample. It must not be",
        "overinterpreted.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--display-manager", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    report = pathlib.Path(args.report)
    raw_root = pathlib.Path(args.output_dir)
    preflight = {
        "tty": os.isatty(0),
        "tty_name": os.ttyname(0) if os.isatty(0) else None,
        "display": os.environ.get("DISPLAY"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "display_manager_service": args.display_manager,
    }
    pairs = []
    try:
        if output.exists() or report.exists() or raw_root.exists():
            raise RuntimeError("qualification outputs already exist")
        tty_name = preflight["tty_name"] or ""
        if not preflight["tty"] or not tty_name.startswith("/dev/tty"):
            raise RuntimeError("a real Linux /dev/tty is required")
        if preflight["display"] or preflight["wayland_display"]:
            raise RuntimeError("DISPLAY and WAYLAND_DISPLAY must be unset")
        dm = capture(["systemctl", "is-active", args.display_manager])
        preflight["display_manager"] = dm
        if dm["stdout"].strip() != "inactive":
            raise RuntimeError(f"{args.display_manager} is not inactive")
        forbidden = processes()
        preflight["forbidden_processes"] = forbidden
        if forbidden:
            raise RuntimeError("desktop, media, browser, monitor, or display processes remain")
        git_status = capture([
            "git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
            "status", "--porcelain=v1", "--untracked-files=all",
        ])
        preflight["git_status"] = git_status
        if git_status["returncode"] or git_status["stdout"]:
            raise RuntimeError("worktree is not clean")
        expected = json.loads(HASH_INVENTORY.read_text())["sha256"]
        hashes = {relative: sha256(ROOT / relative) for relative in expected}
        preflight["harness_sha256"] = hashes
        if hashes != expected:
            raise RuntimeError("qualification infrastructure hash mismatch")
        manifest = json.loads(pathlib.Path(args.manifest).read_text())
        expected_commits = {
            "phase3a3": "a26a0c1218d7ddeaad174c86a33255189ca5c2cc",
            "phase3a4": "6258184d8d9d032ef423b75eddeeaf8168c7e45a",
        }
        if manifest.get("source_commits") != expected_commits:
            raise RuntimeError("qualification binding source identity mismatch")
        patch_path = pathlib.Path(manifest["test_only_native_patch"])
        if sha256(patch_path) != manifest.get("test_only_native_patch_sha256"):
            raise RuntimeError("qualification patch identity mismatch")
        binding_checks = {}
        for variant in ("phase3a3", "phase3a4"):
            libraries = list((pathlib.Path(manifest["bindings"][variant]) / "tinycudann_bindings").glob("_120_C*.so"))
            if len(libraries) != 1:
                raise RuntimeError(f"unexpected {variant} binding layout")
            digest = sha256(libraries[0])
            binding_checks[variant] = {
                "path": str(libraries[0]),
                "sha256": digest,
                "expected_sha256": manifest["binding_sha256"][variant],
                "pass": digest == manifest["binding_sha256"][variant],
            }
        preflight["bindings"] = binding_checks
        if not all(item["pass"] for item in binding_checks.values()):
            raise RuntimeError("binding identity mismatch")
        libraries = [pathlib.Path(binding_checks[v]["path"]).resolve() for v in ("phase3a3", "phase3a4")]
        if libraries[0] == libraries[1] or libraries[0].stat().st_ino == libraries[1].stat().st_ino:
            raise RuntimeError("A3 and A4 binding paths or inodes are not distinct")
        if binding_checks["phase3a3"]["sha256"] == binding_checks["phase3a4"]["sha256"]:
            raise RuntimeError("A3 and A4 binding SHA-256 values are not distinct")
        amd_process = capture([AMD_SMI, "process", "--general", "--gpu", "0", "--json"])
        preflight["amd_smi_process"] = amd_process
        if amd_process["returncode"]:
            raise RuntimeError("AMD-SMI process query failed")
        foreign_gpu = gpu_processes(json.loads(amd_process["stdout"]))
        preflight["foreign_gpu_compute_processes"] = foreign_gpu
        if foreign_gpu:
            raise RuntimeError("foreign GPU processes are active")

        bindings = manifest["bindings"]
        child = ROOT / "scripts/benchmark_phase3a4_conditioned_metric_v4.py"
        for number in range(1, DECLARED_PAIRS + 1):
            order = ("phase3a3", "phase3a4") if number % 2 else ("phase3a4", "phase3a3")
            runs = {}
            for variant in order:
                raw = raw_root / f"pair_{number:02d}_large_1024_w128_relu_forward_{variant}.json"
                completed = subprocess.run([
                    sys.executable, str(child), "--bindings", bindings[variant],
                    "--variant", variant, "--output", str(raw),
                ], text=True, capture_output=True)
                run = json.loads(raw.read_text()) if raw.exists() else {"status": "INFRASTRUCTURE_FAIL", "valid": False}
                run.update({
                    "process_returncode": completed.returncode,
                    "process_stdout": completed.stdout,
                    "process_stderr": completed.stderr,
                })
                runs[variant] = run
            valid = all(run.get("valid", False) for run in runs.values())
            values = {}
            for variant in ("phase3a3", "phase3a4"):
                stationary = runs[variant].get("measurement", {}).get("stationarity")
                values[variant] = stationary.get("overall_median_per_operation_ms") if stationary else None
            pairs.append({
                "pair": number,
                "order": list(order),
                "runs": runs,
                "valid": valid,
                "phase3a3_per_operation_ms": values["phase3a3"],
                "phase3a4_per_operation_ms": values["phase3a4"],
                "ratio": values["phase3a3"] / values["phase3a4"] if valid else None,
            })

        all_runs = [pair["runs"][variant] for pair in pairs for variant in ("phase3a3", "phase3a4")]
        valid_pairs = [pair for pair in pairs if pair["valid"]]
        phase = {
            variant: {
                "total_processes": DECLARED_PAIRS,
                "valid_processes": sum(pair["runs"][variant].get("valid", False) for pair in pairs),
                "stationary_processes": sum(pair["runs"][variant].get("measurement", {}).get("stationarity_pass", False) for pair in pairs),
            }
            for variant in ("phase3a3", "phase3a4")
        }
        counts = {
            "total_processes": 2 * DECLARED_PAIRS,
            "valid_processes": sum(run.get("valid", False) for run in all_runs),
            "valid_pairs": len(valid_pairs),
            "warmups_converged": sum(run.get("steady_state_warmup", {}).get("convergence", {}).get("converged", False) for run in all_runs),
            "stationary_processes": sum(run.get("measurement", {}).get("stationarity_pass", False) for run in all_runs),
            "headroom_passes": sum(run.get("queue_headroom", {}).get("pass", False) for run in all_runs),
            "invariant_passes": sum(run.get("measurement", {}).get("invariants_pass", False) for run in all_runs),
        }
        gates = {"valid_pairs_at_least_18_of_28": counts["valid_pairs"] >= MINIMUM_VALID_PAIRS}
        status = "ENVIRONMENT_QUALIFICATION_PASS" if all(gates.values()) else "ENVIRONMENT_QUALIFICATION_FAIL"
        statistics_doc = log_summary(valid_pairs) if valid_pairs else None
        order_description = {}
        for label, first in (("A3->A4", "phase3a3"), ("A4->A3", "phase3a4")):
            subset = [pair for pair in valid_pairs if pair["order"][0] == first]
            order_description[label] = log_summary(subset) if subset else None
        document = {
            "schema": 2,
            "protocol": "phase3a4_environment_qualification",
            "status": status,
            "qualification_only": True,
            "phase3a4_pass_authorized": False,
            "protocol_v5_created": False,
            "declared_pairs": DECLARED_PAIRS,
            "completed_pairs": len(pairs),
            "preflight": preflight,
            "pairs": pairs,
            "counts": counts,
            "phase_counts": phase,
            "qualification_gates": gates,
            "valid_pair_statistics": statistics_doc,
            "order_description": order_description,
            "performance_gate_evaluated": False,
            "geometric_mean_is_decision_value": False,
            "pass_tag_authorized": False,
        }
    except Exception as error:
        preflight["error"] = f"{type(error).__name__}: {error}"
        document = {
            "schema": 2,
            "protocol": "phase3a4_environment_qualification",
            "status": "INVALID_ENVIRONMENT",
            "qualification_only": True,
            "phase3a4_pass_authorized": False,
            "protocol_v5_created": False,
            "preflight": preflight,
            "pairs": pairs,
            "performance_gate_evaluated": False,
            "pass_tag_authorized": False,
        }
    write_json(output, document)
    write_markdown(report, document)
    print(document["status"])
    if document["status"] == "ENVIRONMENT_QUALIFICATION_PASS":
        return 0
    return 3 if document["status"] == "INVALID_ENVIRONMENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
