#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1B_FP16_FORWARD_001: gate production work on FP16 epilogues."""
import collections
import datetime
import hashlib
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/probe_phase3b1b_fp16_epilogues.cpp"
REPORT = ROOT / "phase3b1_reports/phase3b1b_epilogue_checkpoint.json"
BUILD_LOG = ROOT / "phase3b1_reports/phase3b1b_epilogue_build.log"
RUN_LOG = ROOT / "phase3b1_reports/phase3b1b_epilogue_run.log"
MARKER = "TCNN_RDNA4_P3B1B_FP16_FORWARD_001"
BASE = "22364010853d872702bdf8f63cad26b890b6f47b"
WIDTHS = (16, 32, 64, 128)

def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, **kwargs)

def identity(item):
    a = item.get("algorithm", {})
    return {k: a.get(k) for k in ("index", "solution_name", "kernel_name", "workspace_bytes")}

def classify(item):
    launches = item.get("launches", [])
    memory_ok = len(launches) == 6 and all(
        x.get("matmul_status") == 0 and x.get("synchronize_status") == 0
        and x.get("output_memory", {}).get("prefix_guard_intact")
        and x.get("output_memory", {}).get("suffix_guard_intact")
        and not x.get("output_memory", {}).get("payload_unchanged", True)
        for x in launches)
    ws = item.get("workspace_memory", {})
    memory_ok &= bool(ws.get("prefix_guard_intact") and ws.get("suffix_guard_intact"))
    steady = item.get("steady_state", {})
    steady_ok = all(steady.get(k) == 0 for k in
                    ("new_handles", "new_heuristic_queries", "new_descriptors", "workspace_growth_bytes"))
    num = item.get("numerics", {})
    target = item.get("target_coverage", {})
    semantic_ok = (
        num.get("max_abs", float("inf")) <= 1e-5
        and num.get("nan_count") == 0 and num.get("inf_count") == 0
        and num.get("relu_mask_mismatches") == 0
        and target.get("near_zero_count", 0) > 0
        and target.get("exact_zero_count", 0) > 0
        and target.get("bias_sign_flip_count", 0) > 0
        and target.get("large_finite_bias") is True)
    return item.get("process_status") == "PASS" and item.get("probe_state") == "EXECUTED" and memory_ok and steady_ok and semantic_ok

def main():
    branch = run(["git", "-C", str(ROOT), "branch", "--show-current"]).stdout.strip()
    head = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
    if branch != "phase3b1b-fp16-forward-baseline" or head != BASE:
        raise SystemExit(f"precondition failed: branch={branch} head={head}")
    REPORT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tcnn_p3b1b_epi_") as temp:
        binary = pathlib.Path(temp) / "probe_phase3b1b"
        command = ["/opt/rocm/bin/hipcc", "-std=c++17", "-O2", "--offload-arch=gfx1201",
                   "-I/opt/rocm/include", f"-I{ROOT/'dependencies'}", str(SOURCE),
                   "-L/opt/rocm/lib", "-lhipblaslt", "-lamdhip64", "-o", str(binary)]
        built = run(command)
        BUILD_LOG.write_text(f"MARKER={MARKER}\nCOMMAND={json.dumps(command)}\n{built.stdout}{built.stderr}")
        if built.returncode:
            raise SystemExit(f"probe compilation failed: {built.returncode}")
        cases, raw_log = [], []
        for m in WIDTHS:
            for k in WIDTHS:
                for epilogue in ("BIAS", "RELU_BIAS"):
                    for dtype in ("f16", "f32"):
                        fresh = []
                        for repeat in (1, 2):
                            cmd = [str(binary), str(m), "16", str(k), epilogue, dtype, str(repeat)]
                            completed = run(cmd, cwd=temp)
                            raw_log.append({"command": cmd, "returncode": completed.returncode,
                                            "stdout": completed.stdout, "stderr": completed.stderr})
                            try: data = json.loads(completed.stdout.strip().splitlines()[-1])
                            except Exception as exc: data = {"process_status": "ERROR", "error": str(exc)}
                            data["process_returncode"] = completed.returncode
                            data["process_stderr"] = completed.stderr
                            fresh.append(data)
                        passed = classify(fresh[0]) and classify(fresh[1]) and identity(fresh[0]) == identity(fresh[1])
                        cases.append({"m": m, "n": 16, "k": k, "epilogue": epilogue,
                                      "d_type": dtype, "passed": passed,
                                      "fresh_algorithm_stable": identity(fresh[0]) == identity(fresh[1]),
                                      "fresh_process_runs": fresh})
        RUN_LOG.write_text(json.dumps({"marker": MARKER, "invocations": raw_log}, indent=2) + "\n")
    failures = [c for c in cases if not c["passed"]]
    counts = collections.Counter((c["epilogue"], c["d_type"], c["passed"]) for c in cases)
    doc = {
        "schema": 1, "marker": MARKER, "phase": "3B1-B-stage1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": {"branch": branch, "starting_commit": head},
        "probe_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "matrix": {"direction": "NN", "m_widths": WIDTHS, "k_widths": WIDTHS,
                   "n": 16, "d_types": ["HIP_R_16F", "HIP_R_32F"],
                   "bias_type": "HIP_R_16F", "compute_type": "HIPBLAS_COMPUTE_32F",
                   "fresh_processes_per_signature": 2},
        "summary": {"signatures": len(cases), "fresh_processes": 2*len(cases),
                    "passed": len(cases)-len(failures), "failed": len(failures),
                    "counts": {str(k): v for k, v in counts.items()}},
        "decision": "EPILOGUE_CHECKPOINT_PASS" if not failures and len(cases) == 64 else "PHASE3B1B_BLOCKED",
        "cases": cases,
    }
    REPORT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(doc["decision"]); print(REPORT)
    return 0 if doc["decision"] == "EPILOGUE_CHECKPOINT_PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
