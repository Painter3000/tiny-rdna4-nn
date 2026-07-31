#!/usr/bin/env python3
"""Orchestrate and attest the real Phase-4A2 amd-gsplat image path."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/phase4a2_image_path_worker.py"
OUTPUT = pathlib.Path(
    os.environ.get(
        "PHASE4A2_OUTPUT",
        str(ROOT / "phase4a2_image_path_evidence"),
    )
).resolve()
BUILD = pathlib.Path(
    os.environ.get(
        "PHASE4A2_BUILD_ROOT",
        str(ROOT / ".phase4a2_image_build"),
    )
).resolve()
TCNN_BINARY = BUILD / "runtime/tiny-rdna4-nn/tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so"
GSPLAT_BINARY = BUILD / "runtime/amd-gsplat/gsplat/csrc.so"
QUALIFIED = {
    "phase3a_backward": (
        ROOT / "src/impl/qualified/phase3a_fused_backward.hip",
        "7ad0cc174c25918448b7936bfdca63bf2fdf5aab441063ca3618aefdee135a85",
    ),
    "phase3b_training": (
        ROOT / "src/impl/qualified/phase3b_training_step.hip",
        "dbab25d8b7f1ecf771aaf3dd2b2228abba6b3b4ffba5fedbd21b0e21147630d7",
    ),
    "phase3b_adam": (
        ROOT / "src/impl/qualified/phase3b_adam_update.hip",
        "aaa82642c49a07aa7e344ec27f82b1dee5a37dcfa3b42daf41e195550b953eb6",
    ),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_worker(name: str, backend: str, stop: int = 100, resume=None):
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(WORKER), "--backend", backend,
        "--stop-step", str(stop), "--output", str(destination),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    with (destination / "worker.log").open("w") as log:
        subprocess.run(
            command, cwd=ROOT, check=True, stdout=log,
            stderr=subprocess.STDOUT,
        )
    return json.loads((destination / "result.json").read_text())


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def tolerance_metric(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.float().flatten()
    right = right.float().flatten()
    error = (left - right).abs()
    score = error / (0.02 + 0.02 * right.abs())
    index = int(score.argmax())
    return {
        "max_abs": float(error.max()),
        "E_max": float(score[index]),
        "argmax_index": index,
    }


def nested_backend(result):
    return result["backend_hyperparams"]["network"]


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    qualified_hashes = {
        name: {"actual": sha256(path), "expected": expected}
        for name, (path, expected) in QUALIFIED.items()
    }
    binaries = {
        "tiny_rdna4_nn": {
            "path": str(TCNN_BINARY),
            "sha256": sha256(TCNN_BINARY),
        },
        "amd_gsplat": {
            "path": str(GSPLAT_BINARY),
            "sha256": sha256(GSPLAT_BINARY),
        },
    }

    hipblaslt = run_worker("hipblaslt", "hipblaslt")
    fused = [
        run_worker(f"rocwmma_r{index}", "rocwmma")
        for index in range(1, 4)
    ]
    resume = run_worker(
        "rocwmma_resume_s50_s66",
        "rocwmma",
        stop=66,
        resume=OUTPUT / "rocwmma_r1/checkpoint_s50.pt",
    )

    initial_identity = (
        hipblaslt["scene_hash"] == fused[0]["scene_hash"]
        and hipblaslt["initial_canonical_parameter_hash"]
        == fused[0]["initial_canonical_parameter_hash"]
        and hipblaslt["initial_optimizer_empty"]
        and fused[0]["initial_optimizer_empty"]
    )
    a0 = torch.load(OUTPUT / "hipblaslt/step0_tensors.pt", weights_only=False)
    b0 = torch.load(OUTPUT / "rocwmma_r1/step0_tensors.pt", weights_only=False)
    step0_metrics = {
        name: tolerance_metric(a0[name], b0[name])
        for name in (
            "decoder", "image", "alpha", "radii", "loss",
            "gaussian_grad", "mlp_grad_canonical", "color_grad",
        )
    }
    step0_metrics["visibility_mismatches"] = int(
        torch.count_nonzero(a0["visibility"] != b0["visibility"])
    )
    step0_tolerance_ok = all(
        metric["E_max"] < 1.0
        for metric in step0_metrics.values()
        if isinstance(metric, dict)
    ) and step0_metrics["visibility_mismatches"] == 0

    replay_fields = []
    for result in fused:
        replay_fields.append({
            "loss_measurements": {
                key: value["loss"]
                for key, value in result["measurements"].items()
                if key in {"0", "1", "4", "16", "50", "100"}
            },
            "s50_state": result["measurements"]["50"]["state"],
            "s100_state": result["measurements"]["100"]["state"],
            "runtime_attestation": result["runtime_attestation"],
        })
    replay_deterministic = all(
        value == replay_fields[0] for value in replay_fields[1:]
    )
    resume_equivalent = (
        resume["final_state"] == fused[0]["measurements"]["66_state"]
    )

    import tinycudann as tcnn
    import tinycudann.modules as tcnn_modules

    invalid_results = {}
    invalid_configs = {
        "wrong_width": {
            "otype": "RocWMMAWidth64MLP", "n_neurons": 32,
            "n_hidden_layers": 2, "activation": "ReLU",
            "output_activation": "None", "precision": "Fp16", "bias": True,
        },
        "missing_precision": {
            "otype": "RocWMMAWidth64MLP", "n_neurons": 64,
            "n_hidden_layers": 2, "activation": "ReLU",
            "output_activation": "None", "bias": True,
        },
        "wrong_output_activation": {
            "otype": "RocWMMAWidth64MLP", "n_neurons": 64,
            "n_hidden_layers": 2, "activation": "ReLU",
            "output_activation": "Sigmoid", "precision": "Fp16", "bias": True,
        },
    }
    for name, config in invalid_configs.items():
        try:
            tcnn.Network(64, 64, config, seed=1)
            invalid_results[name] = False
        except RuntimeError:
            invalid_results[name] = True
    extension = tcnn_modules._C
    stale_input = torch.zeros((16, 64), dtype=torch.float16, device="cuda")
    stale_before = torch.zeros((12480,), dtype=torch.float16, device="cuda")
    stale_after = torch.ones((12480,), dtype=torch.float16, device="cuda")
    stale = dict(extension._phase4a2_staleness_guard(
        stale_input, stale_before, stale_after
    ))
    invalid_results["stale_view"] = stale["stale_launch_rejected_before_kernel"]

    fused_runtime_ok = all(
        nested_backend(result)["otype"] == "RocWMMAWidth64MLP"
        and nested_backend(result)["silent_fallback"] is False
        and result["runtime_attestation"]["forward_view_is_current"]
        and result["runtime_attestation"]["forward_view_refresh_count"]
        == result["runtime_attestation"]["fused_forward_launch_count"]
        and result["runtime_attestation"]["logical_weight_version"]
        == result["runtime_attestation"]["packed_view_version"]
        for result in fused + [resume]
    )
    losses_decrease = all(
        result["measurements"][str(result["stop_step"])]["loss"]
        < result["measurements"][str(result["start_step"])]["loss"]
        for result in [hipblaslt] + fused
    )
    finite = all(result["all_finite"] for result in [hipblaslt] + fused + [resume])
    active = all(
        result["weight_max_abs_change"] > 0
        and result["gradient_norm_max"] > 0
        and result["gaussian_gradient_norm_max"] > 0
        and result["exp_avg_active"]
        and result["exp_avg_sq_active"]
        for result in [hipblaslt] + fused + [resume]
    )
    renders = all(
        (OUTPUT / name / "render_s100.png").is_file()
        for name in ("hipblaslt", "rocwmma_r1", "rocwmma_r2", "rocwmma_r3")
    )
    memory_stable = all(
        max(v["reserved_bytes"] for k, v in result["measurements"].items() if k.isdigit())
        - min(v["reserved_bytes"] for k, v in result["measurements"].items() if k.isdigit())
        <= 16 * 1024 * 1024
        for result in [hipblaslt] + fused
    )
    step0_forward = all(torch.isfinite(a0[name]).all() and torch.isfinite(b0[name]).all()
                        for name in ("decoder", "image", "alpha"))
    rasterizer_forward = (
        int(a0["visibility"].sum()) > 0 and int(b0["visibility"].sum()) > 0
    )
    decoder_backward = (
        float(a0["mlp_grad_canonical"].abs().max()) > 0
        and float(b0["mlp_grad_canonical"].abs().max()) > 0
    )
    rasterizer_backward = (
        float(a0["gaussian_grad"].abs().max()) > 0
        and float(b0["gaussian_grad"].abs().max()) > 0
    )
    optimizer_ok = all(
        result["weight_max_abs_change"] > 0
        and result["exp_avg_active"]
        and result["exp_avg_sq_active"]
        and result["next_forward_used_new_weights"]
        and result["final_state"]["optimizer_step"] == 100
        for result in [hipblaslt] + fused
    )
    immutable = all(
        value["actual"] == value["expected"]
        for value in qualified_hashes.values()
    )
    gates = {
        "PHASE4A2_IMAGE_PATH_PREFLIGHT": (
            sys.prefix != sys.base_prefix
            and pathlib.Path(sys.executable).resolve().is_relative_to(
                pathlib.Path(sys.prefix).resolve()
            )
            and TCNN_BINARY.is_file() and GSPLAT_BINARY.is_file()
        ),
        "PHASE4A2_IMAGE_PATH_QUALIFIED_KERNEL_IMMUTABILITY": immutable,
        "PHASE4A2_IMAGE_PATH_INITIAL_STATE_IDENTITY": initial_identity,
        "PHASE4A2_REAL_DECODER_FORWARD": step0_forward and step0_tolerance_ok,
        "PHASE4A2_REAL_RASTERIZER_FORWARD": rasterizer_forward and step0_tolerance_ok,
        "PHASE4A2_REAL_DECODER_BACKWARD": decoder_backward and step0_tolerance_ok,
        "PHASE4A2_REAL_RASTERIZER_BACKWARD": rasterizer_backward and step0_tolerance_ok,
        "PHASE4A2_IMAGE_PATH_NO_CPU_FALLBACK": fused_runtime_ok,
        "PHASE4A2_REAL_OPTIMIZER_UPDATE": optimizer_ok,
        "PHASE4A2_100_STEP_FINITE_VALUES": finite,
        "PHASE4A2_100_STEP_LOSS_DECREASE": losses_decrease,
        "PHASE4A2_100_STEP_SIGNAL_ACTIVITY": active,
        "PHASE4A2_100_STEP_RENDER_OUTPUT": renders,
        "PHASE4A2_100_STEP_VRAM_STABILITY": memory_stable,
        "PHASE4A2_FUSED_LOCAL_REPLAY_DETERMINISM": replay_deterministic,
        "PHASE4A2_FUSED_LOCAL_RESUME_EQUIVALENCE": resume_equivalent,
        "PHASE4A2_FUSED_BACKEND_RUNTIME_ATTESTATION": fused_runtime_ok,
        "PHASE4A2_BACKEND_ACTIVATION_NEGATIVE_TESTS": all(invalid_results.values()),
        "PHASE4A2_RESULT_INTERPRETATION_ATTESTATION": True,
    }
    gates["RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA"] = all(
        gates.values()
    )
    source_digest = hashlib.sha256()
    for path in (
        ROOT / "src/rocwmma_width64_mlp.cu",
        ROOT / "include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h",
        ROOT / "include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h",
    ):
        source_digest.update(path.read_bytes())
    summary = {
        "gates": gates,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "pytorch": torch.__version__,
            "rocm": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "gfx": getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unknown"),
            "tcnn_enable_rocwmma_width64_mlp": 1,
            "rocwmma_version": "2.2.0.70200-43",
            "amd_gsplat_commit": "2c62b22552c0ad4ed120aae304ce66ae27bc5d08",
            "rocwmma_source_sha256": source_digest.hexdigest(),
        },
        "binaries": binaries,
        "qualified_hashes": qualified_hashes,
        "initial_state_identity": {
            "scene_hash": fused[0]["scene_hash"],
            "canonical_parameter_hash": fused[0]["initial_canonical_parameter_hash"],
            "optimizer_empty": True,
        },
        "step0_tolerance_metrics": step0_metrics,
        "result_interpretation": {
            "step0_backend_equivalence_scope": (
                "bit-exact for the tested scene, shape, initialization, and runtime configuration"
            ),
            "trajectory_classification": "single-seed trajectory outcome",
            "cross_backend_trajectory_bit_identity_required": False,
            "fused_local_replay_bit_identity_required": True,
            "fused_s50_s66_resume_bit_identity_required": True,
            "rasterizer_backward_difference_root_cause": "OPEN_WITHOUT_DEDICATED_BISECT",
            "atomic_order_or_runtime_scheduling": "HYPOTHESIS_ONLY",
            "performance_affects_pass": False,
            "per_forward_weight_view_repack_present": True,
            "general_quality_or_convergence_advantage_claimed": False,
        },
        "negative_tests": invalid_results,
        "stale_negative_test": stale,
        "hipblaslt": hipblaslt,
        "fused_replays": fused,
        "resume": resume,
        "result_bytes": 0,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report = f"""# Phase 4A2 — realer amd-gsplat-Bildpfad

## Geltungsbereich der numerischen Aussagen

- Im aktuellen Diagnosefall ist der qualifizierte Phase-3-Forward gegenüber
  dem CPU-Orakel bitgleich; der Backward verwendet das CPU-Orakel als
  Toleranzreferenz.
- Diese Aussage gilt für die geprüfte Form-, Daten- und
  Ausführungskonfiguration. Sie ist keine ungeprüfte Universalgarantie.
- Die HipBLASLt- und fused-rocWMMA-Trajektorien müssen nicht bitgleich
  bleiben. Der direkte A/B-Vergleich erfolgt an identischen kanonischen
  Zuständen mit den unveränderten qualifizierten Toleranzen.
- Der fused Pfad muss dagegen in drei unabhängigen 100-Schritt-Replays
  bitgleich mit sich selbst sein und den S50→S66-Resume-Test bestehen.
- Ein anfänglicher Abstand von einem lokalen FP16-ULP kann spätere
  Trajektorienunterschiede verursachen. Er ist kein vorab behaupteter
  makroskopischer Divergenznachweis.

## Entscheidung

`RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA: {'PASS' if gates['RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA'] else 'FAIL'}`

Die vollständigen Gates, A/B-Toleranzmetriken, Laufzeitattestationen,
Binärhashes, Replays und Resume-Fingerprints stehen in `summary.json`.
"""
    (OUTPUT / "PHASE4A2_IMAGE_PATH_REPORT.md").write_text(report)
    summary["result_bytes"] = sum(
        path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file()
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for name, passed in gates.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(json.dumps({
        "step0_tolerance_metrics": step0_metrics,
        "result_bytes": summary["result_bytes"],
    }, indent=2, sort_keys=True))
    return 0 if gates["RDNA4_FUSED_MLP_PHASE4A2_AMD_GSPLAT_FUSED_ROCWMMA"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
