#!/usr/bin/env python3
"""Fresh-clone runtime smoke for tiny-rdna4-nn on ROCm/gfx1201.

Marker: TCNN_RDNA4_FRESH_CLONE_USER_SMOKE_001

This script is normally launched by scripts/fresh_clone_user_smoke.sh from a
neutral working directory. It verifies package provenance, the documented
HashGrid + PortableMLP path, forward/backward, Adam updates, checkpoint reload,
a second fresh Python process, and the explicit ROCm backend contract.

It is a correctness smoke, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

MARKER = "TCNN_RDNA4_FRESH_CLONE_USER_SMOKE_001"
EXPECTED_ARCH = "gfx1201"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def path_is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def portable_nwie_config() -> dict[str, Any]:
    return {
        "n_input_dims": 3,
        "n_output_dims": 4,
        "encoding_config": {
            "otype": "HashGrid",
            "n_levels": 4,
            "n_features_per_level": 2,
            "log2_hashmap_size": 12,
            "base_resolution": 4,
            "per_level_scale": 2.0,
        },
        "network_config": {
            "otype": "PortableMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 16,
            "n_hidden_layers": 1,
        },
    }


def network_config(backend: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "otype": backend,
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 16,
        "n_hidden_layers": 1,
    }
    if backend == "HipBLASLtMLPFP16":
        config["precision"] = "Fp16"
    return config


def model_parameters(model: Any) -> list[Any]:
    params = list(model.parameters())
    require(bool(params), "Model exposes no trainable parameters")
    return params


def assert_finite_tensor(tensor: Any, label: str) -> None:
    import torch

    require(bool(torch.isfinite(tensor).all()), f"{label} contains NaN or Inf")


def provenance(repo_root: Path) -> tuple[dict[str, Any], Any, Any, Any]:
    import torch
    import tinycudann as tcnn
    from tinycudann.modules import _C

    venv = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    torch_path = Path(torch.__file__).resolve()
    package_path = Path(tcnn.__file__).resolve()
    native_path = Path(_C.__file__).resolve()
    cwd = Path.cwd().resolve()

    require(venv != base_prefix, "Run this smoke inside an activated virtual environment")
    require(path_is_within(torch_path, venv), "Torch does not come from the active virtual environment")
    require(path_is_within(package_path, venv), "tinycudann does not come from the active virtual environment")
    require(path_is_within(native_path, venv), "Native tinycudann binding does not come from the active virtual environment")
    require(not path_is_within(cwd, repo_root), "Runtime smoke must execute from a neutral directory")
    require(torch.version.hip is not None, "This PyTorch build has no ROCm/HIP support")
    require(torch.cuda.is_available(), "No ROCm GPU is visible to PyTorch")

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)
    require(arch == EXPECTED_ARCH, f"Expected {EXPECTED_ARCH}, got {arch!r}")

    report = {
        "marker": MARKER,
        "python_executable": sys.executable,
        "sys_prefix": str(venv),
        "sys_base_prefix": str(base_prefix),
        "cwd": str(cwd),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "torch_path": str(torch_path),
        "tinycudann_path": str(package_path),
        "native_binding_path": str(native_path),
        "native_binding_sha256": sha256_file(native_path),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_arch": arch,
    }
    return report, torch, tcnn, _C


def reject_nvidia_backend_alias(tcnn: Any) -> dict[str, Any]:
    error_text = ""
    try:
        tcnn.Network(
            16,
            16,
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 16,
                "n_hidden_layers": 1,
            },
        )
    except RuntimeError as exc:
        error_text = str(exc)

    require(error_text, "FullyFusedMLP was unexpectedly accepted on the ROCm path")
    require(
        "deliberately excluded" in error_text and "Use PortableMLP" in error_text,
        f"Unexpected FullyFusedMLP rejection message: {error_text}",
    )
    print("EXPLICIT_BACKEND_CONTRACT: PASS")
    return {
        "backend": "FullyFusedMLP",
        "rejected": True,
        "error": error_text,
    }


def run_portable_nwie(
    *,
    torch: Any,
    tcnn: Any,
    evidence_dir: Path,
    launch_child: bool,
) -> dict[str, Any]:
    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)

    config = portable_nwie_config()
    model = tcnn.NetworkWithInputEncoding(**config).to("cuda")
    params = model_parameters(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)

    fixed_input = torch.rand(256, 3, device="cuda", dtype=torch.float32)
    fixed_target = torch.rand(256, 4, device="cuda", dtype=torch.float32)
    before = [param.detach().clone() for param in params]
    losses: list[float] = []

    for step in range(3):
        x = fixed_input.detach().clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)

        y = model(x)
        require(tuple(y.shape) == (256, 4), f"Unexpected output shape: {tuple(y.shape)}")
        assert_finite_tensor(y, "PortableMLP output")

        loss = (y.float() - fixed_target).square().mean()
        assert_finite_tensor(loss, "PortableMLP loss")
        loss.backward()

        require(x.grad is not None, "PortableMLP input gradient is missing")
        assert_finite_tensor(x.grad, "PortableMLP input gradient")
        for index, param in enumerate(params):
            require(param.grad is not None, f"Parameter gradient {index} is missing")
            assert_finite_tensor(param.grad, f"Parameter gradient {index}")

        optimizer.step()
        torch.cuda.synchronize()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        print(f"PORTABLE_NWIE step={step} loss={loss_value:.9f}")

    max_update = max(
        float((param.detach() - original).abs().max().cpu())
        for param, original in zip(params, before)
    )
    require(max_update > 0.0, "Adam did not update any PortableMLP parameter")

    checkpoint = evidence_dir / "portable_nwie_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)

    clone = tcnn.NetworkWithInputEncoding(**config).to("cuda")
    clone.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))

    probe = torch.rand(128, 3, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        output_a = model(probe).float()
        output_b = clone(probe).float()
    torch.cuda.synchronize()
    torch.testing.assert_close(output_b, output_a, rtol=1e-5, atol=1e-6)

    child_report = evidence_dir / "second_process_report.json"
    if launch_child:
        child_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-checkpoint",
            str(checkpoint),
            "--child-report",
            str(child_report),
            "--repo-root",
            str(Path(os.environ["TCNN_SMOKE_REPO_ROOT"]).resolve()),
        ]
        print("SECOND_PROCESS command:", " ".join(child_cmd))
        subprocess.run(child_cmd, cwd=Path.cwd(), check=True)
        child_payload = json.loads(child_report.read_text(encoding="utf-8"))
        require(child_payload.get("result") == "PASS", "Second-process checkpoint smoke failed")
    else:
        child_payload = None

    print("PORTABLEMLP_FORWARD_BACKWARD_ADAM: PASS")
    print("CHECKPOINT_RELOAD: PASS")
    if launch_child:
        print("FRESH_CLONE_USER_SMOKE_PROCESS_2: PASS")

    return {
        "backend": "PortableMLP",
        "model": "NetworkWithInputEncoding",
        "losses": losses,
        "parameter_max_update": max_update,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "second_process": child_payload,
        "passed": True,
    }


def run_backend_case(torch: Any, tcnn: Any, backend: str, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = tcnn.Network(16, 16, network_config(backend), seed=seed).to("cuda")
    params = model_parameters(model)
    optimizer = torch.optim.Adam(params, lr=5e-4)

    x = torch.randn(128, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.sin(x.detach())
    before = [param.detach().clone() for param in params]

    optimizer.zero_grad(set_to_none=True)
    y = model(x)
    require(tuple(y.shape) == (128, 16), f"{backend}: unexpected output shape {tuple(y.shape)}")
    assert_finite_tensor(y, f"{backend} output")

    loss = (y.float() - target.float()).square().mean()
    assert_finite_tensor(loss, f"{backend} loss")
    loss.backward()

    require(x.grad is not None, f"{backend}: input gradient is missing")
    assert_finite_tensor(x.grad, f"{backend} input gradient")
    for index, param in enumerate(params):
        require(param.grad is not None, f"{backend}: parameter gradient {index} is missing")
        assert_finite_tensor(param.grad, f"{backend} parameter gradient {index}")

    optimizer.step()
    torch.cuda.synchronize()

    expected_output_dtype = (
        torch.float16 if backend == "HipBLASLtMLPFP16" else torch.float32
    )
    require(
        y.dtype == expected_output_dtype,
        f"{backend}: expected output dtype {expected_output_dtype}, got {y.dtype}",
    )
    for index, param in enumerate(params):
        require(
            param.dtype == torch.float32,
            f"{backend}: master parameter {index} is not FP32",
        )
        require(
            param.grad.dtype == torch.float32,
            f"{backend}: PyTorch gradient {index} is not FP32",
        )

    max_update = max(
        float((param.detach() - original).abs().max().cpu())
        for param, original in zip(params, before)
    )
    require(max_update > 0.0, f"{backend}: optimizer did not update parameters")

    payload = {
        "backend": backend,
        "output_dtype": str(y.dtype),
        "parameter_dtype": str(params[0].dtype),
        "gradient_dtype": str(params[0].grad.dtype),
        "loss": float(loss.detach().cpu()),
        "parameter_max_update": max_update,
        "passed": True,
    }
    print(f"BACKEND_{backend}: PASS")
    return payload


def child_checkpoint(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    report, torch, tcnn, _ = provenance(repo_root)

    config = portable_nwie_config()
    model = tcnn.NetworkWithInputEncoding(**config).to("cuda")
    model.load_state_dict(
        torch.load(args.child_checkpoint, map_location="cuda", weights_only=True)
    )

    x = torch.rand(64, 3, device="cuda", dtype=torch.float32, requires_grad=True)
    y = model(x)
    loss = y.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    require(tuple(y.shape) == (64, 4), f"Second process: unexpected output shape {tuple(y.shape)}")
    assert_finite_tensor(y, "Second-process output")
    require(x.grad is not None, "Second-process input gradient is missing")
    assert_finite_tensor(x.grad, "Second-process input gradient")
    for index, param in enumerate(model_parameters(model)):
        require(param.grad is not None, f"Second-process parameter gradient {index} is missing")
        assert_finite_tensor(param.grad, f"Second-process parameter gradient {index}")

    payload = {
        "marker": MARKER,
        "result": "PASS",
        "provenance": report,
        "output_shape": list(y.shape),
        "loss": float(loss.detach().cpu()),
    }
    json_write(Path(args.child_report).resolve(), payload)
    print("FRESH_CLONE_USER_SMOKE_PROCESS_2: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "all-backends"), default="quick")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--child-checkpoint")
    parser.add_argument("--child-report")
    args = parser.parse_args()

    if args.child_checkpoint:
        require(bool(args.child_report), "--child-report is required in child mode")
        return child_checkpoint(args)

    require(bool(args.evidence_dir), "--evidence-dir is required")
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve()
    require(not path_is_within(evidence_dir, repo_root), "Evidence directory must be outside the repository")

    report, torch, tcnn, _ = provenance(repo_root)
    print("===== PROVENANCE =====")
    for key in (
        "python_executable",
        "sys_prefix",
        "torch_version",
        "torch_hip_version",
        "torch_path",
        "tinycudann_path",
        "native_binding_path",
        "native_binding_sha256",
        "gpu_name",
        "gpu_arch",
        "cwd",
    ):
        print(f"{key}: {report[key]}")
    print("PROVENANCE_CHECK: PASS")

    report["backend_contract"] = reject_nvidia_backend_alias(tcnn)
    report["portable_nwie"] = run_portable_nwie(
        torch=torch,
        tcnn=tcnn,
        evidence_dir=evidence_dir,
        launch_child=True,
    )

    backend_results: list[dict[str, Any]] = []
    if args.mode == "all-backends":
        for offset, backend in enumerate(
            ("PortableMLP", "HipBLASLtMLP", "HipBLASLtMLPFP16")
        ):
            backend_results.append(
                run_backend_case(torch, tcnn, backend, 20260800 + offset)
            )

    report.update(
        {
            "mode": args.mode,
            "backends": backend_results,
            "result": "PASS",
        }
    )
    json_write(evidence_dir / "runtime_report.json", report)

    print("FRESH_CLONE_CODE_PATH_SMOKE: PASS")
    print("FRESH_CLONE_USER_SMOKE_PROCESS_1: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FRESH_CLONE_USER_SMOKE_RUNTIME: FAIL: {exc}", file=sys.stderr)
        raise
