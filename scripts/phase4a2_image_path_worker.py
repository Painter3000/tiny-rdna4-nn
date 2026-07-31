#!/usr/bin/env python3
"""One fresh-process real amd-gsplat training lane for Phase 4A2."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import time

import torch
from PIL import Image
from gsplat import rasterization
import tinycudann as tcnn
import tinycudann.modules as tcnn_modules

WIDTH = 64
WEIGHT_OFFSETS = (0, 4160, 8320)
MEASUREMENTS = {0, 1, 4, 16, 50, 100}


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def canonicalize(value: torch.Tensor, backend: str) -> torch.Tensor:
    result = value.detach().clone()
    if backend == "hipblaslt":
        for offset in WEIGHT_OFFSETS:
            result[offset : offset + 4096] = (
                value[offset : offset + 4096].reshape(WIDTH, WIDTH).T.reshape(-1)
            )
    return result


def physicalize(value: torch.Tensor, backend: str) -> torch.Tensor:
    return canonicalize(value, backend)


def fp32_recurrence_bits(beta: float, step: int) -> str:
    value = torch.tensor(1.0, dtype=torch.float32)
    factor = torch.tensor(beta, dtype=torch.float32)
    for _ in range(step):
        value = value * factor
    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def save_image(path: pathlib.Path, image: torch.Tensor) -> None:
    pixels = (
        image.detach().float().clamp(0, 1).mul(255).round()
        .to(torch.uint8).cpu().numpy()
    )
    Image.fromarray(pixels).save(path)


def make_scene(seed: int):
    generator = torch.Generator().manual_seed(seed)
    means = torch.empty((128, 3)).uniform_(-0.72, 0.72, generator=generator)
    means[:, 2].uniform_(2.0, 3.2, generator=generator)
    features = torch.empty((128, WIDTH))
    index = torch.arange(128 * WIDTH).reshape(128, WIDTH)
    features.copy_((((index * 43 + 17) % 257) - 128).float() / 128.0)
    features[:, :2] = means[:, :2]
    features[:, 2] = means[:, 2].sub(2.0).div(1.2).mul(2).sub(1)
    target_colors = torch.stack(
        (
            torch.sigmoid(2.2 * features[:, 0] - 0.8 * features[:, 1]),
            torch.sigmoid(2.0 * features[:, 1] + 0.7 * features[:, 2]),
            torch.sigmoid(-1.3 * features[:, 0] + 1.7 * features[:, 2]),
        ),
        dim=1,
    )
    quats = torch.zeros((128, 4))
    quats[:, 0] = 1
    scales = torch.full((128, 3), 0.075)
    opacities = torch.full((128,), 0.86)
    viewmats = torch.eye(4).repeat(2, 1, 1)
    viewmats[1, 0, 3] = 0.12
    intrinsics = torch.tensor(
        [[[55.0, 0.0, 32.0], [0.0, 55.0, 32.0], [0.0, 0.0, 1.0]]]
    ).repeat(2, 1, 1)
    values = [
        means, features, target_colors, quats, scales, opacities,
        viewmats, intrinsics,
    ]
    values = [value.cuda() for value in values]
    means, features, target_colors, quats, scales, opacities, viewmats, intrinsics = values
    with torch.no_grad():
        target, target_alpha, target_meta = rasterization(
            means, quats, scales, opacities, target_colors,
            viewmats, intrinsics, 64, 64, packed=False, render_mode="RGB",
        )
    means.requires_grad_(True)
    scene = {
        "means": means,
        "features": features,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "target": target.detach(),
        "target_alpha": target_alpha.detach(),
        "target_meta": target_meta,
    }
    state_hash = hashlib.sha256()
    for name in sorted(scene):
        value = scene[name]
        if torch.is_tensor(value):
            state_hash.update(name.encode())
            state_hash.update(tensor_bytes(value))
    return scene, state_hash.hexdigest()


def render(scene, colors):
    return rasterization(
        scene["means"], scene["quats"], scene["scales"],
        scene["opacities"], colors, scene["viewmats"],
        scene["intrinsics"], 64, 64, packed=False, render_mode="RGB",
    )


def make_model(backend: str, seed: int):
    config = {
        "otype": (
            "HipBLASLtMLPFP16"
            if backend == "hipblaslt"
            else "RocWMMAWidth64MLP"
        ),
        "n_neurons": WIDTH,
        "n_hidden_layers": 2,
        "activation": "ReLU",
        "output_activation": "None",
        "precision": "Fp16",
        "bias": True,
    }
    model = tcnn.Network(WIDTH, WIDTH, config, seed=seed)
    if backend == "hipblaslt":
        logical = model.params.detach().clone()
        with torch.no_grad():
            model.params.copy_(physicalize(logical, backend))
    return model, config


def optimizer_state_hash(model, optimizer, backend: str) -> str:
    digest = hashlib.sha256()
    digest.update(tensor_bytes(canonicalize(model.params, backend)))
    state = optimizer.state.get(model.params, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            digest.update(tensor_bytes(canonicalize(state[key], backend)))
    if "step" in state:
        digest.update(tensor_bytes(state["step"]))
    return digest.hexdigest()


def state_record(model, optimizer, backend: str, output, image) -> dict:
    state = optimizer.state.get(model.params, {})
    step = int(state.get("step", torch.tensor(0)).item())
    return {
        "state_hash": optimizer_state_hash(model, optimizer, backend),
        "canonical_parameter_hash": tensor_hash(canonicalize(model.params, backend)),
        "exp_avg_hash": (
            tensor_hash(canonicalize(state["exp_avg"], backend))
            if "exp_avg" in state else None
        ),
        "exp_avg_sq_hash": (
            tensor_hash(canonicalize(state["exp_avg_sq"], backend))
            if "exp_avg_sq" in state else None
        ),
        "optimizer_step": step,
        "beta1_power_bits": fp32_recurrence_bits(0.9, step),
        "beta2_power_bits": fp32_recurrence_bits(0.999, step),
        "forward_hash": tensor_hash(output),
        "render_hash": tensor_hash(image),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hipblaslt", "rocwmma"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--stop-step", type=int, default=100)
    parser.add_argument("--resume", type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42004)
    scene, scene_hash = make_scene(42004)
    model, config = make_model(args.backend, 42005)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    extension = tcnn_modules._C
    if args.backend == "rocwmma":
        extension._phase4a2_reset_runtime_attestation()
    initial_canonical = canonicalize(model.params, args.backend)
    initial_hash = tensor_hash(initial_canonical)
    initial_optimizer_empty = not optimizer.state
    measurements = {}
    step0_tensors = {}
    checkpoint_path = args.output / "checkpoint_s50.pt"
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    first_updated_forward_hash = None
    initial_forward_hash = None
    weight_max_abs_change = 0.0
    gradient_norm_max = 0.0
    gaussian_gradient_norm_max = 0.0
    final_output = final_image = None

    for step in range(start_step, args.stop_step + 1):
        optimizer.zero_grad(set_to_none=True)
        scene["means"].grad = None
        output = model(scene["features"])
        colors = torch.sigmoid(output[:, :3].float())
        colors.retain_grad()
        image, alpha, meta = render(scene, colors)
        loss = torch.nn.functional.mse_loss(image, scene["target"])
        if step == 0:
            initial_forward_hash = tensor_hash(output)
            save_image(args.output / "render_s0.png", image[0])
        if step in MEASUREMENTS or step == args.stop_step:
            measurements[str(step)] = {
                "loss": float(loss.detach()),
                "decoder_hash": tensor_hash(output),
                "render_hash": tensor_hash(image),
                "alpha_hash": tensor_hash(alpha),
                "radii_hash": tensor_hash(meta["radii"]),
                "visible_count": int((meta["radii"] > 0).sum()),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
            if step in (50, 100):
                measurements[str(step)]["state"] = state_record(
                    model, optimizer, args.backend, output, image
                )
        if step == args.stop_step:
            final_output, final_image = output.detach(), image.detach()
            break

        loss.backward()
        parameter_grad = model.params.grad.detach()
        gradient_norm = float(parameter_grad.float().norm())
        gaussian_gradient_norm = float(scene["means"].grad.detach().float().norm())
        gradient_norm_max = max(gradient_norm_max, gradient_norm)
        gaussian_gradient_norm_max = max(
            gaussian_gradient_norm_max, gaussian_gradient_norm
        )
        if step == 0:
            step0_tensors = {
                "decoder": output.detach().cpu(),
                "image": image.detach().cpu(),
                "alpha": alpha.detach().cpu(),
                "radii": meta["radii"].detach().cpu(),
                "visibility": (meta["radii"] > 0).detach().cpu(),
                "loss": loss.detach().cpu(),
                "gaussian_grad": scene["means"].grad.detach().cpu(),
                "mlp_grad_canonical": canonicalize(parameter_grad, args.backend).cpu(),
                "color_grad": colors.grad.detach().cpu(),
            }
        before = canonicalize(model.params, args.backend)
        optimizer.step()
        after = canonicalize(model.params, args.backend)
        weight_max_abs_change = max(
            weight_max_abs_change, float((after - before).abs().max())
        )
        completed_step = step + 1
        if completed_step == 1:
            with torch.no_grad():
                first_updated_forward_hash = tensor_hash(model(scene["features"]))
        if completed_step == 50 and args.stop_step >= 50:
            torch.save(
                {
                    "step": 50,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        if completed_step == 66:
            with torch.no_grad():
                s66_output = model(scene["features"])
                s66_image, _, _ = render(
                    scene, torch.sigmoid(s66_output[:, :3].float())
                )
            measurements["66_state"] = state_record(
                model, optimizer, args.backend, s66_output, s66_image
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    save_image(args.output / f"render_s{args.stop_step}.png", final_image[0])
    torch.save(step0_tensors, args.output / "step0_tensors.pt")
    final_state = state_record(
        model, optimizer, args.backend, final_output, final_image
    )
    state = optimizer.state.get(model.params, {})
    runtime = (
        dict(extension._phase4a2_runtime_attestation())
        if args.backend == "rocwmma" else None
    )
    result = {
        "backend": args.backend,
        "backend_config": config,
        "backend_hyperparams": model.native_tcnn_module.hyperparams(),
        "scene_hash": scene_hash,
        "initial_canonical_parameter_hash": initial_hash,
        "initial_optimizer_empty": initial_optimizer_empty,
        "start_step": start_step,
        "stop_step": args.stop_step,
        "measurements": measurements,
        "final_state": final_state,
        "initial_forward_hash": initial_forward_hash,
        "first_updated_forward_hash": first_updated_forward_hash,
        "next_forward_used_new_weights": (
            first_updated_forward_hash is not None
            and first_updated_forward_hash != initial_forward_hash
        ),
        "weight_max_abs_change": weight_max_abs_change,
        "gradient_norm_max": gradient_norm_max,
        "gaussian_gradient_norm_max": gaussian_gradient_norm_max,
        "exp_avg_active": bool(state and torch.count_nonzero(state["exp_avg"])),
        "exp_avg_sq_active": bool(state and torch.count_nonzero(state["exp_avg_sq"])),
        "all_finite": bool(
            torch.isfinite(model.params).all()
            and torch.isfinite(final_output).all()
            and torch.isfinite(final_image).all()
        ),
        "runtime_seconds": elapsed,
        "max_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "runtime_attestation": runtime,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
