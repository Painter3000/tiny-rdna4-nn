#!/usr/bin/env python3
"""TCNN_RDNA4_P3A4_FUSED_RELU_BGRAD_001 validation and regression gate."""
import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys

import torch


def load(bindings):
    if bindings:
        sys.path.insert(0, str(pathlib.Path(bindings).resolve()))
    import tinycudann as tcnn
    return tcnn, importlib.import_module("tinycudann_bindings._120_C")


def config(width=64, layers=4, activation="ReLU"):
    return {"otype": "HipBLASLtMLP", "n_hidden_layers": layers, "n_neurons": width,
            "activation": activation, "output_activation": "None"}


def raw(module, x, grad, initial, mode):
    n = x.shape[0]
    padded = (n + 255) // 256 * 256
    xx = torch.nn.functional.pad(x, (0, 0, 0, padded - n)).contiguous().requires_grad_()
    gg = torch.nn.functional.pad(grad, (0, 0, 0, padded - n)).contiguous()
    params = module.params.detach().clone().requires_grad_()
    ctx, output = module.native_tcnn_module.fwd(xx, params)
    dx, dp = module.native_tcnn_module.bwd_mode(ctx, xx, params, output, gg, initial.clone(), mode)
    torch.cuda.synchronize()
    return output[:n], dx[:n], dp


def targeted(tcnn, native):
    rows = []
    fingerprints = []
    for width in (16, 32, 64, 128):
        model = tcnn.Network(8, 8, config(width, 2), seed=20260721)
        for batch in (1, 7, 31, 64, 257, 1024, 4096):
            generator = torch.Generator(device="cuda").manual_seed(20260721 + width + batch)
            x = torch.randn(batch, 8, device="cuda", generator=generator) * 0.2
            grad = torch.randn(batch, 8, device="cuda", generator=generator) * 0.1
            zero = torch.zeros_like(model.params)
            _, _, overwrite = raw(model, x, grad, zero, "Overwrite")
            _, _, first = raw(model, x, grad, zero, "Accumulate")
            _, _, second = raw(model, x, grad, first, "Accumulate")
            sentinel = torch.linspace(-0.25, 0.25, model.params.numel(), device="cuda")
            _, _, ignored = raw(model, x, grad, sentinel, "Ignore")
            torch.testing.assert_close(first, overwrite, atol=0, rtol=0)
            torch.testing.assert_close(second, 2 * overwrite, atol=2e-5, rtol=2e-5)
            assert torch.equal(ignored, sentinel)
            rows.append({"width": width, "batch": batch, "overwrite_equals_accumulate": True,
                         "double_accumulate": True, "ignore_exact": True})
        # Fixed input repeated 100 times must produce the exact same bytes.
        x = torch.linspace(-1, 1, 257 * 8, device="cuda").reshape(257, 8)
        grad = torch.linspace(1, -1, 257 * 8, device="cuda").reshape(257, 8)
        zero = torch.zeros_like(model.params)
        reference = raw(model, x, grad, zero, "Overwrite")[2]
        for _ in range(99):
            candidate = raw(model, x, grad, zero, "Overwrite")[2]
            assert torch.equal(candidate, reference)
        fingerprints.append({"width": width, "repetitions": 100,
                             "sum": float(reference.double().sum()), "bit_exact": True})
    counters = {name: int(getattr(native, name)()) for name in (
        "_hipblaslt_fused_relu_biasgrad_stage1_launches",
        "_hipblaslt_fused_relu_only_launches",
        "_hipblaslt_biasgrad_finalize_launches",
        "_hipblaslt_fused_relu_biasgrad_fallbacks",
        "_hipblaslt_legacy_activation_grad_launches",
        "_hipblaslt_legacy_bias_grad_launches",
        "_hipblaslt_fused_partial_bytes_live",
        "_hipblaslt_fused_partial_bytes_peak")}
    assert counters["_hipblaslt_fused_relu_biasgrad_stage1_launches"] > 0
    assert counters["_hipblaslt_fused_relu_only_launches"] > 0
    assert counters["_hipblaslt_biasgrad_finalize_launches"] > 0
    assert counters["_hipblaslt_fused_partial_bytes_live"] == 0
    assert counters["_hipblaslt_fused_partial_bytes_peak"] <= 1024 * 1024
    return {"matrix": rows, "determinism": fingerprints, "counters": counters}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-phase3a1", action="store_true")
    args = parser.parse_args()
    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bindings = args.bindings or str(pathlib.Path(__file__).resolve().parents[1] / "bindings" / "torch")
    if not args.skip_phase3a1:
        command = [sys.executable, str(pathlib.Path(__file__).with_name("validate_phase3a1.py")),
                   "--bindings", bindings, "--output", str(out / "phase3a1_regression")]
        completed = subprocess.run(command, env=os.environ.copy())
        if completed.returncode:
            raise SystemExit(completed.returncode)
    tcnn, native = load(bindings)
    properties = torch.cuda.get_device_properties(0)
    assert properties.gcnArchName == "gfx1201" and torch.version.hip
    result = {"result": "PASS", "environment": {"device": torch.cuda.get_device_name(0),
              "arch": properties.gcnArchName, "torch": torch.__version__, "hip": torch.version.hip,
              "binding": native.__file__}, "targeted": targeted(tcnn, native),
              "phase3a1_regression": "SKIPPED" if args.skip_phase3a1 else "PASS"}
    (out / "phase3a4_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PHASE3A4_VALIDATION=PASS")


if __name__ == "__main__":
    main()

