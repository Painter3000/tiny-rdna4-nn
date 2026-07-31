#!/usr/bin/env python3
"""Production inference correctness probe for Phase 4A2-P2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001"
WIDTH = 64
MAX_ABS_TOLERANCE = 0.005
NORMALIZED_L2_TOLERANCE = 0.002
BATCH_CASES = (1, 16, 17, 255, 256, 257)


def find_backend_hyperparams(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("otype") == "RocWMMAWidth64MLP":
            return value
        for child in value.values():
            found = find_backend_hyperparams(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_backend_hyperparams(child)
            if found is not None:
                return found
    return None


def network_config() -> dict[str, Any]:
    return {
        "otype": "RocWMMAWidth64MLP",
        "n_neurons": WIDTH,
        "n_hidden_layers": 2,
        "activation": "ReLU",
        "output_activation": "None",
        "precision": "Fp16",
        "bias": True,
    }


def deterministic_matrix(torch, rows: int, cols: int, salt: int, divisor: float):
    index = torch.arange(
        rows * cols,
        dtype=torch.int64,
        device="cpu",
    ).reshape(rows, cols)
    values = ((index * (37 + salt * 2) + 11 + salt * 19) % 257) - 128
    return (values.to(torch.float32) / divisor).to(torch.float16)


def deterministic_bias(torch, salt: int):
    index = torch.arange(WIDTH, dtype=torch.int64, device="cpu")
    values = ((index * (29 + salt * 2) + 7 + salt * 13) % 97) - 48
    return (values.to(torch.float32) / 2048.0).to(torch.float16)


def deterministic_input(torch, batch: int):
    index = torch.arange(
        batch * WIDTH,
        dtype=torch.int64,
        device="cpu",
    ).reshape(batch, WIDTH)
    values = ((index * 43 + 17) % 257) - 128
    return values.to(torch.float32) / 128.0


def pack_parameters(torch):
    weights = [
        deterministic_matrix(torch, WIDTH, WIDTH, salt, 2048.0)
        for salt in (1, 2, 3)
    ]
    biases = [
        deterministic_bias(torch, salt)
        for salt in (1, 2, 3)
    ]

    pieces = []
    offsets = []
    cursor = 0
    for weight, bias in zip(weights, biases):
        weight_flat_col_major = weight.transpose(0, 1).contiguous().reshape(-1)
        pieces.append(weight_flat_col_major)
        offsets.append(
            {
                "weight": cursor,
                "bias": cursor + WIDTH * WIDTH,
            }
        )
        cursor += WIDTH * WIDTH
        pieces.append(bias.contiguous())
        cursor += WIDTH

    packed = torch.cat(pieces).contiguous()
    assert packed.dtype == torch.float16
    assert packed.numel() == 12480
    return packed, weights, biases, offsets


def cpu_reference(torch, input_fp32, weights, biases):
    current = input_fp32.to(torch.float16)
    hidden_stats = []

    for layer in range(2):
        accumulated = (
            current.to(torch.float32)
            @ weights[layer].to(torch.float32)
        )
        accumulated = accumulated + biases[layer].to(torch.float32)
        positive = int((accumulated > 0).sum().item())
        clamped = int((accumulated <= 0).sum().item())
        current = torch.relu(accumulated).to(torch.float16)
        hidden_stats.append(
            {
                "positive": positive,
                "clamped": clamped,
            }
        )

    output = (
        current.to(torch.float32)
        @ weights[2].to(torch.float32)
    )
    output = output + biases[2].to(torch.float32)
    return output.to(torch.float16), hidden_stats


def normalized_l2(torch, actual, expected) -> float:
    delta = (actual.to(torch.float32) - expected.to(torch.float32)).reshape(-1)
    denominator = torch.linalg.vector_norm(
        expected.to(torch.float32).reshape(-1)
    )
    numerator = torch.linalg.vector_norm(delta)
    return float((numerator / torch.clamp(denominator, min=1.0e-12)).item())


def bitwise_equal(torch, left, right) -> bool:
    left_bits = left.contiguous().view(torch.int16).cpu()
    right_bits = right.contiguous().view(torch.int16).cpu()
    return bool(torch.equal(left_bits, right_bits))


def expect_error(label, function, required_substring):
    try:
        function()
    except Exception as error:
        message = str(error)
        if required_substring not in message:
            raise AssertionError(
                f"{label}: expected {required_substring!r}, got {message!r}"
            ) from error
        return message
    raise AssertionError(f"{label}: expected failure, but call succeeded")


def run_case(torch, model, batch, weights, biases):
    input_cpu = deterministic_input(torch, batch)
    input_gpu = input_cpu.to(device="cuda")

    with torch.no_grad():
        output_1 = model(input_gpu)
        output_2 = model(input_gpu)

    torch.cuda.synchronize()

    expected, hidden_stats = cpu_reference(
        torch,
        input_cpu,
        weights,
        biases,
    )
    actual = output_1.cpu()

    delta = (
        actual.to(torch.float32)
        - expected.to(torch.float32)
    ).abs()
    max_abs = float(delta.max().item())
    nl2 = normalized_l2(torch, actual, expected)
    nonfinite = int((~torch.isfinite(actual)).sum().item())
    repeat_bitwise = bitwise_equal(torch, output_1, output_2)

    if max_abs > MAX_ABS_TOLERANCE:
        raise AssertionError(
            f"batch {batch}: max_abs={max_abs} exceeds "
            f"{MAX_ABS_TOLERANCE}"
        )
    if nl2 > NORMALIZED_L2_TOLERANCE:
        raise AssertionError(
            f"batch {batch}: normalized_l2={nl2} exceeds "
            f"{NORMALIZED_L2_TOLERANCE}"
        )
    if nonfinite != 0:
        raise AssertionError(f"batch {batch}: nonfinite={nonfinite}")
    if not repeat_bitwise:
        raise AssertionError(f"batch {batch}: repeat was not bitwise equal")
    if output_1.dtype != torch.float16:
        raise AssertionError(f"batch {batch}: output dtype is {output_1.dtype}")
    if tuple(output_1.shape) != (batch, WIDTH):
        raise AssertionError(
            f"batch {batch}: output shape is {tuple(output_1.shape)}"
        )

    return {
        "batch": batch,
        "internal_padded_batch": (
            ((batch + 255) // 256) * 256
        ),
        "max_abs": max_abs,
        "normalized_l2": nl2,
        "nonfinite_count": nonfinite,
        "positive_count": int((actual > 0).sum().item()),
        "negative_count": int((actual < 0).sum().item()),
        "zero_count": int((actual == 0).sum().item()),
        "repeat_bitwise_equal": repeat_bitwise,
        "hidden_stats": hidden_stats,
        "actual": actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import tinycudann as tcnn

    if torch.version.hip is None:
        raise RuntimeError("Phase 4A2-P2 requires a ROCm PyTorch build.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 4A2-P2 expects exactly one visible GPU.")

    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture and not architecture.startswith("gfx1201"):
        raise RuntimeError(f"Expected gfx1201, got {architecture!r}.")

    model = tcnn.Network(WIDTH, WIDTH, network_config())
    wrapper_hyperparams = model.native_tcnn_module.hyperparams()
    backend = find_backend_hyperparams(wrapper_hyperparams)
    if backend is None:
        raise AssertionError(
            "Nested RocWMMAWidth64MLP hyperparameters were not found."
        )

    packed, weights, biases, offsets = pack_parameters(torch)
    with torch.no_grad():
        model.params.copy_(
            packed.to(
                device=model.params.device,
                dtype=model.params.dtype,
            )
        )

    assert model.dtype == torch.float16
    assert model.params.numel() == 12480
    assert backend["otype"] == "RocWMMAWidth64MLP"
    assert backend["inference_qualified"] is True
    assert backend["forward_qualified"] is False
    assert backend["backward_qualified"] is False
    assert backend["parameter_elements"] == 12480
    assert backend["batch_tile_rows"] == 16
    assert backend["threads_per_block"] == 128
    assert backend["lds_bytes"] == 2048
    assert backend["source_barriers"] == 3
    assert backend["caller_stream"] is True
    assert backend["host_synchronization"] is False
    assert backend["diagnostic_oracle_arguments"] is False
    assert backend["diagnostic_counters"] is False

    cases = {}
    retained_outputs = {}
    for batch in BATCH_CASES:
        case = run_case(
            torch,
            model,
            batch,
            weights,
            biases,
        )
        retained_outputs[batch] = case.pop("actual")
        cases[str(batch)] = case

    prefix_checks = {
        "batch17_prefix16_equals_batch16": bitwise_equal(
            torch,
            retained_outputs[17][:16],
            retained_outputs[16],
        ),
        "batch255_prefix16_equals_batch16": bitwise_equal(
            torch,
            retained_outputs[255][:16],
            retained_outputs[16],
        ),
        "batch257_prefix256_equals_batch256": bitwise_equal(
            torch,
            retained_outputs[257][:256],
            retained_outputs[256],
        ),
    }
    assert all(prefix_checks.values())

    stream_batch = 33
    stream_input_cpu = deterministic_input(torch, stream_batch)
    stream_input_gpu = stream_input_cpu.to(device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream), torch.no_grad():
        stream_output = model(stream_input_gpu)
    stream.synchronize()

    stream_expected, _ = cpu_reference(
        torch,
        stream_input_cpu,
        weights,
        biases,
    )
    stream_actual = stream_output.cpu()
    stream_max_abs = float(
        (
            stream_actual.to(torch.float32)
            - stream_expected.to(torch.float32)
        ).abs().max().item()
    )
    stream_nl2 = normalized_l2(
        torch,
        stream_actual,
        stream_expected,
    )
    assert stream_max_abs <= MAX_ABS_TOLERANCE
    assert stream_nl2 <= NORMALIZED_L2_TOLERANCE

    training_input = deterministic_input(torch, 16).to(device="cuda")
    training_input.requires_grad_(True)
    training_error = expect_error(
        "training forward",
        lambda: model(training_input),
        "training forward is not qualified",
    )

    largest = retained_outputs[257]
    diverse_output = (
        int((largest > 0).sum().item()) > 0
        and int((largest < 0).sum().item()) > 0
    )
    assert diverse_output

    result = {
        "marker": MARKER,
        "decision": "PHASE4A2_P2_PRODUCTION_INFERENCE_PROCESS_PASS",
        "environment": {
            "torch_hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
            "gcn_arch_name": architecture or "gfx1201",
        },
        "backend": backend,
        "parameter_abi": {
            "dtype": str(model.dtype),
            "elements": int(model.params.numel()),
            "offsets": offsets,
            "layout": "W0_col,b0,W1_col,b1,W2_col,b2",
        },
        "cases": cases,
        "prefix_checks": prefix_checks,
        "nondefault_stream": {
            "batch": stream_batch,
            "max_abs": stream_max_abs,
            "normalized_l2": stream_nl2,
            "pass": True,
        },
        "training_forward_error": training_error,
        "tolerances": {
            "max_abs": MAX_ABS_TOLERANCE,
            "normalized_l2": NORMALIZED_L2_TOLERANCE,
        },
        "gates": {
            "explicit_backend_constructed": True,
            "parameter_abi_12480_fp16": True,
            "all_batch_cases_correct": all(
                case["max_abs"] <= MAX_ABS_TOLERANCE
                and case["normalized_l2"]
                <= NORMALIZED_L2_TOLERANCE
                and case["nonfinite_count"] == 0
                and case["repeat_bitwise_equal"]
                for case in cases.values()
            ),
            "tile_prefix_invariance": all(prefix_checks.values()),
            "nondefault_stream_correct": True,
            "public_output_fp16": all(
                retained_outputs[batch].dtype == torch.float16
                for batch in retained_outputs
            ),
            "output_sign_diversity": diverse_output,
            "training_forward_fail_closed": True,
            "inference_only_hyperparams": (
                backend["inference_qualified"] is True
                and backend["forward_qualified"] is False
                and backend["backward_qualified"] is False
            ),
        },
    }
    assert all(result["gates"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    print("WIDTH64_PRODUCTION_PARAMETER_ABI_12480_FP16: PASS")
    print("WIDTH64_PRODUCTION_BATCH_GRID_16_512: PASS")
    print("WIDTH64_PRODUCTION_INFERENCE_VS_CPU_FP32: PASS")
    print("WIDTH64_PRODUCTION_REPEAT_BITWISE: PASS")
    print("WIDTH64_PRODUCTION_TILE_PREFIX_INVARIANCE: PASS")
    print("WIDTH64_PRODUCTION_NONDEFAULT_STREAM: PASS")
    print("WIDTH64_PRODUCTION_FP16_OUTPUT: PASS")
    print("WIDTH64_TRAINING_FORWARD_FAIL_CLOSED: PASS")
    print("PHASE4A2_P2_PRODUCTION_INFERENCE_PROCESS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
