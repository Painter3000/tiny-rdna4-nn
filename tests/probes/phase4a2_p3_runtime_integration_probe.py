#!/usr/bin/env python3
"""Phase 4A2-P3 runtime integration and lifecycle closure probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P3_RUNTIME_INTEGRATION_CLOSURE_001"
WIDTH = 64
MAX_ABS_TOLERANCE = 0.005
NORMALIZED_L2_TOLERANCE = 0.002
BATCHES = (
    1, 15, 16, 17,
    31, 32, 33,
    63, 64, 65,
    127, 128, 129,
    255, 256, 257,
    511, 512, 513,
    1023,
)
PREFIX_PAIRS = (
    (16, 17),
    (32, 33),
    (64, 65),
    (128, 129),
    (256, 257),
    (512, 513),
)


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


def deterministic_weight(torch, salt: int):
    index = torch.arange(
        WIDTH * WIDTH,
        dtype=torch.int64,
        device="cpu",
    ).reshape(WIDTH, WIDTH)
    values = ((index * (31 + 2 * salt) + 17 * salt + 9) % 251) - 125
    return (values.to(torch.float32) / (2048.0 + salt * 128.0)).to(
        torch.float16
    )


def deterministic_bias(torch, salt: int):
    index = torch.arange(WIDTH, dtype=torch.int64, device="cpu")
    values = ((index * (23 + salt) + 13 * salt + 5) % 101) - 50
    return (values.to(torch.float32) / 2048.0).to(torch.float16)


def deterministic_input(torch, batch: int):
    index = torch.arange(
        batch * WIDTH,
        dtype=torch.int64,
        device="cpu",
    ).reshape(batch, WIDTH)
    values = ((index * 43 + 17) % 257) - 128
    return values.to(torch.float32) / 128.0


def make_parameter_set(torch, base_salt: int):
    weights = [
        deterministic_weight(torch, base_salt + layer)
        for layer in range(3)
    ]
    biases = [
        deterministic_bias(torch, base_salt + layer)
        for layer in range(3)
    ]

    pieces = []
    offsets = []
    cursor = 0
    for weight, bias in zip(weights, biases):
        pieces.append(weight.transpose(0, 1).contiguous().reshape(-1))
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
    return {
        "packed": packed,
        "weights": weights,
        "biases": biases,
        "offsets": offsets,
    }


def cpu_reference(torch, input_fp32, parameter_set):
    current = input_fp32.to(torch.float16)
    for layer in range(2):
        accumulated = (
            current.to(torch.float32)
            @ parameter_set["weights"][layer].to(torch.float32)
        )
        accumulated = accumulated + parameter_set["biases"][layer].to(
            torch.float32
        )
        current = torch.relu(accumulated).to(torch.float16)

    output = (
        current.to(torch.float32)
        @ parameter_set["weights"][2].to(torch.float32)
    )
    output = output + parameter_set["biases"][2].to(torch.float32)
    return output.to(torch.float16)


def bitwise_equal(torch, left, right) -> bool:
    return bool(
        torch.equal(
            left.contiguous().view(torch.int16).cpu(),
            right.contiguous().view(torch.int16).cpu(),
        )
    )


def normalized_l2(torch, actual, expected) -> float:
    delta = (
        actual.to(torch.float32) - expected.to(torch.float32)
    ).reshape(-1)
    numerator = torch.linalg.vector_norm(delta)
    denominator = torch.linalg.vector_norm(
        expected.to(torch.float32).reshape(-1)
    )
    return float(
        (
            numerator
            / torch.clamp(denominator, min=1.0e-12)
        ).item()
    )


def compare(torch, actual, expected, label: str):
    actual_cpu = actual.cpu()
    delta = (
        actual_cpu.to(torch.float32) - expected.to(torch.float32)
    ).abs()
    max_abs = float(delta.max().item())
    nl2 = normalized_l2(torch, actual_cpu, expected)
    nonfinite = int((~torch.isfinite(actual_cpu)).sum().item())

    if max_abs > MAX_ABS_TOLERANCE:
        raise AssertionError(
            f"{label}: max_abs={max_abs} exceeds {MAX_ABS_TOLERANCE}"
        )
    if nl2 > NORMALIZED_L2_TOLERANCE:
        raise AssertionError(
            f"{label}: normalized_l2={nl2} exceeds "
            f"{NORMALIZED_L2_TOLERANCE}"
        )
    if nonfinite != 0:
        raise AssertionError(f"{label}: nonfinite={nonfinite}")

    return {
        "max_abs": max_abs,
        "normalized_l2": nl2,
        "nonfinite_count": nonfinite,
        "positive_count": int((actual_cpu > 0).sum().item()),
        "negative_count": int((actual_cpu < 0).sum().item()),
        "zero_count": int((actual_cpu == 0).sum().item()),
    }


def install_params(model, parameter_set):
    import torch

    with torch.no_grad():
        model.params.copy_(
            parameter_set["packed"].to(
                device=model.params.device,
                dtype=model.params.dtype,
            )
        )


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-launches", type=int, default=64)
    args = parser.parse_args()

    import torch
    import tinycudann as tcnn

    if torch.version.hip is None:
        raise RuntimeError("Phase 4A2-P3 requires ROCm PyTorch.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU.")

    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture and not architecture.startswith("gfx1201"):
        raise RuntimeError(f"Expected gfx1201, got {architecture!r}.")

    model_a = tcnn.Network(WIDTH, WIDTH, network_config())
    model_b = tcnn.Network(WIDTH, WIDTH, network_config())

    wrapper_hyperparams = model_a.native_tcnn_module.hyperparams()
    backend = find_backend_hyperparams(wrapper_hyperparams)
    if backend is None:
        raise AssertionError(
            "Nested RocWMMAWidth64MLP hyperparameters were not found."
        )

    assert backend["inference_qualified"] is True
    assert backend["forward_qualified"] is False
    assert backend["backward_qualified"] is False
    assert backend["silent_fallback"] is False
    assert backend["caller_stream"] is True
    assert backend["host_synchronization"] is False
    assert backend["parameter_elements"] == 12480

    parameter_a = make_parameter_set(torch, 3)
    parameter_b = make_parameter_set(torch, 19)
    install_params(model_a, parameter_a)
    install_params(model_b, parameter_b)

    batch_results = {}
    retained = {}

    for batch in BATCHES:
        input_cpu = deterministic_input(torch, batch)
        input_gpu = input_cpu.to(device="cuda")
        expected = cpu_reference(torch, input_cpu, parameter_a)

        with torch.no_grad():
            output_1 = model_a(input_gpu)
            output_2 = model_a(input_gpu)

        torch.cuda.synchronize()

        metrics = compare(
            torch,
            output_1,
            expected,
            f"batch-{batch}",
        )
        metrics["repeat_bitwise_equal"] = bitwise_equal(
            torch,
            output_1,
            output_2,
        )
        metrics["internal_padded_batch"] = (
            ((batch + 255) // 256) * 256
        )
        metrics["shape"] = list(output_1.shape)
        metrics["dtype"] = str(output_1.dtype)

        assert metrics["repeat_bitwise_equal"]
        assert tuple(output_1.shape) == (batch, WIDTH)
        assert output_1.dtype == torch.float16

        retained[batch] = output_1.cpu()
        batch_results[str(batch)] = metrics

    prefix_checks = {}
    for smaller, larger in PREFIX_PAIRS:
        key = f"batch{larger}_prefix{smaller}"
        prefix_checks[key] = bitwise_equal(
            torch,
            retained[larger][:smaller],
            retained[smaller],
        )
    assert all(prefix_checks.values())

    repeat_batch = 257
    repeat_input_cpu = deterministic_input(torch, repeat_batch)
    repeat_input_gpu = repeat_input_cpu.to(device="cuda")
    with torch.no_grad():
        repeat_baseline = model_a(repeat_input_gpu)
        for _ in range(args.repeat_launches):
            repeated = model_a(repeat_input_gpu)
            if not bitwise_equal(torch, repeated, repeat_baseline):
                raise AssertionError(
                    "Repeated production launch was not bitwise stable."
                )
    torch.cuda.synchronize()

    # A -> B -> A hot swap must restore the exact A result.
    hot_batch = 65
    hot_input_cpu = deterministic_input(torch, hot_batch)
    hot_input_gpu = hot_input_cpu.to(device="cuda")
    with torch.no_grad():
        hot_a_before = model_a(hot_input_gpu)

    install_params(model_a, parameter_b)
    with torch.no_grad():
        hot_b = model_a(hot_input_gpu)

    install_params(model_a, parameter_a)
    with torch.no_grad():
        hot_a_after = model_a(hot_input_gpu)

    torch.cuda.synchronize()

    hot_swap = {
        "a_before_equals_a_after": bitwise_equal(
            torch,
            hot_a_before,
            hot_a_after,
        ),
        "a_differs_from_b": not bitwise_equal(
            torch,
            hot_a_before,
            hot_b,
        ),
    }
    assert all(hot_swap.values())

    expected_a = cpu_reference(torch, hot_input_cpu, parameter_a)
    expected_b = cpu_reference(torch, hot_input_cpu, parameter_b)
    hot_swap["a_metrics"] = compare(
        torch,
        hot_a_after,
        expected_a,
        "hot-swap-A",
    )
    hot_swap["b_metrics"] = compare(
        torch,
        hot_b,
        expected_b,
        "hot-swap-B",
    )

    # Two model instances must remain isolated while launched on two streams.
    stream_batch = 129
    stream_input_cpu = deterministic_input(torch, stream_batch)
    stream_input_gpu_a = stream_input_cpu.to(device="cuda")
    stream_input_gpu_b = stream_input_cpu.to(device="cuda")
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    with torch.cuda.stream(stream_a), torch.no_grad():
        stream_output_a = model_a(stream_input_gpu_a)
    with torch.cuda.stream(stream_b), torch.no_grad():
        stream_output_b = model_b(stream_input_gpu_b)

    stream_a.synchronize()
    stream_b.synchronize()

    stream_metrics = {
        "model_a": compare(
            torch,
            stream_output_a,
            cpu_reference(torch, stream_input_cpu, parameter_a),
            "stream-model-A",
        ),
        "model_b": compare(
            torch,
            stream_output_b,
            cpu_reference(torch, stream_input_cpu, parameter_b),
            "stream-model-B",
        ),
        "outputs_differ": not bitwise_equal(
            torch,
            stream_output_a,
            stream_output_b,
        ),
    }
    assert stream_metrics["outputs_differ"]

    # Re-run model A after concurrent execution to catch cross-model state.
    with torch.no_grad():
        isolation_a_after = model_a(stream_input_gpu_a)
    torch.cuda.synchronize()
    model_isolation = bitwise_equal(
        torch,
        isolation_a_after,
        stream_output_a,
    )
    assert model_isolation

    # Existing backends must remain constructible through the same extension.
    portable = tcnn.Network(
        WIDTH,
        WIDTH,
        {
            "otype": "PortableMLP",
            "n_neurons": WIDTH,
            "n_hidden_layers": 2,
            "activation": "ReLU",
            "output_activation": "None",
            "precision": "Fp32",
        },
    )
    hipblaslt_fp16 = tcnn.Network(
        WIDTH,
        WIDTH,
        {
            "otype": "HipBLASLtMLPFP16",
            "n_neurons": WIDTH,
            "n_hidden_layers": 2,
            "activation": "ReLU",
            "output_activation": "None",
            "precision": "Fp16",
        },
    )
    existing_factories = {
        "PortableMLP_params": int(portable.params.numel()),
        "HipBLASLtMLPFP16_params": int(hipblaslt_fp16.params.numel()),
    }
    assert all(value > 0 for value in existing_factories.values())

    training_input = deterministic_input(torch, 16).to(device="cuda")
    training_input.requires_grad_(True)
    training_error = expect_error(
        "training forward",
        lambda: model_a(training_input),
        "training forward is not qualified",
    )

    maximum_max_abs = max(
        float(value["max_abs"])
        for value in batch_results.values()
    )
    maximum_nl2 = max(
        float(value["normalized_l2"])
        for value in batch_results.values()
    )

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A2_P3_RUNTIME_INTEGRATION_LIFECYCLE_PROCESS_PASS"
        ),
        "environment": {
            "torch_hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
            "gcn_arch_name": architecture or "gfx1201",
        },
        "backend": backend,
        "parameter_abi": {
            "dtype": str(model_a.dtype),
            "elements": int(model_a.params.numel()),
            "set_a_offsets": parameter_a["offsets"],
            "set_b_offsets": parameter_b["offsets"],
        },
        "batch_results": batch_results,
        "prefix_checks": prefix_checks,
        "repeat_launches": {
            "batch": repeat_batch,
            "count": args.repeat_launches,
            "bitwise_stable": True,
        },
        "parameter_hot_swap": hot_swap,
        "dual_stream_model_isolation": {
            "metrics": stream_metrics,
            "model_a_replay_bitwise_equal": model_isolation,
        },
        "existing_factories": existing_factories,
        "training_forward_error": training_error,
        "metrics": {
            "maximum_max_abs": maximum_max_abs,
            "maximum_normalized_l2": maximum_nl2,
        },
        "tolerances": {
            "max_abs": MAX_ABS_TOLERANCE,
            "normalized_l2": NORMALIZED_L2_TOLERANCE,
        },
        "gates": {
            "twenty_batch_cases_correct": len(batch_results) == 20,
            "all_internal_padding_boundaries_exercised": (
                sorted(
                    {
                        value["internal_padded_batch"]
                        for value in batch_results.values()
                    }
                )
                == [256, 512, 768, 1024]
            ),
            "all_repeated_pairs_bitwise": all(
                value["repeat_bitwise_equal"]
                for value in batch_results.values()
            ),
            "prefix_invariance": all(prefix_checks.values()),
            "sixty_four_launch_replay_stable": True,
            "parameter_hot_swap_restores_exact_output": all(
                bool(value)
                for key, value in hot_swap.items()
                if key in (
                    "a_before_equals_a_after",
                    "a_differs_from_b",
                )
            ),
            "dual_nondefault_streams_correct": True,
            "two_model_instances_isolated": model_isolation,
            "existing_factories_construct": all(
                value > 0 for value in existing_factories.values()
            ),
            "training_forward_fail_closed": True,
            "inference_only_contract_preserved": (
                backend["inference_qualified"] is True
                and backend["forward_qualified"] is False
                and backend["backward_qualified"] is False
                and backend["silent_fallback"] is False
            ),
        },
    }
    assert all(result["gates"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    print("WIDTH64_RUNTIME_BATCH_MATRIX_20_CASES: PASS")
    print("WIDTH64_RUNTIME_PADDING_BOUNDARIES_256_1024: PASS")
    print("WIDTH64_RUNTIME_REPEAT_64_LAUNCHES_BITWISE: PASS")
    print("WIDTH64_RUNTIME_PREFIX_INVARIANCE: PASS")
    print("WIDTH64_RUNTIME_PARAMETER_HOT_SWAP_A_B_A: PASS")
    print("WIDTH64_RUNTIME_DUAL_STREAM_MODEL_ISOLATION: PASS")
    print("WIDTH64_RUNTIME_EXISTING_FACTORIES_CONSTRUCT: PASS")
    print("WIDTH64_RUNTIME_TRAINING_FORWARD_FAIL_CLOSED: PASS")
    print("PHASE4A2_P3_RUNTIME_INTEGRATION_LIFECYCLE_PROCESS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
