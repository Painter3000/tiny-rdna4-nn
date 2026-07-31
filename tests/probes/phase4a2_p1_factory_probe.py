#!/usr/bin/env python3
"""Fresh-process factory and fail-closed probe for Phase 4A2-P1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001"


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


def find_backend_hyperparams(value, target_otype):
    """Find the nested backend descriptor inside wrapper hyperparameters."""
    if isinstance(value, dict):
        if value.get("otype") == target_otype:
            return value
        for child in value.values():
            found = find_backend_hyperparams(child, target_otype)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_backend_hyperparams(child, target_otype)
            if found is not None:
                return found
    return None


def config(**updates):
    result = {
        "otype": "RocWMMAWidth64MLP",
        "n_neurons": 64,
        "n_hidden_layers": 2,
        "activation": "ReLU",
        "output_activation": "None",
        "precision": "Fp16",
        "bias": True,
    }
    result.update(updates)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("disabled", "enabled"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import tinycudann as tcnn

    result = {
        "marker": MARKER,
        "mode": args.mode,
        "torch_hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "gates": {},
        "messages": {},
    }

    if args.mode == "disabled":
        result["messages"]["disabled_factory"] = expect_error(
            "disabled factory",
            lambda: tcnn.Network(64, 64, config()),
            "RocWMMAWidth64MLP was not compiled",
        )

        portable = tcnn.Network(
            64,
            64,
            {
                "otype": "PortableMLP",
                "n_neurons": 64,
                "n_hidden_layers": 2,
                "activation": "ReLU",
                "output_activation": "None",
                "precision": "Fp32",
            },
        )
        fp16 = tcnn.Network(
            64,
            64,
            {
                "otype": "HipBLASLtMLPFP16",
                "n_neurons": 64,
                "n_hidden_layers": 2,
                "activation": "ReLU",
                "output_activation": "None",
                "precision": "Fp16",
            },
        )

        result["existing_backends"] = {
            "PortableMLP_params": int(portable.params.numel()),
            "HipBLASLtMLPFP16_params": int(fp16.params.numel()),
        }
        result["gates"] = {
            "default_off_factory_fails_closed": True,
            "portable_mlp_factory_unchanged": portable.params.numel() > 0,
            "hipblaslt_fp16_factory_unchanged": fp16.params.numel() > 0,
        }

    else:
        model = tcnn.Network(64, 64, config())
        wrapper_hyperparams = model.native_tcnn_module.hyperparams()
        hyperparams = find_backend_hyperparams(
            wrapper_hyperparams,
            "RocWMMAWidth64MLP",
        )
        if hyperparams is None:
            raise AssertionError(
                "Could not locate nested RocWMMAWidth64MLP hyperparameters: "
                + json.dumps(wrapper_hyperparams, sort_keys=True)
            )

        result["model"] = {
            "dtype": str(model.dtype),
            "parameter_elements": int(model.params.numel()),
            "wrapper_hyperparams": wrapper_hyperparams,
            "hyperparams": hyperparams,
        }

        assert model.dtype == torch.float16
        assert model.params.numel() == 12480
        assert hyperparams["otype"] == "RocWMMAWidth64MLP"
        assert hyperparams["runtime_architecture"] == "gfx1201"
        assert hyperparams["selection"] == "explicit_otype_only"
        assert hyperparams["silent_fallback"] is False
        assert hyperparams["inference_qualified"] is False
        assert hyperparams["forward_qualified"] is False
        assert hyperparams["backward_qualified"] is False

        invalid_cases = {
            "input_width": (
                lambda: tcnn.Network(32, 64, config()),
                "requires input, hidden, and output widths of 64",
            ),
            "output_width": (
                lambda: tcnn.Network(64, 32, config()),
                "requires input, hidden, and output widths of 64",
            ),
            "hidden_width": (
                lambda: tcnn.Network(64, 64, config(n_neurons=32)),
                "requires input, hidden, and output widths of 64",
            ),
            "hidden_layers": (
                lambda: tcnn.Network(
                    64,
                    64,
                    config(n_hidden_layers=1),
                ),
                "requires exactly two hidden layers",
            ),
            "hidden_activation": (
                lambda: tcnn.Network(
                    64,
                    64,
                    config(activation="None"),
                ),
                "requires hidden activation ReLU",
            ),
            "output_activation": (
                lambda: tcnn.Network(
                    64,
                    64,
                    config(output_activation="ReLU"),
                ),
                "requires output activation None",
            ),
            "precision": (
                lambda: tcnn.Network(
                    64,
                    64,
                    config(precision="Fp32"),
                ),
                "requires precision=Fp16",
            ),
            "bias": (
                lambda: tcnn.Network(
                    64,
                    64,
                    config(bias=False),
                ),
                "requires bias=true",
            ),
        }

        invalid_messages = {}
        for label, (function, substring) in invalid_cases.items():
            invalid_messages[label] = expect_error(
                label,
                function,
                substring,
            )
        result["messages"]["invalid_cases"] = invalid_messages

        x = torch.zeros(
            (16, 64),
            device="cuda",
            dtype=torch.float32,
        )

        with torch.no_grad():
            result["messages"]["inference_fail_closed"] = expect_error(
                "inference",
                lambda: model(x),
                "production inference kernel is not qualified",
            )

        result["messages"]["forward_fail_closed"] = expect_error(
            "forward",
            lambda: model(x),
            "production inference kernel is not qualified",
        )

        result["gates"] = {
            "explicit_factory_constructs": True,
            "fp16_parameter_precision": model.dtype == torch.float16,
            "parameter_abi_12480": model.params.numel() == 12480,
            "hyperparams_fail_closed": (
                hyperparams["silent_fallback"] is False
                and hyperparams["inference_qualified"] is False
                and hyperparams["forward_qualified"] is False
                and hyperparams["backward_qualified"] is False
            ),
            "all_invalid_configs_rejected": len(invalid_messages) == 8,
            "inference_fails_before_kernel": True,
            "forward_fails_before_kernel": True,
        }

    assert all(result["gates"].values())
    result["decision"] = (
        "PHASE4A2_P1_FACTORY_DISABLED_PASS"
        if args.mode == "disabled"
        else "PHASE4A2_P1_FACTORY_ENABLED_SKELETON_PASS"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.mode == "disabled":
        print("WIDTH64_DEFAULT_OFF_FACTORY_FAIL_CLOSED: PASS")
        print("WIDTH64_EXISTING_FACTORY_REGRESSION: PASS")
        print("PHASE4A2_P1_DISABLED_BUILD_FACTORY: PASS")
    else:
        print("WIDTH64_EXPLICIT_FACTORY_CONSTRUCTION: PASS")
        print("WIDTH64_PARAMETER_ABI_12480: PASS")
        print("WIDTH64_INVALID_CONFIG_FAIL_CLOSED: PASS")
        print("WIDTH64_INFERENCE_BEFORE_QUALIFICATION_FAIL_CLOSED: PASS")
        print("WIDTH64_FORWARD_BEFORE_QUALIFICATION_FAIL_CLOSED: PASS")
        print("PHASE4A2_P1_ENABLED_BUILD_FACTORY_SKELETON: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
