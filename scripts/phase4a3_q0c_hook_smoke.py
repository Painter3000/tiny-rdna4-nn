#!/usr/bin/env python3
"""Fail-fast smoke for the Q0c test-only native benchmark hook."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

MARKER = "TCNN_RDNA4_P4A3_Q0C_TOOL_REPAIR_003"
EXPECTED_API_BATCH_GRANULARITY = 256
PUBLIC_BATCHES = (1, 31, 128, 256, 257, 512, 1024, 4096, 16384)


def tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    import tinycudann as tcnn
    from tinycudann.modules import _C

    if not hasattr(_C.Module, "phase4a3_q0c_benchmark_inference"):
        raise RuntimeError("Q0c test-only native hook is missing")

    api_granularity = int(_C.batch_size_granularity())
    if api_granularity != EXPECTED_API_BATCH_GRANULARITY:
        raise RuntimeError(
            f"Expected API batch granularity {EXPECTED_API_BATCH_GRANULARITY}, "
            f"got {api_granularity}"
        )

    def model(otype, seed):
        return tcnn.Network(
            64,
            64,
            {
                "otype": otype,
                "precision": "Fp16",
                "n_neurons": 64,
                "n_hidden_layers": 2,
                "activation": "ReLU",
                "output_activation": "None",
            },
            seed=seed,
        ).cuda().eval()

    candidate = model("RocWMMAWidth64MLP", 2026072701)
    reference = model("HipBLASLtMLPFP16", 2026072701)

    generator = torch.Generator(device="cuda").manual_seed(2026072702)
    master = torch.randn(
        12480,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) * 0.03
    with torch.no_grad():
        candidate.params.copy_(master)
        reference.params.copy_(master)

    results = {}
    for name, instance in (
        ("rocwmma", candidate),
        ("hipblaslt", reference),
    ):
        if instance.params.dtype != torch.float32:
            raise RuntimeError(f"{name}: public master parameters are not FP32")
        native_params = (
            instance.params.detach()
            .to(dtype=instance.dtype)
            .contiguous()
        )
        if native_params.dtype != torch.float16:
            raise RuntimeError(f"{name}: native parameters are not FP16")

        batch_results = {}
        for public_batch in PUBLIC_BATCHES:
            x = torch.randn(
                public_batch,
                64,
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            padded_batch = (
                (public_batch + api_granularity - 1)
                // api_granularity
                * api_granularity
            )
            x32 = torch.nn.functional.pad(
                x, [0, 0, 0, padded_batch - public_batch]
            ).to(torch.float32).contiguous()
            if x32.shape != (padded_batch, 64):
                raise RuntimeError(
                    f"{name}/b{public_batch}: malformed native input shape "
                    f"{tuple(x32.shape)}"
                )
            if padded_batch % api_granularity != 0:
                raise RuntimeError(
                    f"{name}/b{public_batch}: padded batch {padded_batch} "
                    f"is not divisible by {api_granularity}"
                )

            with torch.inference_mode():
                public_output = instance(x)
                single_record, native_output = (
                    instance.native_tcnn_module
                    .phase4a3_q0c_benchmark_inference(
                        x32,
                        native_params,
                        2,
                        True,
                        0,
                    )
                )
                queued_record, queued_output = (
                    instance.native_tcnn_module
                    .phase4a3_q0c_benchmark_inference(
                        x32,
                        native_params,
                        8,
                        False,
                        0,
                    )
                )
            torch.cuda.current_stream().synchronize()

            native_prefix = native_output[:public_batch]
            if not torch.equal(public_output, native_prefix):
                max_abs = float(
                    (public_output.float() - native_prefix.float())
                    .abs()
                    .max()
                )
                raise RuntimeError(
                    f"{name}/b{public_batch}: native hook does not match "
                    f"public path; max_abs={max_abs}"
                )

            single_record = dict(single_record)
            queued_record = dict(queued_record)
            for record_name, record in (
                ("single", single_record),
                ("queued", queued_record),
            ):
                if int(record.get("batch_size", -1)) != padded_batch:
                    raise RuntimeError(
                        f"{name}/b{public_batch}: {record_name} hook recorded "
                        f"batch {record.get('batch_size')}, expected {padded_batch}"
                    )
                if int(record.get("batch_size_granularity", -1)) != api_granularity:
                    raise RuntimeError(
                        f"{name}/b{public_batch}: {record_name} hook recorded "
                        f"granularity {record.get('batch_size_granularity')}, "
                        f"expected {api_granularity}"
                    )
            if len(single_record.get("host_ns", [])) != 2:
                raise RuntimeError(
                    f"{name}/b{public_batch}: malformed single-shot record"
                )
            if float(queued_record.get("event_ms", 0.0)) <= 0.0:
                raise RuntimeError(
                    f"{name}/b{public_batch}: malformed queued event record"
                )

            batch_results[str(public_batch)] = {
                "public_batch": public_batch,
                "api_padded_batch": padded_batch,
                "padding_factor": padded_batch / public_batch,
                "public_output_sha256": tensor_sha256(public_output),
                "native_output_sha256": tensor_sha256(native_prefix),
                "queued_output_sha256": tensor_sha256(
                    queued_output[:public_batch]
                ),
                "single_host_samples": len(single_record["host_ns"]),
                "queued_event_ms": float(queued_record["event_ms"]),
                "queued_submission_ns": int(
                    queued_record["host_submission_ns"]
                ),
            }

        results[name] = {
            "public_params_dtype": str(instance.params.dtype),
            "native_params_dtype": str(native_params.dtype),
            "native_params_sha256": tensor_sha256(native_params),
            "batches": batch_results,
        }

    output = {
        "marker": MARKER,
        "status": "PASS",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "api_batch_size_granularity": api_granularity,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"PHASE4A3_Q0C_API_BATCH_GRANULARITY: {api_granularity}")
    print("PHASE4A3_Q0C_NATIVE_HOOK_SMOKE: PASS")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
