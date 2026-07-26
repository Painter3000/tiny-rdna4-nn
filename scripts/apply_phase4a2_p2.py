#!/usr/bin/env python3
'''Apply the Phase 4A2-P2 production inference bridge transactionally.'''

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001"
EXPECTED_HEAD_PREFIX = "c1b95f6"
EXPECTED_HEAD_SUBJECT = (
    "Add Phase 4A2-P1 rocWMMA Width-64 opt-in backend skeleton"
)
EXPECTED_P1_SOURCE_SHA256 = (
    "f989764e9aa6171e750daeed558fefc8baefb5c7b558f55b6689b97ef6587b8b"
)
EXPECTED_P4_SOURCE_SHA256 = (
    "54e03ee731046bb007d0c554c6d1e6ec2dea99d4f4fe150bb60563e36c4b3382"
)
EXPECTED_P4_HEADER_SHA256 = (
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"
)


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one anchor, found {count}."
        )
    return text.replace(old, new, 1)


def patch_network_with_input_encoding(path: Path) -> str:
    text = path.read_text()
    if MARKER in text:
        return text

    old = '''\t\tconst auto params = network->hyperparams();
\t\tif (equals_case_insensitive(params.value("otype", ""), "HipBLASLtMLPFP16")) {
\t\t\tthrow std::runtime_error{"NetworkWithInputEncoding(shared encoding, shared HipBLASLtMLPFP16) is unqualified; use the JSON/factory constructor."};
\t\t}
'''
    new = '''\t\tconst auto params = network->hyperparams();
\t\tconst std::string backend = params.value("otype", "");
\t\tif (
\t\t\tequals_case_insensitive(backend, "HipBLASLtMLPFP16") ||
\t\t\tequals_case_insensitive(backend, "RocWMMAWidth64MLP")
\t\t) {
\t\t\tthrow std::runtime_error{
\t\t\t\t"NetworkWithInputEncoding(shared encoding, shared explicit FP16 "
\t\t\t\t"backend) is unqualified; use the JSON/factory constructor."
\t\t\t};
\t\t}
'''
    text = replace_once(
        text,
        old,
        new,
        "shared explicit-FP16 constructor guard",
    )

    old = '''\tNetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims, const json& network) : m_encoding{encoding} {
\t\tm_fp16_hipblaslt = equals_case_insensitive(network.value("otype", ""), "HipBLASLtMLPFP16");
\t\tm_logical_encoding_width = encoding->output_width();
'''
    new = '''\tNetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims, const json& network) : m_encoding{encoding} {
\t\tm_fp16_hipblaslt = equals_case_insensitive(network.value("otype", ""), "HipBLASLtMLPFP16");
\t\t// TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001: the rocWMMA backend
\t\t// consumes the same contiguous ColumnMajor [width][batch] bridge.
\t\tm_fp16_rocwmma_width64 = equals_case_insensitive(
\t\t\tnetwork.value("otype", ""),
\t\t\t"RocWMMAWidth64MLP"
\t\t);
\t\tm_logical_encoding_width = encoding->output_width();
'''
    text = replace_once(
        text,
        old,
        new,
        "rocWMMA constructor classification",
    )

    old = '''\t\tif (m_fp16_hipblaslt) {
\t\t\t// TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001: zero padding is a
\t\t\t// backend-bound contract. Standalone encodings and all other
\t\t\t// backends retain the historical padding value of one.
\t\t\tencoding->set_padding_value((T)0.0f);
\t\t\t// TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001: the qualified
\t\t\t// hipBLASLt shapes are discrete, not merely multiples of sixteen.
\t\t\tconst uint32_t logical = m_logical_encoding_width;
\t\t\tif (logical > 128) {
\t\t\t\t// TCNN_RDNA4_P3B1E1_ENCODING_CLOSURE_001: no silent unsupported
\t\t\t\t// width, backend switch, or FP32 fallback is permitted.
\t\t\t\tthrow std::runtime_error{fmt::format("HipBLASLtMLPFP16 encoding width {} exceeds the supported maximum 128.", logical)};
\t\t\t}
\t\t\tconst uint32_t supported = logical <= 16 ? 16 : logical <= 32 ? 32 : logical <= 64 ? 64 : 128;
\t\t\tencoding->set_padded_output_width(supported);
\t\t}

\t\tjson local_network_config = network;
'''
    new = '''\t\tif (m_fp16_hipblaslt) {
\t\t\t// TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001: zero padding is a
\t\t\t// backend-bound contract. Standalone encodings and all other
\t\t\t// backends retain the historical padding value of one.
\t\t\tencoding->set_padding_value((T)0.0f);
\t\t\t// TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001: the qualified
\t\t\t// hipBLASLt shapes are discrete, not merely multiples of sixteen.
\t\t\tconst uint32_t logical = m_logical_encoding_width;
\t\t\tif (logical > 128) {
\t\t\t\t// TCNN_RDNA4_P3B1E1_ENCODING_CLOSURE_001: no silent unsupported
\t\t\t\t// width, backend switch, or FP32 fallback is permitted.
\t\t\t\tthrow std::runtime_error{fmt::format("HipBLASLtMLPFP16 encoding width {} exceeds the supported maximum 128.", logical)};
\t\t\t}
\t\t\tconst uint32_t supported = logical <= 16 ? 16 : logical <= 32 ? 32 : logical <= 64 ? 64 : 128;
\t\t\tencoding->set_padded_output_width(supported);
\t\t}

\t\tif (m_fp16_rocwmma_width64) {
\t\t\tencoding->set_padding_value((T)0.0f);
\t\t\tif (m_logical_encoding_width != 64) {
\t\t\t\tthrow std::runtime_error{fmt::format(
\t\t\t\t\t"RocWMMAWidth64MLP requires an encoding width of exactly "
\t\t\t\t\t"64, but received {}.",
\t\t\t\t\tm_logical_encoding_width
\t\t\t\t)};
\t\t\t}
\t\t\tencoding->set_padded_output_width(64);
\t\t}

\t\tjson local_network_config = network;
'''
    text = replace_once(
        text,
        old,
        new,
        "rocWMMA encoding width/layout contract",
    )

    old = '''\t\treturn m_fp16_hipblaslt ? MatrixLayout::ColumnMajor : m_encoding->preferred_output_layout();
'''
    new = '''\t\treturn (m_fp16_hipblaslt || m_fp16_rocwmma_width64)
\t\t\t? MatrixLayout::ColumnMajor
\t\t\t: m_encoding->preferred_output_layout();
'''
    text = replace_once(
        text,
        old,
        new,
        "rocWMMA network input layout",
    )

    old = '''\tbool m_fp16_hipblaslt = false;
\tuint32_t m_logical_encoding_width = 0;
'''
    new = '''\tbool m_fp16_hipblaslt = false;
\tbool m_fp16_rocwmma_width64 = false;
\tuint32_t m_logical_encoding_width = 0;
'''
    text = replace_once(
        text,
        old,
        new,
        "rocWMMA private state",
    )

    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--p1-json", type=Path, required=True)
    parser.add_argument("--p4-json", type=Path, required=True)
    parser.add_argument("--p4-preparation", type=Path, required=True)
    parser.add_argument("--p4-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract = json.loads(args.contract.read_text())
    p1 = json.loads(args.p1_json.read_text())
    p4 = json.loads(args.p4_json.read_text())
    p4_preparation = json.loads(args.p4_preparation.read_text())

    if contract["marker"] != MARKER:
        raise RuntimeError("P2 contract marker mismatch.")
    if p1["decision"] != (
        "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON_PASS"
    ):
        raise RuntimeError("Phase 4A2-P1 evidence did not pass.")
    if not all(bool(value) for value in p1["gates"].values()):
        raise RuntimeError("Phase 4A2-P1 evidence contains a failed gate.")
    if p4["decision"] != (
        "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_CONSOLIDATED_PASS"
    ):
        raise RuntimeError("Phase 4A1-P4 evidence did not pass.")
    if not all(bool(value) for value in p4["gates"].values()):
        raise RuntimeError("Phase 4A1-P4 evidence contains a failed gate.")
    if p4_preparation["decision"] != "PHASE4A1_P4_PREPARATION_PASS":
        raise RuntimeError("Phase 4A1-P4 preparation did not pass.")

    head = run_git(repo, "rev-parse", "HEAD")
    subject = run_git(repo, "show", "-s", "--format=%s", "HEAD")
    if not head.startswith(EXPECTED_HEAD_PREFIX) or subject != EXPECTED_HEAD_SUBJECT:
        raise RuntimeError(
            f"Unexpected P2 base: {head} / {subject!r}."
        )

    if run_git(repo, "diff", "--name-only") or run_git(
        repo,
        "diff",
        "--cached",
        "--name-only",
    ):
        raise RuntimeError(
            "Tracked files are already modified. P2 refuses to mix changes."
        )

    p1_source = repo / "src/rocwmma_width64_mlp.cu"
    reference_source = (
        repo / contract["baseline"]["p4_reference_source"]
    )
    network_bridge = (
        repo / "include/tiny-cuda-nn/network_with_input_encoding.h"
    )
    mapping_target = (
        repo
        / "include/tiny-cuda-nn/networks/"
        "rocwmma_width64_mapping_gfx1201.h"
    )

    if sha256(p1_source) != EXPECTED_P1_SOURCE_SHA256:
        raise RuntimeError("P1 skeleton source hash mismatch.")
    if sha256(reference_source) != EXPECTED_P4_SOURCE_SHA256:
        raise RuntimeError("Committed P4 reference source hash mismatch.")
    if sha256(args.p4_header) != EXPECTED_P4_HEADER_SHA256:
        raise RuntimeError("P4 mapping header hash mismatch.")
    if p4_preparation["generated_header"]["sha256"] != sha256(
        args.p4_header
    ):
        raise RuntimeError("P4 preparation/header binding mismatch.")
    if p4["preparation_manifest"]["sha256"] != sha256(
        args.p4_preparation
    ):
        raise RuntimeError("P4 final/preparation binding mismatch.")

    production_source = base64.b64decode(
        args.payload.read_text().strip()
    ).decode()
    if MARKER not in production_source:
        raise RuntimeError("Production source payload marker missing.")
    production_source_sha256 = hashlib.sha256(
        production_source.encode()
    ).hexdigest()
    if production_source_sha256 != contract["production_kernel"][
        "source_sha256"
    ]:
        raise RuntimeError("Production source payload hash mismatch.")

    originals = {
        p1_source: p1_source.read_bytes(),
        network_bridge: network_bridge.read_bytes(),
    }
    mapping_existed = mapping_target.exists()
    mapping_original = (
        mapping_target.read_bytes() if mapping_existed else None
    )

    try:
        p1_source.write_text(production_source)
        network_bridge.write_text(
            patch_network_with_input_encoding(network_bridge)
        )
        mapping_target.parent.mkdir(parents=True, exist_ok=True)
        mapping_target.write_bytes(args.p4_header.read_bytes())

        if sha256(mapping_target) != EXPECTED_P4_HEADER_SHA256:
            raise RuntimeError("Installed mapping header hash mismatch.")

        source_text = p1_source.read_text()
        bridge_text = network_bridge.read_text()
        mapping_text = mapping_target.read_text()

        source_gates = {
            "register_file_unpacked_size_api": (
                "static_assert(RegA::size() == SLOTS);" in source_text
                and "static_assert(RegAcc::size() == SLOTS);" in source_text
                and "::num_elements == SLOTS" not in source_text
            ),
            "register_file_size_device_pass_only": (
                "#if defined(__HIP_DEVICE_COMPILE__)" in source_text
                and "gfx1201/Wave32 register view only" in source_text
            ),
            "one_production_kernel": source_text.count("__global__ void") == 1,
            "three_mma_sites": source_text.count(
                "rocwmma::mma_sync("
            ) == 3,
            "three_source_barriers": source_text.count(
                "__syncthreads();"
            ) == 3,
            "one_lds_buffer": source_text.count(
                "__shared__ __align__(16) Half hidden_lds"
            ) == 1,
            "block_indexed_tiles": "blockIdx.x" in source_text,
            "fp16_parameter_bias": (
                "const __half* bias_0" in source_text
                and "static_cast<F32>(bias[global_col])" in source_text
            ),
            "fp16_public_output": (
                "accumulator_bias_to_matrix_a<false>" in source_text
                and "reinterpret_cast<Half*>(output.data())" in source_text
            ),
            "caller_stream_launch": (
                "hipLaunchKernelGGL(" in source_text
                and "stream," in source_text
            ),
            "no_host_sync": (
                "hipDeviceSynchronize" not in source_text
                and "hipStreamSynchronize" not in source_text
            ),
            "no_oracle_or_diagnostics": all(
                token not in source_text
                for token in (
                    "expected_hidden",
                    "diagnostics",
                    "atomicAdd(",
                )
            ),
            "training_fail_closed": (
                "training forward is not " in source_text
                and "qualified; use inference/no-grad." in source_text
                and "backward is not qualified" in source_text
                and "no fallback was executed" in source_text
            ),
            "network_bridge_column_major": (
                "m_fp16_rocwmma_width64" in bridge_text
                and "MatrixLayout::ColumnMajor" in bridge_text
            ),
            "validated_mapping_namespace": (
                "namespace phase4a1_p2_generated" in mapping_text
                and "kAccLaneForATargetA" in mapping_text
                and "kAccumulatorColumn" in mapping_text
            ),
        }
        if not all(source_gates.values()):
            raise RuntimeError(
                "P2 source gate failure: "
                + json.dumps(
                    {
                        key: value
                        for key, value in source_gates.items()
                        if not value
                    },
                    sort_keys=True,
                )
            )

    except Exception:
        for path, content in originals.items():
            path.write_bytes(content)
        if mapping_existed:
            assert mapping_original is not None
            mapping_target.write_bytes(mapping_original)
        elif mapping_target.exists():
            mapping_target.unlink()
        raise

    manifest: dict[str, Any] = {
        "marker": MARKER,
        "decision": "PHASE4A2_P2_APPLY_PASS",
        "baseline": {
            "head": head,
            "subject": subject,
            "p1_json": str(args.p1_json.resolve()),
            "p1_json_sha256": sha256(args.p1_json),
            "p4_json": str(args.p4_json.resolve()),
            "p4_json_sha256": sha256(args.p4_json),
            "p4_preparation": str(args.p4_preparation.resolve()),
            "p4_preparation_sha256": sha256(args.p4_preparation),
            "p4_reference_source_sha256": sha256(reference_source),
            "p4_mapping_header_sha256": sha256(args.p4_header),
        },
        "installed": {
            "production_source": {
                "path": str(p1_source.resolve()),
                "before_sha256": EXPECTED_P1_SOURCE_SHA256,
                "after_sha256": sha256(p1_source),
            },
            "network_bridge": {
                "path": str(network_bridge.resolve()),
                "before_sha256": hashlib.sha256(
                    originals[network_bridge]
                ).hexdigest(),
                "after_sha256": sha256(network_bridge),
            },
            "mapping_header": {
                "path": str(mapping_target.resolve()),
                "sha256": sha256(mapping_target),
                "copied_byte_exact": (
                    mapping_target.read_bytes()
                    == args.p4_header.read_bytes()
                ),
            },
        },
        "source_gates": source_gates,
        "gates": {
            "p1_evidence_bound": True,
            "p1_commit_bound": True,
            "p4_evidence_bound": True,
            "p4_source_bound": True,
            "p4_mapping_bound": True,
            "transactional_apply": True,
            "production_source_gates_pass": all(
                source_gates.values()
            ),
        },
    }

    if not all(manifest["gates"].values()):
        raise RuntimeError("P2 apply gate failed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    args.payload.unlink()

    print("phase4a2_p1_commit: " + head)
    print(
        "p4_reference_source_sha256: "
        + manifest["baseline"]["p4_reference_source_sha256"]
    )
    print(
        "mapping_header_sha256: "
        + manifest["baseline"]["p4_mapping_header_sha256"]
    )
    print("WIDTH64_P4_DATAFLOW_BOUND: PASS")
    print("WIDTH64_PARAMETER_ABI_ADAPTATION: PASS")
    print("WIDTH64_COLUMN_MAJOR_BATCH_BRIDGE: PASS")
    print("WIDTH64_NO_ORACLE_DIAGNOSTIC_ARGUMENTS: PASS")
    print("WIDTH64_INFERENCE_ONLY_FAIL_CLOSED_SCOPE: PASS")
    print("PHASE4A2_P2_APPLY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
