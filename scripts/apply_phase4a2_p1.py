#!/usr/bin/env python3
"""Apply Phase 4A2-P1 as a narrow, anchor-checked production patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001"
EXPECTED_HEAD_PREFIX = "977714b"
EXPECTED_HEAD_SUBJECT = (
    "Add Phase 4A2-P0 Width-64 production integration contract"
)

NEW_FILES = (
    "include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h",
    "src/rocwmma_width64_mlp.cu",
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


def check_new_files(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in NEW_FILES:
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"Bundle file is missing: {relative}")
        content = path.read_text()
        if MARKER not in content:
            raise RuntimeError(f"Bundle marker missing from {relative}")
        hashes[relative] = sha256(path)
    return hashes


def patch_setup(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        return

    anchor = '''if "--no-networks" in sys.argv:
\tinclude_networks = False
\tsys.argv.remove("--no-networks")
\tprint("Building >> without << neural networks (just the input encodings)")
'''
    insertion = anchor + '''
# TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: explicit default-OFF production
# skeleton. Invalid boolean values fail before compilation.
def _tcnn_env_flag(name, default=False):
\tvalue = os.environ.get(name)
\tif value is None:
\t\treturn default
\tnormalized = value.strip().lower()
\tif normalized in ["1", "true", "on", "yes"]:
\t\treturn True
\tif normalized in ["0", "false", "off", "no"]:
\t\treturn False
\traise EnvironmentError(f"{name} expects 0/1, false/true, off/on, or no/yes.")

enable_rocwmma_width64_mlp = _tcnn_env_flag(
\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP",
\tFalse,
)

if enable_rocwmma_width64_mlp:
\tif not is_rocm:
\t\traise EnvironmentError(
\t\t\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP is available only in the ROCm build."
\t\t)
\tif not include_networks:
\t\traise EnvironmentError(
\t\t\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP requires neural-network sources."
\t\t)
\tif rocm_arch != "gfx1201":
\t\traise EnvironmentError(
\t\t\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP requires PYTORCH_ROCM_ARCH=gfx1201."
\t\t)

print(
\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: "
\t+ ("ON" if enable_rocwmma_width64_mlp else "OFF")
)
'''
    text = replace_once(
        text,
        anchor,
        insertion,
        "setup opt-in parser",
    )

    anchor = '''else:
\tbase_definitions.append("-DTCNN_NO_NETWORKS")

# RTC/JIT is deliberately excluded from the ROCm encoding-only baseline.
'''
    insertion = '''else:
\tbase_definitions.append("-DTCNN_NO_NETWORKS")

# TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: source inclusion is explicit.
if is_rocm and include_networks and enable_rocwmma_width64_mlp:
\tbase_source_files.append("../../src/rocwmma_width64_mlp.cu")

# RTC/JIT is deliberately excluded from the ROCm encoding-only baseline.
'''
    text = replace_once(
        text,
        anchor,
        insertion,
        "setup source opt-in",
    )

    anchor = '''\t\tif include_networks:
\t\t\tdefinitions.append("-DTCNN_PORTABLE_MLP_ONLY")
\t\telse:
\t\t\tdefinitions.append("-DTCNN_NO_NETWORKS")
\t\thip_flags = base_nvcc_flags + [
'''
    insertion = '''\t\tif include_networks:
\t\t\tdefinitions.append("-DTCNN_PORTABLE_MLP_ONLY")
\t\telse:
\t\t\tdefinitions.append("-DTCNN_NO_NETWORKS")
\t\tif enable_rocwmma_width64_mlp:
\t\t\tdefinitions.append("-DTCNN_WITH_ROCWMMA_WIDTH64_MLP")
\t\thip_flags = base_nvcc_flags + [
'''
    text = replace_once(
        text,
        anchor,
        insertion,
        "setup compile definition",
    )

    path.write_text(text)


def patch_cpp_api(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        return

    old = '''\tif (equals_case_insensitive(network.value("otype", "PortableMLP"), "HipBLASLtMLPFP16")) {
\t\treturn new DifferentiableObject<__half>{new tcnn::NetworkWithInputEncoding<__half>{n_input_dims, n_output_dims, encoding, network}};
\t}
'''
    new = '''\t// TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: the explicit rocWMMA
\t// skeleton uses the existing FP16 parameter/output module wrapper.
\tconst std::string requested_otype = network.value("otype", "PortableMLP");
\tif (
\t\tequals_case_insensitive(requested_otype, "HipBLASLtMLPFP16") ||
\t\tequals_case_insensitive(requested_otype, "RocWMMAWidth64MLP")
\t) {
\t\treturn new DifferentiableObject<__half>{new tcnn::NetworkWithInputEncoding<__half>{n_input_dims, n_output_dims, encoding, network}};
\t}
'''
    text = replace_once(text, old, new, "cpp_api FP16 dispatch")
    path.write_text(text)


def patch_portable_network(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        return

    old = '''#include <tiny-cuda-nn/networks/portable_mlp.h>

#include <type_traits>
'''
    new = '''#include <tiny-cuda-nn/networks/portable_mlp.h>

#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
// TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: explicit opt-in only.
#include <tiny-cuda-nn/networks/rocwmma_width64_mlp.h>
#endif

#include <type_traits>
'''
    text = replace_once(text, old, new, "portable factory include")

    old = '''\tif (equals_case_insensitive(requested, "HipBLASLtMLPFP16")) return "HipBLASLtMLPFP16";
\tif (
'''
    new = '''\tif (equals_case_insensitive(requested, "HipBLASLtMLPFP16")) return "HipBLASLtMLPFP16";
\t// TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001: never aliases another backend.
\tif (equals_case_insensitive(requested, "RocWMMAWidth64MLP")) return "RocWMMAWidth64MLP";
\tif (
'''
    text = replace_once(text, old, new, "portable select_network")

    old = '''\tif (equals_case_insensitive(selected, "HipBLASLtMLPFP16")) return HipBLASLtMLPFP16::REQUIRED_ALIGNMENT();
\tthrow std::runtime_error{"AMD network selection failed."};
'''
    new = '''\tif (equals_case_insensitive(selected, "HipBLASLtMLPFP16")) return HipBLASLtMLPFP16::REQUIRED_ALIGNMENT();
\tif (equals_case_insensitive(selected, "RocWMMAWidth64MLP")) {
#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
\t\treturn RocWMMAWidth64MLP::REQUIRED_ALIGNMENT();
#else
\t\tthrow std::runtime_error{
\t\t\t"RocWMMAWidth64MLP was not compiled. Set "
\t\t\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 and rebuild."
\t\t};
#endif
\t}
\tthrow std::runtime_error{"AMD network selection failed."};
'''
    text = replace_once(text, old, new, "portable minimum_alignment")

    old = '''\tif constexpr (std::is_same<T,__half>::value) {
\t\tif (!equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
\t\t\tthrow std::runtime_error{"Only HipBLASLtMLPFP16 is available through the AMD FP16 network factory."};
\t\tif (!network.contains("precision") || !equals_case_insensitive(network.at("precision").get<std::string>(), "Fp16"))
\t\t\tthrow std::runtime_error{"HipBLASLtMLPFP16 requires precision=Fp16; no implicit precision selection is allowed."};
\t\treturn new HipBLASLtMLPFP16{
'''
    new = '''\tif constexpr (std::is_same<T,__half>::value) {
\t\tif (equals_case_insensitive(selected, "RocWMMAWidth64MLP")) {
\t\t\tif (!network.contains("precision") || !equals_case_insensitive(network.at("precision").get<std::string>(), "Fp16"))
\t\t\t\tthrow std::runtime_error{"RocWMMAWidth64MLP requires precision=Fp16; no implicit precision selection is allowed."};
\t\t\tif (!network.value("bias", true))
\t\t\t\tthrow std::runtime_error{"RocWMMAWidth64MLP requires bias=true."};
#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
\t\t\treturn new RocWMMAWidth64MLP{
\t\t\t\tnetwork.at("n_input_dims").get<uint32_t>(),
\t\t\t\tnetwork.value("n_neurons", 64u),
\t\t\t\tnetwork.at("n_output_dims").get<uint32_t>(),
\t\t\t\tnetwork.value("n_hidden_layers", 2u),
\t\t\t\tstring_to_activation(network.value("activation", "ReLU")),
\t\t\t\tstring_to_activation(network.value("output_activation", "None"))
\t\t\t};
#else
\t\t\tthrow std::runtime_error{
\t\t\t\t"RocWMMAWidth64MLP was not compiled. Set "
\t\t\t\t"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 and rebuild."
\t\t\t};
#endif
\t\t}
\t\tif (!equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
\t\t\tthrow std::runtime_error{"Only HipBLASLtMLPFP16 and the explicitly compiled RocWMMAWidth64MLP are available through the AMD FP16 network factory."};
\t\tif (!network.contains("precision") || !equals_case_insensitive(network.at("precision").get<std::string>(), "Fp16"))
\t\t\tthrow std::runtime_error{"HipBLASLtMLPFP16 requires precision=Fp16; no implicit precision selection is allowed."};
\t\treturn new HipBLASLtMLPFP16{
'''
    text = replace_once(text, old, new, "portable half factory")

    old = '''\tif constexpr (std::is_same<T,float>::value) {
\t\tif (equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
\t\t\tthrow std::runtime_error{"HipBLASLtMLPFP16 requires the explicit FP16 API path."};
'''
    new = '''\tif constexpr (std::is_same<T,float>::value) {
\t\tif (equals_case_insensitive(selected, "HipBLASLtMLPFP16"))
\t\t\tthrow std::runtime_error{"HipBLASLtMLPFP16 requires the explicit FP16 API path."};
\t\tif (equals_case_insensitive(selected, "RocWMMAWidth64MLP"))
\t\t\tthrow std::runtime_error{"RocWMMAWidth64MLP requires the explicit FP16 API path."};
'''
    text = replace_once(text, old, new, "portable float rejection")

    old = '''std::vector<std::string> builtin_networks() {
\treturn {"PortableMLP", "HipBLASLtMLP", "HipBLASLtMLPFP16"};
}
'''
    new = '''std::vector<std::string> builtin_networks() {
\tstd::vector<std::string> result{
\t\t"PortableMLP",
\t\t"HipBLASLtMLP",
\t\t"HipBLASLtMLPFP16",
\t};
#if defined(TCNN_WITH_ROCWMMA_WIDTH64_MLP)
\tresult.emplace_back("RocWMMAWidth64MLP");
#endif
\treturn result;
}
'''
    text = replace_once(text, old, new, "portable builtin list")
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--p0-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    p0 = json.loads(args.p0_json.read_text())

    if p0["decision"] != "PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_PASS":
        raise RuntimeError("Phase 4A2-P0 evidence did not pass.")
    if not all(bool(value) for value in p0["gates"].values()):
        raise RuntimeError("Phase 4A2-P0 evidence contains a failed gate.")

    head = run_git(repo, "rev-parse", "HEAD")
    subject = run_git(repo, "show", "-s", "--format=%s", "HEAD")
    if not head.startswith(EXPECTED_HEAD_PREFIX) or subject != EXPECTED_HEAD_SUBJECT:
        raise RuntimeError(
            f"Unexpected P1 base: {head} / {subject!r}. "
            "Expected the committed Phase 4A2-P0 checkpoint."
        )

    if run_git(repo, "diff", "--name-only") or run_git(
        repo, "diff", "--cached", "--name-only"
    ):
        raise RuntimeError(
            "Tracked files are already modified. P1 refuses to mix changes."
        )

    new_file_hashes = check_new_files(repo)

    targets = {
        "bindings/torch/setup.py": patch_setup,
        "src/cpp_api.cu": patch_cpp_api,
        "src/portable_network.cu": patch_portable_network,
    }

    before: dict[str, str] = {}
    after: dict[str, str] = {}
    originals: dict[str, bytes] = {}

    for relative in targets:
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"Required integration surface missing: {relative}")
        originals[relative] = path.read_bytes()
        before[relative] = sha256(path)

    try:
        for relative, patcher in targets.items():
            path = repo / relative
            patcher(path)
            after[relative] = sha256(path)
            if MARKER not in path.read_text():
                raise RuntimeError(
                    f"Patch marker missing after update: {relative}"
                )
    except Exception:
        for relative, content in originals.items():
            (repo / relative).write_bytes(content)
        raise

    if before == after:
        raise RuntimeError("P1 did not change any tracked integration surface.")

    manifest: dict[str, Any] = {
        "marker": MARKER,
        "decision": "PHASE4A2_P1_APPLY_PASS",
        "baseline": {
            "head": head,
            "subject": subject,
            "p0_json": str(args.p0_json.resolve()),
            "p0_json_sha256": sha256(args.p0_json),
        },
        "modified_surfaces": {
            relative: {
                "before_sha256": before[relative],
                "after_sha256": after[relative],
            }
            for relative in targets
        },
        "new_files": {
            relative: {"sha256": digest}
            for relative, digest in new_file_hashes.items()
        },
        "gates": {
            "p0_evidence_bound": True,
            "p0_commit_bound": True,
            "tracked_tree_clean_before_apply": True,
            "anchors_unique": True,
            "new_files_marked": True,
            "no_kernel_installed": "__global__" not in (
                repo / "src/rocwmma_width64_mlp.cu"
            ).read_text(),
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("P1 apply gate failed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("phase4a2_p0_commit: " + head)
    print("p0_json_sha256: " + manifest["baseline"]["p0_json_sha256"])
    print("PHASE4A2_P1_ANCHOR_PATCH: PASS")
    print("WIDTH64_P1_NO_PRODUCTION_KERNEL_INSTALLED: PASS")
    print("PHASE4A2_P1_APPLY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
