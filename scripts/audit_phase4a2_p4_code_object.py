#!/usr/bin/env python3
"""Extract and audit the gfx1201 code object for the P4 production kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable

MARKER = "TCNN_RDNA4_P4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_001"
KERNEL_TOKEN = "rocwmma_width64_inference_kernel"
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_tool(candidates: Iterable[str]) -> str:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.is_file() and path.stat().st_mode & 0o111:
            return str(path)
    raise RuntimeError(
        "None of the required tool candidates was found: "
        + ", ".join(candidates)
    )


def run(
    command: list[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if check and result.returncode != 0:
        stdout = result.stdout if text else b"<binary>"
        stderr = result.stderr if text else b"<binary>"
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + f"\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def dump_section(
    objcopy: str,
    source: Path,
    section: str,
    destination: Path,
) -> bool:
    destination.unlink(missing_ok=True)
    result = run(
        [
            objcopy,
            "--dump-section",
            f"{section}={destination}",
            str(source),
        ],
        check=False,
    )
    return (
        result.returncode == 0
        and destination.is_file()
        and destination.stat().st_size > 0
    )


def parse_simple_clang_bundle(data: bytes) -> list[dict[str, Any]]:
    start = data.find(BUNDLE_MAGIC)
    if start < 0:
        return []

    cursor = start + len(BUNDLE_MAGIC)
    if cursor + 8 > len(data):
        return []

    count = struct.unpack_from("<Q", data, cursor)[0]
    cursor += 8
    if count == 0 or count > 128:
        return []

    entries = []
    for _ in range(count):
        if cursor + 24 > len(data):
            return []
        offset, size, identifier_size = struct.unpack_from(
            "<QQQ",
            data,
            cursor,
        )
        cursor += 24
        if identifier_size > 4096 or cursor + identifier_size > len(data):
            return []
        identifier = data[
            cursor : cursor + identifier_size
        ].decode(errors="replace")
        cursor += identifier_size
        entries.append(
            {
                "id": identifier,
                "offset": int(offset),
                "size": int(size),
            }
        )

    valid = []
    for entry in entries:
        offset = entry["offset"]
        size = entry["size"]
        if offset + size <= len(data):
            valid.append(entry)
    return valid


def bundler_list(
    bundler: str,
    payload: Path,
) -> tuple[list[str], list[str]]:
    attempts = [
        [
            bundler,
            "--list",
            "--type=o",
            f"--inputs={payload}",
        ],
        [
            bundler,
            "-list",
            "-type=o",
            f"-inputs={payload}",
        ],
    ]
    logs = []
    for command in attempts:
        result = run(command, check=False)
        logs.append(
            "$ "
            + " ".join(command)
            + "\n"
            + result.stdout
            + result.stderr
        )
        if result.returncode == 0:
            ids = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            if ids:
                return ids, logs
    return [], logs


def bundler_unbundle(
    bundler: str,
    payload: Path,
    target: str,
    output: Path,
) -> tuple[bool, list[str]]:
    attempts = [
        [
            bundler,
            "--unbundle",
            "--type=o",
            f"--inputs={payload}",
            f"--targets={target}",
            f"--outputs={output}",
        ],
        [
            bundler,
            "-unbundle",
            "-type=o",
            f"-inputs={payload}",
            f"-targets={target}",
            f"-outputs={output}",
        ],
    ]
    logs = []
    for command in attempts:
        output.unlink(missing_ok=True)
        result = run(command, check=False)
        logs.append(
            "$ "
            + " ".join(command)
            + "\n"
            + result.stdout
            + result.stderr
        )
        if (
            result.returncode == 0
            and output.is_file()
            and output.stat().st_size > 0
        ):
            return True, logs
    return False, logs


def extract_code_object(
    *,
    object_path: Path,
    output_dir: Path,
    objcopy: str,
    bundler: str,
) -> dict[str, Any]:
    extraction_logs = []
    candidates = []

    for section in (".hip_fatbin", ".llvm.offloading"):
        destination = output_dir / (
            section.lstrip(".").replace(".", "_") + ".bin"
        )
        if dump_section(objcopy, object_path, section, destination):
            candidates.append(
                {
                    "section": section,
                    "path": destination,
                }
            )

    # Some HIP toolchains leave a bundle directly in the object passed here.
    candidates.append({"section": "<object>", "path": object_path})

    for candidate in candidates:
        payload = candidate["path"]
        data = payload.read_bytes()

        if data.startswith(b"\x7fELF") and candidate["section"] != "<object>":
            output = output_dir / "gfx1201_code_object.hsaco"
            shutil.copyfile(payload, output)
            return {
                "method": "raw_elf_section",
                "source_section": candidate["section"],
                "bundle_target": "raw-elf-gfx1201-unverified",
                "bundle_ids": [],
                "code_object": output,
                "logs": extraction_logs,
            }

        ids, logs = bundler_list(bundler, payload)
        extraction_logs.extend(logs)
        target_ids = [
            identifier
            for identifier in ids
            if "gfx1201" in identifier
            and (
                identifier.startswith("hip-")
                or identifier.startswith("hipv4-")
            )
        ]
        if len(target_ids) == 1:
            output = output_dir / "gfx1201_code_object.hsaco"
            success, logs = bundler_unbundle(
                bundler,
                payload,
                target_ids[0],
                output,
            )
            extraction_logs.extend(logs)
            if success and output.read_bytes().startswith(b"\x7fELF"):
                return {
                    "method": "clang_offload_bundler",
                    "source_section": candidate["section"],
                    "bundle_target": target_ids[0],
                    "bundle_ids": ids,
                    "code_object": output,
                    "logs": extraction_logs,
                }

        entries = parse_simple_clang_bundle(data)
        target_entries = [
            entry
            for entry in entries
            if "gfx1201" in entry["id"]
            and (
                entry["id"].startswith("hip-")
                or entry["id"].startswith("hipv4-")
            )
        ]
        if len(target_entries) == 1:
            entry = target_entries[0]
            output = output_dir / "gfx1201_code_object.hsaco"
            output.write_bytes(
                data[
                    entry["offset"] :
                    entry["offset"] + entry["size"]
                ]
            )
            if output.read_bytes().startswith(b"\x7fELF"):
                return {
                    "method": "python_clang_bundle_parser",
                    "source_section": candidate["section"],
                    "bundle_target": entry["id"],
                    "bundle_ids": [
                        value["id"] for value in entries
                    ],
                    "code_object": output,
                    "logs": extraction_logs,
                }

    raise RuntimeError(
        "Unable to extract exactly one gfx1201 HIP code object.\n"
        + "\n".join(extraction_logs)
    )


def parse_labels(disassembly: str) -> list[tuple[int, str, int]]:
    labels = []
    pattern = re.compile(
        r"^\s*([0-9a-fA-F]+)\s+<(.+)>:\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(disassembly):
        labels.append(
            (
                match.start(),
                match.group(2),
                match.end(),
            )
        )
    return labels


def extract_kernel_block(disassembly: str) -> tuple[str, str]:
    labels = parse_labels(disassembly)
    matches = [
        (index, value)
        for index, value in enumerate(labels)
        if KERNEL_TOKEN in value[1] and not value[1].endswith(".kd")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one disassembly label containing "
            f"{KERNEL_TOKEN!r}; found {len(matches)}."
        )

    label_index, label = matches[0]
    start = label[0]
    end = (
        labels[label_index + 1][0]
        if label_index + 1 < len(labels)
        else len(disassembly)
    )
    return label[1], disassembly[start:end]


def instruction_mnemonics(kernel_block: str) -> list[str]:
    """Parse both AMD-style and address-prefixed llvm-objdump syntax.

    ROCm's llvm-objdump commonly prints the mnemonic first and places the
    address/encoding after ``//``:

        s_load_dword ... // 0000000000000000: C002...

    Other LLVM builds print an address before the instruction:

        0000000000000000: ... s_load_dword ...

    The previous parser accepted only the second form and therefore recorded
    zero instructions even though the production-kernel block was correct.
    """

    mnemonics = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.])("
        r"(?:v|s|ds|global|flat|buffer|scratch)"
        r"_[A-Za-z0-9_.]+"
        r")\b"
    )

    for raw_line in kernel_block.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if re.match(r"^[0-9a-fA-F]+\s+<.+>:$", stripped):
            continue

        # TCNN_RDNA4_P4A2_P4_AMDGPU_OBJDUMP_SYNTAX_005:
        # Strip address/encoding comments before searching, while retaining
        # GNU/address-prefixed instruction text on the left side.
        instruction_text = raw_line.split("//", 1)[0]
        instruction_text = instruction_text.split(";", 1)[0]

        match = pattern.search(instruction_text)
        if match:
            mnemonics.append(match.group(1).lower())

    return mnemonics


def find_resource_block(
    disassembly: str,
    raw_symbol: str,
) -> str:
    pattern = re.compile(
        r"\.amdhsa_kernel\s+([^\s]+)(.*?)"
        r"\.end_amdhsa_kernel",
        re.DOTALL,
    )
    blocks = []
    for match in pattern.finditer(disassembly):
        name = match.group(1)
        block = match.group(0)
        if (
            raw_symbol in name
            or KERNEL_TOKEN in name
            or KERNEL_TOKEN in block
        ):
            blocks.append(block)
    if len(blocks) == 1:
        return blocks[0]
    return ""


def integer_field(text: str, patterns: list[str]) -> int | None:
    for expression in patterns:
        match = re.search(expression, text)
        if match:
            return int(match.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(args.contract.read_text())
    expected = contract["code_object_audit"]

    objcopy = find_tool(
        (
            "/opt/rocm/llvm/bin/llvm-objcopy",
            "llvm-objcopy",
        )
    )
    objdump = find_tool(
        (
            "/opt/rocm/llvm/bin/llvm-objdump",
            "llvm-objdump",
        )
    )
    readelf = find_tool(
        (
            "/opt/rocm/llvm/bin/llvm-readelf",
            "llvm-readelf",
        )
    )
    # llvm-nm is not shipped by every ROCm LLVM package. Symbol inventory
    # therefore uses llvm-readelf for raw names and llvm-objdump for the
    # optional demangled view; both tools are already mandatory below.
    bundler = find_tool(
        (
            "/opt/rocm/llvm/bin/clang-offload-bundler",
            "clang-offload-bundler",
        )
    )

    extraction = extract_code_object(
        object_path=args.object,
        output_dir=args.output_dir,
        objcopy=objcopy,
        bundler=bundler,
    )
    code_object: Path = extraction["code_object"]

    (args.output_dir / "extraction.log").write_text(
        "\n\n".join(extraction["logs"]) + "\n"
    )
    (args.output_dir / "bundle_ids.txt").write_text(
        "\n".join(extraction["bundle_ids"]) + "\n"
    )

    elf_header = run(
        [readelf, "--file-header", str(code_object)]
    ).stdout
    (args.output_dir / "elf_header.txt").write_text(elf_header)

    if "AMDGPU" not in elf_header.upper():
        raise RuntimeError(
            "Extracted code object is not reported as an AMDGPU ELF."
        )

    # TCNN_RDNA4_P4A2_P4_SYMBOL_TOOL_FALLBACK_002:
    # llvm-readelf --symbols --wide is a portable raw-symbol fallback for
    # ROCm installations that omit llvm-nm. llvm-objdump supplies a
    # demangled audit view when supported.
    raw_symbols = run(
        [readelf, "--symbols", "--wide", str(code_object)]
    ).stdout

    demangled_result = run(
        [objdump, "--syms", "--demangle", str(code_object)],
        check=False,
    )
    demangled_symbols = (
        demangled_result.stdout
        if demangled_result.returncode == 0
        else raw_symbols
    )

    (args.output_dir / "symbols_raw.txt").write_text(raw_symbols)
    (args.output_dir / "symbols_demangled.txt").write_text(
        demangled_symbols
    )

    # TCNN_RDNA4_P4A2_P4_KERNEL_SYMBOL_COMPANION_FILTER_004:
    # AMDGPU code objects may expose compiler-generated metadata companions
    # beside the actual kernel symbol, for example:
    #   <kernel>.num_vgpr
    #   <kernel>.num_agpr
    #   <kernel>.private_seg_size
    #   <kernel>.has_recursion
    #   <kernel>.has_dyn_sized_stack
    # These are scalar metadata symbols, not additional executable kernels.
    raw_kernel_symbol_candidates = []
    for line in raw_symbols.splitlines():
        if KERNEL_TOKEN not in line:
            continue
        fields = line.split()
        if not fields or "UND" in fields:
            continue
        raw_kernel_symbol_candidates.append(fields[-1])

    raw_kernel_symbol_candidates = sorted(
        set(raw_kernel_symbol_candidates)
    )
    raw_kernel_symbols = sorted(
        symbol
        for symbol in raw_kernel_symbol_candidates
        if "." not in symbol
    )
    metadata_companion_symbols = sorted(
        symbol
        for symbol in raw_kernel_symbol_candidates
        if "." in symbol
    )

    if len(raw_kernel_symbols) != 1:
        raise RuntimeError(
            "Expected exactly one executable raw kernel symbol after "
            "metadata-companion filtering; executable="
            + repr(raw_kernel_symbols)
            + ", companions="
            + repr(metadata_companion_symbols)
        )
    raw_symbol = raw_kernel_symbols[0]

    if not all(
        symbol.startswith(raw_symbol + ".")
        for symbol in metadata_companion_symbols
    ):
        raise RuntimeError(
            "Kernel metadata companion did not share the exact kernel "
            "symbol prefix: "
            + repr(metadata_companion_symbols)
        )

    disassembly_result = run(
        [
            objdump,
            "--disassemble",
            "--demangle",
            "--mcpu=gfx1201",
            str(code_object),
        ],
        check=False,
    )
    if disassembly_result.returncode != 0:
        disassembly_result = run(
            [
                objdump,
                "--disassemble",
                "--demangle",
                str(code_object),
            ]
        )
    disassembly = disassembly_result.stdout
    (args.output_dir / "code_object.disassembly.txt").write_text(
        disassembly
    )

    kernel_label, kernel_block = extract_kernel_block(disassembly)
    (args.output_dir / "production_kernel.isa.txt").write_text(
        kernel_block
    )

    metadata_result = run(
        [
            objdump,
            "--amdgpu-code-object-metadata",
            str(code_object),
        ],
        check=False,
    )
    metadata_text = metadata_result.stdout + metadata_result.stderr
    if metadata_result.returncode != 0:
        metadata_text = run(
            [readelf, "--notes", str(code_object)],
            check=False,
        ).stdout
    (args.output_dir / "code_object.metadata.txt").write_text(
        metadata_text
    )

    resource_block = find_resource_block(
        disassembly,
        raw_symbol,
    )
    (args.output_dir / "production_kernel.resources.txt").write_text(
        resource_block
    )
    resource_text = resource_block + "\n" + metadata_text

    next_free_vgpr = integer_field(
        resource_text,
        [
            r"\.amdhsa_next_free_vgpr\s+(\d+)",
            r"\.vgpr_count:\s*(\d+)",
            r'"\.vgpr_count"\s*:\s*(\d+)',
        ],
    )
    next_free_sgpr = integer_field(
        resource_text,
        [
            r"\.amdhsa_next_free_sgpr\s+(\d+)",
            r"\.sgpr_count:\s*(\d+)",
            r'"\.sgpr_count"\s*:\s*(\d+)',
        ],
    )
    group_segment = integer_field(
        resource_text,
        [
            r"\.amdhsa_group_segment_fixed_size\s+(\d+)",
            r"\.group_segment_fixed_size:\s*(\d+)",
            r'"\.group_segment_fixed_size"\s*:\s*(\d+)',
        ],
    )
    private_segment = integer_field(
        resource_text,
        [
            r"\.amdhsa_private_segment_fixed_size\s+(\d+)",
            r"\.private_segment_fixed_size:\s*(\d+)",
            r'"\.private_segment_fixed_size"\s*:\s*(\d+)',
        ],
    )

    mnemonics = instruction_mnemonics(kernel_block)
    counts: dict[str, int] = {}
    for mnemonic in mnemonics:
        counts[mnemonic] = counts.get(mnemonic, 0) + 1

    mfma_or_wmma = sum(
        count
        for mnemonic, count in counts.items()
        if mnemonic.startswith("v_mfma")
        or mnemonic.startswith("v_wmma")
    )
    # TCNN_RDNA4_P4A2_P4_PRODUCTION_DS_INVENTORY_006:
    # Count LDS memory transfers separately from DS cross-lane permutations.
    lds_load_instructions = sum(
        count
        for mnemonic, count in counts.items()
        if mnemonic.startswith("ds_load")
        or mnemonic.startswith("ds_read")
    )
    lds_store_instructions = sum(
        count
        for mnemonic, count in counts.items()
        if mnemonic.startswith("ds_store")
        or mnemonic.startswith("ds_write")
    )
    ds_bpermute_b32_instructions = counts.get(
        "ds_bpermute_b32",
        0,
    )
    barriers = sum(
        count
        for mnemonic, count in counts.items()
        if mnemonic.startswith("s_barrier")
    )
    scratch = sum(
        count
        for mnemonic, count in counts.items()
        if mnemonic.startswith("scratch_")
        or mnemonic.startswith("flat_scratch")
    )
    global_memory = {
        mnemonic: count
        for mnemonic, count in sorted(counts.items())
        if mnemonic.startswith(
            (
                "global_load",
                "global_store",
                "flat_load",
                "flat_store",
                "buffer_load",
                "buffer_store",
            )
        )
    }
    ds_mnemonics = sorted(
        mnemonic
        for mnemonic in counts
        if mnemonic.startswith("ds_")
    )

    gates = {
        "contract_marker": contract["marker"] == MARKER,
        "exactly_one_kernel_symbol": (
            len(raw_kernel_symbols)
            == contract["kernel"]["expected_kernel_symbols"]
        ),
        "metadata_companions_classified": all(
            symbol.startswith(raw_symbol + ".")
            for symbol in metadata_companion_symbols
        ),
        "kernel_label_found": KERNEL_TOKEN in kernel_label,
        "amdgpu_objdump_instructions_parsed": len(mnemonics) > 0,
        "gfx1201_bundle_target": (
            "gfx1201" in extraction["bundle_target"]
            or "gfx1201" in elf_header
            or "gfx1201" in metadata_text
        ),
        "group_segment_2048": (
            group_segment
            == expected["group_segment_fixed_size"]
        ),
        "private_segment_zero": (
            private_segment
            == expected["private_segment_fixed_size"]
        ),
        "resource_register_counts_recorded": (
            next_free_vgpr is not None
            and next_free_sgpr is not None
            and next_free_vgpr > 0
            and next_free_sgpr > 0
        ),
        "twelve_mfma_or_wmma": (
            mfma_or_wmma
            == expected["mfma_or_wmma_instructions"]
        ),
        "eight_lds_load_instructions": (
            lds_load_instructions
            == expected["lds_load_instructions"]
        ),
        "two_lds_store_instructions": (
            lds_store_instructions
            == expected["lds_store_instructions"]
        ),
        "one_hundred_ninety_two_ds_bpermute_b32": (
            ds_bpermute_b32_instructions
            == expected["ds_bpermute_b32_instructions"]
        ),
        "six_block_barriers": (
            barriers == expected["block_barriers"]
        ),
        "no_scratch_instructions": (
            scratch == expected["scratch_instruction_count"]
        ),
        "exact_ds_mnemonic_inventory": (
            ds_mnemonics
            == sorted(expected["required_ds_mnemonics"])
        ),
        "global_memory_traffic_recorded": bool(global_memory),
    }

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_PASS"
            if all(gates.values())
            else "PHASE4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_FAIL"
        ),
        "tools": {
            "llvm_objcopy": objcopy,
            "llvm_objdump": objdump,
            "llvm_readelf": readelf,
            "raw_symbol_inventory": (
                f"{readelf} --symbols --wide"
            ),
            "demangled_symbol_inventory": (
                f"{objdump} --syms --demangle"
            ),
            "llvm_nm_required": False,
            "clang_offload_bundler": bundler,
        },
        "inputs": {
            "object": str(args.object.resolve()),
            "object_sha256": sha256(args.object),
            "contract": str(args.contract.resolve()),
            "contract_sha256": sha256(args.contract),
        },
        "extraction": {
            "method": extraction["method"],
            "source_section": extraction["source_section"],
            "bundle_target": extraction["bundle_target"],
            "bundle_ids": extraction["bundle_ids"],
            "code_object": str(code_object.resolve()),
            "code_object_sha256": sha256(code_object),
        },
        "kernel": {
            "raw_symbol": raw_symbol,
            "demangled_label": kernel_label,
            "metadata_companion_symbols": metadata_companion_symbols,
            "metadata_companion_count": len(
                metadata_companion_symbols
            ),
        },
        "resources": {
            "next_free_vgpr_or_vgpr_count": next_free_vgpr,
            "next_free_sgpr_or_sgpr_count": next_free_sgpr,
            "group_segment_fixed_size": group_segment,
            "private_segment_fixed_size": private_segment,
        },
        "isa": {
            "instruction_parser": (
                "AMD-mnemonic-first and address-prefixed llvm-objdump"
            ),
            "parsed_instruction_count": len(mnemonics),
            "mfma_or_wmma_instructions": mfma_or_wmma,
            "lds_load_instructions": lds_load_instructions,
            "lds_store_instructions": lds_store_instructions,
            "ds_bpermute_b32_instructions": (
                ds_bpermute_b32_instructions
            ),
            "block_barriers": barriers,
            "scratch_instruction_count": scratch,
            "ds_mnemonics": ds_mnemonics,
            "global_memory_mnemonics": global_memory,
            "all_mnemonic_counts": dict(sorted(counts.items())),
        },
        "claim_boundaries": {
            "register_counts_do_not_establish_occupancy": True,
            "global_mnemonics_do_not_establish_pointer_provenance": True,
            "performance_not_measured": True,
        },
        "gates": gates,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    if not all(gates.values()):
        failed = {
            key: value
            for key, value in gates.items()
            if not value
        }
        raise RuntimeError(
            "P4 code-object audit gates failed: "
            + json.dumps(failed, sort_keys=True)
        )

    print("WIDTH64_GFX1201_CODE_OBJECT_EXTRACTED: PASS")
    print("WIDTH64_EXACT_PRODUCTION_KERNEL_SYMBOL: PASS")
    print("WIDTH64_KERNEL_METADATA_COMPANIONS_CLASSIFIED: PASS")
    print("WIDTH64_AMDGPU_OBJDUMP_SYNTAX_PARSED: PASS")
    print("WIDTH64_RESOURCE_METADATA_RECORDED: PASS")
    print("WIDTH64_GROUP_SEGMENT_2048: PASS")
    print("WIDTH64_PRIVATE_SEGMENT_ZERO: PASS")
    print("WIDTH64_MFMA_OR_WMMA_12: PASS")
    print("WIDTH64_LDS_LOADS_8_STORES_2: PASS")
    print("WIDTH64_DS_BPERMUTE_B32_192: PASS")
    print("WIDTH64_EXACT_DS_MNEMONIC_INVENTORY: PASS")
    print("WIDTH64_BLOCK_BARRIERS_6: PASS")
    print("WIDTH64_SCRATCH_INSTRUCTIONS_ZERO: PASS")
    print("WIDTH64_GLOBAL_MEMORY_MNEMONICS_RECORDED: PASS")
    print("PHASE4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
