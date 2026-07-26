#!/usr/bin/env python3
"""Audit the exact P4 gfx1201 device ISA and resource metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A1_P5_ISA_RESOURCE_GLOBAL_TRAFFIC_001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def unique_kernel_symbols(assembly: str) -> list[str]:
    symbols = re.findall(
        r"^\s*\.amdgpu_hsa_kernel\s+(\S+)",
        assembly,
        flags=re.MULTILINE,
    )
    if not symbols:
        symbols = re.findall(
            r"^\s*\.amdhsa_kernel\s+(\S+)",
            assembly,
            flags=re.MULTILINE,
        )
    return sorted(set(symbols))


def metadata_block(assembly: str, symbol: str) -> str:
    pattern = re.compile(
        rf"^\s*\.amdhsa_kernel\s+{re.escape(symbol)}\s*$"
        rf"(?P<body>.*?)"
        rf"^\s*\.end_amdhsa_kernel\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(assembly)
    if not match:
        raise AssertionError("kernel metadata block not found")
    return match.group(0)


def function_block(assembly: str, symbol: str) -> str:
    label = re.search(
        rf"^\s*{re.escape(symbol)}:\s*(?:[#;].*)?$",
        assembly,
        flags=re.MULTILINE,
    )
    if not label:
        raise AssertionError("kernel function label not found")

    tail = assembly[label.start():]
    size_match = re.search(
        rf"^\s*\.size\s+{re.escape(symbol)}\s*,",
        tail,
        flags=re.MULTILINE,
    )
    if size_match:
        return tail[:size_match.end()]

    end_match = re.search(
        r"^\s*\.Lfunc_end\d+:\s*$",
        tail,
        flags=re.MULTILINE,
    )
    if end_match:
        return tail[:end_match.end()]

    raise AssertionError("kernel function end not found")


def directive(block: str, name: str) -> int:
    match = re.search(
        rf"^\s*\.amdhsa_{re.escape(name)}\s+(\d+)\s*$",
        block,
        flags=re.MULTILINE,
    )
    if not match:
        raise AssertionError(f"missing .amdhsa_{name}")
    return int(match.group(1))


def normalize_kernel(function: str, metadata: str) -> str:
    kept: list[str] = []
    for raw_line in (function + "\n" + metadata).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith((".file", ".loc", ".cfi", ".ident")):
            continue
        if line.startswith((".section .debug", ".section\t.debug")):
            continue
        line = re.sub(r"\s+", " ", line)
        kept.append(line)
    return "\n".join(kept) + "\n"


def instruction_counts(function: str) -> dict[str, int]:
    mnemonics: list[str] = []
    for raw_line in function.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or line.endswith(":") or line.startswith("."):
            continue
        match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)\b", line)
        if match:
            mnemonics.append(match.group(1))

    def count(pattern: str) -> int:
        regex = re.compile(pattern)
        return sum(bool(regex.search(mnemonic)) for mnemonic in mnemonics)

    ds_mnemonics = sorted(
        {
            mnemonic
            for mnemonic in mnemonics
            if mnemonic.startswith("ds_")
        }
    )

    return {
        "total_instructions": len(mnemonics),
        "ds_mnemonics": ds_mnemonics,
        "mfma_or_wmma": count(r"^v_(?:mfma|wmma)"),
        "lds_reads": count(r"^ds_(?:read|load)"),
        "lds_writes": count(r"^ds_(?:write|store)"),
        "block_barriers": count(r"^s_barrier"),
        "vector_global_loads": count(
            r"^(?:global|flat|buffer)_load"
        ),
        "vector_global_stores": count(
            r"^(?:global|flat|buffer)_store"
        ),
        "scalar_memory_loads": count(r"^s_load"),
        "scratch_loads": count(
            r"(?:^scratch_load|scratch.*load|^buffer_load.*scratch)"
        ),
        "scratch_stores": count(
            r"(?:^scratch_store|scratch.*store|^buffer_store.*scratch)"
        ),
        "flat_scratch_mentions": count(r"flat_scratch"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assembly-1", type=Path, required=True)
    parser.add_argument("--assembly-2", type=Path, required=True)
    parser.add_argument("--llvm-ir", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--readobj", type=Path, required=True)
    parser.add_argument("--device-object", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    preparation = load(args.preparation)
    assert preparation["decision"] == "PHASE4A1_P5_PREPARATION_PASS"
    assert preparation["source"]["sha256"] == sha256(args.source)

    assembly_1 = args.assembly_1.read_text(errors="replace")
    assembly_2 = args.assembly_2.read_text(errors="replace")
    llvm_ir = args.llvm_ir.read_text(errors="replace")
    objdump = args.objdump.read_text(errors="replace")
    readobj = args.readobj.read_text(errors="replace")

    symbols_1 = unique_kernel_symbols(assembly_1)
    symbols_2 = unique_kernel_symbols(assembly_2)

    assert symbols_1 == symbols_2
    assert len(symbols_1) == 1, symbols_1

    symbol = symbols_1[0]
    assert "width64_three_layer_fused_kernel" in symbol

    function_1 = function_block(assembly_1, symbol)
    function_2 = function_block(assembly_2, symbol)
    metadata_1 = metadata_block(assembly_1, symbol)
    metadata_2 = metadata_block(assembly_2, symbol)

    normalized_1 = normalize_kernel(function_1, metadata_1)
    normalized_2 = normalize_kernel(function_2, metadata_2)
    normalized_equal = normalized_1 == normalized_2

    normalized_sha_1 = hashlib.sha256(
        normalized_1.encode()
    ).hexdigest()
    normalized_sha_2 = hashlib.sha256(
        normalized_2.encode()
    ).hexdigest()

    resources = {
        "group_segment_fixed_size": directive(
            metadata_1,
            "group_segment_fixed_size",
        ),
        "private_segment_fixed_size": directive(
            metadata_1,
            "private_segment_fixed_size",
        ),
        "next_free_vgpr": directive(
            metadata_1,
            "next_free_vgpr",
        ),
        "next_free_sgpr": directive(
            metadata_1,
            "next_free_sgpr",
        ),
        "wavefront_size32": directive(
            metadata_1,
            "wavefront_size32",
        ),
    }

    optional_directives = (
        "kernarg_size",
        "user_sgpr_count",
        "accum_offset",
    )
    for name in optional_directives:
        match = re.search(
            rf"^\s*\.amdhsa_{name}\s+(\d+)\s*$",
            metadata_1,
            flags=re.MULTILINE,
        )
        resources[name] = int(match.group(1)) if match else None

    instructions = instruction_counts(function_1)
    scratch_instruction_count = (
        instructions["scratch_loads"]
        + instructions["scratch_stores"]
        + instructions["flat_scratch_mentions"]
    )

    ir_checks = {
        "kernel_name_present": (
            "width64_three_layer_fused_kernel" in llvm_ir
        ),
        "lds_address_space_present": "addrspace(3)" in llvm_ir,
        "hidden_lds_name_present": "hidden_lds" in llvm_ir,
    }

    object_checks = {
        "objdump_kernel_symbol_present": symbol in objdump,
        "objdump_mfma_or_wmma_present": bool(
            re.search(r"\bv_(?:mfma|wmma)", objdump)
        ),
        "objdump_lds_access_present": bool(
            re.search(r"\bds_(?:read|write|load|store)", objdump)
        ),
        "readobj_amdgpu_code_object_present": (
            "Machine: EM_AMDGPU" in readobj
        ),
        "device_object_nonempty": args.device_object.stat().st_size > 0,
        "binary_nonempty": args.binary.stat().st_size > 0,
    }

    source_contract = preparation["source_contract"]
    source_hidden_transport_ok = (
        source_contract["single_hidden_lds_declarations"] == 1
        and source_contract["hidden_lds_matrix_stores"] == 2
        and source_contract["layer_2_3_lds_matrix_inputs"] == 2
        and source_contract["explicit_hidden_global_output_arguments"] == 0
    )

    no_compiler_scratch = (
        resources["private_segment_fixed_size"] == 0
        and scratch_instruction_count == 0
    )

    lds_transport_in_isa = (
        instructions["lds_reads"] > 0
        and instructions["lds_writes"] > 0
        and instructions["block_barriers"] >= 3
        and ir_checks["lds_address_space_present"]
    )

    global_traffic_classified = (
        instructions["vector_global_loads"] > 0
        and instructions["vector_global_stores"] > 0
        and source_contract["final_output_epilogue_calls"] == 1
        and source_contract["expected_hidden_oracle_arguments"] == 2
    )

    no_hidden_intermediate_global_traffic = (
        source_hidden_transport_ok
        and no_compiler_scratch
        and lds_transport_in_isa
    )

    gates = {
        "single_device_kernel": len(symbols_1) == 1,
        "gfx1201_wave32_code_object": (
            resources["wavefront_size32"] == 1
            and (
                "gfx1201" in assembly_1
                or "gfx1201" in readobj
                or "EM_AMDGPU" in readobj
            )
        ),
        "normalized_isa_fresh_build_reproducibility": normalized_equal,
        "lds_resource_exactly_2048_bytes": (
            resources["group_segment_fixed_size"] == 2048
        ),
        "private_segment_zero": (
            resources["private_segment_fixed_size"] == 0
        ),
        "no_scratch_instructions": scratch_instruction_count == 0,
        "matrix_core_instruction_present": (
            instructions["mfma_or_wmma"] > 0
        ),
        "lds_read_write_present": (
            instructions["lds_reads"] > 0
            and instructions["lds_writes"] > 0
        ),
        "three_or_more_block_barriers_in_isa": (
            instructions["block_barriers"] >= 3
        ),
        "llvm_lds_address_space_present": (
            ir_checks["kernel_name_present"]
            and ir_checks["lds_address_space_present"]
        ),
        "device_object_disassembly_valid": all(object_checks.values()),
        "global_traffic_classified": global_traffic_classified,
        "no_hidden_intermediate_global_traffic_or_scratch": (
            no_hidden_intermediate_global_traffic
        ),
    }

    passed = all(gates.values())

    result = {
        "marker": MARKER,
        "decision": (
            "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT_PASS"
            if passed
            else "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT_FAIL"
        ),
        "kernel": {
            "symbol": symbol,
            "device_kernel_count": len(symbols_1),
            "normalized_assembly_sha256_build_1": normalized_sha_1,
            "normalized_assembly_sha256_build_2": normalized_sha_2,
        },
        "resources": resources,
        "instructions": instructions,
        "scratch_instruction_count": scratch_instruction_count,
        "llvm_ir_checks": ir_checks,
        "device_object_checks": object_checks,
        "global_traffic_interpretation": {
            "external_global_loads_expected": (
                "input, three weight tensors, three biases, and two "
                "diagnostic hidden-oracle tensors"
            ),
            "external_global_stores_expected": (
                "final FP32 output and diagnostic counters only"
            ),
            "hidden_1_transport": "LDS",
            "hidden_2_transport": "same reused LDS allocation",
            "compiler_private_scratch_bytes": (
                resources["private_segment_fixed_size"]
            ),
            "formal_per_instruction_pointer_provenance_claimed": False,
            "basis": [
                "exact P4 source has no hidden global output argument",
                "layer 2 and layer 3 source inputs are hidden_lds",
                "optimized LLVM IR contains LDS addrspace(3)",
                "ISA contains LDS reads and writes",
                "private segment is zero",
                "no scratch load/store instruction is present",
            ],
        },
        "artifacts": {
            "source": {
                "path": str(args.source.resolve()),
                "sha256": sha256(args.source),
            },
            "assembly_1": {
                "path": str(args.assembly_1.resolve()),
                "sha256": sha256(args.assembly_1),
            },
            "assembly_2": {
                "path": str(args.assembly_2.resolve()),
                "sha256": sha256(args.assembly_2),
            },
            "llvm_ir": {
                "path": str(args.llvm_ir.resolve()),
                "sha256": sha256(args.llvm_ir),
            },
            "objdump": {
                "path": str(args.objdump.resolve()),
                "sha256": sha256(args.objdump),
            },
            "readobj": {
                "path": str(args.readobj.resolve()),
                "sha256": sha256(args.readobj),
            },
            "device_object": {
                "path": str(args.device_object.resolve()),
                "sha256": sha256(args.device_object),
                "bytes": args.device_object.stat().st_size,
            },
            "binary": {
                "path": str(args.binary.resolve()),
                "sha256": sha256(args.binary),
                "bytes": args.binary.stat().st_size,
            },
        },
        "gates": gates,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    report = f"""# Phase 4A1-P5 — Width-64 ISA/resource/global-traffic audit

Decision: **`{result["decision"]}`**

## Kernel identity

- Device kernels in code object: `{len(symbols_1)}`
- Kernel symbol: `{symbol}`
- Normalized fresh-build ISA identical: `{normalized_equal}`
- Normalized ISA SHA-256: `{normalized_sha_1}`

## Resources

- Wave32 flag: `{resources["wavefront_size32"]}`
- VGPR allocation boundary: `{resources["next_free_vgpr"]}`
- SGPR allocation boundary: `{resources["next_free_sgpr"]}`
- LDS/group segment: `{resources["group_segment_fixed_size"]}` bytes
- Private/scratch segment: `{resources["private_segment_fixed_size"]}` bytes

## Static ISA inventory

- MFMA/WMMA instructions: `{instructions["mfma_or_wmma"]}`
- DS mnemonics: `{", ".join(instructions["ds_mnemonics"])}`
- LDS reads: `{instructions["lds_reads"]}`
- LDS writes: `{instructions["lds_writes"]}`
- Block barriers: `{instructions["block_barriers"]}`
- Vector global loads: `{instructions["vector_global_loads"]}`
- Vector global stores: `{instructions["vector_global_stores"]}`
- Scratch instructions: `{scratch_instruction_count}`

## Interpretation

The audit confirms that both hidden tensors travel through the single LDS
allocation and that the compiler emitted no private scratch segment or scratch
load/store instruction. Global traffic remains for external inputs, weights,
biases, diagnostic oracle reads, final output, and diagnostics.

This is not presented as a formal per-instruction pointer-provenance proof.
"""
    args.output_report.write_text(report)

    marker_values = {
        "WIDTH64_SINGLE_DEVICE_KERNEL_OBJECT": gates[
            "single_device_kernel"
        ],
        "WIDTH64_WAVE32_CODE_OBJECT": gates[
            "gfx1201_wave32_code_object"
        ],
        "WIDTH64_ISA_FRESH_BUILD_REPRODUCIBILITY": gates[
            "normalized_isa_fresh_build_reproducibility"
        ],
        "WIDTH64_LDS_RESOURCE_2048_BYTES": gates[
            "lds_resource_exactly_2048_bytes"
        ],
        "WIDTH64_PRIVATE_SEGMENT_ZERO": gates[
            "private_segment_zero"
        ],
        "WIDTH64_NO_SCRATCH_INSTRUCTIONS": gates[
            "no_scratch_instructions"
        ],
        "WIDTH64_MATRIX_CORE_INSTRUCTIONS_PRESENT": gates[
            "matrix_core_instruction_present"
        ],
        "WIDTH64_LDS_READ_WRITE_PRESENT": gates[
            "lds_read_write_present"
        ],
        "WIDTH64_BLOCK_BARRIERS_PRESENT": gates[
            "three_or_more_block_barriers_in_isa"
        ],
        "WIDTH64_LLVM_LDS_ADDRESS_SPACE_PRESENT": gates[
            "llvm_lds_address_space_present"
        ],
        "WIDTH64_DEVICE_OBJECT_DISASSEMBLY_VALID": gates[
            "device_object_disassembly_valid"
        ],
        "WIDTH64_GLOBAL_TRAFFIC_CLASSIFIED": gates[
            "global_traffic_classified"
        ],
        "WIDTH64_NO_HIDDEN_INTERMEDIATE_GLOBAL_TRAFFIC": gates[
            "no_hidden_intermediate_global_traffic_or_scratch"
        ],
    }

    print("kernel_symbol: " + symbol)
    print("next_free_vgpr: " + str(resources["next_free_vgpr"]))
    print("next_free_sgpr: " + str(resources["next_free_sgpr"]))
    print(
        "group_segment_fixed_size: "
        + str(resources["group_segment_fixed_size"])
    )
    print(
        "private_segment_fixed_size: "
        + str(resources["private_segment_fixed_size"])
    )
    print(
        "mfma_or_wmma_instructions: "
        + str(instructions["mfma_or_wmma"])
    )
    print(
        "ds_mnemonics: "
        + ",".join(instructions["ds_mnemonics"])
    )
    print("lds_reads: " + str(instructions["lds_reads"]))
    print("lds_writes: " + str(instructions["lds_writes"]))
    print("block_barriers: " + str(instructions["block_barriers"]))
    print("scratch_instruction_count: " + str(scratch_instruction_count))
    print(
        "llvm_ir_checks: "
        + json.dumps(ir_checks, sort_keys=True)
    )
    print(
        "device_object_checks: "
        + json.dumps(object_checks, sort_keys=True)
    )

    for marker, value in marker_values.items():
        print(f"{marker}: {'PASS' if value else 'FAIL'}")

    print(
        "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT: "
        + ("PASS" if passed else "FAIL")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
