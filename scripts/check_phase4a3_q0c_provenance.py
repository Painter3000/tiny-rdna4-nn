#!/usr/bin/env python3
"""Fail-closed Q0c-P gate for the exact measurement object and extension."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

from phase4a3_q0c_common import MARKER, enumerate_bundles, load_contract, sha256


P4_AUDIT_PASS = "PHASE4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_PASS"


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    value = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and value.returncode:
        raise RuntimeError("$ " + " ".join(command) + "\n" + value.stdout + value.stderr)
    return value


def tool(name: str) -> str:
    value = shutil.which(name) or shutil.which("/opt/rocm/llvm/bin/" + name)
    if not value:
        candidate = pathlib.Path("/opt/rocm/llvm/bin") / name
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError(f"required tool missing: {name}")
    return value


def version(path: str) -> str:
    value = run([path, "--version"], check=False)
    return (value.stdout + value.stderr).splitlines()[0]


def kernel_isa(objdump: str, payload: pathlib.Path, token: str) -> str | None:
    value = run([objdump, "--disassemble", "--demangle", "--mcpu=gfx1201", str(payload)], check=False)
    if value.returncode:
        value = run([objdump, "--disassemble", "--demangle", str(payload)], check=False)
    labels = []
    import re
    for match in re.finditer(r"^\s*[0-9a-fA-F]+\s+<(.+)>:\s*$", value.stdout, re.MULTILINE):
        labels.append((match.start(), match.end(), match.group(1)))
    matches = [(i, label) for i, label in enumerate(labels) if token in label[2] and not label[2].endswith(".kd")]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("multiple Width-64 kernel labels in one code object")
    index, label = matches[0]
    end = labels[index + 1][0] if index + 1 < len(labels) else len(value.stdout)
    return value.stdout[label[0]:end]


def validate_build_object(obj: pathlib.Path, expected_basename: str) -> pathlib.Path:
    obj = obj.resolve()
    if not obj.is_file():
        raise RuntimeError(f"Q0c build object is missing: {obj}")
    if obj.name != expected_basename:
        raise RuntimeError(
            "unexpected Q0c build object basename: "
            f"{obj.name!r}, expected {expected_basename!r}"
        )
    return obj


def validate_p4_audit(audit_result: object, kernel_token: str) -> tuple[str, list[str]]:
    if not isinstance(audit_result, dict):
        raise RuntimeError("P4 structural audit result is not a JSON object")
    if audit_result.get("decision") != P4_AUDIT_PASS:
        raise RuntimeError("exact measurement object did not pass the P4 structural audit")

    gates = audit_result.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise RuntimeError("P4 structural audit contains no gates")
    failed_gates = sorted(
        name for name, passed in gates.items()
        if not isinstance(name, str) or passed is not True
    )
    if failed_gates:
        raise RuntimeError(
            "P4 structural audit contains failed gates: " + ", ".join(map(str, failed_gates))
        )

    kernel = audit_result.get("kernel")
    if not isinstance(kernel, dict):
        raise RuntimeError("P4 structural audit contains no kernel classification")
    raw_symbol = kernel.get("raw_symbol")
    companions = kernel.get("metadata_companion_symbols")
    if not isinstance(raw_symbol, str) or not raw_symbol:
        raise RuntimeError("P4 structural audit contains no executable kernel symbol")
    if kernel_token not in raw_symbol:
        raise RuntimeError(
            "audited executable kernel symbol does not contain "
            f"{kernel_token!r}: {raw_symbol!r}"
        )
    if not isinstance(companions, list) or not all(isinstance(symbol, str) for symbol in companions):
        raise RuntimeError("P4 structural audit contains invalid kernel metadata companions")
    invalid_companions = [
        symbol for symbol in companions if not symbol.startswith(raw_symbol + ".")
    ]
    if invalid_companions:
        raise RuntimeError(
            "unclassified kernel metadata companions: " + repr(invalid_companions)
        )
    return raw_symbol, companions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--object", type=pathlib.Path, required=True)
    parser.add_argument("--extension", type=pathlib.Path, required=True)
    parser.add_argument("--p4-audit", type=pathlib.Path, required=True)
    parser.add_argument("--p4-contract", type=pathlib.Path, required=True)
    parser.add_argument("--build-command-file", type=pathlib.Path, required=True)
    parser.add_argument("--link-command-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    cfg = contract["provenance"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    obj = validate_build_object(args.object, cfg["build_object_basename"])
    if not args.extension.is_file():
        raise RuntimeError("final loaded extension is missing")
    for record in (args.build_command_file, args.link_command_file):
        if not record.is_file() or not record.read_text().strip():
            raise RuntimeError(f"missing command provenance: {record}")

    audit_dir = args.output_dir / "object_p4_audit"
    audit_json = args.output_dir / "object_p4_audit.json"
    audit = run([sys.executable, str(args.p4_audit), "--object", str(obj), "--contract", str(args.p4_contract), "--output-dir", str(audit_dir), "--output-json", str(audit_json)], check=False)
    (args.output_dir / "object_p4_audit.log").write_text(audit.stdout + audit.stderr)
    if audit.returncode or not audit_json.is_file():
        raise RuntimeError("P4 structural audit failed on exact measurement object")

    audit_result = json.loads(audit_json.read_text())
    raw_symbol, metadata_companions = validate_p4_audit(
        audit_result, cfg["kernel_token"]
    )
    print("PHASE4A3_Q0C_OBJECT_KERNEL_SYMBOL_CLASSIFIED: PASS")

    objdump = tool("llvm-objdump")
    data = args.extension.read_bytes()
    bundles = enumerate_bundles(data)
    if not bundles:
        raise RuntimeError("no clang offload bundles found in final extension")
    code_objects, matching = [], []
    with tempfile.TemporaryDirectory(prefix="q0c_code_objects_", dir=args.output_dir) as temporary:
        temp = pathlib.Path(temporary)
        for bundle_index, bundle in enumerate(bundles):
            for entry_index, entry in enumerate(bundle["entries"]):
                payload = entry["payload"]
                record = {"bundle_index": bundle_index, "entry_index": entry_index, "id": entry["id"], "offset": entry["offset"], "size": entry["size"]}
                if payload.startswith(b"\x7fELF"):
                    path = temp / f"b{bundle_index}_e{entry_index}.hsaco"
                    path.write_bytes(payload)
                    record["sha256"] = sha256(path)
                    isa = kernel_isa(objdump, path, cfg["kernel_token"])
                    if isa is not None:
                        isa_path = args.output_dir / f"extension_kernel_b{bundle_index}_e{entry_index}.isa.txt"
                        isa_path.write_text(isa)
                        record["kernel_isa_sha256"] = sha256(isa_path)
                        matching.append(record)
                code_objects.append(record)
    if len(matching) != cfg["exact_final_extension_kernels"]:
        raise RuntimeError(f"expected exactly one final-extension Width-64 kernel, found {len(matching)}")
    if matching[0]["kernel_isa_sha256"] != cfg["p4_reference_kernel_isa_sha256"]:
        raise RuntimeError("final-extension kernel ISA does not match frozen P4 reference")

    result = {
        "marker": MARKER,
        "subphase": "P",
        "decision": contract["decisions"]["P_pass"],
        "object": {
            "path": str(obj),
            "sha256": sha256(obj),
            "raw_symbol": raw_symbol,
            "metadata_companion_symbols": metadata_companions,
        },
        "extension": {"path": str(args.extension.resolve()), "sha256": sha256(args.extension)},
        "bundle_count": len(bundles),
        "code_objects": code_objects,
        "matching_kernel": matching[0],
        "frozen_p4_reference_kernel_isa_sha256": cfg["p4_reference_kernel_isa_sha256"],
        "p4_audit_json_sha256": sha256(audit_json),
        "build_command": args.build_command_file.read_text().strip(),
        "link_command": args.link_command_file.read_text().strip(),
        "tools": {"objdump": version(objdump)}
    }
    (args.output_dir / "phase4a3_q0c_provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(contract["decisions"]["P_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
