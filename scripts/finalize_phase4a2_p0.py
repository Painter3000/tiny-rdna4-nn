#!/usr/bin/env python3
"""Finalize deterministic Phase 4A2-P0 contract evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER = "TCNN_RDNA4_P4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-1", type=Path, required=True)
    parser.add_argument("--inventory-2", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    first = load(args.inventory_1)
    second = load(args.inventory_2)

    assert first["marker"] == MARKER
    assert second["marker"] == MARKER
    assert first["decision"] == "PHASE4A2_P0_PREPARATION_PASS"
    assert second["decision"] == "PHASE4A2_P0_PREPARATION_PASS"
    assert all(bool(value) for value in first["gates"].values())
    assert all(bool(value) for value in second["gates"].values())
    assert all(bool(value) for value in first["contract"]["gates"].values())
    assert all(bool(value) for value in second["contract"]["gates"].values())

    deterministic = args.inventory_1.read_bytes() == args.inventory_2.read_bytes()
    assert deterministic

    contract = first["contract"]["data"]
    bridge = contract["p4_to_production_bridge_deltas"]

    consolidated = {
        "marker": MARKER,
        "decision": "PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_PASS",
        "inventory": {
            "sha256": sha256(args.inventory_1),
            "fresh_process_exact_match": deterministic,
            "data": first,
        },
        "locked_backend": {
            "otype": contract["public_api"]["otype"],
            "class_name": contract["public_api"]["class_name"],
            "build_option": contract["public_api"]["build_option"],
            "compile_definition": contract["public_api"]["compile_definition"],
            "eligibility": contract["eligibility"],
            "parameter_abi": contract["parameter_abi"],
            "execution_contract": contract["execution_contract"],
            "scope": contract["scope"],
            "bridge_deltas": bridge,
        },
        "gates": {
            "phase4a1_pass_tag_bound": first["gates"]["phase4a1_tag_bound"],
            "phase4a1_p5_evidence_bound": first["gates"]["phase4a1_p5_evidence_bound"],
            "production_surfaces_frozen": first["gates"]["required_surfaces_present"],
            "production_sources_unchanged": first["gates"]["production_sources_clean"],
            "reserved_identifiers_collision_free": first["gates"]["reserved_identifiers_collision_free"],
            "contract_fail_closed": (
                first["contract"]["gates"]["explicit_otype_only"]
                and first["contract"]["gates"]["default_disabled"]
                and first["contract"]["gates"]["no_silent_fallback"]
            ),
            "network_abi_locked": (
                first["contract"]["gates"]["width64_topology"]
                and first["contract"]["gates"]["two_hidden_layers"]
                and first["contract"]["gates"]["relu_none_activations"]
            ),
            "parameter_abi_locked": first["contract"]["gates"]["fp16_parameter_abi"],
            "batch_layout_locked": first["contract"]["gates"]["batch_tile_16"],
            "inference_only_scope_locked": first["contract"]["gates"]["inference_only"],
            "existing_backends_preserved": first["contract"]["gates"]["existing_backends_unchanged"],
            "bridge_delta_ledger_locked": first["contract"]["gates"]["bridge_delta_ledger_complete"],
            "fresh_process_reproducible": deterministic,
        },
    }
    assert all(consolidated["gates"].values())

    args.evidence.mkdir(parents=True, exist_ok=True)
    output_json = args.evidence / "phase4a2_p0_production_integration_contract.json"
    output_json.write_text(json.dumps(consolidated, indent=2, sort_keys=True) + "\n")

    resources = first["baseline"]["p5_resources"]
    surface_count = len(first["integration_surfaces"])

    report = f"""# Phase 4A2-P0 — Width-64 production integration contract

Decision: **`{consolidated["decision"]}`**

## Bound baseline

- Tag: `{first["baseline"]["tag"]}`
- Commit: `{first["baseline"]["tag_commit"]}`
- P5 evidence: `{first["baseline"]["p5_json_sha256"]}`
- P5 resource facts: VGPR boundary `{resources["next_free_vgpr"]}`,
  SGPR boundary `{resources["next_free_sgpr"]}`, LDS
  `{resources["group_segment_fixed_size"]}` bytes, private segment
  `{resources["private_segment_fixed_size"]}` bytes, scratch instructions
  `{resources["scratch_instruction_count"]}`.

## Locked public backend

- JSON `otype`: `{contract["public_api"]["otype"]}`
- Build option: `{contract["public_api"]["build_option"]}` — default OFF
- Compile definition: `{contract["public_api"]["compile_definition"]}`
- Selection: explicit only; unsupported requests fail closed.
- Initial capability: inference only.
- Existing backends remain unchanged.

## Exact eligibility

```text
HIP AMD / gfx1201
Network<__half>
64 input -> 64 ReLU -> 64 ReLU -> 64 None
bias enabled
batch rows >= 16 and divisible by 16
ColumnMajor [64][batch]
caller-supplied hipStream_t
```

## Parameter ABI

```text
layer 0: W0[64x64 FP16 column-major], b0[64 FP16]
layer 1: W1[64x64 FP16 column-major], b1[64 FP16]
layer 2: W2[64x64 FP16 column-major], b2[64 FP16]
total: 12,480 FP16 elements
biases are promoted to FP32 in the epilogue
```

## Production bridge deltas

The P4 standalone kernel is not copied blindly. P1/P2 must close all
{len(bridge)} recorded deltas: multi-block batching, GPUMatrix layout,
single-buffer FP16 parameter ABI, FP16 public output, diagnostic removal and
caller-stream execution.

## Frozen integration surfaces

`{surface_count}` required/optional production surfaces were hashed and
inventoried. No reserved P4A2 identifier existed and no production source was
modified during P0.

No performance or training/backward claim is made.
"""
    (args.evidence / "PHASE4A2_P0_REPORT.md").write_text(report)

    print("PHASE4A2_P0_FRESH_PROCESS_REPRODUCIBILITY: PASS")
    print("PHASE4A2_P0_CONSOLIDATED_EVIDENCE: RECORDED")
    print("PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
