#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPARE="$ROOT/scripts/prepare_phase4a2_p0.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p0.py"
CONTRACT="$ROOT/contracts/phase4a2_p0_width64_production_integration_contract.json"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
P5_FILENAME="phase4a1_p5_width64_isa_resource_global_traffic.json"

resolve_p5_evidence() {
    if [[ -n "${PHASE4A1_P5_SOURCE_EVIDENCE:-}" ]]; then
        printf '%s\n' "$PHASE4A1_P5_SOURCE_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/$P5_FILENAME" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a1_p5_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A1-P5 evidence directory found." >&2
    return 1
}

P5_DIR="$(resolve_p5_evidence)"
P5_JSON="$P5_DIR/$P5_FILENAME"

EVIDENCE="${1:-${PHASE4A2_P0_EVIDENCE:-$EVIDENCE_ROOT/phase4a2_p0_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a2_p0_*)
        ;;
    *)
        echo "Refusing unsafe evidence path: $EVIDENCE" >&2
        exit 2
        ;;
esac

if [[ -d "$EVIDENCE" ]] && \
   [[ -n "$(find "$EVIDENCE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty evidence directory: $EVIDENCE" >&2
    exit 2
fi

for path in "$PREPARE" "$FINALIZER" "$CONTRACT" "$P5_JSON"; do
    test -f "$path"
done

mkdir -p "$EVIDENCE/process_1" "$EVIDENCE/process_2"

echo "===== PHASE 4A1 TAG BASELINE ====="
TAG="phase4a1-width64-rocwmma-fused-forward-gfx1201-pass"
HEAD_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
TAG_COMMIT="$(git -C "$ROOT" rev-parse "$TAG^{}")"
test "$HEAD_COMMIT" = "$TAG_COMMIT"
echo "head: $HEAD_COMMIT"
echo "tag_commit: $TAG_COMMIT"
echo "PHASE4A1_PASS_TAG_BOUND: PASS"

echo
echo "===== PYTHON SELF-CHECK ====="
python -m py_compile "$PREPARE" "$FINALIZER"
python - "$CONTRACT" <<'PY_CONTRACT'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["marker"] == "TCNN_RDNA4_P4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_001"
assert data["public_api"]["otype"] == "RocWMMAWidth64MLP"
assert data["public_api"]["default_enabled"] is False
assert data["public_api"]["silent_fallback"] is False
assert data["scope"]["phase_4a2_initial_capability"] == "inference_only"
assert data["parameter_abi"]["total_parameter_elements"] == 12480
print("PHASE4A2_P0_CONTRACT_SELF_CHECK: PASS")
PY_CONTRACT
echo "PHASE4A2_P0_PYTHON_SELF_CHECK: PASS"

{
    echo "===== PHASE 4A2-P0 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $HEAD_COMMIT"
    echo "phase4a1_tag: $TAG"
    echo "phase4a1_tag_commit: $TAG_COMMIT"
    echo "phase4a1_p5_evidence: $P5_DIR"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
} > "$EVIDENCE/context.txt"

run_inventory() {
    local process="$1"
    python "$PREPARE" \
        --repo "$ROOT" \
        --contract "$CONTRACT" \
        --p5-json "$P5_JSON" \
        --output "$EVIDENCE/$process/inventory.json" \
        2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== FRESH PROCESS 1 ====="
run_inventory process_1

echo
echo "===== FRESH PROCESS 2 ====="
run_inventory process_2

preparation_markers=(
    "PHASE4A1_PASS_TAG_BOUND: PASS"
    "PHASE4A1_P5_EVIDENCE_BOUND: PASS"
    "WIDTH64_PRODUCTION_SURFACES_INVENTORIED: PASS"
    "WIDTH64_PUBLIC_OTYPE_COLLISION_FREE: PASS"
    "WIDTH64_OPT_IN_FAIL_CLOSED_CONTRACT: PASS"
    "WIDTH64_NETWORK_ABI_CONTRACT: PASS"
    "WIDTH64_PARAMETER_LAYOUT_CONTRACT: PASS"
    "WIDTH64_BATCH_LAYOUT_CONTRACT: PASS"
    "WIDTH64_INFERENCE_ONLY_SCOPE_LOCKED: PASS"
    "WIDTH64_EXISTING_BACKENDS_UNCHANGED_CONTRACT: PASS"
    "PHASE4A2_P0_PREPARATION: PASS"
)

for process in process_1 process_2; do
    for marker in "${preparation_markers[@]}"; do
        grep -Fx "$marker" "$EVIDENCE/$process/run.log"
    done
done

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --inventory-1 "$EVIDENCE/process_1/inventory.json" \
    --inventory-2 "$EVIDENCE/process_2/inventory.json" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

final_markers=(
    "PHASE4A2_P0_FRESH_PROCESS_REPRODUCIBILITY: PASS"
    "PHASE4A2_P0_CONSOLIDATED_EVIDENCE: RECORDED"
    "PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT: PASS"
)

for marker in "${final_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/finalize.log"
done

sha256sum \
    "$CONTRACT" \
    "$PREPARE" \
    "$FINALIZER" \
    "$P5_JSON" \
    "$EVIDENCE/process_1/inventory.json" \
    "$EVIDENCE/process_2/inventory.json" \
    "$EVIDENCE/phase4a2_p0_production_integration_contract.json" \
    "$EVIDENCE/PHASE4A2_P0_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

python - "$EVIDENCE/phase4a2_p0_production_integration_contract.json" <<'PY_AUDIT'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == "PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT_PASS"
assert all(data["gates"].values())
inventory = data["inventory"]["data"]
assert inventory["decision"] == "PHASE4A2_P0_PREPARATION_PASS"
assert all(inventory["gates"].values())
assert all(inventory["contract"]["gates"].values())
assert data["locked_backend"]["otype"] == "RocWMMAWidth64MLP"
assert data["locked_backend"]["parameter_abi"]["total_parameter_elements"] == 12480
assert data["locked_backend"]["execution_contract"]["lds_bytes"] == 2048
print("PHASE4A2_P0_JSON_AUDIT: PASS")
PY_AUDIT

echo
echo "PHASE4A1_PASS_TAG_BOUND: PASS"
echo "PHASE4A1_P5_EVIDENCE_BOUND: PASS"
echo "WIDTH64_PRODUCTION_SURFACES_INVENTORIED: PASS"
echo "WIDTH64_PUBLIC_OTYPE_COLLISION_FREE: PASS"
echo "WIDTH64_OPT_IN_FAIL_CLOSED_CONTRACT: PASS"
echo "WIDTH64_NETWORK_ABI_CONTRACT: PASS"
echo "WIDTH64_PARAMETER_LAYOUT_CONTRACT: PASS"
echo "WIDTH64_BATCH_LAYOUT_CONTRACT: PASS"
echo "WIDTH64_INFERENCE_ONLY_SCOPE_LOCKED: PASS"
echo "WIDTH64_EXISTING_BACKENDS_UNCHANGED_CONTRACT: PASS"
echo "PHASE4A2_P0_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "PHASE4A2_P0_JSON_AUDIT: PASS"
echo "PHASE4A2_P0_MAP_CONTEXT: RECORDED"
echo "PHASE4A2_P0_PRODUCTION_INTEGRATION_CONTRACT: PASS"
echo "Evidence: $EVIDENCE"
