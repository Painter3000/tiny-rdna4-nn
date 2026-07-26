#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P3_RUNTIME_INTEGRATION_CLOSURE_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="$ROOT/probes/phase4a2_p3_runtime_integration_probe.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p3.py"
CONTRACT="$ROOT/contracts/phase4a2_p3_runtime_integration_closure_contract.json"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
P2_FILENAME="phase4a2_p2_width64_production_inference_parameter_abi.json"

resolve_p2_evidence() {
    if [[ -n "${PHASE4A2_P2_SOURCE_EVIDENCE:-}" ]]; then
        printf '%s\n' "$PHASE4A2_P2_SOURCE_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/$P2_FILENAME" ]] && \
           [[ -d "$candidate/build/lib" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a2_p2_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A2-P2 evidence directory found." >&2
    return 1
}

P2_DIR="$(resolve_p2_evidence)"
P2_JSON="$P2_DIR/$P2_FILENAME"
BINDINGS_DIR="$ROOT/bindings/torch"

EVIDENCE="${1:-${PHASE4A2_P3_EVIDENCE:-$EVIDENCE_ROOT/phase4a2_p3_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a2_p3_*)
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

for path in "$PROBE" "$FINALIZER" "$CONTRACT" "$P2_JSON"; do
    test -f "$path"
done

shopt -s nullglob
EXTENSIONS=("$P2_DIR"/build/lib/tinycudann_bindings/_120_C*.so)
shopt -u nullglob

if [[ "${#EXTENSIONS[@]}" -ne 1 ]]; then
    echo "Expected exactly one P2 extension artifact, found ${#EXTENSIONS[@]}." >&2
    exit 1
fi

EXTENSION="${EXTENSIONS[0]}"

mkdir -p \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

echo "===== PHASE 4A2-P2 BASELINE ====="
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
SUBJECT="$(git -C "$ROOT" show -s --format=%s HEAD)"
echo "head: $HEAD"
echo "subject: $SUBJECT"
test "${HEAD:0:7}" = "204a807"
test "$SUBJECT" = \
    "Add Phase 4A2-P2 rocWMMA Width-64 production inference"
echo "PHASE4A2_P2_COMMIT_BOUND: PASS"

echo
echo "===== SELF-CHECK ====="
PYTHONPYCACHEPREFIX="$EVIDENCE/pycache" \
    python -m py_compile "$PROBE" "$FINALIZER"

python - "$CONTRACT" "$P2_JSON" <<'PY_CHECK'
import json
import pathlib
import sys

contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
p2 = json.loads(pathlib.Path(sys.argv[2]).read_text())

assert contract["marker"] == (
    "TCNN_RDNA4_P4A2_P3_RUNTIME_INTEGRATION_CLOSURE_001"
)
assert contract["baseline"]["required_head_prefix"] == "204a807"
assert contract["baseline"]["rebuild"] is False
assert contract["runtime_matrix"]["repeat_launches"] == 64
assert len(contract["runtime_matrix"]["batch_sizes"]) == 20
assert p2["decision"] == (
    "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI_PASS"
)
assert all(p2["gates"].values())
print("PHASE4A2_P3_CONTRACT_AND_P2_EVIDENCE: PASS")
PY_CHECK

SOURCE="$ROOT/src/rocwmma_width64_mlp.cu"
MAPPING="$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h"

test "$(sha256sum "$SOURCE" | awk '{print $1}')" = \
    "7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4"
test "$(sha256sum "$MAPPING" | awk '{print $1}')" = \
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"

echo "WIDTH64_P2_PRODUCTION_SOURCE_EXACT: PASS"
echo "WIDTH64_P2_MAPPING_HEADER_EXACT: PASS"

{
    echo "===== PHASE 4A2-P3 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $HEAD"
    echo "phase4a2_p2_evidence: $P2_DIR"
    echo "reused_extension: $EXTENSION"
    echo "reused_extension_sha256: $(sha256sum "$EXTENSION" | awk '{print $1}')"
    echo
    echo "===== STATUS BEFORE RUNTIME PROBE ====="
    git -C "$ROOT" status --short --untracked-files=all
} > "$EVIDENCE/context.txt"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
test -x "$PYTHON_BIN"

run_probe() {
    local process="$1"
    (
        cd "$EVIDENCE/$process"
        env \
            PYTHONNOUSERSITE=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH="$P2_DIR/build/lib:$BINDINGS_DIR" \
            "$PYTHON_BIN" "$PROBE" \
                --repeat-launches 64 \
                --output "$EVIDENCE/$process/result.json"
    ) 2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== RUNTIME INTEGRATION FRESH PROCESS 1 ====="
run_probe process_1

echo
echo "===== RUNTIME INTEGRATION FRESH PROCESS 2 ====="
run_probe process_2

process_markers=(
    "WIDTH64_RUNTIME_BATCH_MATRIX_20_CASES: PASS"
    "WIDTH64_RUNTIME_PADDING_BOUNDARIES_256_1024: PASS"
    "WIDTH64_RUNTIME_REPEAT_64_LAUNCHES_BITWISE: PASS"
    "WIDTH64_RUNTIME_PREFIX_INVARIANCE: PASS"
    "WIDTH64_RUNTIME_PARAMETER_HOT_SWAP_A_B_A: PASS"
    "WIDTH64_RUNTIME_DUAL_STREAM_MODEL_ISOLATION: PASS"
    "WIDTH64_RUNTIME_EXISTING_FACTORIES_CONSTRUCT: PASS"
    "WIDTH64_RUNTIME_TRAINING_FORWARD_FAIL_CLOSED: PASS"
    "PHASE4A2_P3_RUNTIME_INTEGRATION_LIFECYCLE_PROCESS: PASS"
)

for process in process_1 process_2; do
    for marker in "${process_markers[@]}"; do
        grep -Fx "$marker" "$EVIDENCE/$process/run.log"
    done
done

cmp \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_2/result.json"

echo
echo "===== FINALIZE ====="
"$PYTHON_BIN" "$FINALIZER" \
    --repo "$ROOT" \
    --contract "$CONTRACT" \
    --p2-json "$P2_JSON" \
    --extension "$EXTENSION" \
    --process-1 "$EVIDENCE/process_1/result.json" \
    --process-2 "$EVIDENCE/process_2/result.json" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

final_markers=(
    "WIDTH64_P2_EXTENSION_REUSED: PASS"
    "WIDTH64_RUNTIME_MATRIX_20_BATCHES: PASS"
    "WIDTH64_RUNTIME_PADDING_256_1024: PASS"
    "WIDTH64_RUNTIME_64_LAUNCH_BITWISE: PASS"
    "WIDTH64_RUNTIME_PARAMETER_HOT_SWAP: PASS"
    "WIDTH64_RUNTIME_DUAL_STREAM_MODEL_ISOLATION: PASS"
    "WIDTH64_RUNTIME_EXISTING_FACTORIES: PASS"
    "WIDTH64_RUNTIME_FRESH_PROCESS_REPRODUCIBILITY: PASS"
    "WIDTH64_P3_NO_PRODUCTION_CODE_CHANGE: PASS"
    "PHASE4A2_P3_CONSOLIDATED_EVIDENCE: RECORDED"
    "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE: PASS"
)

for marker in "${final_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/finalize.log"
done

sha256sum \
    "$P2_JSON" \
    "$EXTENSION" \
    "$CONTRACT" \
    "$PROBE" \
    "$FINALIZER" \
    "$SOURCE" \
    "$MAPPING" \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_2/result.json" \
    "$EVIDENCE/phase4a2_p3_width64_runtime_integration_lifecycle_closure.json" \
    "$EVIDENCE/PHASE4A2_P3_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

"$PYTHON_BIN" - "$EVIDENCE/phase4a2_p3_width64_runtime_integration_lifecycle_closure.json" <<'PY_JSON'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE_PASS"
)
assert all(data["gates"].values())
assert all(data["static_gates"].values())
assert data["fresh_processes"]["exact_match"] is True
assert len(data["metrics"]["batch_cases"]) == 20
assert data["metrics"]["repeat_launches"] == 64
print("PHASE4A2_P3_JSON_AUDIT: PASS")
PY_JSON

echo
echo "WIDTH64_P2_EXTENSION_REUSED: PASS"
echo "WIDTH64_RUNTIME_MATRIX_20_BATCHES: PASS"
echo "WIDTH64_RUNTIME_PADDING_256_1024: PASS"
echo "WIDTH64_RUNTIME_64_LAUNCH_BITWISE: PASS"
echo "WIDTH64_RUNTIME_PARAMETER_HOT_SWAP: PASS"
echo "WIDTH64_RUNTIME_DUAL_STREAM_MODEL_ISOLATION: PASS"
echo "WIDTH64_RUNTIME_EXISTING_FACTORIES: PASS"
echo "WIDTH64_RUNTIME_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "WIDTH64_P3_NO_PRODUCTION_CODE_CHANGE: PASS"
echo "PHASE4A2_P3_JSON_AUDIT: PASS"
echo "PHASE4A2_P3_MAP_CONTEXT: RECORDED"
echo "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE: PASS"
echo "Evidence: $EVIDENCE"
