#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT="$ROOT/contracts/phase4a2_p4_production_code_object_audit_contract.json"
AUDITOR="$ROOT/scripts/audit_phase4a2_p4_code_object.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p4.py"
P3_PROBE="$ROOT/probes/phase4a2_p3_runtime_integration_probe.py"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
P3_FILENAME="phase4a2_p3_width64_runtime_integration_lifecycle_closure.json"

resolve_p3_evidence() {
    if [[ -n "${PHASE4A2_P3_SOURCE_EVIDENCE:-}" ]]; then
        printf '%s\n' "$PHASE4A2_P3_SOURCE_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/$P3_FILENAME" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a2_p3_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A2-P3 evidence directory found." >&2
    return 1
}

P3_DIR="$(resolve_p3_evidence)"
P3_JSON="$P3_DIR/$P3_FILENAME"

EVIDENCE="${1:-${PHASE4A2_P4_EVIDENCE:-$EVIDENCE_ROOT/phase4a2_p4_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a2_p4_*)
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

for path in \
    "$CONTRACT" \
    "$AUDITOR" \
    "$FINALIZER" \
    "$P3_PROBE" \
    "$P3_JSON"; do
    test -f "$path"
done

mkdir -p \
    "$EVIDENCE/build/temp" \
    "$EVIDENCE/build/lib" \
    "$EVIDENCE/code_object" \
    "$EVIDENCE/replay_1" \
    "$EVIDENCE/replay_2"

echo "===== PHASE 4A2-P4 GIT CONTEXT ====="
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
SUBJECT="$(git -C "$ROOT" show -s --format=%s HEAD)"
PARENT="$(git -C "$ROOT" rev-parse HEAD^)"
PARENT_SUBJECT="$(git -C "$ROOT" show -s --format=%s HEAD^)"

if [[ "${HEAD:0:7}" == "de76469" ]] && \
   [[ "$SUBJECT" == "Add Phase 4A2-P3 rocWMMA Width-64 runtime integration closure" ]]; then
    GIT_CONTEXT_MODE="p3_bundle_precommit"
elif [[ "$SUBJECT" == "Add Phase 4A2-P4 rocWMMA Width-64 production code-object audit" ]] && \
     [[ "${PARENT:0:7}" == "de76469" ]] && \
     [[ "$PARENT_SUBJECT" == "Add Phase 4A2-P3 rocWMMA Width-64 runtime integration closure" ]]; then
    GIT_CONTEXT_MODE="p4_release_commit"
else
    echo "Unsupported P4 git context." >&2
    echo "HEAD: $HEAD" >&2
    echo "Subject: $SUBJECT" >&2
    echo "Parent: $PARENT" >&2
    echo "Parent subject: $PARENT_SUBJECT" >&2
    exit 1
fi

echo "mode: $GIT_CONTEXT_MODE"
echo "head: $HEAD"
echo "subject: $SUBJECT"
echo "parent: $PARENT"
echo "parent subject: $PARENT_SUBJECT"
echo "PHASE4A2_P4_GIT_CONTEXT_BOUND: PASS"

echo
echo "===== SELF-CHECK ====="
PYTHONPYCACHEPREFIX="$EVIDENCE/pycache" \
    python -m py_compile "$AUDITOR" "$FINALIZER" "$P3_PROBE"

python - "$CONTRACT" "$P3_JSON" <<'PY_CHECK'
import json
import pathlib
import sys

contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
p3 = json.loads(pathlib.Path(sys.argv[2]).read_text())

assert contract["marker"] == (
    "TCNN_RDNA4_P4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_001"
)
assert contract["baseline"]["required_head_prefix"] == "de76469"
assert contract["baseline"]["required_head_subject"] == (
    "Add Phase 4A2-P3 rocWMMA Width-64 runtime integration closure"
)
assert contract["release_closure"]["marker"] == (
    "TCNN_RDNA4_P4A2_P4_RELEASE_REPRODUCIBILITY_007"
)
assert contract["baseline"]["fresh_build"] is True
assert contract["code_object_audit"]["group_segment_fixed_size"] == 2048
assert contract["code_object_audit"]["private_segment_fixed_size"] == 0
assert contract["code_object_audit"]["mfma_or_wmma_instructions"] == 12
assert contract["code_object_audit"]["lds_load_instructions"] == 8
assert contract["code_object_audit"]["lds_store_instructions"] == 2
assert contract["code_object_audit"]["ds_bpermute_b32_instructions"] == 192
assert contract["code_object_audit"]["block_barriers"] == 6
assert contract["code_object_audit"]["scratch_instruction_count"] == 0
assert p3["decision"] == (
    "PHASE4A2_P3_WIDTH64_RUNTIME_INTEGRATION_LIFECYCLE_CLOSURE_PASS"
)
assert all(p3["gates"].values())
print("PHASE4A2_P4_CONTRACT_AND_P3_EVIDENCE: PASS")
PY_CHECK

SOURCE="$ROOT/src/rocwmma_width64_mlp.cu"
MAPPING="$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h"

test "$(sha256sum "$SOURCE" | awk '{print $1}')" = \
    "7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4"
test "$(sha256sum "$MAPPING" | awk '{print $1}')" = \
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"

echo "WIDTH64_P2_PRODUCTION_SOURCE_EXACT: PASS"
echo "WIDTH64_P2_MAPPING_HEADER_EXACT: PASS"

if [[ -n "$(git -C "$ROOT" diff --name-only)" ]] || \
   [[ -n "$(git -C "$ROOT" diff --cached --name-only)" ]]; then
    echo "Tracked worktree is not clean before P4." >&2
    git -C "$ROOT" status --short
    exit 1
fi

{
    echo "===== PHASE 4A2-P4 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $HEAD"
    echo "phase4a2_p3_evidence: $P3_DIR"
    echo
    echo "===== STATUS BEFORE BUILD ====="
    git -C "$ROOT" status --short --untracked-files=all
} > "$EVIDENCE/context.txt"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BINDINGS_DIR="$ROOT/bindings/torch"
DEPENDENCY_ROOT="${TCNN_DEPENDENCY_ROOT:-$ROOT/dependencies}"
MAX_JOBS="${MAX_JOBS:-4}"

test -x "$PYTHON_BIN"
test -d "$DEPENDENCY_ROOT"

echo
echo "===== FRESH EXPLICIT-ON PRODUCTION BUILD ====="
(
    cd "$BINDINGS_DIR"
    env \
        PYTORCH_ROCM_ARCH=gfx1201 \
        TCNN_DEPENDENCY_ROOT="$DEPENDENCY_ROOT" \
        TCNN_HALF_PRECISION=1 \
        TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 \
        MAX_JOBS="$MAX_JOBS" \
        "$PYTHON_BIN" setup.py build_ext \
            --build-temp "$EVIDENCE/build/temp" \
            --build-lib "$EVIDENCE/build/lib"
) 2>&1 | tee "$EVIDENCE/build.log"

grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" \
    "$EVIDENCE/build.log"
grep -q "rocwmma_width64_mlp.cu" "$EVIDENCE/build.log"
grep -q "TCNN_WITH_ROCWMMA_WIDTH64_MLP" "$EVIDENCE/build.log"

mapfile -t OBJECTS < <(
    find "$EVIDENCE" \
        -type f \
        -name 'rocwmma_width64_mlp.o' \
        -print |
    sort
)

if [[ "${#OBJECTS[@]}" -ne 1 ]]; then
    echo "Expected exactly one production object, found ${#OBJECTS[@]}." >&2
    printf '%s\n' "${OBJECTS[@]:-}"
    exit 1
fi
OBJECT="${OBJECTS[0]}"

shopt -s nullglob
EXTENSIONS=("$EVIDENCE"/build/lib/tinycudann_bindings/_120_C*.so)
shopt -u nullglob
if [[ "${#EXTENSIONS[@]}" -ne 1 ]]; then
    echo "Expected exactly one extension, found ${#EXTENSIONS[@]}." >&2
    exit 1
fi
EXTENSION="${EXTENSIONS[0]}"

echo "production_object: $OBJECT"
echo "production_extension: $EXTENSION"
echo "WIDTH64_FRESH_PRODUCTION_BUILD: PASS"

echo
echo "===== CODE-OBJECT EXTRACTION AND ISA AUDIT ====="
"$PYTHON_BIN" "$AUDITOR" \
    --object "$OBJECT" \
    --contract "$CONTRACT" \
    --output-dir "$EVIDENCE/code_object" \
    --output-json "$EVIDENCE/code_object_audit.json" \
    2>&1 | tee "$EVIDENCE/code_object_audit.log"

audit_markers=(
    "WIDTH64_GFX1201_CODE_OBJECT_EXTRACTED: PASS"
    "WIDTH64_EXACT_PRODUCTION_KERNEL_SYMBOL: PASS"
    "WIDTH64_KERNEL_METADATA_COMPANIONS_CLASSIFIED: PASS"
    "WIDTH64_AMDGPU_OBJDUMP_SYNTAX_PARSED: PASS"
    "WIDTH64_RESOURCE_METADATA_RECORDED: PASS"
    "WIDTH64_GROUP_SEGMENT_2048: PASS"
    "WIDTH64_PRIVATE_SEGMENT_ZERO: PASS"
    "WIDTH64_MFMA_OR_WMMA_12: PASS"
    "WIDTH64_LDS_LOADS_8_STORES_2: PASS"
    "WIDTH64_DS_BPERMUTE_B32_192: PASS"
    "WIDTH64_EXACT_DS_MNEMONIC_INVENTORY: PASS"
    "WIDTH64_BLOCK_BARRIERS_6: PASS"
    "WIDTH64_SCRATCH_INSTRUCTIONS_ZERO: PASS"
    "WIDTH64_GLOBAL_MEMORY_MNEMONICS_RECORDED: PASS"
    "PHASE4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT: PASS"
)

for marker in "${audit_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/code_object_audit.log"
done

run_replay() {
    local process="$1"
    (
        cd "$EVIDENCE/$process"
        env \
            PYTHONNOUSERSITE=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH="$EVIDENCE/build/lib:$BINDINGS_DIR" \
            "$PYTHON_BIN" "$P3_PROBE" \
                --repeat-launches 64 \
                --output "$EVIDENCE/$process/result.json"
    ) 2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== FRESH BUILD RUNTIME REPLAY 1 ====="
run_replay replay_1

echo
echo "===== FRESH BUILD RUNTIME REPLAY 2 ====="
run_replay replay_2

replay_markers=(
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

for process in replay_1 replay_2; do
    for marker in "${replay_markers[@]}"; do
        grep -Fx "$marker" "$EVIDENCE/$process/run.log"
    done
done

cmp \
    "$EVIDENCE/replay_1/result.json" \
    "$EVIDENCE/replay_2/result.json"

echo
echo "===== FINALIZE ====="
"$PYTHON_BIN" "$FINALIZER" \
    --repo "$ROOT" \
    --contract "$CONTRACT" \
    --p3-json "$P3_JSON" \
    --build-log "$EVIDENCE/build.log" \
    --object "$OBJECT" \
    --extension "$EXTENSION" \
    --audit-json "$EVIDENCE/code_object_audit.json" \
    --replay-1 "$EVIDENCE/replay_1/result.json" \
    --replay-2 "$EVIDENCE/replay_2/result.json" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

final_markers=(
    "WIDTH64_FRESH_PRODUCTION_BUILD: PASS"
    "WIDTH64_GFX1201_CODE_OBJECT_BOUND: PASS"
    "WIDTH64_EXACT_PRODUCTION_KERNEL_BOUND: PASS"
    "WIDTH64_RESOURCE_FIELDS_RECORDED: PASS"
    "WIDTH64_GROUP_2048_PRIVATE_ZERO: PASS"
    "WIDTH64_MFMA12_LDSLOAD8_STORE2_BARRIER6: PASS"
    "WIDTH64_DS_BPERMUTE_B32_192: PASS"
    "WIDTH64_SCRATCH_ZERO: PASS"
    "WIDTH64_FRESH_RUNTIME_REPLAY_TWICE: PASS"
    "WIDTH64_P4_NO_PRODUCTION_CODE_CHANGE: PASS"
    "PHASE4A2_P4_CONSOLIDATED_EVIDENCE: RECORDED"
    "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT: PASS"
)

for marker in "${final_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/finalize.log"
done

sha256sum \
    "$P3_JSON" \
    "$CONTRACT" \
    "$AUDITOR" \
    "$FINALIZER" \
    "$SOURCE" \
    "$MAPPING" \
    "$OBJECT" \
    "$EXTENSION" \
    "$EVIDENCE/code_object/gfx1201_code_object.hsaco" \
    "$EVIDENCE/code_object/production_kernel.isa.txt" \
    "$EVIDENCE/code_object/production_kernel.resources.txt" \
    "$EVIDENCE/code_object_audit.json" \
    "$EVIDENCE/replay_1/result.json" \
    "$EVIDENCE/replay_2/result.json" \
    "$EVIDENCE/phase4a2_p4_width64_production_code_object_resource_audit.json" \
    "$EVIDENCE/PHASE4A2_P4_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

"$PYTHON_BIN" - "$EVIDENCE/phase4a2_p4_width64_production_code_object_resource_audit.json" <<'PY_JSON'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT_PASS"
)
assert all(data["gates"].values())
assert all(data["static_gates"].values())
assert data["runtime_replays"]["exact_match"] is True
summary = data["resource_summary"]
assert summary["group_segment_fixed_size"] == 2048
assert summary["private_segment_fixed_size"] == 0
assert summary["mfma_or_wmma_instructions"] == 12
assert summary["lds_load_instructions"] == 8
assert summary["lds_store_instructions"] == 2
assert summary["ds_bpermute_b32_instructions"] == 192
assert summary["ds_mnemonics"] == [
    "ds_bpermute_b32",
    "ds_load_b128",
    "ds_store_b128",
]
assert summary["block_barriers"] == 6
assert summary["scratch_instruction_count"] == 0
print("PHASE4A2_P4_JSON_AUDIT: PASS")
PY_JSON

echo
echo "WIDTH64_FRESH_PRODUCTION_BUILD: PASS"
echo "WIDTH64_GFX1201_CODE_OBJECT_BOUND: PASS"
echo "WIDTH64_EXACT_PRODUCTION_KERNEL_BOUND: PASS"
echo "WIDTH64_RESOURCE_FIELDS_RECORDED: PASS"
echo "WIDTH64_GROUP_2048_PRIVATE_ZERO: PASS"
echo "WIDTH64_MFMA12_LDSLOAD8_STORE2_BARRIER6: PASS"
echo "WIDTH64_DS_BPERMUTE_B32_192: PASS"
echo "WIDTH64_SCRATCH_ZERO: PASS"
echo "WIDTH64_FRESH_RUNTIME_REPLAY_TWICE: PASS"
echo "WIDTH64_P4_NO_PRODUCTION_CODE_CHANGE: PASS"
echo "PHASE4A2_P4_JSON_AUDIT: PASS"
echo "PHASE4A2_P4_MAP_CONTEXT: RECORDED"
echo "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT: PASS"
echo "Evidence: $EVIDENCE"
