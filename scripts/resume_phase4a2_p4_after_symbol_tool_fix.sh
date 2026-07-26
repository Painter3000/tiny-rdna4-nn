#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P4_EVIDENCE_OBJECT_PATH_FIX_003
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT="$ROOT/contracts/phase4a2_p4_production_code_object_audit_contract.json"
AUDITOR="$ROOT/scripts/audit_phase4a2_p4_code_object.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p4.py"
P3_PROBE="$ROOT/probes/phase4a2_p3_runtime_integration_probe.py"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
P3_FILENAME="phase4a2_p3_width64_runtime_integration_lifecycle_closure.json"

resolve_p4_evidence() {
    if [[ -n "${PHASE4A2_P4_RESUME_EVIDENCE:-}" ]]; then
        printf '%s\n' "$PHASE4A2_P4_RESUME_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/build.log" ]] && \
           [[ -d "$candidate/build/temp" ]] && \
           [[ -d "$candidate/build/lib" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a2_p4_*' \
            -print |
        sort -r
    )

    echo "No resumable Phase 4A2-P4 evidence directory found." >&2
    return 1
}

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

EVIDENCE="$(resolve_p4_evidence)"
P3_DIR="$(resolve_p3_evidence)"
P3_JSON="$P3_DIR/$P3_FILENAME"
BINDINGS_DIR="$ROOT/bindings/torch"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

case "$(basename "$EVIDENCE")" in
    phase4a2_p4_*)
        ;;
    *)
        echo "Refusing unsafe evidence path: $EVIDENCE" >&2
        exit 2
        ;;
esac

for path in \
    "$CONTRACT" \
    "$AUDITOR" \
    "$FINALIZER" \
    "$P3_PROBE" \
    "$P3_JSON" \
    "$EVIDENCE/build.log"; do
    test -f "$path"
done

# setuptools preserves the source-relative "../../src" component in the
# object output path. After path normalization, the production object can be
# located under "$EVIDENCE/src", not below "$EVIDENCE/build/temp".
# Search the complete evidence tree, matching the original P4 runner.
mapfile -t OBJECTS < <(
    find "$EVIDENCE" \
        -type f \
        -name 'rocwmma_width64_mlp.o' \
        -print |
    sort -u
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

echo "===== RESUME PHASE 4A2-P4 AFTER OBJECT-PATH FIX ====="
echo "Evidence: $EVIDENCE"
echo "Production object: $OBJECT"
echo "Production extension: $EXTENSION"

PYTHONPYCACHEPREFIX="$EVIDENCE/pycache_symbol_tool_fix" \
    "$PYTHON_BIN" -m py_compile "$AUDITOR" "$FINALIZER" "$P3_PROBE"

# TCNN_RDNA4_P4A2_P4_RELEASE_REPRODUCIBILITY_007:
# Preserve the prior successful P4 evidence before regenerating it. The
# finalizer will require byte identity of the production object, linked
# extension, extracted code object, source/mapping/bridge hashes, and both
# runtime replay JSON files.
PRIOR_DIR="$EVIDENCE/prior_p4_release_closure"
PRIOR_P4_JSON="$PRIOR_DIR/phase4a2_p4_prior.json"
PRIOR_REPLAY_1="$PRIOR_DIR/replay_1_prior.json"
PRIOR_REPLAY_2="$PRIOR_DIR/replay_2_prior.json"

test -f \
    "$EVIDENCE/phase4a2_p4_width64_production_code_object_resource_audit.json"
test -f "$EVIDENCE/replay_1/result.json"
test -f "$EVIDENCE/replay_2/result.json"

rm -rf "$PRIOR_DIR"
mkdir -p "$PRIOR_DIR"

cp --preserve=mode,timestamps \
    "$EVIDENCE/phase4a2_p4_width64_production_code_object_resource_audit.json" \
    "$PRIOR_P4_JSON"
cp --preserve=mode,timestamps \
    "$EVIDENCE/replay_1/result.json" \
    "$PRIOR_REPLAY_1"
cp --preserve=mode,timestamps \
    "$EVIDENCE/replay_2/result.json" \
    "$PRIOR_REPLAY_2"

# The fresh build already passed. Reuse it byte-for-byte and restart only the
# code-object audit, runtime replays, and finalizer.
rm -rf \
    "$EVIDENCE/code_object" \
    "$EVIDENCE/replay_1" \
    "$EVIDENCE/replay_2"

rm -f \
    "$EVIDENCE/code_object_audit.json" \
    "$EVIDENCE/code_object_audit.log" \
    "$EVIDENCE/finalize.log" \
    "$EVIDENCE/phase4a2_p4_width64_production_code_object_resource_audit.json" \
    "$EVIDENCE/PHASE4A2_P4_REPORT.md" \
    "$EVIDENCE/SHA256SUMS"

mkdir -p \
    "$EVIDENCE/code_object" \
    "$EVIDENCE/replay_1" \
    "$EVIDENCE/replay_2"

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
    --prior-p4-json "$PRIOR_P4_JSON" \
    --prior-replay-1 "$PRIOR_REPLAY_1" \
    --prior-replay-2 "$PRIOR_REPLAY_2" \
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
    "WIDTH64_P4_RELEASE_GIT_CONTEXT: PASS"
    "WIDTH64_PRIOR_P4_PRODUCTION_ARTIFACTS_BYTE_IDENTICAL: PASS"
    "WIDTH64_PRIOR_P4_RUNTIME_REPLAYS_BYTE_IDENTICAL: PASS"
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
    "$OBJECT" \
    "$EXTENSION" \
    "$EVIDENCE/code_object/gfx1201_code_object.hsaco" \
    "$EVIDENCE/code_object/production_kernel.isa.txt" \
    "$EVIDENCE/code_object/production_kernel.resources.txt" \
    "$EVIDENCE/code_object_audit.json" \
    "$EVIDENCE/replay_1/result.json" \
    "$EVIDENCE/replay_2/result.json" \
    "$PRIOR_P4_JSON" \
    "$PRIOR_REPLAY_1" \
    "$PRIOR_REPLAY_2" \
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
assert data["git_context"]["mode"] == "p4_release_commit"
assert data["prior_p4_equivalence"]["provided"] is True
assert data["prior_p4_equivalence"]["all_equal"] is True
assert all(data["prior_p4_equivalence"]["gates"].values())
audit = data["code_object_audit"]["data"]
assert audit["tools"]["llvm_nm_required"] is False
print("PHASE4A2_P4_JSON_AUDIT: PASS")
PY_JSON

echo
echo "WIDTH64_SYMBOL_INVENTORY_READELF_FALLBACK: PASS"
echo "WIDTH64_EVIDENCE_OBJECT_PATH_RESOLVED: PASS"
echo "PHASE4A2_P4_JSON_AUDIT: PASS"
echo "PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT: PASS"
echo "Evidence: $EVIDENCE"
