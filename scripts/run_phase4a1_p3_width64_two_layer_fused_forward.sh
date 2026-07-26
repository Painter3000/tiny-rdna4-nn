#!/usr/bin/env bash
# TCNN_RDNA4_P4A1_P3_WIDTH64_TWO_LAYER_FUSED_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPARE="$ROOT/scripts/prepare_phase4a1_p3.py"
SOURCE="$ROOT/scripts/phase4a1_p3_width64_two_layer_fused.cpp"
FINALIZER="$ROOT/scripts/finalize_phase4a1_p3.py"
BUILD_DIR="$ROOT/build/phase4a1_p3"
GENERATED_HEADER="$BUILD_DIR/phase4a1_p3_mapping_generated.hpp"
BINARY="$BUILD_DIR/phase4a1_p3_width64_two_layer_fused"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"

resolve_evidence() {
    local explicit="$1"
    local pattern="$2"
    local filename="$3"
    local label="$4"
    if [[ -n "$explicit" ]]; then
        printf '%s\n' "$explicit"
        return
    fi
    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/$filename" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(find "$EVIDENCE_ROOT" -maxdepth 1 -type d -name "$pattern" -print | sort -r)
    echo "No valid $label evidence directory found." >&2
    return 1
}

P0_DIR="$(resolve_evidence "${P0_EVIDENCE:-}" 'phase4a1_p0_*' 'cpu_oracle.json' 'Phase 4A1-P0')"
P1_DIR="$(resolve_evidence "${P1_EVIDENCE:-}" 'phase4a1_p1_*' 'phase4a1_p1_width64_single_layer.json' 'Phase 4A1-P1')"
P2_DIR="$(resolve_evidence "${PHASE4A1_P2_SOURCE_EVIDENCE:-}" 'phase4a1_p2_*' 'phase4a1_p2_width64_hidden_lds.json' 'Phase 4A1-P2')"

P0_JSON="$P0_DIR/cpu_oracle.json"
P1_JSON="$P1_DIR/phase4a1_p1_width64_single_layer.json"
P2_JSON="$P2_DIR/phase4a1_p2_width64_hidden_lds.json"
P2_MANIFEST="$P2_DIR/source_manifest.json"
P2_HEADER="$P2_DIR/phase4a1_p2_mapping_generated.hpp"
HASH_JSON="$P0_DIR/tensor_hashes.json"
INPUT_BIN="$P0_DIR/input_fp16_row_major.bin"
WEIGHT_1_BIN="$P0_DIR/weight_1_fp16_col_major.bin"
WEIGHT_2_BIN="$P0_DIR/weight_2_fp16_col_major.bin"
BIAS_1_BIN="$P0_DIR/bias_1_fp32.bin"
BIAS_2_BIN="$P0_DIR/bias_2_fp32.bin"
EXPECTED_HIDDEN_1_BIN="$P0_DIR/hidden_1_fp16_row_major.bin"
EXPECTED_HIDDEN_2_BIN="$P0_DIR/hidden_2_fp16_row_major.bin"

EVIDENCE="${1:-${PHASE4A1_P3_EVIDENCE:-$EVIDENCE_ROOT/phase4a1_p3_$(date -u +%Y%m%dT%H%M%SZ)}}"
case "$(basename "$EVIDENCE")" in
    phase4a1_p3_*) ;;
    *) echo "Refusing unsafe evidence path: $EVIDENCE" >&2; exit 2 ;;
esac
if [[ -d "$EVIDENCE" ]] && [[ -n "$(find "$EVIDENCE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty evidence directory: $EVIDENCE" >&2
    exit 2
fi

mkdir -p "$BUILD_DIR" "$EVIDENCE/process_1" "$EVIDENCE/process_2"
for path in "$PREPARE" "$SOURCE" "$FINALIZER" "$P0_JSON" "$P1_JSON" "$P2_JSON" "$P2_MANIFEST" "$P2_HEADER" "$HASH_JSON" "$INPUT_BIN" "$WEIGHT_1_BIN" "$WEIGHT_2_BIN" "$BIAS_1_BIN" "$BIAS_2_BIN" "$EXPECTED_HIDDEN_1_BIN" "$EXPECTED_HIDDEN_2_BIN"; do
    test -f "$path"
done
test -x "$HIPCC"
test -f /opt/rocm/include/rocwmma/rocwmma.hpp
test -f /opt/rocm/include/rocwmma/rocwmma_transforms.hpp

echo "===== PREREQUISITE CHAIN ====="
python - "$P0_JSON" "$P1_JSON" "$P2_JSON" <<'PY_A1P3_PREREQ'
import json, pathlib, sys
p0, p1, p2 = [json.loads(pathlib.Path(path).read_text()) for path in sys.argv[1:]]
assert p0["decision"] == "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
assert all(p0["gates"].values())
assert p1["decision"] == "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
assert all(p1["gates"].values())
assert p2["decision"] == "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_CONSOLIDATED_PASS"
assert all(p2["gates"].values())
assert p2["result"]["metrics"]["mismatch_count"] == 0
print("PHASE4A1_P0_PREREQUISITE: PASS")
print("PHASE4A1_P1_PREREQUISITE: PASS")
print("PHASE4A1_P2_PREREQUISITE: PASS")
print("PHASE4A1_P3_PREREQUISITE_CHAIN: PASS")
PY_A1P3_PREREQ

{
    echo "===== PHASE 4A1-P3 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p0_evidence: $P0_DIR"
    echo "p1_evidence: $P1_DIR"
    echo "p2_evidence: $P2_DIR"
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo "===== HIPCC ====="
    "$HIPCC" --version
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== PYTHON SELF-CHECK ====="
python -m py_compile "$PREPARE" "$FINALIZER"
echo "PHASE4A1_P3_PYTHON_SELF_CHECK: PASS"

echo
echo "===== PREPARATION ====="
python "$PREPARE" \
    --p0-json "$P0_JSON" --p1-json "$P1_JSON" --p2-json "$P2_JSON" \
    --p2-manifest "$P2_MANIFEST" --p2-header "$P2_HEADER" \
    --input-bin "$INPUT_BIN" --weight-1-bin "$WEIGHT_1_BIN" --weight-2-bin "$WEIGHT_2_BIN" \
    --bias-1-bin "$BIAS_1_BIN" --bias-2-bin "$BIAS_2_BIN" \
    --expected-hidden-1-bin "$EXPECTED_HIDDEN_1_BIN" --expected-hidden-2-bin "$EXPECTED_HIDDEN_2_BIN" \
    --tensor-hashes "$HASH_JSON" --output-header "$GENERATED_HEADER" \
    --output-manifest "$EVIDENCE/preparation_manifest.json" \
    2>&1 | tee "$EVIDENCE/preparation.log"
grep -Fx "PHASE4A1_P3_PREPARATION: PASS" "$EVIDENCE/preparation.log"
cp "$GENERATED_HEADER" "$EVIDENCE/phase4a1_p3_mapping_generated.hpp"

compile=("$HIPCC" -std=c++17 -O2 -gline-tables-only --offload-arch=gfx1201 -I/opt/rocm/include -I"$BUILD_DIR" "$SOURCE" -o "$BINARY")
printf '%q ' "${compile[@]}" > "$EVIDENCE/compile_command.txt"
printf '\n' >> "$EVIDENCE/compile_command.txt"

echo
echo "===== BUILD ====="
"${compile[@]}" 2>&1 | tee "$EVIDENCE/build.log"

sha256sum "$PREPARE" "$SOURCE" "$FINALIZER" "$GENERATED_HEADER" "$P0_JSON" "$P1_JSON" "$P2_JSON" "$P2_MANIFEST" "$INPUT_BIN" "$WEIGHT_1_BIN" "$WEIGHT_2_BIN" "$BIAS_1_BIN" "$BIAS_2_BIN" "$EXPECTED_HIDDEN_1_BIN" "$EXPECTED_HIDDEN_2_BIN" "$BINARY" > "$EVIDENCE/SHA256SUMS"

run_process() {
    local process="$1"
    "$BINARY" "$INPUT_BIN" "$WEIGHT_1_BIN" "$WEIGHT_2_BIN" "$BIAS_1_BIN" "$BIAS_2_BIN" "$EXPECTED_HIDDEN_1_BIN" "$EXPECTED_HIDDEN_2_BIN" "$EVIDENCE/$process/result.json" "$EVIDENCE/$process/hidden_2.csv" 2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== FRESH PROCESS 1 ====="
run_process process_1
echo
echo "===== FRESH PROCESS 2 ====="
run_process process_2

markers=(
    "WIDTH64_LAYER1_HIDDEN_BITWISE_CORRECTNESS: PASS"
    "WIDTH64_LAYER1_LDS_PUBLICATION: PASS"
    "WIDTH64_LAYER1_CROSS_WAVE_VISIBILITY: PASS"
    "WIDTH64_LAYER2_INPUT_FROM_LDS_ONLY: PASS"
    "WIDTH64_LAYER2_FOUR_K_TILE_ACCUMULATION: PASS"
    "WIDTH64_LAYER2_HIDDEN_BITWISE_CORRECTNESS: PASS"
    "WIDTH64_NO_INTERMEDIATE_GLOBAL_STORE_RELOAD: PASS"
    "RDNA4_WIDTH64_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: PASS"
    "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_PROCESS: PASS"
)
for process in process_1 process_2; do
    for marker in "${markers[@]}"; do
        grep -Fx "$marker" "$EVIDENCE/$process/run.log"
    done
done

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --process-1-json "$EVIDENCE/process_1/result.json" \
    --process-2-json "$EVIDENCE/process_2/result.json" \
    --process-1-csv "$EVIDENCE/process_1/hidden_2.csv" \
    --process-2-csv "$EVIDENCE/process_2/hidden_2.csv" \
    --preparation-manifest "$EVIDENCE/preparation_manifest.json" \
    --p0-json "$P0_JSON" --p2-json "$P2_JSON" --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

grep -Fx "WIDTH64_TWO_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS" "$EVIDENCE/finalize.log"
grep -Fx "PHASE4A1_P3_CONSOLIDATED_EVIDENCE: RECORDED" "$EVIDENCE/finalize.log"
grep -Fx "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED: PASS" "$EVIDENCE/finalize.log"

python - "$EVIDENCE/phase4a1_p3_width64_two_layer_fused.json" <<'PY_A1P3_AUDIT'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_CONSOLIDATED_PASS"
assert all(data["gates"].values())
result = data["result"]
assert result["decision"] == "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED_PASS"
assert all(result["gates"].values())
assert result["diagnostics"]["hidden_1_lds_mismatch_count"] == 0
assert result["diagnostics"]["layer_2_k_tile_counts"] == [4, 4, 4, 4]
assert result["metrics"]["hidden_2_mismatch_count"] == 0
assert result["metrics"]["nonfinite_count"] == 0
assert result["metrics"]["max_abs"] == 0
assert result["topology"]["hidden_1_transport"] == "LDS_only"
assert not result["topology"]["hidden_1_global_store"]
assert not result["topology"]["hidden_1_global_reload"]
print("PHASE4A1_P3_JSON_AUDIT: PASS")
PY_A1P3_AUDIT

sha256sum "$EVIDENCE/preparation_manifest.json" "$EVIDENCE/phase4a1_p3_mapping_generated.hpp" "$EVIDENCE/phase4a1_p3_width64_two_layer_fused.json" "$EVIDENCE/phase4a1_p3_hidden_2_output.csv" "$EVIDENCE/PHASE4A1_P3_REPORT.md" >> "$EVIDENCE/SHA256SUMS"

echo
echo "WIDTH64_LAYER1_HIDDEN_BITWISE_CORRECTNESS: PASS"
echo "WIDTH64_LAYER1_LDS_PUBLICATION: PASS"
echo "WIDTH64_LAYER1_CROSS_WAVE_VISIBILITY: PASS"
echo "WIDTH64_LAYER2_INPUT_FROM_LDS_ONLY: PASS"
echo "WIDTH64_LAYER2_FOUR_K_TILE_ACCUMULATION: PASS"
echo "WIDTH64_LAYER2_HIDDEN_BITWISE_CORRECTNESS: PASS"
echo "WIDTH64_NO_INTERMEDIATE_GLOBAL_STORE_RELOAD: PASS"
echo "WIDTH64_TWO_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "RDNA4_WIDTH64_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: PASS"
echo "PHASE4A1_P3_MAP_CONTEXT: RECORDED"
echo "PHASE4A1_P3_WIDTH64_TWO_LAYER_FUSED: PASS"
echo "Evidence: $EVIDENCE"
