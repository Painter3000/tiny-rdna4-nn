#!/usr/bin/env bash
# TCNN_RDNA4_P4A1_P2_HIDDEN_EPILOGUE_LDS_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GENERATOR="$ROOT/scripts/generate_phase4a1_p2_mapping_header.py"
SOURCE="$ROOT/scripts/phase4a1_p2_width64_hidden_lds.cpp"
FINALIZER="$ROOT/scripts/finalize_phase4a1_p2.py"

BUILD_DIR="$ROOT/build/phase4a1_p2"
GENERATED_HEADER="$BUILD_DIR/phase4a1_p2_mapping_generated.hpp"
BINARY="$BUILD_DIR/phase4a1_p2_width64_hidden_lds"

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
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name "$pattern" \
            -print |
        sort -r
    )

    echo "No valid $label evidence directory found." >&2
    return 1
}

P0_DIR="$(
    resolve_evidence \
        "${P0_EVIDENCE:-}" \
        'phase4a1_p0_*' \
        'cpu_oracle.json' \
        'Phase 4A1-P0'
)"

P1_DIR="$(
    resolve_evidence \
        "${P1_EVIDENCE:-}" \
        'phase4a1_p1_*' \
        'phase4a1_p1_width64_single_layer.json' \
        'Phase 4A1-P1'
)"

P3_DIR="$(
    resolve_evidence \
        "${P3_EVIDENCE:-}" \
        'phase4a0_p3_*' \
        'phase4a0_p3_analysis.json' \
        'Phase 4A0-P3'
)"

P4_DIR="$(
    resolve_evidence \
        "${P4_EVIDENCE:-}" \
        'phase4a0_p4_*' \
        'phase4a0_p4_relay_result.json' \
        'Phase 4A0-P4'
)"

P0_JSON="$P0_DIR/cpu_oracle.json"
P1_JSON="$P1_DIR/phase4a1_p1_width64_single_layer.json"
P3_JSON="$P3_DIR/phase4a0_p3_analysis.json"
P4_JSON="$P4_DIR/phase4a0_p4_relay_result.json"

HASH_JSON="$P0_DIR/tensor_hashes.json"
INPUT_BIN="$P0_DIR/input_fp16_row_major.bin"
WEIGHT_BIN="$P0_DIR/weight_1_fp16_col_major.bin"
BIAS_BIN="$P0_DIR/bias_1_fp32.bin"
EXPECTED_BIN="$P0_DIR/hidden_1_fp16_row_major.bin"

EVIDENCE="${1:-${PHASE4A1_P2_EVIDENCE:-$EVIDENCE_ROOT/phase4a1_p2_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a1_p2_*)
        ;;
    *)
        echo "Refusing unsafe evidence path: $EVIDENCE" >&2
        echo "Expected a basename beginning with phase4a1_p2_." >&2
        exit 2
        ;;
esac

if [[ -d "$EVIDENCE" ]] && \
   [[ -n "$(find "$EVIDENCE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty evidence directory: $EVIDENCE" >&2
    exit 2
fi

mkdir -p \
    "$BUILD_DIR" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

test -x "$HIPCC"
test -f "$GENERATOR"
test -f "$SOURCE"
test -f "$FINALIZER"

test -f "$P0_JSON"
test -f "$P1_JSON"
test -f "$P3_JSON"
test -f "$P4_JSON"

test -f "$HASH_JSON"
test -f "$INPUT_BIN"
test -f "$WEIGHT_BIN"
test -f "$BIAS_BIN"
test -f "$EXPECTED_BIN"

test -f /opt/rocm/include/rocwmma/rocwmma.hpp
test -f /opt/rocm/include/rocwmma/rocwmma_transforms.hpp

echo "===== PREREQUISITE CHAIN ====="

python - \
    "$P0_JSON" \
    "$P1_JSON" \
    "$P3_JSON" \
    "$P4_JSON" \
    "$HASH_JSON" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$BIAS_BIN" \
    "$EXPECTED_BIN" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    p0_path,
    p1_path,
    p3_path,
    p4_path,
    hashes_path,
    input_path,
    weight_path,
    bias_path,
    expected_path,
) = map(pathlib.Path, sys.argv[1:])

p0 = json.loads(p0_path.read_text())
p1 = json.loads(p1_path.read_text())
p3 = json.loads(p3_path.read_text())
p4 = json.loads(p4_path.read_text())
hashes = json.loads(hashes_path.read_text())

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert p0["decision"] == (
    "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
)
assert all(p0["gates"].values())

assert p1["decision"] == (
    "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
)
assert all(p1["gates"].values())
assert p1["p0_json_sha256"] == sha256(p0_path)

assert p3["decision"] == (
    "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"
)
assert all(p3["gates"].values())

assert p4["decision"] == (
    "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS"
)
assert all(p4["gates"].values())
assert p4["p3_sha256"] == sha256(p3_path)

assert input_path.stat().st_size == 16 * 64 * 2
assert weight_path.stat().st_size == 64 * 64 * 2
assert bias_path.stat().st_size == 64 * 4
assert expected_path.stat().st_size == 16 * 64 * 2

assert sha256(input_path) == (
    hashes["input_fp16_row_major"]["sha256"]
)
assert sha256(weight_path) == (
    hashes["weight_1_fp16_physical_col_major"]["sha256"]
)
assert sha256(bias_path) == (
    hashes["bias_1_fp32"]["sha256"]
)
assert sha256(expected_path) == (
    hashes["hidden_1_fp16_row_major"]["sha256"]
)

layer_1 = p0["cpu_oracle"]["layer_1_stats"]
assert layer_1["relu_clamped"] > 0
assert layer_1["relu_positive"] > 0
assert layer_1["fp16_cast_changed"] > 0

print("PHASE4A1_P0_PREREQUISITE: PASS")
print("PHASE4A1_P1_PREREQUISITE: PASS")
print("PHASE4A0_P3_MAP_PREREQUISITE: PASS")
print("PHASE4A0_P4_RELAY_PREREQUISITE: PASS")
print("PHASE4A1_P2_INPUT_HASHES: VERIFIED")
print("PHASE4A1_P2_PREREQUISITE_CHAIN: PASS")
PY

{
    echo "===== PHASE 4A1-P2 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p0_evidence: $P0_DIR"
    echo "p1_evidence: $P1_DIR"
    echo "p3_evidence: $P3_DIR"
    echo "p4_evidence: $P4_DIR"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo
    echo "===== HIPCC ====="
    "$HIPCC" --version
    echo
    echo "===== ROCWMMA PACKAGE ====="
    dpkg-query -W -f='${Package} ${Version}\n' \
        'rocwmma*' 2>/dev/null || true
    echo
    echo "===== INPUT HASHES ====="
    sha256sum \
        "$GENERATOR" \
        "$SOURCE" \
        "$FINALIZER" \
        "$P0_JSON" \
        "$P1_JSON" \
        "$P3_JSON" \
        "$P4_JSON" \
        "$INPUT_BIN" \
        "$WEIGHT_BIN" \
        "$BIAS_BIN" \
        "$EXPECTED_BIN" \
        /opt/rocm/include/rocwmma/rocwmma.hpp \
        /opt/rocm/include/rocwmma/rocwmma_transforms.hpp
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== PYTHON SELF-CHECK ====="

python -m py_compile \
    "$GENERATOR" \
    "$FINALIZER"

echo "PHASE4A1_P2_PYTHON_SELF_CHECK: PASS"

echo
echo "===== MAPPING HEADER GENERATION ====="

python "$GENERATOR" \
    --p0-json "$P0_JSON" \
    --p1-json "$P1_JSON" \
    --p3-json "$P3_JSON" \
    --p4-json "$P4_JSON" \
    --output-header "$GENERATED_HEADER" \
    --output-manifest "$EVIDENCE/source_manifest.json" \
    2>&1 | tee "$EVIDENCE/source_generation.log"

grep -Fx \
    "PHASE4A1_P2_MAPPING_HEADER_GENERATION: PASS" \
    "$EVIDENCE/source_generation.log"

cp \
    "$GENERATED_HEADER" \
    "$EVIDENCE/phase4a1_p2_mapping_generated.hpp"

compile=(
    "$HIPCC"
    -std=c++17
    -O2
    -gline-tables-only
    --offload-arch=gfx1201
    -I/opt/rocm/include
    -I"$BUILD_DIR"
    "$SOURCE"
    -o "$BINARY"
)

printf '%q ' "${compile[@]}" \
    > "$EVIDENCE/compile_command.txt"
printf '\n' >> "$EVIDENCE/compile_command.txt"

echo
echo "===== BUILD ====="

"${compile[@]}" \
    2>&1 | tee "$EVIDENCE/build.log"

sha256sum \
    "$GENERATOR" \
    "$SOURCE" \
    "$FINALIZER" \
    "$GENERATED_HEADER" \
    "$P0_JSON" \
    "$P1_JSON" \
    "$P3_JSON" \
    "$P4_JSON" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$BIAS_BIN" \
    "$EXPECTED_BIN" \
    "$BINARY" \
    > "$EVIDENCE/SHA256SUMS"

echo
echo "===== FRESH PROCESS 1 ====="

"$BINARY" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$BIAS_BIN" \
    "$EXPECTED_BIN" \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_1/hidden.csv" \
    2>&1 | tee "$EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="

"$BINARY" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$BIAS_BIN" \
    "$EXPECTED_BIN" \
    "$EVIDENCE/process_2/result.json" \
    "$EVIDENCE/process_2/hidden.csv" \
    2>&1 | tee "$EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    grep -Fx \
        "WIDTH64_HIDDEN_EPILOGUE: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_ACCUMULATOR_TO_A_RELAY: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_LDS_ALL_FOUR_WAVES_PUBLICATION: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_CROSS_WAVE_ROTATED_READBACK: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_LDS_HIDDEN_BITWISE_CORRECTNESS: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_LDS_BARRIER_VISIBILITY: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_PROCESS: PASS" \
        "$EVIDENCE/$process/run.log"
done

echo
echo "===== FINALIZE ====="

python "$FINALIZER" \
    --process-1-json "$EVIDENCE/process_1/result.json" \
    --process-2-json "$EVIDENCE/process_2/result.json" \
    --process-1-csv "$EVIDENCE/process_1/hidden.csv" \
    --process-2-csv "$EVIDENCE/process_2/hidden.csv" \
    --manifest "$EVIDENCE/source_manifest.json" \
    --p0-json "$P0_JSON" \
    --p1-json "$P1_JSON" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

grep -Fx \
    "WIDTH64_HIDDEN_LDS_FRESH_PROCESS_REPRODUCIBILITY: PASS" \
    "$EVIDENCE/finalize.log"

grep -Fx \
    "PHASE4A1_P2_CONSOLIDATED_EVIDENCE: RECORDED" \
    "$EVIDENCE/finalize.log"

grep -Fx \
    "PHASE4A1_P2_WIDTH64_HIDDEN_LDS: PASS" \
    "$EVIDENCE/finalize.log"

python - \
    "$EVIDENCE/phase4a1_p2_width64_hidden_lds.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())

assert data["decision"] == (
    "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_CONSOLIDATED_PASS"
)
assert all(data["gates"].values())

result = data["result"]

assert result["decision"] == (
    "PHASE4A1_P2_WIDTH64_HIDDEN_LDS_PASS"
)
assert all(result["gates"].values())

assert result["metrics"]["mismatch_count"] == 0
assert result["metrics"]["nonfinite_count"] == 0
assert result["metrics"]["max_abs"] == 0
assert result["metrics"]["relay_moved_entries_per_tile"] == 240

assert result["diagnostics"]["wave_entry_counts"] == [1, 1, 1, 1]
assert result["diagnostics"]["publication_counts"] == [1, 1, 1, 1]
assert result["diagnostics"]["consumer_counts"] == [1, 1, 1, 1]
assert result["diagnostics"]["producer_visibility_counts"] == [1, 1, 1, 1]

assert result["cross_wave_readback"] == [
    {"consumer_wave": 0, "producer_wave": 1},
    {"consumer_wave": 1, "producer_wave": 2},
    {"consumer_wave": 2, "producer_wave": 3},
    {"consumer_wave": 3, "producer_wave": 0},
]

print("PHASE4A1_P2_JSON_AUDIT: PASS")
PY

sha256sum \
    "$EVIDENCE/source_manifest.json" \
    "$EVIDENCE/phase4a1_p2_mapping_generated.hpp" \
    "$EVIDENCE/phase4a1_p2_width64_hidden_lds.json" \
    "$EVIDENCE/phase4a1_p2_hidden_readback.csv" \
    "$EVIDENCE/PHASE4A1_P2_REPORT.md" \
    >> "$EVIDENCE/SHA256SUMS"

echo
echo "WIDTH64_HIDDEN_EPILOGUE: PASS"
echo "WIDTH64_ACCUMULATOR_TO_A_RELAY: PASS"
echo "WIDTH64_LDS_ALL_FOUR_WAVES_PUBLICATION: PASS"
echo "WIDTH64_CROSS_WAVE_ROTATED_READBACK: PASS"
echo "WIDTH64_LDS_HIDDEN_BITWISE_CORRECTNESS: PASS"
echo "WIDTH64_LDS_BARRIER_VISIBILITY: PASS"
echo "WIDTH64_HIDDEN_LDS_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "PHASE4A1_P2_MAP_CONTEXT: RECORDED"
echo "PHASE4A1_P2_WIDTH64_HIDDEN_LDS: PASS"
echo "Evidence: $EVIDENCE"
