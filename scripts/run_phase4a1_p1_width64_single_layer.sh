#!/usr/bin/env bash
# TCNN_RDNA4_P4A1_P1_WIDTH64_FOUR_K_TILE_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/scripts/phase4a1_p1_width64_single_layer.cpp"
FINALIZER="$ROOT/scripts/finalize_phase4a1_p1.py"
BUILD_DIR="$ROOT/build/phase4a1_p1"
BINARY="$BUILD_DIR/phase4a1_p1_width64_single_layer"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"

resolve_p0_evidence() {
    if [[ -n "${P0_EVIDENCE:-}" ]]; then
        printf '%s\n' "$P0_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/cpu_oracle.json" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a1_p0_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A1-P0 evidence directory found." >&2
    return 1
}

P0_DIR="$(resolve_p0_evidence)"
P0_JSON="$P0_DIR/cpu_oracle.json"
INPUT_BIN="$P0_DIR/input_fp16_row_major.bin"
WEIGHT_BIN="$P0_DIR/weight_1_fp16_col_major.bin"
HASH_JSON="$P0_DIR/tensor_hashes.json"

EVIDENCE="${1:-${P1_EVIDENCE:-$EVIDENCE_ROOT/phase4a1_p1_$(date -u +%Y%m%dT%H%M%SZ)}}"

mkdir -p \
    "$BUILD_DIR" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

test -x "$HIPCC"
test -f "$SOURCE"
test -f "$FINALIZER"
test -f "$P0_JSON"
test -f "$INPUT_BIN"
test -f "$WEIGHT_BIN"
test -f "$HASH_JSON"
test -f /opt/rocm/include/rocwmma/rocwmma.hpp

echo "===== P0 PREREQUISITE ====="
python - \
    "$P0_JSON" \
    "$HASH_JSON" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" <<'PY'
import hashlib
import json
import pathlib
import sys

oracle_path = pathlib.Path(sys.argv[1])
hash_path = pathlib.Path(sys.argv[2])
input_path = pathlib.Path(sys.argv[3])
weight_path = pathlib.Path(sys.argv[4])

oracle = json.loads(oracle_path.read_text())
hashes = json.loads(hash_path.read_text())

assert oracle["decision"] == (
    "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
)
assert all(oracle["gates"].values())

plan = oracle["tile_plan"]
assert plan["decision"] == "WIDTH64_FUSED_MLP_TILE_PLAN_LOCKED"
assert plan["execution"]["waves_per_block"] == 4
assert plan["execution"]["threads_per_block"] == 128
assert plan["execution"]["output_tiles_per_layer"] == 4
assert plan["execution"]["k_tiles_per_output_tile"] == 4

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert input_path.stat().st_size == 16 * 64 * 2
assert weight_path.stat().st_size == 64 * 64 * 2

assert sha256(input_path) == (
    hashes["input_fp16_row_major"]["sha256"]
)
assert sha256(weight_path) == (
    hashes["weight_1_fp16_physical_col_major"]["sha256"]
)

print("PHASE4A1_P0_PREREQUISITE: PASS")
print("PHASE4A1_P1_INPUT_HASHES: VERIFIED")
PY

{
    echo "===== PHASE 4A1-P1 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p0_evidence: $P0_DIR"
    echo "p0_json: $P0_JSON"
    echo "input_bin: $INPUT_BIN"
    echo "weight_bin: $WEIGHT_BIN"
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
        "$SOURCE" \
        "$FINALIZER" \
        "$P0_JSON" \
        "$INPUT_BIN" \
        "$WEIGHT_BIN" \
        /opt/rocm/include/rocwmma/rocwmma.hpp
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== FINALIZER SELF-CHECK ====="
python -m py_compile "$FINALIZER"
echo "PHASE4A1_P1_FINALIZER_PY_COMPILE: PASS"

compile=(
    "$HIPCC"
    -std=c++17
    -O2
    -gline-tables-only
    --offload-arch=gfx1201
    -I/opt/rocm/include
    "$SOURCE"
    -o "$BINARY"
)

printf '%q ' "${compile[@]}" \
    > "$EVIDENCE/compile_command.txt"
printf '\n' >> "$EVIDENCE/compile_command.txt"

echo
echo "===== BUILD ====="
"${compile[@]}" 2>&1 | tee "$EVIDENCE/build.log"

sha256sum \
    "$SOURCE" \
    "$FINALIZER" \
    "$P0_JSON" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$BINARY" \
    > "$EVIDENCE/SHA256SUMS"

echo
echo "===== FRESH PROCESS 1 ====="
"$BINARY" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_1/stage_outputs.csv" \
    2>&1 | tee "$EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="
"$BINARY" \
    "$INPUT_BIN" \
    "$WEIGHT_BIN" \
    "$EVIDENCE/process_2/result.json" \
    "$EVIDENCE/process_2/stage_outputs.csv" \
    2>&1 | tee "$EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    for stage in 1 2 3 4; do
        grep -Fx \
            "WIDTH64_K_TILE_STAGE_${stage}: PASS" \
            "$EVIDENCE/$process/run.log"
    done

    grep -Fx \
        "WIDTH64_FOUR_K_TILE_ACCUMULATION: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_ALL_FOUR_WAVES_OUTPUT_COVERAGE: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "WIDTH64_SINGLE_LAYER_VS_CPU_FP64: PASS" \
        "$EVIDENCE/$process/run.log"

    grep -Fx \
        "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_PROCESS: PASS" \
        "$EVIDENCE/$process/run.log"
done

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --process-1-json "$EVIDENCE/process_1/result.json" \
    --process-2-json "$EVIDENCE/process_2/result.json" \
    --process-1-csv "$EVIDENCE/process_1/stage_outputs.csv" \
    --process-2-csv "$EVIDENCE/process_2/stage_outputs.csv" \
    --evidence "$EVIDENCE" \
    --p0-json "$P0_JSON" \
    2>&1 | tee "$EVIDENCE/finalize.log"

grep -Fx \
    "WIDTH64_SINGLE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS" \
    "$EVIDENCE/finalize.log"

grep -Fx \
    "PHASE4A1_P1_CONSOLIDATED_EVIDENCE: RECORDED" \
    "$EVIDENCE/finalize.log"

grep -Fx \
    "PHASE4A1_P1_WIDTH64_SINGLE_LAYER: PASS" \
    "$EVIDENCE/finalize.log"

python - \
    "$EVIDENCE/phase4a1_p1_width64_single_layer.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())

assert data["decision"] == (
    "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_CONSOLIDATED_PASS"
)
assert all(data["gates"].values())

result = data["result"]
assert result["decision"] == (
    "PHASE4A1_P1_WIDTH64_SINGLE_LAYER_PASS"
)
assert all(result["gates"].values())

assert [stage["k_terms"] for stage in result["stages"]] == [
    16,
    32,
    48,
    64,
]
assert all(stage["passed"] for stage in result["stages"])
assert all(tile["passed"] for tile in result["wave_tiles"])
assert result["diagnostics"]["wave_entry_counts"] == [1, 1, 1, 1]
assert result["diagnostics"]["wave_exit_counts"] == [1, 1, 1, 1]

print("PHASE4A1_P1_JSON_AUDIT: PASS")
PY

sha256sum \
    "$EVIDENCE/phase4a1_p1_width64_single_layer.json" \
    "$EVIDENCE/phase4a1_p1_stage_outputs.csv" \
    "$EVIDENCE/PHASE4A1_P1_REPORT.md" \
    >> "$EVIDENCE/SHA256SUMS"

echo
echo "WIDTH64_FOUR_K_TILE_ACCUMULATION: PASS"
echo "WIDTH64_ALL_FOUR_WAVES_OUTPUT_COVERAGE: PASS"
echo "WIDTH64_SINGLE_LAYER_VS_CPU_FP64: PASS"
echo "WIDTH64_SINGLE_LAYER_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "PHASE4A1_P1_MAP_CONTEXT: RECORDED"
echo "PHASE4A1_P1_WIDTH64_SINGLE_LAYER: PASS"
echo "Evidence: $EVIDENCE"
