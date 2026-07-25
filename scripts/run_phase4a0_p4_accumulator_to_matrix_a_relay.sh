#!/usr/bin/env bash
# TCNN_RDNA4_P4A0_P4_ACC_TO_A_RELAY_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR="$ROOT/scripts/generate_phase4a0_p4_relay_source.py"
BUILD_DIR="$ROOT/build/phase4a0_p4"
GENERATED_SOURCE="$BUILD_DIR/phase4a0_p4_acc_to_a_relay.cpp"
BINARY="$BUILD_DIR/phase4a0_p4_acc_to_a_relay"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"

resolve_p3_evidence() {
    if [[ -n "${P3_EVIDENCE:-}" ]]; then
        printf '%s\n' "$P3_EVIDENCE"
        return
    fi

    if [[ $# -ge 1 && -n "${1:-}" ]]; then
        printf '%s\n' "$1"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/phase4a0_p3_analysis.json" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a0_p3_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A0-P3 evidence directory found." >&2
    return 1
}

P3_DIR="$(resolve_p3_evidence "${1:-}")"
P3_JSON="$P3_DIR/phase4a0_p3_analysis.json"
EVIDENCE="${2:-${P4_EVIDENCE:-$EVIDENCE_ROOT/phase4a0_p4_$(date -u +%Y%m%dT%H%M%SZ)}}"

mkdir -p \
    "$BUILD_DIR" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

test -x "$HIPCC"
test -f "$GENERATOR"
test -f "$P3_JSON"
test -f /opt/rocm/include/rocwmma/rocwmma.hpp
test -f /opt/rocm/include/rocwmma/rocwmma_transforms.hpp

echo "===== P3 PREREQUISITE ====="
python - "$P3_JSON" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"
assert all(data["gates"].values())
assert data["context"]["arch"].startswith("gfx1201")
assert int(data["context"]["warp_size"]) == 32
assert (
    "matrix_a_to_accumulator"
    in data["kernel_policy"]["coordinate_reindex_required_pairs"]
)
print("PHASE4A0_P3_PREREQUISITE: PASS")
PY

{
    echo "===== PHASE 4A0-P4 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p3_evidence: $P3_DIR"
    echo "p3_json: $P3_JSON"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo
    echo "===== HIPCC ====="
    "$HIPCC" --version
    echo
    echo "===== ROCWMMA PACKAGE ====="
    dpkg-query -W -f='${Package} ${Version}\n' 'rocwmma*' 2>/dev/null || true
    echo
    echo "===== INPUT HASHES ====="
    sha256sum \
        "$GENERATOR" \
        "$P3_JSON" \
        /opt/rocm/include/rocwmma/rocwmma.hpp \
        /opt/rocm/include/rocwmma/rocwmma_transforms.hpp
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== GENERATOR SELF-CHECK ====="
python -m py_compile "$GENERATOR"
echo "PHASE4A0_P4_GENERATOR_PY_COMPILE: PASS"

echo
echo "===== SOURCE GENERATION ====="
python "$GENERATOR" \
    --p3-json "$P3_JSON" \
    --output-source "$GENERATED_SOURCE" \
    --output-manifest "$EVIDENCE/source_manifest.json" \
    2>&1 | tee "$EVIDENCE/source_generation.log"

grep -Fx \
    "PHASE4A0_P4_SOURCE_GENERATION: PASS" \
    "$EVIDENCE/source_generation.log"

cp "$GENERATED_SOURCE" "$EVIDENCE/phase4a0_p4_acc_to_a_relay.cpp"

compile=(
    "$HIPCC"
    -std=c++17
    -O2
    -gline-tables-only
    --offload-arch=gfx1201
    -I/opt/rocm/include
    "$GENERATED_SOURCE"
    -o "$BINARY"
)

printf '%q ' "${compile[@]}" > "$EVIDENCE/compile_command.txt"
printf '\n' >> "$EVIDENCE/compile_command.txt"

echo
echo "===== BUILD ====="
"${compile[@]}" 2>&1 | tee "$EVIDENCE/build.log"

sha256sum \
    "$GENERATOR" \
    "$P3_JSON" \
    "$GENERATED_SOURCE" \
    "$BINARY" \
    > "$EVIDENCE/SHA256SUMS"

echo
echo "===== FRESH PROCESS 1 ====="
"$BINARY" \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_1/matrix.csv" \
    2>&1 | tee "$EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="
"$BINARY" \
    "$EVIDENCE/process_2/result.json" \
    "$EVIDENCE/process_2/matrix.csv" \
    2>&1 | tee "$EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    grep -Fx "ROCWMMA_P4_WAVE32_CONTEXT: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P4_P3_MAPPING_EMBEDDED: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P4_GUARD_REGIONS: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P4_ACCUMULATOR_EPILOGUE: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_ACC_TO_A_REINDEX: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_ACC_TO_A_FP16_CAST: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_ACC_TO_A_STORED_MATRIX: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx \
        "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PROCESS: PASS" \
        "$EVIDENCE/$process/run.log"
done

echo
echo "===== REPRODUCIBILITY ====="
cmp \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_2/result.json"
cmp \
    "$EVIDENCE/process_1/matrix.csv" \
    "$EVIDENCE/process_2/matrix.csv"

echo "ROCWMMA_ACC_TO_A_FRESH_PROCESS_REPRODUCIBILITY: PASS"

cp "$EVIDENCE/process_1/result.json" \
   "$EVIDENCE/phase4a0_p4_relay_result.json"
cp "$EVIDENCE/process_1/matrix.csv" \
   "$EVIDENCE/phase4a0_p4_relay_matrix.csv"

python - "$EVIDENCE/phase4a0_p4_relay_result.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())

assert data["decision"] == (
    "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS"
)
assert all(data["gates"].values())
assert data["metrics"]["mismatch_count"] == 0
assert data["metrics"]["max_abs"] == 0
assert data["metrics"]["relu_clamped_count"] > 0
assert data["metrics"]["positive_count"] > 0
assert data["metrics"]["fp16_cast_changed_count"] > 0
assert data["metrics"]["input_quantization_changed"] > 0

print("PHASE4A0_P4_JSON_AUDIT: PASS")
PY

cat > "$EVIDENCE/PHASE4A0_P4_RELAY.md" <<EOF
# Phase 4A0-P4 — Accumulator to matrix-A relay proof

Decision: **PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS**

- P3 evidence: \`$P3_DIR\`
- P3 JSON SHA256: \`$(sha256sum "$P3_JSON" | awk '{print $1}')\`
- Target: \`gfx1201\`, Wave32
- Tile: \`16×16×16\`
- Accumulator epilogue: FP32 output-column bias + ReLU
- Cross-role relay: P3-derived Wave32 shuffle permutation
- Cast: FP32 → FP16
- Destination: \`from_register_file<FragA>\`
- Final validation: bitwise-equal full 16×16 matrix
- Fresh processes: 2/2 identical
EOF

sha256sum \
    "$EVIDENCE/source_manifest.json" \
    "$EVIDENCE/phase4a0_p4_acc_to_a_relay.cpp" \
    "$EVIDENCE/phase4a0_p4_relay_result.json" \
    "$EVIDENCE/phase4a0_p4_relay_matrix.csv" \
    "$EVIDENCE/PHASE4A0_P4_RELAY.md" \
    >> "$EVIDENCE/SHA256SUMS"

echo
echo "ROCWMMA_P4_MAP_CONTEXT: RECORDED"
echo "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY: PASS"
echo "Evidence: $EVIDENCE"
