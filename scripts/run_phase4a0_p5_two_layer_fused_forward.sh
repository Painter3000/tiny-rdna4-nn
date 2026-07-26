#!/usr/bin/env bash
# TCNN_RDNA4_P4A0_P5_TWO_LAYER_FUSED_FORWARD_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR="$ROOT/scripts/generate_phase4a0_p5_two_layer_fused_source.py"
BUILD_DIR="$ROOT/build/phase4a0_p5"
GENERATED_SOURCE="$BUILD_DIR/phase4a0_p5_two_layer_fused_forward.cpp"
BINARY="$BUILD_DIR/phase4a0_p5_two_layer_fused_forward"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"

resolve_evidence() {
    local phase="$1"
    local explicit="$2"
    local pattern="$3"
    local filename="$4"

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

    echo "No valid $phase evidence directory found." >&2
    return 1
}

P3_DIR="$(
    resolve_evidence \
        "Phase 4A0-P3" \
        "${P3_EVIDENCE:-}" \
        'phase4a0_p3_*' \
        'phase4a0_p3_analysis.json'
)"
P4_DIR="$(
    resolve_evidence \
        "Phase 4A0-P4" \
        "${P4_EVIDENCE:-}" \
        'phase4a0_p4_*' \
        'phase4a0_p4_relay_result.json'
)"

P3_JSON="$P3_DIR/phase4a0_p3_analysis.json"
P4_JSON="$P4_DIR/phase4a0_p4_relay_result.json"
EVIDENCE="${1:-${P5_EVIDENCE:-$EVIDENCE_ROOT/phase4a0_p5_$(date -u +%Y%m%dT%H%M%SZ)}}"

mkdir -p \
    "$BUILD_DIR" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

test -x "$HIPCC"
test -f "$GENERATOR"
test -f "$P3_JSON"
test -f "$P4_JSON"
test -f /opt/rocm/include/rocwmma/rocwmma.hpp
test -f /opt/rocm/include/rocwmma/rocwmma_transforms.hpp

echo "===== P3 / P4 PREREQUISITES ====="
python - "$P3_JSON" "$P4_JSON" <<'PY'
import hashlib
import json
import pathlib
import sys

p3_path = pathlib.Path(sys.argv[1])
p4_path = pathlib.Path(sys.argv[2])

p3 = json.loads(p3_path.read_text())
p4 = json.loads(p4_path.read_text())

assert p3["decision"] == (
    "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"
)
assert all(p3["gates"].values())

assert p4["decision"] == (
    "PHASE4A0_P4_ACCUMULATOR_TO_MATRIX_A_RELAY_PASS"
)
assert all(p4["gates"].values())

p3_sha = hashlib.sha256(p3_path.read_bytes()).hexdigest()
assert p4["p3_sha256"] == p3_sha

print("PHASE4A0_P3_PREREQUISITE: PASS")
print("PHASE4A0_P4_PREREQUISITE: PASS")
print("PHASE4A0_P5_PREREQUISITE_CHAIN: PASS")
PY

{
    echo "===== PHASE 4A0-P5 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
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
        "$P3_JSON" \
        "$P4_JSON" \
        /opt/rocm/include/rocwmma/rocwmma.hpp \
        /opt/rocm/include/rocwmma/rocwmma_transforms.hpp
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== GENERATOR SELF-CHECK ====="
python -m py_compile "$GENERATOR"
echo "PHASE4A0_P5_GENERATOR_PY_COMPILE: PASS"

echo
echo "===== SOURCE GENERATION ====="
python "$GENERATOR" \
    --p3-json "$P3_JSON" \
    --p4-json "$P4_JSON" \
    --output-source "$GENERATED_SOURCE" \
    --output-manifest "$EVIDENCE/source_manifest.json" \
    2>&1 | tee "$EVIDENCE/source_generation.log"

grep -Fx \
    "PHASE4A0_P5_SOURCE_GENERATION: PASS" \
    "$EVIDENCE/source_generation.log"

cp \
    "$GENERATED_SOURCE" \
    "$EVIDENCE/phase4a0_p5_two_layer_fused_forward.cpp"

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

printf '%q ' "${compile[@]}" \
    > "$EVIDENCE/compile_command.txt"
printf '\n' >> "$EVIDENCE/compile_command.txt"

echo
echo "===== BUILD ====="
"${compile[@]}" 2>&1 | tee "$EVIDENCE/build.log"

sha256sum \
    "$GENERATOR" \
    "$P3_JSON" \
    "$P4_JSON" \
    "$GENERATED_SOURCE" \
    "$BINARY" \
    > "$EVIDENCE/SHA256SUMS"

echo
echo "===== FRESH PROCESS 1 ====="
"$BINARY" \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_1/hidden.csv" \
    "$EVIDENCE/process_1/output.csv" \
    2>&1 | tee "$EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="
"$BINARY" \
    "$EVIDENCE/process_2/result.json" \
    "$EVIDENCE/process_2/hidden.csv" \
    "$EVIDENCE/process_2/output.csv" \
    2>&1 | tee "$EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    grep -Fx \
        "ROCWMMA_P5_WAVE32_CONTEXT: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_P3_MAPPING_EMBEDDED: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_GUARD_REGIONS: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_QUANTIZATION_EXERCISED: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_HIDDEN_EPILOGUE: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_HIDDEN_BITWISE_CORRECTNESS: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_NO_INTERMEDIATE_GLOBAL_RELOAD: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P5_OUTPUT_VS_CPU_FP64: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "RDNA4_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PROCESS: PASS" \
        "$EVIDENCE/$process/run.log"
done

echo
echo "===== REPRODUCIBILITY ====="
cmp \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_2/result.json"
cmp \
    "$EVIDENCE/process_1/hidden.csv" \
    "$EVIDENCE/process_2/hidden.csv"
cmp \
    "$EVIDENCE/process_1/output.csv" \
    "$EVIDENCE/process_2/output.csv"

echo "ROCWMMA_P5_FRESH_PROCESS_REPRODUCIBILITY: PASS"

cp \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/phase4a0_p5_result.json"
cp \
    "$EVIDENCE/process_1/hidden.csv" \
    "$EVIDENCE/phase4a0_p5_hidden.csv"
cp \
    "$EVIDENCE/process_1/output.csv" \
    "$EVIDENCE/phase4a0_p5_output.csv"

python - "$EVIDENCE/phase4a0_p5_result.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())

assert data["decision"] == (
    "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PASS"
)
assert all(data["gates"].values())

metrics = data["metrics"]
tolerances = data["tolerances"]

assert metrics["hidden_mismatch_count"] == 0
assert metrics["output_nonfinite_count"] == 0
assert metrics["output_max_abs"] <= tolerances["output_max_abs"]
assert (
    metrics["output_normalized_l2"]
    <= tolerances["output_normalized_l2"]
)
assert metrics["hidden_relu_clamped_count"] > 0
assert metrics["hidden_positive_count"] > 0
assert metrics["hidden_fp16_cast_changed_count"] > 0
assert data["topology"]["kernel_launches"] == 1
assert not data["topology"]["intermediate_global_reload"]

print("PHASE4A0_P5_JSON_AUDIT: PASS")
PY

cat > "$EVIDENCE/PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD.md" <<EOF
# Phase 4A0-P5 — Two-layer fused-forward correctness

Decision: **PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PASS**

- P3 evidence: \`$P3_DIR\`
- P4 evidence: \`$P4_DIR\`
- Target: \`gfx1201\`, Wave32
- Topology: \`16 → 16 → 16\`, batch rows \`16\`
- Inputs and weights: FP16
- Accumulators and biases: FP32
- Hidden activation: ReLU
- Hidden relay: P3-derived accumulator→matrix-A permutation
- Kernel launches: 1
- Intermediate global reload: none
- Hidden global store: diagnostic only
- Hidden validation: bitwise equal
- Final validation: FP32 output vs independent CPU-FP64
- Fresh processes: 2/2 identical
EOF

sha256sum \
    "$EVIDENCE/source_manifest.json" \
    "$EVIDENCE/phase4a0_p5_two_layer_fused_forward.cpp" \
    "$EVIDENCE/phase4a0_p5_result.json" \
    "$EVIDENCE/phase4a0_p5_hidden.csv" \
    "$EVIDENCE/phase4a0_p5_output.csv" \
    "$EVIDENCE/PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD.md" \
    >> "$EVIDENCE/SHA256SUMS"

echo
echo "ROCWMMA_P5_MAP_CONTEXT: RECORDED"
echo "RDNA4_TWO_LAYER_FUSED_FORWARD_CORRECTNESS: PASS"
echo "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD: PASS"
echo "Evidence: $EVIDENCE"
