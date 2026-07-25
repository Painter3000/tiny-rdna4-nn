#!/usr/bin/env bash
# TCNN_RDNA4_P4A0_P2_RAW_LANE_FRAGMENT_MAP_003
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/scripts/phase4a0_p2_raw_lane_fragment_map.cpp"
FINALIZER="$ROOT/scripts/finalize_phase4a0_p2.py"
BUILD_DIR="$ROOT/build/phase4a0_p2"
BINARY="$BUILD_DIR/phase4a0_p2_raw_lane_fragment_map"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"

EVIDENCE="${1:-$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p2_$(date -u +%Y%m%dT%H%M%SZ)}"
P1_EVIDENCE="${P1_EVIDENCE:-$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p1_20260725T061312Z}"

mkdir -p "$EVIDENCE/process_1" "$EVIDENCE/process_2" "$BUILD_DIR"

test -x "$HIPCC"
test -f "$SOURCE"
test -f "$FINALIZER"
test -f /opt/rocm/include/rocwmma/rocwmma.hpp
test -f /opt/rocm/include/rocwmma/rocwmma_transforms.hpp
test -f "$P1_EVIDENCE/phase4a0_p1_minimal_rocwmma_gemm.json"

python - "$P1_EVIDENCE/phase4a0_p1_minimal_rocwmma_gemm.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == "PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM_PASS"
assert all(data["gates"].values())
print("PHASE4A0_P1_PREREQUISITE: PASS")
PY

{
    echo "===== PHASE 4A0-P2 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p1_evidence: $P1_EVIDENCE"
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
    echo "===== HASHES ====="
    sha256sum \
        "$SOURCE" \
        "$FINALIZER" \
        /opt/rocm/include/rocwmma/rocwmma.hpp \
        /opt/rocm/include/rocwmma/rocwmma_transforms.hpp
} > "$EVIDENCE/context.txt" 2>&1

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

printf '%q ' "${compile[@]}" > "$EVIDENCE/compile_command.txt"
printf '\n' >> "$EVIDENCE/compile_command.txt"

echo "===== BUILD ====="
"${compile[@]}" 2>&1 | tee "$EVIDENCE/build.log"

sha256sum "$SOURCE" "$FINALIZER" "$BINARY" > "$EVIDENCE/SHA256SUMS"

echo
echo "===== FRESH PROCESS 1 ====="
"$BINARY" "$EVIDENCE/process_1/capture.tsv" \
    2>&1 | tee "$EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="
"$BINARY" "$EVIDENCE/process_2/capture.tsv" \
    2>&1 | tee "$EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    grep -Fx "ROCWMMA_P2_DEVICE_GEOMETRY_CAPTURE: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_WAVE32_CONTEXT: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_GUARD_REGIONS: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_WRITE_OWNERSHIP: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_MATRIX_A_REGISTER_MAP: CAPTURED" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_MATRIX_B_REGISTER_MAP: CAPTURED" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_ACCUMULATOR_REGISTER_MAP: CAPTURED" "$EVIDENCE/$process/run.log"
    grep -Fx "ROCWMMA_P2_STORED_OUTPUT_VALIDATION: PASS" "$EVIDENCE/$process/run.log"
    grep -Fx "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_PROCESS: PASS" "$EVIDENCE/$process/run.log"
done

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --process-1 "$EVIDENCE/process_1/capture.tsv" \
    --process-2 "$EVIDENCE/process_2/capture.tsv" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

grep -Fx "ROCWMMA_P2_DEVICE_GEOMETRY_CAPTURE: PASS" "$EVIDENCE/finalize.log"
grep -Fx "ROCWMMA_P2_FRESH_PROCESS_REPRODUCIBILITY: PASS" "$EVIDENCE/finalize.log"
grep -Fx "ROCWMMA_P2_MAP_CONTEXT: RECORDED" "$EVIDENCE/finalize.log"
grep -Fx "ROCWMMA_RAW_LANE_FRAGMENT_MAP: CAPTURED" "$EVIDENCE/finalize.log"
grep -Fx "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP: PASS" "$EVIDENCE/finalize.log"

python - "$EVIDENCE/phase4a0_p2_fragment_map.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_PASS"
assert all(data["gates"].values())
assert all(
    len(data["maps"][role]) == 256
    for role in ("matrix_a", "matrix_b", "accumulator")
)
assert len(data["stored_output"]) == 256
assert data["context"]["matrix_a_device_slots_per_lane"] == "8"
assert data["context"]["matrix_b_device_slots_per_lane"] == "8"
assert data["context"]["accumulator_device_slots_per_lane"] == "8"
print("PHASE4A0_P2_JSON_AUDIT: PASS")
PY

echo
echo "Evidence: $EVIDENCE"
