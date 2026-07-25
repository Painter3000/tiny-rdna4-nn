#!/usr/bin/env bash
# TCNN_RDNA4_P4A0_P1_MINIMAL_ROCWMMA_GEMM_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/scripts/phase4a0_p1_minimal_rocwmma_gemm.cpp"
BUILD_DIR="$ROOT/build/phase4a0_p1"
BINARY="$BUILD_DIR/phase4a0_p1_minimal_rocwmma_gemm"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"

EVIDENCE="${1:-$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a0_p1_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$EVIDENCE" "$BUILD_DIR"

test -x "$HIPCC"
test -f /opt/rocm/include/rocwmma/rocwmma.hpp
test -f "$SOURCE"

{
    echo "===== PHASE 4A0-P1 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo
    echo "===== SUBMODULES ====="
    git -C "$ROOT" submodule status --recursive
    echo
    echo "===== HIPCC ====="
    "$HIPCC" --version
    echo
    echo "===== ROCWMMA PACKAGE ====="
    dpkg-query -W -f='${Package} ${Version}\n' 'rocwmma*' 2>/dev/null || true
    echo
    echo "===== SOURCE SHA256 ====="
    sha256sum "$SOURCE"
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

sha256sum "$SOURCE" "$BINARY" > "$EVIDENCE/SHA256SUMS"

echo
echo "===== RUN ====="
"$BINARY" "$EVIDENCE/phase4a0_p1_minimal_rocwmma_gemm.json" \
    2>&1 | tee "$EVIDENCE/run.log"

grep -Fx "ROCWMMA_P1_WAVE32_CONTEXT: PASS" "$EVIDENCE/run.log"
grep -Fx "ROCWMMA_P1_INPUT_QUANTIZATION: PASS" "$EVIDENCE/run.log"
grep -Fx "ROCWMMA_NUMERICAL_RESULT_VS_CPU: PASS" "$EVIDENCE/run.log"
grep -Fx "PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM: PASS" "$EVIDENCE/run.log"

python - "$EVIDENCE/phase4a0_p1_minimal_rocwmma_gemm.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())

assert data["marker"] == "TCNN_RDNA4_P4A0_P1_MINIMAL_ROCWMMA_GEMM_001"
assert data["decision"] == "PHASE4A0_P1_MINIMAL_ROCWMMA_GEMM_PASS"
assert all(data["gates"].values())

print("PHASE4A0_P1_JSON_AUDIT: PASS")
PY

echo
echo "Evidence: $EVIDENCE"
