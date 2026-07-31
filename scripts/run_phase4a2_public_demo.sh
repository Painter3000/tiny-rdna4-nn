#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PHASE4A2_BUILD_ROOT:-$ROOT/.phase4a2_image_build}"
fail() { echo "PHASE4A2_PUBLIC_DEMO: FAIL: $*" >&2; exit 1; }
[[ -n "${VIRTUAL_ENV:-}" ]] || fail "activate a Python virtual environment"
[[ -f "$BUILD_ROOT/runtime/tiny-rdna4-nn/tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so" ]] || fail "run the public build command first"
[[ -f "$BUILD_ROOT/runtime/amd-gsplat/gsplat/csrc.so" ]] || fail "run the public build command first"
export PHASE4A2_OUTPUT="${PHASE4A2_OUTPUT:-$ROOT/phase4a2_public_demo_result}"
exec python "$ROOT/scripts/run_phase4a2_image_path.py"
