#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PHASE4A2_BUILD_ROOT:-$ROOT/.phase4a2_image_build}"

fail() {
    echo "PHASE4A2_PUBLIC_DEMO: FAIL: $*" >&2
    exit 1
}

if [[ -n "${PHASE4A2_PYTHON:-}" ]]; then
    PYTHON_BIN="$PHASE4A2_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    fail "activate a Python virtual environment or set PHASE4A2_PYTHON"
fi

[[ -x "$PYTHON_BIN" ]] ||
    fail "Python interpreter unavailable: $PYTHON_BIN"

TCNN_RUNTIME="$BUILD_ROOT/runtime/tiny-rdna4-nn"
GSPLAT_RUNTIME="$BUILD_ROOT/runtime/amd-gsplat"

[[ -f "$TCNN_RUNTIME/tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so" ]] ||
    fail "run the public build command first"
[[ -f "$GSPLAT_RUNTIME/gsplat/csrc.so" ]] ||
    fail "run the public build command first"

export PHASE4A2_BUILD_ROOT="$BUILD_ROOT"
export PHASE4A2_OUTPUT="${PHASE4A2_OUTPUT:-$ROOT/phase4a2_public_demo_result}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TCNN_RUNTIME:$GSPLAT_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

PHASE4A2_TCNN_RUNTIME="$TCNN_RUNTIME" \
PHASE4A2_GSPLAT_RUNTIME="$GSPLAT_RUNTIME" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import gsplat
import tinycudann
import tinycudann.modules as tcnn_modules


def require_under(module_file: str | None, expected_root: str, label: str) -> None:
    if not module_file:
        raise SystemExit(f"{label}_IMPORT_HAS_NO_FILE")

    actual = Path(module_file).resolve()
    root = Path(expected_root).resolve()

    try:
        actual.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"{label}_IMPORT_OUTSIDE_BUILD_RUNTIME: actual={actual} expected_root={root}"
        )


require_under(
    tinycudann.__file__,
    os.environ["PHASE4A2_TCNN_RUNTIME"],
    "TINYCUDANN",
)
require_under(
    getattr(tcnn_modules._C, "__file__", None),
    os.environ["PHASE4A2_TCNN_RUNTIME"],
    "TINYCUDANN_NATIVE",
)
require_under(
    gsplat.__file__,
    os.environ["PHASE4A2_GSPLAT_RUNTIME"],
    "GSPLAT",
)

print("PHASE4A2_PUBLIC_DEMO_IMPORT_PROVENANCE: PASS")
print("tinycudann:", Path(tinycudann.__file__).resolve())
print("tinycudann_native:", Path(tcnn_modules._C.__file__).resolve())
print("gsplat:", Path(gsplat.__file__).resolve())
PY

exec "$PYTHON_BIN" "$ROOT/scripts/run_phase4a2_image_path.py"
