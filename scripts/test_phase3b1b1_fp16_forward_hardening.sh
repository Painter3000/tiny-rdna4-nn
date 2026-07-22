#!/usr/bin/env bash
# TCNN_RDNA4_P3B1B1_FP16_FORWARD_HARDENING_001: run the hardening matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TCNN_PYTHON:-/home/oem/therock_test/venv/bin/python}"
BUILD_LIB="${ROOT}/bindings/torch/build/lib.linux-x86_64-cpython-312"
export PYTHONPATH="${BUILD_LIB}:${ROOT}/bindings/torch${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" "${ROOT}/scripts/test_phase3b1b1_fp16_forward_hardening.py" "$@"
