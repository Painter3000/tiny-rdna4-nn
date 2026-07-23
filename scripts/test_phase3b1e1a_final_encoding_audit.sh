#!/usr/bin/env bash
# TCNN_RDNA4_P3B1E1A_FINAL_ENCODING_AUDIT_001
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo/bindings/torch/build/lib.linux-x86_64-cpython-312:$repo/bindings/torch:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
/home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1e1a_final_encoding_audit.py"
/home/oem/therock_test/venv/bin/python "$repo/scripts/finalize_phase3b1e1a_report.py"
