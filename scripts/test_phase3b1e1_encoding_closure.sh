#!/usr/bin/env bash
# TCNN_RDNA4_P3B1E1_ENCODING_CLOSURE_001
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo/bindings/torch/build/lib.linux-x86_64-cpython-312:$repo/bindings/torch:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
/home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1e1_encoding_closure.py" --output /tmp/phase3b1e1_encoding_closure_raw.json
/home/oem/therock_test/venv/bin/python "$repo/scripts/finalize_phase3b1e1_report.py"
