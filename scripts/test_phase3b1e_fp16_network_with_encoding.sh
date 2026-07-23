#!/usr/bin/env bash
# TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo/bindings/torch/build/lib.linux-x86_64-cpython-312:$repo/bindings/torch${PYTHONPATH:+:$PYTHONPATH}"
/home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1e_fp16_network_with_encoding.py" --output /tmp/phase3b1e_fp16_network_with_encoding_raw.json
/home/oem/therock_test/venv/bin/python "$repo/scripts/finalize_phase3b1e_report.py"
