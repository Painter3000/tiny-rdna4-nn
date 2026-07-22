#!/usr/bin/env bash
# TCNN_RDNA4_P3B1D1_TRAINING_AUDIT_001
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";export PYTHONPATH="$repo/bindings/torch:$repo/bindings/torch/build/lib.linux-x86_64-cpython-312${PYTHONPATH:+:$PYTHONPATH}"
exec /home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1d1_training_audit.py" --output /tmp/phase3b1d1_training_audit_raw.json
