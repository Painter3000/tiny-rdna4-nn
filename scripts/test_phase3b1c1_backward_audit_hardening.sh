#!/usr/bin/env bash
# TCNN_RDNA4_P3B1C1_BACKWARD_AUDIT_HARDENING_001
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_dir/bindings/torch/build/lib.linux-x86_64-cpython-312"
export PYTHONPATH="$repo_dir/bindings/torch:$build_dir${PYTHONPATH:+:$PYTHONPATH}"
exec /home/oem/therock_test/venv/bin/python "$repo_dir/scripts/test_phase3b1c_fp16_backward.py" \
  --output "$repo_dir/phase3b1_reports/phase3b1c1_backward_audit_hardening_raw.json"
