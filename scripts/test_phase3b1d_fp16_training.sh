#!/usr/bin/env bash
# TCNN_RDNA4_P3B1D_FP16_TRAINING_001
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; build="$repo/bindings/torch/build/lib.linux-x86_64-cpython-312"
export PYTHONPATH="$repo/bindings/torch:$build${PYTHONPATH:+:$PYTHONPATH}"
exec /home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1d_fp16_training.py" --baseline /tmp/phase3b1d_fresh_gpu_baseline.json --output /tmp/phase3b1d_fp16_training_raw.json
