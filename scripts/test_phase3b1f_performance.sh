#!/usr/bin/env bash
# TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo/bindings/torch/build/lib.linux-x86_64-cpython-312:$repo/bindings/torch:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--execute-full" ]]; then
	shift
	exec /home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1f_performance.py" --execute-full "$@"
fi

exec /home/oem/therock_test/venv/bin/python "$repo/scripts/test_phase3b1f_performance.py" \
	--protocol-audit --output /tmp/phase3b1f_protocol_audit.json
