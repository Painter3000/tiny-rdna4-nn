#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

echo "Phase 3C: validating portable counter stream, fresh-building, and running GPU smoke"
exec python3 tools/phase3c_portable_smoke.py "$@"
