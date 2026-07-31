#!/usr/bin/env bash
set -euo pipefail

ROCM_REAL="${PHASE4A_ROCM_REAL:-/opt/rocm}"
ROCM_CLANGXX="${PHASE4A_ROCM_CLANGXX:-$ROCM_REAL/lib/llvm/bin/clang++}"

[[ -x "$ROCM_CLANGXX" ]] || {
    echo "PHASE4A_HIPCC_COMPAT: FAIL: ROCM_CLANGXX_UNAVAILABLE: $ROCM_CLANGXX" >&2
    exit 127
}

args=()
is_compile=0

for arg in "$@"; do
    case "$arg" in
        --rocm-path=*)
            arg="--rocm-path=$ROCM_REAL"
            ;;
        -c)
            is_compile=1
            ;;
    esac
    args+=("$arg")
done

if (( is_compile )); then
    exec "$ROCM_CLANGXX" -x hip "${args[@]}"
fi

exec "$ROCM_CLANGXX" "${args[@]}"
