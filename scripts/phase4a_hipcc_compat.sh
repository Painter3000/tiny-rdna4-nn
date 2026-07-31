#!/usr/bin/env bash
set -euo pipefail

ROCM_REAL="${PHASE4A_ROCM_REAL:-/opt/rocm}"
args=()
is_compile=0
for arg in "$@"; do
    case "$arg" in
        --rocm-path=*) arg="--rocm-path=$ROCM_REAL" ;;
        -c) is_compile=1 ;;
    esac
    args+=("$arg")
done

if (( is_compile )); then
    exec "$ROCM_REAL/llvm/bin/clang++" -x hip "${args[@]}"
fi
exec "$ROCM_REAL/llvm/bin/clang++" "${args[@]}"
