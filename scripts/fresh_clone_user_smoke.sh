#!/usr/bin/env bash
# Fresh-clone build + runtime smoke for tiny-rdna4-nn.
#
# Marker: TCNN_RDNA4_FRESH_CLONE_USER_SMOKE_001
#
# Default behavior:
#   * requires an activated, independent Python venv with ROCm PyTorch
#   * requires tinycudann to be absent before the build
#   * requires a clean recursive Git checkout
#   * builds a persistent wheel with --no-deps and --no-cache-dir
#   * installs that exact wheel
#   * runs the runtime smoke from a neutral directory
#   * checks native ROCm/PyTorch linkage
#   * confirms the repository and submodules remain clean
#
# This is a correctness/reproducibility smoke, not a performance benchmark.

set -Eeuo pipefail

MARKER="TCNN_RDNA4_FRESH_CLONE_USER_SMOKE_001"
MODE="quick"
BUILD_MODE="full"
ALLOW_DIRTY=0
EVIDENCE_DIR=""
MAX_JOBS_VALUE="${MAX_JOBS:-1}"
ROCM_PATH_VALUE="${ROCM_PATH:-/opt/rocm}"
ARCH_VALUE="${PYTORCH_ROCM_ARCH:-gfx1201}"

usage() {
    cat <<'EOF'
Usage:
  scripts/fresh_clone_user_smoke.sh [options]

Options:
  --quick                    Test the documented HashGrid + PortableMLP path.
                             This is the default.
  --all-backends             Also test PortableMLP, HipBLASLtMLP, and
                             HipBLASLtMLPFP16 as explicit network backends.
  --runtime-only             Skip wheel build/install and test the already
                             installed tinycudann package. Intended for local
                             development of this smoke script.
  --allow-dirty              Permit a dirty checkout. Intended only while
                             developing the smoke script; never use for final
                             fresh-clone evidence.
  --evidence-dir PATH        Store logs and reports outside the repository.
                             Default: a timestamped directory under $TMPDIR.
  -h, --help                 Show this help.

Full-mode prerequisites:
  * activated Python venv
  * ROCm-enabled PyTorch already installed in that venv
  * tinycudann not importable before the run
  * recursive submodules initialized
  * PYTORCH_ROCM_ARCH=gfx1201 (set automatically if absent)
EOF
}

while (($#)); do
    case "$1" in
        --quick)
            MODE="quick"
            shift
            ;;
        --all-backends)
            MODE="all-backends"
            shift
            ;;
        --runtime-only)
            BUILD_MODE="runtime-only"
            shift
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
            shift
            ;;
        --evidence-dir)
            [[ $# -ge 2 ]] || { echo "--evidence-dir requires a path" >&2; exit 2; }
            EVIDENCE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
PY_SCRIPT="$REPO_ROOT/scripts/fresh_clone_user_smoke.py"

[[ -f "$PY_SCRIPT" ]] || {
    echo "Missing runtime script: $PY_SCRIPT" >&2
    exit 1
}

if [[ -z "$EVIDENCE_DIR" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    EVIDENCE_DIR="${TMPDIR:-/tmp}/tiny-rdna4-nn-fresh-clone-smoke-${stamp}-$$"
fi
mkdir -p "$EVIDENCE_DIR"
EVIDENCE_DIR="$(realpath -m "$EVIDENCE_DIR")"
REPO_ROOT="$(realpath "$REPO_ROOT")"

case "$EVIDENCE_DIR/" in
    "$REPO_ROOT/"*)
        echo "Evidence directory must be outside the repository: $EVIDENCE_DIR" >&2
        exit 1
        ;;
esac

MASTER_LOG="$EVIDENCE_DIR/fresh_clone_user_smoke.log"
FAILURE_FILE="$EVIDENCE_DIR/FAILURE.txt"

on_error() {
    local rc=$?
    local line="${BASH_LINENO[0]:-unknown}"
    {
        echo "marker=$MARKER"
        echo "result=FAIL"
        echo "exit_code=$rc"
        echo "line=$line"
        echo "mode=$MODE"
        echo "build_mode=$BUILD_MODE"
        echo "repository=$REPO_ROOT"
    } > "$FAILURE_FILE"
    echo "FRESH_CLONE_USER_SMOKE: FAIL (exit=$rc line=$line)" >&2
    echo "Evidence: $EVIDENCE_DIR" >&2
    exit "$rc"
}
trap on_error ERR

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "===== FRESH CLONE USER SMOKE ====="
echo "marker: $MARKER"
echo "repository: $REPO_ROOT"
echo "mode: $MODE"
echo "build mode: $BUILD_MODE"
echo "evidence: $EVIDENCE_DIR"
echo "ROCM_PATH: $ROCM_PATH_VALUE"
echo "PYTORCH_ROCM_ARCH: $ARCH_VALUE"
echo "MAX_JOBS: $MAX_JOBS_VALUE"

if [[ "$ARCH_VALUE" != "gfx1201" ]]; then
    echo "Expected PYTORCH_ROCM_ARCH=gfx1201, got $ARCH_VALUE" >&2
    exit 1
fi
if [[ ! -d "$ROCM_PATH_VALUE" ]]; then
    echo "ROCm path does not exist: $ROCM_PATH_VALUE" >&2
    exit 1
fi

# Prevent hidden package provenance from shell-level Python path overrides.
ORIGINAL_PYTHONPATH="${PYTHONPATH-}"
ORIGINAL_PYTHONHOME="${PYTHONHOME-}"
unset PYTHONPATH
unset PYTHONHOME
export ROCM_PATH="$ROCM_PATH_VALUE"
export PYTORCH_ROCM_ARCH="$ARCH_VALUE"
export MAX_JOBS="$MAX_JOBS_VALUE"
export TCNN_SMOKE_REPO_ROOT="$REPO_ROOT"

{
    echo "original_PYTHONPATH=${ORIGINAL_PYTHONPATH}"
    echo "original_PYTHONHOME=${ORIGINAL_PYTHONHOME}"
    echo "effective_ROCM_PATH=${ROCM_PATH}"
    echo "effective_PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    echo "effective_MAX_JOBS=${MAX_JOBS}"
} > "$EVIDENCE_DIR/environment_overrides.txt"

echo
echo "===== PYTHON / TORCH PREFLIGHT ====="
command -v python
python --version
python -m pip --version
python - <<'PY'
import pathlib
import sys
import torch

prefix = pathlib.Path(sys.prefix).resolve()
base = pathlib.Path(sys.base_prefix).resolve()
torch_path = pathlib.Path(torch.__file__).resolve()

print("sys.executable:", sys.executable)
print("sys.prefix:", prefix)
print("sys.base_prefix:", base)
print("torch:", torch.__version__)
print("torch path:", torch_path)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

assert prefix != base, "An activated virtual environment is required"
assert prefix in torch_path.parents, "Torch does not come from the active venv"
assert torch.version.hip is not None, "This PyTorch build has no ROCm support"
assert torch.cuda.is_available(), "No ROCm GPU is visible"

props = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print("Architecture:", props.gcnArchName)
assert props.gcnArchName == "gfx1201", props.gcnArchName

for module in ("setuptools", "wheel", "packaging", "ninja"):
    __import__(module)
print("FRESH_ROCM_TORCH_ENV: PASS")
PY

python -m pip show torch > "$EVIDENCE_DIR/pip_show_torch.txt"
python -m pip freeze > "$EVIDENCE_DIR/pip_freeze_before.txt"

echo
echo "===== CHECKOUT IDENTITY ====="
{
    echo "remote:"
    git -C "$REPO_ROOT" remote -v
    echo
    echo "head:"
    git -C "$REPO_ROOT" rev-parse HEAD
    echo
    echo "branch:"
    git -C "$REPO_ROOT" branch --show-current
    echo
    echo "submodules:"
    git -C "$REPO_ROOT" submodule status --recursive
    echo
    echo "status:"
    git -C "$REPO_ROOT" status --short --untracked-files=all
} | tee "$EVIDENCE_DIR/clone_identity_before.txt"

SUBMODULE_STATUS="$(git -C "$REPO_ROOT" submodule status --recursive)"
if grep -Eq '^[-+U]' <<<"$SUBMODULE_STATUS"; then
    echo "Submodules are missing, mismatched, or conflicted" >&2
    exit 1
fi

STATUS_BEFORE="$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
printf '%s\n' "$STATUS_BEFORE" > "$EVIDENCE_DIR/git_status_before.txt"
if [[ "$ALLOW_DIRTY" -ne 1 && -n "$STATUS_BEFORE" ]]; then
    echo "Fresh-clone smoke requires a clean checkout:" >&2
    printf '%s\n' "$STATUS_BEFORE" >&2
    exit 1
fi

WHEEL_PATH=""
WHEEL_SHA256=""
if [[ "$BUILD_MODE" == "full" ]]; then
    echo
    echo "===== CLEAN PACKAGE PRECONDITION ====="
    python - <<'PY'
import importlib.util

package_spec = importlib.util.find_spec("tinycudann")
binding_spec = importlib.util.find_spec("tinycudann_bindings")
print("tinycudann spec before build:", package_spec)
print("tinycudann_bindings spec before build:", binding_spec)
assert package_spec is None and binding_spec is None, (
    "tinycudann or its native binding is already importable. Use a clean venv, "
    "or use --runtime-only only while developing the smoke script."
)
print("CLEAN_ENV_WITHOUT_TINYCUDANN: PASS")
PY

    WHEEL_DIR="$EVIDENCE_DIR/wheel"
    mkdir -p "$WHEEL_DIR"

    echo
    echo "===== BUILD WHEEL ====="
    python -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --no-cache-dir \
        --wheel-dir "$WHEEL_DIR" \
        "$REPO_ROOT/bindings/torch" \
        2>&1 | tee "$EVIDENCE_DIR/wheel_build.log"

    mapfile -t wheels < <(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'tinycudann-*.whl' -print | sort)
    [[ "${#wheels[@]}" -eq 1 ]] || {
        echo "Expected exactly one tinycudann wheel, found ${#wheels[@]}" >&2
        printf '%s\n' "${wheels[@]}" >&2
        exit 1
    }
    WHEEL_PATH="${wheels[0]}"
    WHEEL_SHA256="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
    printf '%s  %s\n' "$WHEEL_SHA256" "$(basename "$WHEEL_PATH")" \
        | tee "$EVIDENCE_DIR/wheel_sha256.txt"

    echo
    echo "===== INSTALL EXACT WHEEL ====="
    python -m pip install \
        --no-deps \
        --force-reinstall \
        --no-cache-dir \
        "$WHEEL_PATH" \
        2>&1 | tee "$EVIDENCE_DIR/wheel_install.log"
else
    echo
    echo "===== RUNTIME-ONLY PACKAGE PRECONDITION ====="
    python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("tinycudann")
print("tinycudann spec:", spec)
assert spec is not None, "tinycudann is not installed"
print("RUNTIME_ONLY_PACKAGE_VISIBLE: PASS")
PY
fi

echo
echo "===== PIP CHECK ====="
python -m pip check | tee "$EVIDENCE_DIR/pip_check.txt"
python -m pip freeze > "$EVIDENCE_DIR/pip_freeze_after.txt"

NEUTRAL_DIR="$EVIDENCE_DIR/neutral"
RUNTIME_EVIDENCE="$EVIDENCE_DIR/runtime"
mkdir -p "$NEUTRAL_DIR" "$RUNTIME_EVIDENCE"

echo
echo "===== RUNTIME SMOKE FROM NEUTRAL DIRECTORY ====="
(
    cd "$NEUTRAL_DIR"
    python "$PY_SCRIPT" \
        --mode "$MODE" \
        --repo-root "$REPO_ROOT" \
        --evidence-dir "$RUNTIME_EVIDENCE"
) 2>&1 | tee "$EVIDENCE_DIR/runtime.log"

if grep -Fq \
    "GPUMemoryArena: GPU 0 does not support virtual memory" \
    "$EVIDENCE_DIR/runtime.log"
then
    MEMORY_WARNING_STATUS="OBSERVED_EXPECTED_NONFATAL"
else
    MEMORY_WARNING_STATUS="NOT_OBSERVED"
fi
echo "EXPECTED_GPU_MEMORY_ARENA_WARNING: $MEMORY_WARNING_STATUS" \
    | tee "$EVIDENCE_DIR/memory_warning_status.txt"

echo
echo "===== NATIVE LINKAGE ====="
NATIVE_SO="$(
    cd "$NEUTRAL_DIR"
    python - <<'PY'
from tinycudann.modules import _C
print(_C.__file__)
PY
)"
TORCH_LIB="$(
    python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

echo "native binding: $NATIVE_SO"
echo "torch libraries: $TORCH_LIB"

env \
    LD_LIBRARY_PATH="$TORCH_LIB:$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ldd "$NATIVE_SO" \
    2>&1 | tee "$EVIDENCE_DIR/native_binding_ldd.txt"

if grep -q 'not found' "$EVIDENCE_DIR/native_binding_ldd.txt"; then
    echo "NATIVE_LIBRARY_RESOLUTION: FAIL" >&2
    exit 1
fi

if grep -Eq '(^|[[:space:]])lib(cudart|cuda|nvrtc)\.so([.[:space:]]|$)' \
    "$EVIDENCE_DIR/native_binding_ldd.txt"
then
    echo "Unexpected NVIDIA runtime library in native linkage" >&2
    exit 1
fi

grep -E 'libamdhip64|libhipblaslt|libc10|libtorch' \
    "$EVIDENCE_DIR/native_binding_ldd.txt" \
    > "$EVIDENCE_DIR/native_binding_key_libraries.txt" || true

echo "NATIVE_LIBRARY_RESOLUTION_WITH_TORCH_PATH: PASS"

echo
echo "===== POST-BUILD CHECKOUT STATE ====="
{
    echo "main repository:"
    git -C "$REPO_ROOT" status --short --untracked-files=all
    echo
    echo "submodule pointers:"
    git -C "$REPO_ROOT" submodule status --recursive
    echo
    echo "submodule worktrees:"
    git -C "$REPO_ROOT" submodule foreach --recursive \
        'echo "--- $name"; git status --short --untracked-files=all'
} | tee "$EVIDENCE_DIR/checkout_state_after.txt"

STATUS_AFTER="$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
printf '%s\n' "$STATUS_AFTER" > "$EVIDENCE_DIR/git_status_after.txt"

if [[ "$ALLOW_DIRTY" -eq 1 ]]; then
    if [[ "$STATUS_AFTER" != "$STATUS_BEFORE" ]]; then
        echo "The smoke run changed the pre-existing dirty checkout" >&2
        diff -u \
            "$EVIDENCE_DIR/git_status_before.txt" \
            "$EVIDENCE_DIR/git_status_after.txt" || true
        exit 1
    fi
else
    if [[ -n "$STATUS_AFTER" ]]; then
        echo "The smoke run left repository changes:" >&2
        printf '%s\n' "$STATUS_AFTER" >&2
        exit 1
    fi
fi

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
RUNTIME_REPORT="$RUNTIME_EVIDENCE/runtime_report.json"

python - "$EVIDENCE_DIR/summary.json" <<PY
import json
from pathlib import Path

runtime = json.loads(Path(${RUNTIME_REPORT@Q}).read_text(encoding="utf-8"))
summary = {
    "marker": ${MARKER@Q},
    "result": "PASS",
    "mode": ${MODE@Q},
    "build_mode": ${BUILD_MODE@Q},
    "repository": ${REPO_ROOT@Q},
    "commit": ${COMMIT@Q},
    "branch": ${BRANCH@Q},
    "rocm_path": ${ROCM_PATH@Q},
    "pytorch_rocm_arch": ${PYTORCH_ROCM_ARCH@Q},
    "wheel_path": ${WHEEL_PATH@Q} or None,
    "wheel_sha256": ${WHEEL_SHA256@Q} or None,
    "expected_gpu_memory_arena_warning": ${MEMORY_WARNING_STATUS@Q},
    "runtime_report": runtime,
}
Path(__import__("sys").argv[1]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

rm -f "$FAILURE_FILE"

echo
echo "===== FINAL DECISION ====="
if [[ "$BUILD_MODE" == "full" ]]; then
    echo "FRESH_CLONE_WHEEL_BUILD_INSTALL: PASS"
else
    echo "FRESH_CLONE_WHEEL_BUILD_INSTALL: SKIPPED_RUNTIME_ONLY"
fi
echo "FRESH_CLONE_PROVENANCE: PASS"
echo "FRESH_CLONE_FUNCTIONAL_RUNTIME: PASS"
echo "FRESH_CLONE_SECOND_PROCESS: PASS"
echo "FRESH_CLONE_NATIVE_LINKAGE: PASS"
echo "FRESH_CLONE_REPOSITORY_CLEANLINESS: PASS"
echo "FRESH_CLONE_USER_SMOKE: PASS"
echo "Evidence: $EVIDENCE_DIR"
