#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P1_OPT_IN_SKELETON_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY="$ROOT/scripts/apply_phase4a2_p1.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p1.py"
PROBE="$ROOT/probes/phase4a2_p1_factory_probe.py"
ADDENDUM="$ROOT/contracts/phase4a2_p1_integration_surface_addendum.json"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
P0_FILENAME="phase4a2_p0_production_integration_contract.json"

resolve_p0_evidence() {
    if [[ -n "${PHASE4A2_P0_SOURCE_EVIDENCE:-}" ]]; then
        printf '%s\n' "$PHASE4A2_P0_SOURCE_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/$P0_FILENAME" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a2_p0_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A2-P0 evidence directory found." >&2
    return 1
}

P0_DIR="$(resolve_p0_evidence)"
P0_JSON="$P0_DIR/$P0_FILENAME"

EVIDENCE="${1:-${PHASE4A2_P1_EVIDENCE:-$EVIDENCE_ROOT/phase4a2_p1_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a2_p1_*)
        ;;
    *)
        echo "Refusing unsafe evidence path: $EVIDENCE" >&2
        exit 2
        ;;
esac

if [[ -d "$EVIDENCE" ]] && \
   [[ -n "$(find "$EVIDENCE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty evidence directory: $EVIDENCE" >&2
    exit 2
fi

for path in \
    "$APPLY" \
    "$FINALIZER" \
    "$PROBE" \
    "$ADDENDUM" \
    "$P0_JSON" \
    "$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h" \
    "$ROOT/src/rocwmma_width64_mlp.cu"; do
    test -f "$path"
done

mkdir -p \
    "$EVIDENCE/build_off/temp" \
    "$EVIDENCE/build_off/lib" \
    "$EVIDENCE/build_on/temp" \
    "$EVIDENCE/build_on/lib" \
    "$EVIDENCE/process_disabled" \
    "$EVIDENCE/process_enabled_1" \
    "$EVIDENCE/process_enabled_2"

echo "===== PHASE 4A2-P0 BASELINE ====="
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
SUBJECT="$(git -C "$ROOT" show -s --format=%s HEAD)"
echo "head: $HEAD"
echo "subject: $SUBJECT"
test "${HEAD:0:7}" = "977714b"
test "$SUBJECT" = "Add Phase 4A2-P0 Width-64 production integration contract"
echo "PHASE4A2_P0_COMMIT_BOUND: PASS"

echo
echo "===== SELF-CHECK ====="
PYTHONPYCACHEPREFIX="$EVIDENCE/pycache" \
    python -m py_compile "$APPLY" "$FINALIZER" "$PROBE"
python - "$ADDENDUM" <<'PY_ADDENDUM'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["marker"] == "TCNN_RDNA4_P4A2_P1_SURFACE_ADDENDUM_001"
assert data["p0_contract_preserved"] is True
assert len(data["additional_required_surfaces"]) == 2
print("PHASE4A2_P1_SURFACE_ADDENDUM_SELF_CHECK: PASS")
PY_ADDENDUM
echo "PHASE4A2_P1_PYTHON_SELF_CHECK: PASS"

{
    echo "===== PHASE 4A2-P1 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head_before_apply: $HEAD"
    echo "phase4a2_p0_evidence: $P0_DIR"
    echo
    echo "===== STATUS BEFORE APPLY ====="
    git -C "$ROOT" status --short --untracked-files=all
} > "$EVIDENCE/context_before_apply.txt"

echo
echo "===== APPLY ====="
python "$APPLY" \
    --repo "$ROOT" \
    --p0-json "$P0_JSON" \
    --output "$EVIDENCE/apply_manifest.json" \
    2>&1 | tee "$EVIDENCE/apply.log"

grep -Fx "PHASE4A2_P1_ANCHOR_PATCH: PASS" "$EVIDENCE/apply.log"
grep -Fx "WIDTH64_P1_NO_PRODUCTION_KERNEL_INSTALLED: PASS" "$EVIDENCE/apply.log"
grep -Fx "PHASE4A2_P1_APPLY: PASS" "$EVIDENCE/apply.log"

git -C "$ROOT" diff --check

echo
echo "===== STATIC BUILD-SWITCH AUDIT ====="
python - "$ROOT" <<'PY_STATIC'
from pathlib import Path
import sys

root = Path(sys.argv[1])
setup = (root / "bindings/torch/setup.py").read_text()
factory = (root / "src/portable_network.cu").read_text()
source = (root / "src/rocwmma_width64_mlp.cu").read_text()
header = (
    root / "include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h"
).read_text()

assert '"TCNN_ENABLE_ROCWMMA_WIDTH64_MLP",' in setup
assert '\tFalse,' in setup
assert 'base_source_files.append("../../src/rocwmma_width64_mlp.cu")' in setup
assert 'definitions.append("-DTCNN_WITH_ROCWMMA_WIDTH64_MLP")' in setup
assert "RocWMMAWidth64MLP was not compiled" in factory
assert "TOTAL_PARAMETER_ELEMENTS == 12480" in header
assert "__global__" not in source
assert "rocwmma::" not in source
assert "mma_sync" not in source

print("WIDTH64_BUILD_SWITCH_STATIC_AUDIT: PASS")
print("WIDTH64_FACTORY_FAIL_CLOSED_STATIC_AUDIT: PASS")
print("WIDTH64_NO_PRODUCTION_KERNEL_STATIC_AUDIT: PASS")
PY_STATIC

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BINDINGS_DIR="$ROOT/bindings/torch"
DEPENDENCY_ROOT="${TCNN_DEPENDENCY_ROOT:-$ROOT/dependencies}"
MAX_JOBS="${MAX_JOBS:-4}"

test -x "$PYTHON_BIN"
test -d "$DEPENDENCY_ROOT"

build_variant() {
    local mode="$1"
    local build_root="$2"
    local log="$3"

    rm -rf "$build_root/temp" "$build_root/lib"
    mkdir -p "$build_root/temp" "$build_root/lib"

    (
        cd "$BINDINGS_DIR"
        env \
            PYTORCH_ROCM_ARCH=gfx1201 \
            TCNN_DEPENDENCY_ROOT="$DEPENDENCY_ROOT" \
            TCNN_HALF_PRECISION=1 \
            TCNN_ENABLE_ROCWMMA_WIDTH64_MLP="$mode" \
            MAX_JOBS="$MAX_JOBS" \
            "$PYTHON_BIN" setup.py build_ext \
                --build-temp "$build_root/temp" \
                --build-lib "$build_root/lib"
    ) 2>&1 | tee "$log"
}

echo
echo "===== DEFAULT-OFF BUILD ====="
build_variant \
    0 \
    "$EVIDENCE/build_off" \
    "$EVIDENCE/build_off.log"

grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: OFF" \
    "$EVIDENCE/build_off.log"

if grep -q "rocwmma_width64_mlp.cu" "$EVIDENCE/build_off.log"; then
    echo "Default-OFF build unexpectedly compiled rocwmma_width64_mlp.cu." >&2
    exit 1
fi

if grep -q "TCNN_WITH_ROCWMMA_WIDTH64_MLP" "$EVIDENCE/build_off.log"; then
    echo "Default-OFF build unexpectedly defined the rocWMMA backend." >&2
    exit 1
fi

echo
echo "===== DEFAULT-OFF FACTORY PROBE ====="
(
    cd "$EVIDENCE/process_disabled"
    env \
        PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$EVIDENCE/build_off/lib:$BINDINGS_DIR" \
        "$PYTHON_BIN" "$PROBE" \
            --mode disabled \
            --output "$EVIDENCE/process_disabled/result.json"
) 2>&1 | tee "$EVIDENCE/process_disabled/run.log"

grep -Fx "WIDTH64_DEFAULT_OFF_FACTORY_FAIL_CLOSED: PASS" \
    "$EVIDENCE/process_disabled/run.log"
grep -Fx "WIDTH64_EXISTING_FACTORY_REGRESSION: PASS" \
    "$EVIDENCE/process_disabled/run.log"
grep -Fx "PHASE4A2_P1_DISABLED_BUILD_FACTORY: PASS" \
    "$EVIDENCE/process_disabled/run.log"

echo
echo "===== EXPLICIT-ON BUILD ====="
build_variant \
    1 \
    "$EVIDENCE/build_on" \
    "$EVIDENCE/build_on.log"

grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" \
    "$EVIDENCE/build_on.log"
grep -q "rocwmma_width64_mlp.cu" \
    "$EVIDENCE/build_on.log"
grep -q "TCNN_WITH_ROCWMMA_WIDTH64_MLP" \
    "$EVIDENCE/build_on.log"

run_enabled_probe() {
    local process="$1"
    (
        cd "$EVIDENCE/$process"
        env \
            PYTHONNOUSERSITE=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH="$EVIDENCE/build_on/lib:$BINDINGS_DIR" \
            "$PYTHON_BIN" "$PROBE" \
                --mode enabled \
                --output "$EVIDENCE/$process/result.json"
    ) 2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== EXPLICIT-ON FACTORY FRESH PROCESS 1 ====="
run_enabled_probe process_enabled_1

echo
echo "===== EXPLICIT-ON FACTORY FRESH PROCESS 2 ====="
run_enabled_probe process_enabled_2

enabled_markers=(
    "WIDTH64_EXPLICIT_FACTORY_CONSTRUCTION: PASS"
    "WIDTH64_PARAMETER_ABI_12480: PASS"
    "WIDTH64_INVALID_CONFIG_FAIL_CLOSED: PASS"
    "WIDTH64_INFERENCE_BEFORE_QUALIFICATION_FAIL_CLOSED: PASS"
    "WIDTH64_FORWARD_BEFORE_QUALIFICATION_FAIL_CLOSED: PASS"
    "PHASE4A2_P1_ENABLED_BUILD_FACTORY_SKELETON: PASS"
)

for process in process_enabled_1 process_enabled_2; do
    for marker in "${enabled_markers[@]}"; do
        grep -Fx "$marker" "$EVIDENCE/$process/run.log"
    done
done

cmp \
    "$EVIDENCE/process_enabled_1/result.json" \
    "$EVIDENCE/process_enabled_2/result.json"

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --repo "$ROOT" \
    --apply "$EVIDENCE/apply_manifest.json" \
    --disabled "$EVIDENCE/process_disabled/result.json" \
    --enabled-1 "$EVIDENCE/process_enabled_1/result.json" \
    --enabled-2 "$EVIDENCE/process_enabled_2/result.json" \
    --build-off-log "$EVIDENCE/build_off.log" \
    --build-on-log "$EVIDENCE/build_on.log" \
    --surface-addendum "$ADDENDUM" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

final_markers=(
    "WIDTH64_DEFAULT_OFF_BUILD: PASS"
    "WIDTH64_DEFAULT_OFF_FACTORY_FAIL_CLOSED: PASS"
    "WIDTH64_EXISTING_FACTORY_REGRESSION: PASS"
    "WIDTH64_EXPLICIT_ON_BUILD: PASS"
    "WIDTH64_EXPLICIT_FACTORY_CONSTRUCTION: PASS"
    "WIDTH64_PARAMETER_ABI_12480: PASS"
    "WIDTH64_INVALID_CONFIG_FAIL_CLOSED: PASS"
    "WIDTH64_INFERENCE_FORWARD_PREKERNEL_FAIL_CLOSED: PASS"
    "WIDTH64_ENABLED_FRESH_PROCESS_REPRODUCIBILITY: PASS"
    "WIDTH64_NO_PRODUCTION_KERNEL_INSTALLED: PASS"
    "PHASE4A2_P1_SURFACE_ADDENDUM: RECORDED"
    "PHASE4A2_P1_CONSOLIDATED_EVIDENCE: RECORDED"
    "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON: PASS"
)

for marker in "${final_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/finalize.log"
done

sha256sum \
    "$P0_JSON" \
    "$ADDENDUM" \
    "$APPLY" \
    "$FINALIZER" \
    "$PROBE" \
    "$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mlp.h" \
    "$ROOT/src/rocwmma_width64_mlp.cu" \
    "$ROOT/bindings/torch/setup.py" \
    "$ROOT/src/cpp_api.cu" \
    "$ROOT/src/portable_network.cu" \
    "$EVIDENCE/apply_manifest.json" \
    "$EVIDENCE/process_disabled/result.json" \
    "$EVIDENCE/process_enabled_1/result.json" \
    "$EVIDENCE/process_enabled_2/result.json" \
    "$EVIDENCE/phase4a2_p1_opt_in_class_build_factory_skeleton.json" \
    "$EVIDENCE/PHASE4A2_P1_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

python - "$EVIDENCE/phase4a2_p1_opt_in_class_build_factory_skeleton.json" <<'PY_JSON'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON_PASS"
)
assert all(data["gates"].values())
assert all(data["static_gates"].values())
assert data["builds"]["explicit_on"]["fresh_process_exact_match"] is True
assert data["builds"]["explicit_on"]["fresh_process_1"]["model"][
    "parameter_elements"
] == 12480
print("PHASE4A2_P1_JSON_AUDIT: PASS")
PY_JSON

echo
echo "WIDTH64_DEFAULT_OFF_BUILD: PASS"
echo "WIDTH64_DEFAULT_OFF_FACTORY_FAIL_CLOSED: PASS"
echo "WIDTH64_EXISTING_FACTORY_REGRESSION: PASS"
echo "WIDTH64_EXPLICIT_ON_BUILD: PASS"
echo "WIDTH64_EXPLICIT_FACTORY_CONSTRUCTION: PASS"
echo "WIDTH64_PARAMETER_ABI_12480: PASS"
echo "WIDTH64_INVALID_CONFIG_FAIL_CLOSED: PASS"
echo "WIDTH64_INFERENCE_FORWARD_PREKERNEL_FAIL_CLOSED: PASS"
echo "WIDTH64_ENABLED_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "WIDTH64_NO_PRODUCTION_KERNEL_INSTALLED: PASS"
echo "PHASE4A2_P1_SURFACE_ADDENDUM: RECORDED"
echo "PHASE4A2_P1_JSON_AUDIT: PASS"
echo "PHASE4A2_P1_MAP_CONTEXT: RECORDED"
echo "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON: PASS"
echo "Evidence: $EVIDENCE"
