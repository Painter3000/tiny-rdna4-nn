#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P1_PROBE_HOTFIX_002
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="$ROOT/probes/phase4a2_p1_factory_probe.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p1.py"
ADDENDUM="$ROOT/contracts/phase4a2_p1_integration_surface_addendum.json"
BINDINGS_DIR="$ROOT/bindings/torch"

EVIDENCE="${PHASE4A2_P1_RESUME_EVIDENCE:-}"

if [[ -z "$EVIDENCE" ]]; then
    echo "Set PHASE4A2_P1_RESUME_EVIDENCE to the failed P1 evidence directory." >&2
    exit 2
fi

case "$(basename "$EVIDENCE")" in
    phase4a2_p1_*)
        ;;
    *)
        echo "Refusing unsafe evidence path: $EVIDENCE" >&2
        exit 2
        ;;
esac

required=(
    "$PROBE"
    "$FINALIZER"
    "$ADDENDUM"
    "$EVIDENCE/apply_manifest.json"
    "$EVIDENCE/build_off.log"
    "$EVIDENCE/build_on.log"
    "$EVIDENCE/process_disabled/result.json"
    "$EVIDENCE/process_disabled/run.log"
)

for path in "${required[@]}"; do
    test -f "$path"
done

shopt -s nullglob
OFF_LIBS=("$EVIDENCE"/build_off/lib/tinycudann_bindings/_120_C*.so)
ON_LIBS=("$EVIDENCE"/build_on/lib/tinycudann_bindings/_120_C*.so)
shopt -u nullglob

if [[ "${#OFF_LIBS[@]}" -ne 1 || "${#ON_LIBS[@]}" -ne 1 ]]; then
    echo "Expected exactly one OFF and one ON extension artifact." >&2
    exit 1
fi

grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: OFF" "$EVIDENCE/build_off.log"
grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" "$EVIDENCE/build_on.log"
grep -q "rocwmma_width64_mlp.cu" "$EVIDENCE/build_on.log"
grep -q "TCNN_WITH_ROCWMMA_WIDTH64_MLP" "$EVIDENCE/build_on.log"
grep -Fx "PHASE4A2_P1_DISABLED_BUILD_FACTORY: PASS" \
    "$EVIDENCE/process_disabled/run.log"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
test -x "$PYTHON_BIN"

PYTHONPYCACHEPREFIX="$EVIDENCE/pycache_hotfix" \
    "$PYTHON_BIN" -m py_compile "$PROBE" "$FINALIZER"

rm -rf \
    "$EVIDENCE/process_enabled_1" \
    "$EVIDENCE/process_enabled_2"

mkdir -p \
    "$EVIDENCE/process_enabled_1" \
    "$EVIDENCE/process_enabled_2"

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

echo "===== CORRECTED EXPLICIT-ON FACTORY FRESH PROCESS 1 ====="
run_enabled_probe process_enabled_1

echo
echo "===== CORRECTED EXPLICIT-ON FACTORY FRESH PROCESS 2 ====="
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

rm -f \
    "$EVIDENCE/finalize.log" \
    "$EVIDENCE/phase4a2_p1_opt_in_class_build_factory_skeleton.json" \
    "$EVIDENCE/PHASE4A2_P1_REPORT.md" \
    "$EVIDENCE/SHA256SUMS"

echo
echo "===== FINALIZE AFTER PROBE FIX ====="
"$PYTHON_BIN" "$FINALIZER" \
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
    "$PROBE" \
    "$FINALIZER" \
    "$ADDENDUM" \
    "$EVIDENCE/apply_manifest.json" \
    "$EVIDENCE/build_off.log" \
    "$EVIDENCE/build_on.log" \
    "$EVIDENCE/process_disabled/result.json" \
    "$EVIDENCE/process_enabled_1/result.json" \
    "$EVIDENCE/process_enabled_2/result.json" \
    "$EVIDENCE/phase4a2_p1_opt_in_class_build_factory_skeleton.json" \
    "$EVIDENCE/PHASE4A2_P1_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

"$PYTHON_BIN" - "$EVIDENCE/phase4a2_p1_opt_in_class_build_factory_skeleton.json" <<'PY_JSON'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON_PASS"
)
assert all(data["gates"].values())
assert all(data["static_gates"].values())
backend = data["builds"]["explicit_on"]["fresh_process_1"]["model"]["hyperparams"]
assert backend["otype"] == "RocWMMAWidth64MLP"
assert backend["parameter_elements"] == 12480
print("PHASE4A2_P1_JSON_AUDIT: PASS")
PY_JSON

echo
echo "PHASE4A2_P1_PROBE_HOTFIX: PASS"
echo "PHASE4A2_P1_JSON_AUDIT: PASS"
echo "PHASE4A2_P1_OPT_IN_CLASS_BUILD_FACTORY_SKELETON: PASS"
echo "Evidence: $EVIDENCE"
