#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY="$ROOT/scripts/apply_phase4a2_p2.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p2.py"
PROBE="$ROOT/probes/phase4a2_p2_inference_probe.py"
CONTRACT="$ROOT/contracts/phase4a2_p2_production_inference_contract.json"
PAYLOAD="$ROOT/scripts/phase4a2_p2_production_source.b64"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
P1_FILENAME="phase4a2_p1_opt_in_class_build_factory_skeleton.json"
P4_FILENAME="phase4a1_p4_width64_three_layer_fused.json"

resolve_evidence() {
    local explicit="$1"
    local pattern="$2"
    local filename="$3"
    local label="$4"

    if [[ -n "$explicit" ]]; then
        printf '%s\n' "$explicit"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/$filename" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name "$pattern" \
            -print |
        sort -r
    )

    echo "No valid $label evidence directory found." >&2
    return 1
}

P1_DIR="$(
    resolve_evidence \
        "${PHASE4A2_P1_SOURCE_EVIDENCE:-}" \
        'phase4a2_p1_*' \
        "$P1_FILENAME" \
        'Phase 4A2-P1'
)"
P1_JSON="$P1_DIR/$P1_FILENAME"

P4_DIR="$(
    resolve_evidence \
        "${PHASE4A1_P4_SOURCE_EVIDENCE:-}" \
        'phase4a1_p4_*' \
        "$P4_FILENAME" \
        'Phase 4A1-P4'
)"
P4_JSON="$P4_DIR/$P4_FILENAME"
P4_PREPARATION="$P4_DIR/preparation_manifest.json"
P4_HEADER="$P4_DIR/phase4a1_p4_bindings_generated.hpp"

EVIDENCE="${1:-${PHASE4A2_P2_EVIDENCE:-$EVIDENCE_ROOT/phase4a2_p2_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a2_p2_*)
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
    "$CONTRACT" \
    "$PAYLOAD" \
    "$P1_JSON" \
    "$P4_JSON" \
    "$P4_PREPARATION" \
    "$P4_HEADER"; do
    test -f "$path"
done

mkdir -p \
    "$EVIDENCE/build/temp" \
    "$EVIDENCE/build/lib" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

echo "===== PHASE 4A2-P1 BASELINE ====="
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
SUBJECT="$(git -C "$ROOT" show -s --format=%s HEAD)"
echo "head: $HEAD"
echo "subject: $SUBJECT"
test "${HEAD:0:7}" = "c1b95f6"
test "$SUBJECT" = \
    "Add Phase 4A2-P1 rocWMMA Width-64 opt-in backend skeleton"
echo "PHASE4A2_P1_COMMIT_BOUND: PASS"

echo
echo "===== SELF-CHECK ====="
PYTHONPYCACHEPREFIX="$EVIDENCE/pycache" \
    python -m py_compile "$APPLY" "$FINALIZER" "$PROBE"

python - "$CONTRACT" <<'PY_CONTRACT'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["marker"] == "TCNN_RDNA4_P4A2_P2_PRODUCTION_INFERENCE_001"
assert data["production_kernel"]["parameter_elements"] == 12480
assert data["production_kernel"]["lds_bytes"] == 2048
assert data["production_kernel"]["source_barriers"] == 3
assert data["scope"]["inference"] == "qualified_by_P2"
assert data["scope"]["training_forward"] == "fail_closed"
assert data["scope"]["backward"] == "fail_closed"
assert data["scope"]["performance_claim"] == "none"
print("PHASE4A2_P2_CONTRACT_SELF_CHECK: PASS")
PY_CONTRACT
echo "PHASE4A2_P2_PYTHON_SELF_CHECK: PASS"

{
    echo "===== PHASE 4A2-P2 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head_before_apply: $HEAD"
    echo "phase4a2_p1_evidence: $P1_DIR"
    echo "phase4a1_p4_evidence: $P4_DIR"
    echo
    echo "===== STATUS BEFORE APPLY ====="
    git -C "$ROOT" status --short --untracked-files=all
} > "$EVIDENCE/context_before_apply.txt"

echo
echo "===== APPLY PRODUCTION BRIDGE ====="
python "$APPLY" \
    --repo "$ROOT" \
    --contract "$CONTRACT" \
    --payload "$PAYLOAD" \
    --p1-json "$P1_JSON" \
    --p4-json "$P4_JSON" \
    --p4-preparation "$P4_PREPARATION" \
    --p4-header "$P4_HEADER" \
    --output "$EVIDENCE/apply_manifest.json" \
    2>&1 | tee "$EVIDENCE/apply.log"

apply_markers=(
    "WIDTH64_P4_DATAFLOW_BOUND: PASS"
    "WIDTH64_PARAMETER_ABI_ADAPTATION: PASS"
    "WIDTH64_COLUMN_MAJOR_BATCH_BRIDGE: PASS"
    "WIDTH64_NO_ORACLE_DIAGNOSTIC_ARGUMENTS: PASS"
    "WIDTH64_INFERENCE_ONLY_FAIL_CLOSED_SCOPE: PASS"
    "PHASE4A2_P2_APPLY: PASS"
)

for marker in "${apply_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/apply.log"
done

test ! -e "$PAYLOAD"
git -C "$ROOT" diff --check

echo
echo "===== PRODUCTION SOURCE AUDIT ====="
python - "$ROOT" "$P4_HEADER" <<'PY_SOURCE'
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1])
p4_header = Path(sys.argv[2])
source = (root / "src/rocwmma_width64_mlp.cu").read_text()
bridge = (
    root / "include/tiny-cuda-nn/network_with_input_encoding.h"
).read_text()
mapping_path = (
    root
    / "include/tiny-cuda-nn/networks/"
    "rocwmma_width64_mapping_gfx1201.h"
)
mapping = mapping_path.read_text()

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert source.count("__global__ void") == 1
assert source.count("rocwmma::mma_sync(") == 3
assert source.count("__syncthreads();") == 3
assert source.count("__shared__ __align__(16) Half hidden_lds") == 1
assert "blockIdx.x" in source
assert "input.n() / TILE_ROWS" in source
assert "const __half* bias_0" in source
assert "accumulator_bias_to_matrix_a<false>" in source
assert "hipLaunchKernelGGL(" in source
assert "hipDeviceSynchronize" not in source
assert "hipStreamSynchronize" not in source
assert "expected_hidden" not in source
assert "diagnostics" not in source
assert "atomicAdd(" not in source
assert "m_fp16_rocwmma_width64" in bridge
assert "set_padded_output_width(64)" in bridge
assert sha(mapping_path) == sha(p4_header)
assert sha(mapping_path) == (
    "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517"
)
assert "namespace phase4a1_p2_generated" in mapping

print("WIDTH64_PRODUCTION_SOURCE_DATAFLOW_AUDIT: PASS")
print("WIDTH64_PRODUCTION_MAPPING_HEADER_EXACT: PASS")
print("WIDTH64_PRODUCTION_NO_HOST_SYNC_STATIC: PASS")
print("WIDTH64_PRODUCTION_NO_DIAGNOSTIC_PATH_STATIC: PASS")
PY_SOURCE

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BINDINGS_DIR="$ROOT/bindings/torch"
DEPENDENCY_ROOT="${TCNN_DEPENDENCY_ROOT:-$ROOT/dependencies}"
MAX_JOBS="${MAX_JOBS:-4}"

test -x "$PYTHON_BIN"
test -d "$DEPENDENCY_ROOT"

rm -rf "$EVIDENCE/build/temp" "$EVIDENCE/build/lib"
mkdir -p "$EVIDENCE/build/temp" "$EVIDENCE/build/lib"

echo
echo "===== EXPLICIT-ON PRODUCTION BUILD ====="
(
    cd "$BINDINGS_DIR"
    env \
        PYTORCH_ROCM_ARCH=gfx1201 \
        TCNN_DEPENDENCY_ROOT="$DEPENDENCY_ROOT" \
        TCNN_HALF_PRECISION=1 \
        TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 \
        MAX_JOBS="$MAX_JOBS" \
        "$PYTHON_BIN" setup.py build_ext \
            --build-temp "$EVIDENCE/build/temp" \
            --build-lib "$EVIDENCE/build/lib"
) 2>&1 | tee "$EVIDENCE/build.log"

grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" \
    "$EVIDENCE/build.log"
grep -q "rocwmma_width64_mlp.cu" \
    "$EVIDENCE/build.log"
grep -q "TCNN_WITH_ROCWMMA_WIDTH64_MLP" \
    "$EVIDENCE/build.log"

echo "WIDTH64_PRODUCTION_BUILD: PASS"

run_probe() {
    local process="$1"
    (
        cd "$EVIDENCE/$process"
        env \
            PYTHONNOUSERSITE=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH="$EVIDENCE/build/lib:$BINDINGS_DIR" \
            "$PYTHON_BIN" "$PROBE" \
                --output "$EVIDENCE/$process/result.json"
    ) 2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== PRODUCTION INFERENCE FRESH PROCESS 1 ====="
run_probe process_1

echo
echo "===== PRODUCTION INFERENCE FRESH PROCESS 2 ====="
run_probe process_2

process_markers=(
    "WIDTH64_PRODUCTION_PARAMETER_ABI_12480_FP16: PASS"
    "WIDTH64_PRODUCTION_BATCH_GRID_16_512: PASS"
    "WIDTH64_PRODUCTION_INFERENCE_VS_CPU_FP32: PASS"
    "WIDTH64_PRODUCTION_REPEAT_BITWISE: PASS"
    "WIDTH64_PRODUCTION_TILE_PREFIX_INVARIANCE: PASS"
    "WIDTH64_PRODUCTION_NONDEFAULT_STREAM: PASS"
    "WIDTH64_PRODUCTION_FP16_OUTPUT: PASS"
    "WIDTH64_TRAINING_FORWARD_FAIL_CLOSED: PASS"
    "PHASE4A2_P2_PRODUCTION_INFERENCE_PROCESS: PASS"
)

for process in process_1 process_2; do
    for marker in "${process_markers[@]}"; do
        grep -Fx "$marker" "$EVIDENCE/$process/run.log"
    done
done

cmp \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_2/result.json"

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --repo "$ROOT" \
    --contract "$CONTRACT" \
    --apply "$EVIDENCE/apply_manifest.json" \
    --process-1 "$EVIDENCE/process_1/result.json" \
    --process-2 "$EVIDENCE/process_2/result.json" \
    --build-log "$EVIDENCE/build.log" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

final_markers=(
    "WIDTH64_PRODUCTION_KERNEL_INSTALLED: PASS"
    "WIDTH64_P4_SOURCE_AND_MAPPING_BOUND: PASS"
    "WIDTH64_FP16_PARAMETER_ABI_12480: PASS"
    "WIDTH64_COLUMN_MAJOR_BATCH_BRIDGE: PASS"
    "WIDTH64_MULTI_BLOCK_BATCH_CORRECTNESS: PASS"
    "WIDTH64_INFERENCE_VS_CPU_REFERENCE: PASS"
    "WIDTH64_FRESH_PROCESS_REPRODUCIBILITY: PASS"
    "WIDTH64_NONDEFAULT_STREAM_CORRECTNESS: PASS"
    "WIDTH64_TRAINING_BACKWARD_FAIL_CLOSED: PASS"
    "WIDTH64_NO_ORACLE_DIAGNOSTIC_PATH: PASS"
    "PHASE4A2_P2_CONSOLIDATED_EVIDENCE: RECORDED"
    "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI: PASS"
)

for marker in "${final_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/finalize.log"
done

sha256sum \
    "$P1_JSON" \
    "$P4_JSON" \
    "$P4_PREPARATION" \
    "$P4_HEADER" \
    "$CONTRACT" \
    "$APPLY" \
    "$FINALIZER" \
    "$PROBE" \
    "$ROOT/src/rocwmma_width64_mlp.cu" \
    "$ROOT/include/tiny-cuda-nn/network_with_input_encoding.h" \
    "$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h" \
    "$EVIDENCE/apply_manifest.json" \
    "$EVIDENCE/build.log" \
    "$EVIDENCE/process_1/result.json" \
    "$EVIDENCE/process_2/result.json" \
    "$EVIDENCE/phase4a2_p2_width64_production_inference_parameter_abi.json" \
    "$EVIDENCE/PHASE4A2_P2_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

python - "$EVIDENCE/phase4a2_p2_width64_production_inference_parameter_abi.json" <<'PY_JSON'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI_PASS"
)
assert all(data["gates"].values())
assert all(data["static_gates"].values())
assert data["fresh_processes"]["exact_match"] is True
assert data["metrics"]["batch_cases"] == [1, 16, 17, 255, 256, 257]
assert data["fresh_processes"]["process_1"]["backend"][
    "inference_qualified"
] is True
assert data["fresh_processes"]["process_1"]["backend"][
    "forward_qualified"
] is False
assert data["fresh_processes"]["process_1"]["backend"][
    "backward_qualified"
] is False
print("PHASE4A2_P2_JSON_AUDIT: PASS")
PY_JSON

echo
echo "WIDTH64_PRODUCTION_KERNEL_INSTALLED: PASS"
echo "WIDTH64_P4_SOURCE_AND_MAPPING_BOUND: PASS"
echo "WIDTH64_FP16_PARAMETER_ABI_12480: PASS"
echo "WIDTH64_COLUMN_MAJOR_BATCH_BRIDGE: PASS"
echo "WIDTH64_MULTI_BLOCK_BATCH_CORRECTNESS: PASS"
echo "WIDTH64_INFERENCE_VS_CPU_REFERENCE: PASS"
echo "WIDTH64_FRESH_PROCESS_REPRODUCIBILITY: PASS"
echo "WIDTH64_NONDEFAULT_STREAM_CORRECTNESS: PASS"
echo "WIDTH64_TRAINING_BACKWARD_FAIL_CLOSED: PASS"
echo "WIDTH64_NO_ORACLE_DIAGNOSTIC_PATH: PASS"
echo "PHASE4A2_P2_JSON_AUDIT: PASS"
echo "PHASE4A2_P2_MAP_CONTEXT: RECORDED"
echo "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI: PASS"
echo "Evidence: $EVIDENCE"
