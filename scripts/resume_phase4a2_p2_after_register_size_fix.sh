#!/usr/bin/env bash
# TCNN_RDNA4_P4A2_P2_HIP_HOST_PASS_HOTFIX_004
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/src/rocwmma_width64_mlp.cu"
CONTRACT="$ROOT/contracts/phase4a2_p2_production_inference_contract.json"
PAYLOAD="$ROOT/scripts/phase4a2_p2_production_source.b64"
PROBE="$ROOT/probes/phase4a2_p2_inference_probe.py"
FINALIZER="$ROOT/scripts/finalize_phase4a2_p2.py"
BINDINGS_DIR="$ROOT/bindings/torch"

EVIDENCE="${PHASE4A2_P2_RESUME_EVIDENCE:-}"

if [[ -z "$EVIDENCE" ]]; then
    while IFS= read -r candidate; do
        if [[ -f "$candidate/apply_manifest.json" ]] && \
           [[ -f "$candidate/build.log" ]]; then
            EVIDENCE="$candidate"
            break
        fi
    done < <(
        find "$HOME/therock_test/tcnn_rdna4_port/workspace/evidence" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a2_p2_*' \
            -print |
        sort -r
    )
fi

if [[ -z "$EVIDENCE" ]]; then
    echo "No resumable Phase 4A2-P2 evidence directory found." >&2
    exit 2
fi

case "$(basename "$EVIDENCE")" in
    phase4a2_p2_*)
        ;;
    *)
        echo "Refusing unsafe evidence path: $EVIDENCE" >&2
        exit 2
        ;;
esac

for path in \
    "$SOURCE" \
    "$CONTRACT" \
    "$PAYLOAD" \
    "$PROBE" \
    "$FINALIZER" \
    "$EVIDENCE/apply_manifest.json"; do
    test -f "$path"
done

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DEPENDENCY_ROOT="${TCNN_DEPENDENCY_ROOT:-$ROOT/dependencies}"
MAX_JOBS="${MAX_JOBS:-4}"

test -x "$PYTHON_BIN"
test -d "$DEPENDENCY_ROOT"

echo "===== PATCH HIP HOST-PASS ASSERTION SCOPE ====="
"$PYTHON_BIN" - "$SOURCE" "$CONTRACT" "$EVIDENCE/apply_manifest.json" <<'PY_PATCH'
import hashlib
import json
import pathlib
import sys

source_path = pathlib.Path(sys.argv[1])
contract_path = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(sys.argv[3])

desired = [
    "// TCNN_RDNA4_P4A2_P2_HIP_HOST_PASS_HOTFIX_005:",
    "// The HIP host pass has no target-specific Wave32 geometry.",
    "// Validate the gfx1201/Wave32 register view only in the device pass.",
    "#if defined(__HIP_DEVICE_COMPILE__)",
    "static_assert(RegA::size() == SLOTS);",
    "static_assert(RegAcc::size() == SLOTS);",
    "#endif",
]

lines = source_path.read_text().splitlines()

reg_a_indices = [
    i for i, line in enumerate(lines)
    if "static_assert(RegA::" in line and "== SLOTS" in line
]
reg_acc_indices = [
    i for i, line in enumerate(lines)
    if "static_assert(RegAcc::" in line and "== SLOTS" in line
]

if len(reg_a_indices) != 1 or len(reg_acc_indices) != 1:
    raise RuntimeError(
        "Expected exactly one RegA and one RegAcc SLOTS assertion, got "
        f"{len(reg_a_indices)} and {len(reg_acc_indices)}."
    )

a_index = reg_a_indices[0]
acc_index = reg_acc_indices[0]

if acc_index <= a_index or acc_index - a_index > 4:
    raise RuntimeError(
        f"Unexpected assertion ordering: RegA={a_index}, RegAcc={acc_index}."
    )

# Normalize both old forms:
#   num_elements == SLOTS
#   size() == SLOTS
# and an already partially guarded form. Remove only the immediately
# associated comments/guard, never neighboring static_asserts.
block_start = a_index
while block_start > 0 and lines[block_start - 1].strip().startswith("//"):
    block_start -= 1

if block_start > 0 and lines[block_start - 1].strip() == (
    "#if defined(__HIP_DEVICE_COMPILE__)"
):
    block_start -= 1
    while block_start > 0 and lines[block_start - 1].strip().startswith("//"):
        block_start -= 1

block_end = acc_index + 1
while block_end < len(lines) and not lines[block_end].strip():
    # Preserve blank lines by stopping before them.
    break

if block_end < len(lines) and lines[block_end].strip() == "#endif":
    block_end += 1

lines[block_start:block_end] = desired
source_path.write_text("\n".join(lines) + "\n")

source = source_path.read_text()
required = (
    "#if defined(__HIP_DEVICE_COMPILE__)",
    "static_assert(RegA::size() == SLOTS);",
    "static_assert(RegAcc::size() == SLOTS);",
    "TCNN_RDNA4_P4A2_P2_HIP_HOST_PASS_HOTFIX_005",
)
for token in required:
    if source.count(token) != 1:
        raise RuntimeError(
            f"Expected one occurrence of {token!r}, got {source.count(token)}."
        )

if "::num_elements == SLOTS" in source:
    raise RuntimeError("Packed num_elements assertion survived.")

digest = hashlib.sha256(source_path.read_bytes()).hexdigest()

contract = json.loads(contract_path.read_text())
contract["production_kernel"]["source_sha256"] = digest
contract["production_kernel"]["register_file_element_api"] = (
    "fragment::size() unpacked elements"
)
contract["production_kernel"]["register_size_assert_scope"] = (
    "__HIP_DEVICE_COMPILE__"
)
contract["production_kernel"]["host_pass_hotfix_marker"] = (
    "TCNN_RDNA4_P4A2_P2_HIP_HOST_PASS_HOTFIX_005"
)
contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

manifest = json.loads(manifest_path.read_text())
manifest["installed"]["production_source"]["after_sha256"] = digest
manifest["hip_host_pass_hotfix"] = {
    "marker": "TCNN_RDNA4_P4A2_P2_HIP_HOST_PASS_HOTFIX_005",
    "reason": (
        "Normalize either size()/num_elements assertion form and guard the "
        "gfx1201 Wave32 register-size proof with __HIP_DEVICE_COMPILE__."
    ),
    "source_sha256": digest,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print("WIDTH64_REGISTER_FILE_DEVICE_PASS_ASSERTION: PASS")
print("corrected_source_sha256: " + digest)
PY_PATCH

grep -qFx \
    "#if defined(__HIP_DEVICE_COMPILE__)" \
    "$SOURCE"
grep -qFx \
    "static_assert(RegA::size() == SLOTS);" \
    "$SOURCE"
grep -qFx \
    "static_assert(RegAcc::size() == SLOTS);" \
    "$SOURCE"

if grep -q "::num_elements == SLOTS" "$SOURCE"; then
    echo "Packed num_elements assertion survived unexpectedly." >&2
    exit 1
fi

# The payload is a transient apply artifact and must not remain in the final
# changed-file set.
rm -f "$PAYLOAD"

PYTHONPYCACHEPREFIX="$EVIDENCE/pycache_host_pass_hotfix" \
    "$PYTHON_BIN" -m py_compile "$PROBE" "$FINALIZER"

rm -rf \
    "$EVIDENCE/build/temp" \
    "$EVIDENCE/build/lib" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

mkdir -p \
    "$EVIDENCE/build/temp" \
    "$EVIDENCE/build/lib" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

echo
echo "===== REBUILD AFTER HIP HOST-PASS FIX ====="
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
grep -q "rocwmma_width64_mlp.cu" "$EVIDENCE/build.log"
grep -q "TCNN_WITH_ROCWMMA_WIDTH64_MLP" "$EVIDENCE/build.log"
echo "WIDTH64_PRODUCTION_BUILD_AFTER_HOST_PASS_FIX: PASS"

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

rm -f \
    "$EVIDENCE/finalize.log" \
    "$EVIDENCE/phase4a2_p2_width64_production_inference_parameter_abi.json" \
    "$EVIDENCE/PHASE4A2_P2_REPORT.md" \
    "$EVIDENCE/SHA256SUMS"

echo
echo "===== FINALIZE AFTER HIP HOST-PASS FIX ====="
"$PYTHON_BIN" "$FINALIZER" \
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

"$PYTHON_BIN" - "$EVIDENCE/phase4a2_p2_width64_production_inference_parameter_abi.json" <<'PY_JSON'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI_PASS"
)
assert all(data["gates"].values())
assert all(data["static_gates"].values())
assert data["static_gates"]["register_file_unpacked_size_api"] is True
assert data["static_gates"]["register_file_size_device_pass_only"] is True
assert data["fresh_processes"]["exact_match"] is True
print("PHASE4A2_P2_JSON_AUDIT: PASS")
PY_JSON

echo
echo "PHASE4A2_P2_HIP_HOST_PASS_HOTFIX: PASS"
echo "PHASE4A2_P2_JSON_AUDIT: PASS"
echo "PHASE4A2_P2_WIDTH64_PRODUCTION_INFERENCE_AND_PARAMETER_ABI: PASS"
echo "Evidence: $EVIDENCE"
