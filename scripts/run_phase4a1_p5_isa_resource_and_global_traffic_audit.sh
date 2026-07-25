#!/usr/bin/env bash
# TCNN_RDNA4_P4A1_P5_ISA_RESOURCE_GLOBAL_TRAFFIC_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPARE="$ROOT/scripts/prepare_phase4a1_p5.py"
AUDITOR="$ROOT/scripts/audit_phase4a1_p5_isa.py"
FINALIZER="$ROOT/scripts/finalize_phase4a1_p5.py"
SOURCE="$ROOT/scripts/phase4a1_p4_width64_three_layer_fused.cpp"

HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
LLVM_ROOT="${LLVM_ROOT:-/opt/rocm/llvm/bin}"
OBJDUMP="${OBJDUMP:-$LLVM_ROOT/llvm-objdump}"
READOBJ="${READOBJ:-$LLVM_ROOT/llvm-readobj}"
OBJCOPY="${OBJCOPY:-$LLVM_ROOT/llvm-objcopy}"
OFFLOAD_BUNDLER="${OFFLOAD_BUNDLER:-$LLVM_ROOT/clang-offload-bundler}"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"

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

P0_DIR="$(
    resolve_evidence \
        "${P0_EVIDENCE:-}" \
        'phase4a1_p0_*' \
        'cpu_oracle.json' \
        'Phase 4A1-P0'
)"

P4_DIR="$(
    resolve_evidence \
        "${PHASE4A1_P4_SOURCE_EVIDENCE:-}" \
        'phase4a1_p4_*' \
        'phase4a1_p4_width64_three_layer_fused.json' \
        'Phase 4A1-P4'
)"

P4_JSON="$P4_DIR/phase4a1_p4_width64_three_layer_fused.json"
P4_PREPARATION="$P4_DIR/preparation_manifest.json"
P4_HEADER="$P4_DIR/phase4a1_p4_bindings_generated.hpp"
P4_SHA256SUMS="$P4_DIR/SHA256SUMS"
P4_REFERENCE_JSON="$P4_DIR/process_1/result.json"
P4_REFERENCE_CSV="$P4_DIR/process_1/final_output.csv"

INPUT_BIN="$P0_DIR/input_fp16_row_major.bin"
WEIGHT_1_BIN="$P0_DIR/weight_1_fp16_col_major.bin"
WEIGHT_2_BIN="$P0_DIR/weight_2_fp16_col_major.bin"
WEIGHT_3_BIN="$P0_DIR/weight_3_fp16_col_major.bin"
BIAS_1_BIN="$P0_DIR/bias_1_fp32.bin"
BIAS_2_BIN="$P0_DIR/bias_2_fp32.bin"
BIAS_3_BIN="$P0_DIR/bias_3_fp32.bin"
EXPECTED_HIDDEN_1_BIN="$P0_DIR/hidden_1_fp16_row_major.bin"
EXPECTED_HIDDEN_2_BIN="$P0_DIR/hidden_2_fp16_row_major.bin"
EXPECTED_OUTPUT_BIN="$P0_DIR/output_fp64_row_major.bin"

EVIDENCE="${1:-${PHASE4A1_P5_EVIDENCE:-$EVIDENCE_ROOT/phase4a1_p5_$(date -u +%Y%m%dT%H%M%SZ)}}"

case "$(basename "$EVIDENCE")" in
    phase4a1_p5_*)
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

BUILD_ROOT="$ROOT/build/phase4a1_p5"
BUILD_1="$BUILD_ROOT/device_build_1"
BUILD_2="$BUILD_ROOT/device_build_2"
EXEC_BUILD="$BUILD_ROOT/executable"
BINARY="$EXEC_BUILD/phase4a1_p5_audited_p4_binary"

rm -rf "$BUILD_1" "$BUILD_2" "$EXEC_BUILD"

mkdir -p \
    "$BUILD_1" \
    "$BUILD_2" \
    "$EXEC_BUILD" \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

for path in \
    "$PREPARE" "$AUDITOR" "$FINALIZER" "$SOURCE" \
    "$P4_JSON" "$P4_PREPARATION" "$P4_HEADER" "$P4_SHA256SUMS" \
    "$P4_REFERENCE_JSON" "$P4_REFERENCE_CSV" \
    "$INPUT_BIN" "$WEIGHT_1_BIN" "$WEIGHT_2_BIN" "$WEIGHT_3_BIN" \
    "$BIAS_1_BIN" "$BIAS_2_BIN" "$BIAS_3_BIN" \
    "$EXPECTED_HIDDEN_1_BIN" "$EXPECTED_HIDDEN_2_BIN" \
    "$EXPECTED_OUTPUT_BIN"; do
    test -f "$path"
done

for tool in \
    "$HIPCC" \
    "$OBJDUMP" \
    "$READOBJ" \
    "$OBJCOPY" \
    "$OFFLOAD_BUNDLER"; do
    test -x "$tool"
done

cp "$P4_HEADER" "$BUILD_1/phase4a1_p4_bindings_generated.hpp"
cp "$P4_HEADER" "$BUILD_2/phase4a1_p4_bindings_generated.hpp"
cp "$P4_HEADER" "$EXEC_BUILD/phase4a1_p4_bindings_generated.hpp"

echo "===== P4 PREREQUISITE ====="
python - "$P4_JSON" <<'PY_P5_PREREQ'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == (
    "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_CONSOLIDATED_PASS"
)
assert all(data["gates"].values())
assert data["result"]["topology"]["layers"] == 3
assert data["result"]["topology"]["lds_buffers"] == 1
assert data["result"]["topology"]["lds_bytes"] == 2048
assert data["result"]["topology"]["barriers"] == 3
print("PHASE4A1_P4_PREREQUISITE: PASS")
PY_P5_PREREQ

{
    echo "===== PHASE 4A1-P5 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p0_evidence: $P0_DIR"
    echo "p4_evidence: $P4_DIR"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo
    echo "===== HIPCC ====="
    "$HIPCC" --version
    echo
    echo "===== LLVM TOOLS ====="
    "$OBJDUMP" --version
    "$READOBJ" --version
    "$OBJCOPY" --version
    "$OFFLOAD_BUNDLER" --version
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== PYTHON SELF-CHECK ====="
python -m py_compile "$PREPARE" "$AUDITOR" "$FINALIZER"
echo "PHASE4A1_P5_PYTHON_SELF_CHECK: PASS"

echo
echo "===== PREPARATION ====="
python "$PREPARE" \
    --p4-json "$P4_JSON" \
    --p4-preparation "$P4_PREPARATION" \
    --p4-header "$P4_HEADER" \
    --p4-sha256s "$P4_SHA256SUMS" \
    --source "$SOURCE" \
    --output "$EVIDENCE/preparation_manifest.json" \
    2>&1 | tee "$EVIDENCE/preparation.log"

grep -Fx "PHASE4A1_P5_SOURCE_CONTRACT: PASS" \
    "$EVIDENCE/preparation.log"
grep -Fx "PHASE4A1_P5_PREPARATION: PASS" \
    "$EVIDENCE/preparation.log"

device_common=(
    -std=c++17
    -O2
    -gline-tables-only
    --offload-arch=gfx1201
    -I/opt/rocm/include
)

build_device_companion() {
    local build_dir="$1"
    local label="$2"

    echo "===== DEVICE COMPANION $label ====="

    "$HIPCC" \
        --offload-device-only \
        -S \
        "${device_common[@]}" \
        -I"$build_dir" \
        "$SOURCE" \
        -o "$build_dir/device.s" \
        2>&1 | tee "$EVIDENCE/${label}_assembly_build.log"
}

echo
echo "===== TWO FRESH DEVICE BUILDS ====="
build_device_companion "$BUILD_1" "device_build_1"
build_device_companion "$BUILD_2" "device_build_2"

echo
echo "===== OPTIMIZED LLVM IR ====="
"$HIPCC" \
    --offload-device-only \
    -emit-llvm \
    -S \
    "${device_common[@]}" \
    -I"$BUILD_1" \
    "$SOURCE" \
    -o "$BUILD_1/device.ll" \
    2>&1 | tee "$EVIDENCE/llvm_ir_build.log"

echo
echo "===== EXACT P4 EXECUTABLE REBUILD ====="
compile_executable=(
    "$HIPCC"
    -std=c++17
    -O2
    -gline-tables-only
    --offload-arch=gfx1201
    -I/opt/rocm/include
    -I"$EXEC_BUILD"
    "$SOURCE"
    -o "$BINARY"
)

printf '%q ' "${compile_executable[@]}" \
    > "$EVIDENCE/executable_compile_command.txt"
printf '\n' >> "$EVIDENCE/executable_compile_command.txt"

"${compile_executable[@]}" \
    2>&1 | tee "$EVIDENCE/executable_build.log"

echo
echo "===== EXECUTABLE HIP FATBIN EXTRACTION ====="

EXECUTABLE_SECTIONS="$EVIDENCE/executable_sections.txt"
FATBIN="$EVIDENCE/phase4a1_p5_hip_fatbin.bin"
BUNDLE_IDS="$EVIDENCE/hip_fatbin_bundle_ids.txt"
BUNDLER_LIST_STDERR="$EVIDENCE/hip_fatbin_bundle_list.stderr"
BUNDLER_UNBUNDLE_STDERR="$EVIDENCE/hip_fatbin_unbundle.stderr"
CODE_OBJECT="$EVIDENCE/phase4a1_p5_gfx1201_code_object.hsaco"

"$READOBJ" \
    --sections \
    "$BINARY" \
    > "$EXECUTABLE_SECTIONS"

grep -q "\.hip_fatbin" "$EXECUTABLE_SECTIONS"

"$OBJCOPY" \
    --dump-section ".hip_fatbin=$FATBIN" \
    "$BINARY"

test -s "$FATBIN"

if "$OFFLOAD_BUNDLER" \
    --type=o \
    --inputs="$FATBIN" \
    --list \
    > "$BUNDLE_IDS" \
    2> "$BUNDLER_LIST_STDERR"; then
    BUNDLER_ARGUMENT_STYLE="plural"
else
    "$OFFLOAD_BUNDLER" \
        --type=o \
        --input="$FATBIN" \
        --list \
        > "$BUNDLE_IDS" \
        2> "$BUNDLER_LIST_STDERR"
    BUNDLER_ARGUMENT_STYLE="singular"
fi

mapfile -t GFX1201_BUNDLE_TARGETS < <(
    sed \
        -e 's/^[[:space:]]*//' \
        -e 's/[[:space:]]*$//' \
        "$BUNDLE_IDS" |
    grep -E \
        '^(hip|hipv4)-amdgcn-amd-amdhsa--gfx1201([:+-].*)?$' \
        || true
)

if [[ "${#GFX1201_BUNDLE_TARGETS[@]}" -ne 1 ]]; then
    echo "Expected exactly one gfx1201 HIP bundle ID; got ${#GFX1201_BUNDLE_TARGETS[@]}." >&2
    echo "Recorded IDs:" >&2
    cat "$BUNDLE_IDS" >&2
    exit 1
fi

GFX1201_BUNDLE_TARGET="${GFX1201_BUNDLE_TARGETS[0]}"

if [[ "$BUNDLER_ARGUMENT_STYLE" == "plural" ]]; then
    "$OFFLOAD_BUNDLER" \
        --type=o \
        --inputs="$FATBIN" \
        --outputs="$CODE_OBJECT" \
        --targets="$GFX1201_BUNDLE_TARGET" \
        --unbundle \
        2> "$BUNDLER_UNBUNDLE_STDERR"
else
    "$OFFLOAD_BUNDLER" \
        --type=o \
        --input="$FATBIN" \
        --output="$CODE_OBJECT" \
        --targets="$GFX1201_BUNDLE_TARGET" \
        --unbundle \
        2> "$BUNDLER_UNBUNDLE_STDERR"
fi

test -s "$CODE_OBJECT"

"$READOBJ" \
    --file-headers \
    "$CODE_OBJECT" \
    > "$EVIDENCE/extracted_code_object_header.txt"

grep -q "Machine: EM_AMDGPU" \
    "$EVIDENCE/extracted_code_object_header.txt"

echo "gfx1201_bundle_target: $GFX1201_BUNDLE_TARGET"
echo "PHASE4A1_P5_HIP_FATBIN_SECTION_EXTRACTION: PASS"
echo "PHASE4A1_P5_GFX1201_BUNDLE_UNBUNDLE: PASS"

echo
echo "===== EXTRACTED CODE OBJECT INSPECTION ====="

"$OBJDUMP" \
    -d \
    --triple=amdgcn-amd-amdhsa \
    --mcpu=gfx1201 \
    "$CODE_OBJECT" \
    > "$EVIDENCE/device_objdump.txt"

"$READOBJ" \
    --file-headers \
    --sections \
    --symbols \
    --notes \
    "$CODE_OBJECT" \
    > "$EVIDENCE/device_readobj.txt"

file \
    "$CODE_OBJECT" \
    "$BINARY" \
    > "$EVIDENCE/file_types.txt"

grep -q "Machine: EM_AMDGPU" "$EVIDENCE/device_readobj.txt"
grep -q "width64_three_layer_fused_kernel" "$EVIDENCE/device_objdump.txt"

echo "PHASE4A1_P5_EXTRACTED_CODE_OBJECT_INSPECTION: PASS"

run_replay() {
    local process="$1"
    "$BINARY" \
        "$INPUT_BIN" \
        "$WEIGHT_1_BIN" "$WEIGHT_2_BIN" "$WEIGHT_3_BIN" \
        "$BIAS_1_BIN" "$BIAS_2_BIN" "$BIAS_3_BIN" \
        "$EXPECTED_HIDDEN_1_BIN" "$EXPECTED_HIDDEN_2_BIN" \
        "$EXPECTED_OUTPUT_BIN" \
        "$EVIDENCE/$process/result.json" \
        "$EVIDENCE/$process/final_output.csv" \
        phase4a1_p4 \
        2>&1 | tee "$EVIDENCE/$process/run.log"
}

echo
echo "===== AUDITED BINARY FRESH PROCESS 1 ====="
run_replay process_1

echo
echo "===== AUDITED BINARY FRESH PROCESS 2 ====="
run_replay process_2

for process in process_1 process_2; do
    grep -Fx \
        "RDNA4_WIDTH64_THREE_LAYER_FUSED_FORWARD_CORRECTNESS: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "PHASE4A1_P4_WIDTH64_THREE_LAYER_FUSED_PROCESS: PASS" \
        "$EVIDENCE/$process/run.log"
done

echo
echo "===== ISA / RESOURCE / TRAFFIC AUDIT ====="
python "$AUDITOR" \
    --preparation "$EVIDENCE/preparation_manifest.json" \
    --source "$SOURCE" \
    --assembly-1 "$BUILD_1/device.s" \
    --assembly-2 "$BUILD_2/device.s" \
    --llvm-ir "$BUILD_1/device.ll" \
    --objdump "$EVIDENCE/device_objdump.txt" \
    --readobj "$EVIDENCE/device_readobj.txt" \
    --device-object "$CODE_OBJECT" \
    --binary "$BINARY" \
    --output-json "$EVIDENCE/isa_resource_audit.json" \
    --output-report "$EVIDENCE/PHASE4A1_P5_ISA_REPORT.md" \
    2>&1 | tee "$EVIDENCE/isa_audit.log"

test -s "$FATBIN"
test -s "$CODE_OBJECT"
grep -Fx "$GFX1201_BUNDLE_TARGET" "$BUNDLE_IDS"

audit_markers=(
    "WIDTH64_SINGLE_DEVICE_KERNEL_OBJECT: PASS"
    "WIDTH64_WAVE32_CODE_OBJECT: PASS"
    "WIDTH64_ISA_FRESH_BUILD_REPRODUCIBILITY: PASS"
    "WIDTH64_LDS_RESOURCE_2048_BYTES: PASS"
    "WIDTH64_PRIVATE_SEGMENT_ZERO: PASS"
    "WIDTH64_NO_SCRATCH_INSTRUCTIONS: PASS"
    "WIDTH64_MATRIX_CORE_INSTRUCTIONS_PRESENT: PASS"
    "WIDTH64_LDS_READ_WRITE_PRESENT: PASS"
    "WIDTH64_BLOCK_BARRIERS_PRESENT: PASS"
    "WIDTH64_LLVM_LDS_ADDRESS_SPACE_PRESENT: PASS"
    "WIDTH64_DEVICE_OBJECT_DISASSEMBLY_VALID: PASS"
    "WIDTH64_GLOBAL_TRAFFIC_CLASSIFIED: PASS"
    "WIDTH64_NO_HIDDEN_INTERMEDIATE_GLOBAL_TRAFFIC: PASS"
    "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT: PASS"
)

for marker in "${audit_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/isa_audit.log"
done

echo
echo "===== FINALIZE ====="
python "$FINALIZER" \
    --preparation "$EVIDENCE/preparation_manifest.json" \
    --audit-json "$EVIDENCE/isa_resource_audit.json" \
    --audit-report "$EVIDENCE/PHASE4A1_P5_ISA_REPORT.md" \
    --p4-json "$P4_JSON" \
    --p4-reference-json "$P4_REFERENCE_JSON" \
    --p4-reference-csv "$P4_REFERENCE_CSV" \
    --process-1-json "$EVIDENCE/process_1/result.json" \
    --process-1-csv "$EVIDENCE/process_1/final_output.csv" \
    --process-2-json "$EVIDENCE/process_2/result.json" \
    --process-2-csv "$EVIDENCE/process_2/final_output.csv" \
    --evidence "$EVIDENCE" \
    2>&1 | tee "$EVIDENCE/finalize.log"

final_markers=(
    "WIDTH64_AUDITED_BINARY_FUNCTIONAL_REPLAY: PASS"
    "WIDTH64_AUDITED_BINARY_EXACT_P4_REPLAY: PASS"
    "PHASE4A1_P5_CONSOLIDATED_EVIDENCE: RECORDED"
    "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT: PASS"
)

for marker in "${final_markers[@]}"; do
    grep -Fx "$marker" "$EVIDENCE/finalize.log"
done

cp "$BUILD_1/device.s" "$EVIDENCE/device_build_1.s"
cp "$BUILD_2/device.s" "$EVIDENCE/device_build_2.s"
cp "$BUILD_1/device.ll" "$EVIDENCE/device_optimized.ll"
cp "$BINARY" "$EVIDENCE/phase4a1_p5_audited_binary"

sha256sum \
    "$PREPARE" "$AUDITOR" "$FINALIZER" "$SOURCE" \
    "$P4_JSON" "$P4_PREPARATION" "$P4_HEADER" \
    "$EVIDENCE/preparation_manifest.json" \
    "$EVIDENCE/isa_resource_audit.json" \
    "$EVIDENCE/phase4a1_p5_width64_isa_resource_global_traffic.json" \
    "$EVIDENCE/device_build_1.s" \
    "$EVIDENCE/device_build_2.s" \
    "$EVIDENCE/device_optimized.ll" \
    "$EVIDENCE/executable_sections.txt" \
    "$EVIDENCE/phase4a1_p5_hip_fatbin.bin" \
    "$EVIDENCE/hip_fatbin_bundle_ids.txt" \
    "$EVIDENCE/device_objdump.txt" \
    "$EVIDENCE/device_readobj.txt" \
    "$EVIDENCE/phase4a1_p5_gfx1201_code_object.hsaco" \
    "$EVIDENCE/phase4a1_p5_audited_binary" \
    "$EVIDENCE/PHASE4A1_P5_REPORT.md" \
    > "$EVIDENCE/SHA256SUMS"

python - \
    "$EVIDENCE/phase4a1_p5_width64_isa_resource_global_traffic.json" \
    <<'PY_P5_JSON_AUDIT'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())

assert data["decision"] == (
    "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT_PASS"
)
assert all(data["gates"].values())

audit = data["isa_audit"]["data"]
assert audit["decision"] == (
    "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_AUDIT_PASS"
)
assert all(audit["gates"].values())
assert audit["kernel"]["device_kernel_count"] == 1
assert audit["resources"]["wavefront_size32"] == 1
assert audit["resources"]["group_segment_fixed_size"] == 2048
assert audit["resources"]["private_segment_fixed_size"] == 0
assert audit["scratch_instruction_count"] == 0
assert audit["instructions"]["mfma_or_wmma"] > 0
assert audit["instructions"]["lds_reads"] > 0
assert audit["instructions"]["lds_writes"] > 0
assert audit["instructions"]["block_barriers"] >= 3

print("PHASE4A1_P5_JSON_AUDIT: PASS")
PY_P5_JSON_AUDIT

echo
echo "PHASE4A1_P5_HIP_FATBIN_SECTION_EXTRACTION: PASS"
echo "PHASE4A1_P5_GFX1201_BUNDLE_UNBUNDLE: PASS"
echo "PHASE4A1_P5_EXTRACTED_CODE_OBJECT_INSPECTION: PASS"
echo "WIDTH64_SINGLE_DEVICE_KERNEL_OBJECT: PASS"
echo "WIDTH64_WAVE32_CODE_OBJECT: PASS"
echo "WIDTH64_LDS_RESOURCE_2048_BYTES: PASS"
echo "WIDTH64_PRIVATE_SEGMENT_ZERO: PASS"
echo "WIDTH64_NO_SCRATCH_INSTRUCTIONS: PASS"
echo "WIDTH64_MATRIX_CORE_INSTRUCTIONS_PRESENT: PASS"
echo "WIDTH64_LDS_READ_WRITE_PRESENT: PASS"
echo "WIDTH64_BLOCK_BARRIERS_PRESENT: PASS"
echo "WIDTH64_LLVM_LDS_ADDRESS_SPACE_PRESENT: PASS"
echo "WIDTH64_DEVICE_OBJECT_DISASSEMBLY_VALID: PASS"
echo "WIDTH64_GLOBAL_TRAFFIC_CLASSIFIED: PASS"
echo "WIDTH64_NO_HIDDEN_INTERMEDIATE_GLOBAL_TRAFFIC: PASS"
echo "WIDTH64_AUDITED_BINARY_FUNCTIONAL_REPLAY: PASS"
echo "PHASE4A1_P5_JSON_AUDIT: PASS"
echo "PHASE4A1_P5_MAP_CONTEXT: RECORDED"
echo "PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT: PASS"
echo "Evidence: $EVIDENCE"
