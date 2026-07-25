#!/usr/bin/env bash
# TCNN_RDNA4_P4A1_P0_WIDTH64_TILE_PLAN_CPU_ORACLE_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORACLE="$ROOT/scripts/phase4a1_p0_width64_tile_plan_and_cpu_oracle.py"
EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
BASELINE_TAG="phase4a0-rocwmma-fused-forward-gfx1201-pass"

resolve_p5_evidence() {
    if [[ -n "${P5_EVIDENCE:-}" ]]; then
        printf '%s\n' "$P5_EVIDENCE"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/phase4a0_p5_result.json" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a0_p5_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A0-P5 evidence directory found." >&2
    return 1
}

P5_DIR="$(resolve_p5_evidence)"
P5_JSON="$P5_DIR/phase4a0_p5_result.json"
EVIDENCE="${1:-${P0_EVIDENCE:-$EVIDENCE_ROOT/phase4a1_p0_$(date -u +%Y%m%dT%H%M%SZ)}}"

mkdir -p \
    "$EVIDENCE/process_1" \
    "$EVIDENCE/process_2"

test -f "$ORACLE"
test -f "$P5_JSON"

echo "===== PHASE 4A0 BASELINE ====="
git -C "$ROOT" rev-parse -q --verify "refs/tags/$BASELINE_TAG" >/dev/null
echo "PHASE4A0_BASELINE_TAG: PASS"

python - "$P5_JSON" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["decision"] == "PHASE4A0_P5_TWO_LAYER_FUSED_FORWARD_PASS"
assert all(data["gates"].values())
assert data["context"]["arch"].startswith("gfx1201")
assert int(data["context"]["warp_size"]) == 32
assert data["topology"]["kernel_launches"] == 1
assert not data["topology"]["intermediate_global_reload"]
print("PHASE4A0_P5_PREREQUISITE: PASS")
PY

{
    echo "===== PHASE 4A1-P0 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "baseline_tag: $BASELINE_TAG"
    echo "baseline_tag_commit: $(git -C "$ROOT" rev-list -n1 "$BASELINE_TAG")"
    echo "p5_evidence: $P5_DIR"
    echo "p5_json: $P5_JSON"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo
    echo "===== PYTHON ====="
    python --version
    echo
    echo "===== INPUT HASHES ====="
    sha256sum "$ORACLE" "$P5_JSON"
} > "$EVIDENCE/context.txt" 2>&1

echo
echo "===== ORACLE SELF-TEST ====="
python -m py_compile "$ORACLE"
python "$ORACLE" --self-test \
    2>&1 | tee "$EVIDENCE/self_test.log"

grep -Fx \
    "PHASE4A1_P0_ORACLE_SELF_TEST: PASS" \
    "$EVIDENCE/self_test.log"

echo
echo "===== FRESH PROCESS 1 ====="
python "$ORACLE" \
    --p5-json "$P5_JSON" \
    --output-dir "$EVIDENCE/process_1" \
    2>&1 | tee "$EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="
python "$ORACLE" \
    --p5-json "$P5_JSON" \
    --output-dir "$EVIDENCE/process_2" \
    2>&1 | tee "$EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    grep -Fx "WIDTH64_TILE_COVERAGE: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx "WIDTH64_LDS_SINGLE_BUFFER_PLAN: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx "WIDTH64_SCALAR_VS_TILED_ORACLE: PASS" \
        "$EVIDENCE/$process/run.log"
    grep -Fx "WIDTH64_HIDDEN1_FP16_ORACLE: RECORDED" \
        "$EVIDENCE/$process/run.log"
    grep -Fx "WIDTH64_HIDDEN2_FP16_ORACLE: RECORDED" \
        "$EVIDENCE/$process/run.log"
    grep -Fx "WIDTH64_OUTPUT_FP64_ORACLE: RECORDED" \
        "$EVIDENCE/$process/run.log"
    grep -Fx \
        "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PROCESS: PASS" \
        "$EVIDENCE/$process/run.log"
done

echo
echo "===== REPRODUCIBILITY ====="
files=(
    tile_plan.json
    cpu_oracle.json
    tensor_hashes.json
    oracle_probes.csv
    PHASE4A1_P0_REPORT.md
    input_fp16_row_major.bin
    hidden_1_fp16_row_major.bin
    hidden_2_fp16_row_major.bin
    output_fp64_row_major.bin
    weight_1_fp16_col_major.bin
    weight_2_fp16_col_major.bin
    weight_3_fp16_col_major.bin
    bias_1_fp32.bin
    bias_2_fp32.bin
    bias_3_fp32.bin
)

for file in "${files[@]}"; do
    cmp \
        "$EVIDENCE/process_1/$file" \
        "$EVIDENCE/process_2/$file"
    cp "$EVIDENCE/process_1/$file" "$EVIDENCE/$file"
done

echo "PHASE4A1_P0_FRESH_PROCESS_REPRODUCIBILITY: PASS"

python - "$EVIDENCE/cpu_oracle.json" "$EVIDENCE/tile_plan.json" <<'PY'
import json
import pathlib
import sys

oracle = json.loads(pathlib.Path(sys.argv[1]).read_text())
plan = json.loads(pathlib.Path(sys.argv[2]).read_text())

assert oracle["decision"] == (
    "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE_PASS"
)
assert all(oracle["gates"].values())
assert plan["decision"] == "WIDTH64_FUSED_MLP_TILE_PLAN_LOCKED"

execution = plan["execution"]
assert execution["waves_per_block"] == 4
assert execution["wave_size"] == 32
assert execution["threads_per_block"] == 128
assert execution["mma_sync_calls_per_wave"] == 12
assert execution["mma_sync_calls_per_block"] == 48
assert not execution["intermediate_global_store"]
assert not execution["intermediate_global_reload"]

assert plan["lds"]["buffers"] == 1
assert plan["lds"]["bytes"] == 2048
assert len(plan["barriers"]) == 3
assert len(plan["operations"]) == 48

cpu = oracle["cpu_oracle"]
assert cpu["scalar_vs_tiled"]["hidden_1"]["bitwise_equal"]
assert cpu["scalar_vs_tiled"]["hidden_2"]["bitwise_equal"]
assert cpu["scalar_vs_tiled"]["output"]["bitwise_equal"]
assert cpu["output_statistics"]["nonfinite_count"] == 0

assert oracle["scope_boundary"]["this_phase_is_cpu_only"]
assert not oracle["scope_boundary"]["gpu_correctness_claimed"]
assert not oracle["scope_boundary"]["performance_claimed"]

print("PHASE4A1_P0_JSON_AUDIT: PASS")
PY

sha256sum \
    "$ORACLE" \
    "$P5_JSON" \
    "$EVIDENCE"/*.json \
    "$EVIDENCE"/*.csv \
    "$EVIDENCE"/*.md \
    "$EVIDENCE"/*.bin \
    > "$EVIDENCE/SHA256SUMS"

echo
echo "WIDTH64_FUSED_MLP_TILE_PLAN: LOCKED"
echo "WIDTH64_CPU_FP64_ORACLE: RECORDED"
echo "PHASE4A1_P0_MAP_CONTEXT: RECORDED"
echo "PHASE4A1_P0_WIDTH64_TILE_PLAN_AND_CPU_ORACLE: PASS"
echo "Evidence: $EVIDENCE"
