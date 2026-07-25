#!/usr/bin/env bash
# TCNN_RDNA4_P4A0_P3_FRAGMENT_MAP_INTERPRETATION_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYZER="$ROOT/scripts/analyze_phase4a0_p3_fragment_maps.py"
EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"

resolve_p2_evidence() {
    if [[ -n "${P2_EVIDENCE:-}" ]]; then
        printf '%s\n' "$P2_EVIDENCE"
        return
    fi

    if [[ $# -ge 1 && -n "${1:-}" ]]; then
        printf '%s\n' "$1"
        return
    fi

    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/phase4a0_p2_fragment_map.json" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done < <(
        find "$EVIDENCE_ROOT" \
            -maxdepth 1 \
            -type d \
            -name 'phase4a0_p2_*' \
            -print |
        sort -r
    )

    echo "No valid Phase 4A0-P2 evidence directory found." >&2
    return 1
}

P2_DIR="$(resolve_p2_evidence "${1:-}")"
P2_JSON="$P2_DIR/phase4a0_p2_fragment_map.json"
P3_EVIDENCE="${2:-${P3_EVIDENCE:-$EVIDENCE_ROOT/phase4a0_p3_$(date -u +%Y%m%dT%H%M%SZ)}}"

mkdir -p \
    "$P3_EVIDENCE/process_1" \
    "$P3_EVIDENCE/process_2"

test -f "$ANALYZER"
test -f "$P2_JSON"

echo "===== P2 PREREQUISITE ====="
python - "$P2_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())

assert data["decision"] == "PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP_PASS"
assert all(data["gates"].values())
assert data["context"]["arch"].startswith("gfx1201")
assert int(data["context"]["warp_size"]) == 32

print("PHASE4A0_P2_PREREQUISITE: PASS")
PY

{
    echo "===== PHASE 4A0-P3 CONTEXT ====="
    echo "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $ROOT"
    echo "branch: $(git -C "$ROOT" branch --show-current)"
    echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "p2_evidence: $P2_DIR"
    echo "p2_json: $P2_JSON"
    echo
    echo "===== GIT STATUS ====="
    git -C "$ROOT" status --short
    echo
    echo "===== PYTHON ====="
    python --version
    echo
    echo "===== SHA256 ====="
    sha256sum "$ANALYZER" "$P2_JSON"
} > "$P3_EVIDENCE/context.txt" 2>&1

echo
echo "===== ANALYZER SELF-TEST ====="
python -m py_compile "$ANALYZER"
python "$ANALYZER" --self-test \
    2>&1 | tee "$P3_EVIDENCE/self_test.log"

grep -Fx \
    "PHASE4A0_P3_ANALYZER_SELF_TEST: PASS" \
    "$P3_EVIDENCE/self_test.log"

echo
echo "===== FRESH PROCESS 1 ====="
python "$ANALYZER" \
    --input "$P2_JSON" \
    --output-dir "$P3_EVIDENCE/process_1" \
    2>&1 | tee "$P3_EVIDENCE/process_1/run.log"

echo
echo "===== FRESH PROCESS 2 ====="
python "$ANALYZER" \
    --input "$P2_JSON" \
    --output-dir "$P3_EVIDENCE/process_2" \
    2>&1 | tee "$P3_EVIDENCE/process_2/run.log"

for process in process_1 process_2; do
    grep -E \
        '^ROCWMMA_P3_ROLE_AFFINE_GF2_MODELS: (DERIVED|LOOKUP_FALLBACK)$' \
        "$P3_EVIDENCE/$process/run.log"
    grep -E \
        '^ROCWMMA_P3_PAIRWISE_AFFINE_GF2_MODELS: (DERIVED|LOOKUP_FALLBACK)$' \
        "$P3_EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P3_ROLE_EQUIVALENCE: CLASSIFIED" \
        "$P3_EVIDENCE/$process/run.log"
    grep -Fx \
        "ROCWMMA_P3_KERNEL_POLICY: RECORDED" \
        "$P3_EVIDENCE/$process/run.log"
    grep -Fx \
        "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION: PASS" \
        "$P3_EVIDENCE/$process/run.log"
done

echo
echo "===== REPRODUCIBILITY ====="
cmp \
    "$P3_EVIDENCE/process_1/phase4a0_p3_analysis.json" \
    "$P3_EVIDENCE/process_2/phase4a0_p3_analysis.json"
cmp \
    "$P3_EVIDENCE/process_1/PHASE4A0_P3_ANALYSIS.md" \
    "$P3_EVIDENCE/process_2/PHASE4A0_P3_ANALYSIS.md"
cmp \
    "$P3_EVIDENCE/process_1/fragment_role_slot_maps.csv" \
    "$P3_EVIDENCE/process_2/fragment_role_slot_maps.csv"
cmp \
    "$P3_EVIDENCE/process_1/fragment_role_reindex_tables.csv" \
    "$P3_EVIDENCE/process_2/fragment_role_reindex_tables.csv"

echo "ROCWMMA_P3_FRESH_PROCESS_REPRODUCIBILITY: PASS"

cp \
    "$P3_EVIDENCE/process_1/phase4a0_p3_analysis.json" \
    "$P3_EVIDENCE/phase4a0_p3_analysis.json"
cp \
    "$P3_EVIDENCE/process_1/PHASE4A0_P3_ANALYSIS.md" \
    "$P3_EVIDENCE/PHASE4A0_P3_ANALYSIS.md"
cp \
    "$P3_EVIDENCE/process_1/fragment_role_slot_maps.csv" \
    "$P3_EVIDENCE/fragment_role_slot_maps.csv"
cp \
    "$P3_EVIDENCE/process_1/fragment_role_reindex_tables.csv" \
    "$P3_EVIDENCE/fragment_role_reindex_tables.csv"

python - "$P3_EVIDENCE/phase4a0_p3_analysis.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())

assert data["decision"] == "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION_PASS"
assert all(data["gates"].values())
assert set(data["role_models"]) == {
    "matrix_a",
    "matrix_b",
    "accumulator",
}
assert set(data["pairwise_role_equivalence"]) == {
    "matrix_a_to_matrix_b",
    "matrix_a_to_accumulator",
    "matrix_b_to_accumulator",
}

for role in data["role_models"].values():
    assert role["entry_count"] == 256
    assert len(role["coordinate_to_lane_slot"]) == 256

for pair in data["pairwise_role_equivalence"].values():
    assert len(
        pair["coordinate_preserving_reindex"]["entries"]
    ) == 256

print("PHASE4A0_P3_JSON_AUDIT: PASS")
PY

sha256sum \
    "$ANALYZER" \
    "$P2_JSON" \
    "$P3_EVIDENCE/phase4a0_p3_analysis.json" \
    "$P3_EVIDENCE/PHASE4A0_P3_ANALYSIS.md" \
    "$P3_EVIDENCE/fragment_role_slot_maps.csv" \
    "$P3_EVIDENCE/fragment_role_reindex_tables.csv" \
    > "$P3_EVIDENCE/SHA256SUMS"

echo
echo "ROCWMMA_P3_MAP_CONTEXT: RECORDED"
echo "PHASE4A0_P3_FRAGMENT_MAP_INTERPRETATION: PASS"
echo "Evidence: $P3_EVIDENCE"
