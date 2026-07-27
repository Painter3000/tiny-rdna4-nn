#!/usr/bin/env bash
# TCNN_RDNA4_P4A3_Q0B_APPARATUS_REDESIGN_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT="$ROOT/contracts/phase4a3_q0b_apparatus_contract.json"
Q0A="$ROOT/prior_evidence/phase4a3_q0a_forensic_summary.json"
PATCHER="$ROOT/scripts/patch_phase4a3_q0b_binding.py"
WORKER="$ROOT/scripts/phase4a3_q0b_worker.py"
FINALIZER="$ROOT/scripts/finalize_phase4a3_q0b.py"
SMOKE="$ROOT/scripts/phase4a3_q0b_hook_smoke.py"
BRIDGE_SOURCE="$ROOT/scripts/phase4a3_q0b_hip_runtime_bridge.cpp"
BINDING="$ROOT/bindings/torch/tinycudann/bindings.cpp"
SETUP="$ROOT/bindings/torch/setup.py"
SOURCE="$ROOT/src/rocwmma_width64_mlp.cu"
MAPPING="$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h"

EVIDENCE_ROOT="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence"
EVIDENCE="${1:-${PHASE4A3_Q0B_EVIDENCE:-$EVIDENCE_ROOT/phase4a3_q0b_$(date -u +%Y%m%dT%H%M%SZ)}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
MAX_JOBS="${MAX_JOBS:-1}"
BENCH_CPU="${PHASE4A3_BENCH_CPU:-$(( $(nproc) - 1 ))}"
TAG="phase4a2-width64-production-inference-gfx1201-pass"
EXPECTED_COMMIT="cd1330a21452f7e2edab9e676567b7a040f922bc"

case "$(basename "$EVIDENCE")" in
	phase4a3_q0b_*) ;;
	*) echo "Unsafe evidence path: $EVIDENCE" >&2; exit 2 ;;
esac
if [[ -d "$EVIDENCE" ]] && [[ -n "$(find "$EVIDENCE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
	echo "Refusing non-empty evidence directory." >&2
	exit 2
fi
mkdir -p "$EVIDENCE"/{build/temp,build/lib,workers,pycache,backup}
exec > >(tee "$EVIDENCE/run.log") 2>&1

BACKUP="$EVIDENCE/backup/bindings.cpp"
PATCH_ACTIVE=false
restore_binding() {
	if [[ "$PATCH_ACTIVE" == true && -f "$BACKUP" ]]; then
		"$PYTHON_BIN" "$PATCHER" --file "$BINDING" --backup "$BACKUP" --mode restore || true
		PATCH_ACTIVE=false
	fi
}
trap restore_binding EXIT INT TERM HUP ERR

echo "===== PHASE 4A3-Q0b APPARATUS REDESIGN ====="
echo "Evidence: $EVIDENCE"
echo "CPU: $BENCH_CPU"

echo
echo "===== HARD PREFLIGHT ====="
TTY_NAME="$(tty || true)"
REAL_TTY=false
[[ "$TTY_NAME" =~ ^/dev/tty[0-9]+$ ]] && REAL_TTY=true
[[ "$REAL_TTY" == true ]] || { echo "Q0b requires /dev/ttyN." >&2; exit 10; }
DISPLAY_MANAGER_INACTIVE=true
for service in display-manager.service lightdm.service gdm.service gdm3.service sddm.service; do
	if systemctl is-active --quiet "$service" 2>/dev/null; then
		DISPLAY_MANAGER_INACTIVE=false
	fi
done
[[ "$DISPLAY_MANAGER_INACTIVE" == true ]] || { echo "Display manager is active." >&2; exit 11; }
FORBIDDEN_ENV_CLEAR=true
for name in HIP_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY; do
	value="${!name:-}"
	if [[ -n "$value" && "$value" != 0 ]]; then
		echo "Forbidden: $name=$value" >&2
		FORBIDDEN_ENV_CLEAR=false
	fi
done
[[ "$FORBIDDEN_ENV_CLEAR" == true ]] || exit 12
echo "WIDTH64_Q0B_ENVIRONMENT_PREFLIGHT: PASS"

echo
echo "===== RELEASE / SOURCE IDENTITY ====="
TAG_COMMIT="$(git -C "$ROOT" rev-list -n 1 "$TAG")"
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ "$TAG_COMMIT" == "$EXPECTED_COMMIT" ]]
git -C "$ROOT" merge-base --is-ancestor "$TAG_COMMIT" "$HEAD"
RELEASE_IDENTITY_PASS=true
[[ "$(sha256sum "$SOURCE" | awk '{print $1}')" == "7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4" ]]
[[ "$(sha256sum "$MAPPING" | awk '{print $1}')" == "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517" ]]

git -C "$ROOT" show "$TAG_COMMIT:bindings/torch/tinycudann/bindings.cpp" > "$EVIDENCE/backup/bindings.cpp.release"
git -C "$ROOT" show "$TAG_COMMIT:bindings/torch/setup.py" > "$EVIDENCE/backup/setup.py.release"
cmp -s "$BINDING" "$EVIDENCE/backup/bindings.cpp.release"
cmp -s "$SETUP" "$EVIDENCE/backup/setup.py.release"
PRODUCTION_DIFF="$(git -C "$ROOT" diff --name-only "$TAG_COMMIT" -- src include bindings/torch/setup.py bindings/torch/tinycudann)"
[[ -z "$PRODUCTION_DIFF" ]] || {
	echo "Production paths differ before Q0b:" >&2
	printf '%s\n' "$PRODUCTION_DIFF" >&2
	exit 13
}
PRODUCTION_IDENTITY_PASS=true
echo "WIDTH64_Q0B_RELEASE_AND_PRODUCTION_IDENTITY: PASS"

echo
echo "===== CONTRACT SELF-CHECK ====="
PYTHONPYCACHEPREFIX="$EVIDENCE/pycache/preflight" "$PYTHON_BIN" -m py_compile "$PATCHER" "$WORKER" "$FINALIZER" "$SMOKE"
bash -n "$0"
"$PYTHON_BIN" - "$CONTRACT" "$Q0A" <<'PY_CHECK'
import hashlib, json, pathlib, sys
contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
q0a_path = pathlib.Path(sys.argv[2])
q0a = json.loads(q0a_path.read_text())
assert contract["marker"] == "TCNN_RDNA4_P4A3_Q0B_APPARATUS_REDESIGN_001"
assert q0a["decision"] == "PROCEED_TO_PHASE4A3_Q0B_APPARATUS_REDESIGN"
assert hashlib.sha256(q0a_path.read_bytes()).hexdigest() == contract["prior_evidence"]["q0a_summary_sha256"]
assert contract["comparison"]["no_ratio_direction_expected"] is True
assert contract["analysis_pipeline_frozen"]["outlier_removal"] == "none"
assert contract["analysis_pipeline_frozen"]["posthoc_block_dropping"] is False
assert contract["floors"]["subtract_any_floor"] is False
assert contract["floors"]["common_additive_S_assumption_forbidden"] is True
assert contract["claims"]["performance_claim_allowed"] is False
assert contract["comparison"]["required_api_batch_size_granularity"] == 256
assert contract["comparison"]["hardcoded_backend_tile_granularity_for_cpp_api_forbidden"] is True
print("PHASE4A3_Q0B_PREREGISTRATION_SELF_CHECK: PASS")
PY_CHECK

echo
echo "===== TEST-ONLY BRIDGE ====="
BRIDGE="$EVIDENCE/build/libphase4a3_q0b_bridge.so"
"$HIPCC" -std=c++17 -O2 -fPIC -shared "$BRIDGE_SOURCE" -o "$BRIDGE"
test -s "$BRIDGE"
echo "WIDTH64_Q0B_BRIDGE_BUILD: PASS"

echo
echo "===== TEMPORARY TEST-ONLY BINDING HOOK ====="
"$PYTHON_BIN" "$PATCHER" --file "$BINDING" --backup "$BACKUP" --mode apply
PATCH_ACTIVE=true
cp "$BINDING" "$EVIDENCE/backup/bindings.cpp.patched"

echo
echo "===== FRESH TEST-BINDING BUILD ====="
(
	cd "$ROOT/bindings/torch"
	env \
		PYTORCH_ROCM_ARCH=gfx1201 \
		TCNN_DEPENDENCY_ROOT="$ROOT/dependencies" \
		TCNN_HALF_PRECISION=1 \
		TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 \
		MAX_JOBS="$MAX_JOBS" \
		"$PYTHON_BIN" setup.py build_ext \
			--build-temp "$EVIDENCE/build/temp" \
			--build-lib "$EVIDENCE/build/lib"
) 2>&1 | tee "$EVIDENCE/build.log"
grep -Fx "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP: ON" "$EVIDENCE/build.log"
grep -q "rocwmma_width64_mlp.cu" "$EVIDENCE/build.log"
shopt -s nullglob
EXTENSIONS=("$EVIDENCE"/build/lib/tinycudann_bindings/_120_C*.so)
shopt -u nullglob
[[ "${#EXTENSIONS[@]}" -eq 1 ]] || { echo "Expected exactly one extension." >&2; exit 14; }
EXTENSION="${EXTENSIONS[0]}"
FRESH_TEST_BINDING=true
echo "extension: $EXTENSION"
echo "extension_sha256: $(sha256sum "$EXTENSION" | awk '{print $1}')"

echo
echo "===== RESTORE REPOSITORY BINDING BEFORE EXECUTION ====="
"$PYTHON_BIN" "$PATCHER" --file "$BINDING" --backup "$BACKUP" --mode restore
PATCH_ACTIVE=false
BINDING_RESTORED=true
cmp -s "$BINDING" "$EVIDENCE/backup/bindings.cpp.release"
PRODUCTION_DIFF_AFTER="$(git -C "$ROOT" diff --name-only "$TAG_COMMIT" -- src include bindings/torch/setup.py bindings/torch/tinycudann)"
[[ -z "$PRODUCTION_DIFF_AFTER" ]]
echo "WIDTH64_Q0B_BINDING_RESTORED_BEFORE_EXECUTION: PASS"

echo
echo "===== FAIL-FAST NATIVE HOOK SMOKE ====="
env \
	PYTHONPATH="$EVIDENCE/build/lib:$ROOT/bindings/torch" \
	PYTHONPYCACHEPREFIX="$EVIDENCE/pycache/hook_smoke" \
	"$PYTHON_BIN" "$SMOKE" \
		--output "$EVIDENCE/phase4a3_q0b_hook_smoke.json" \
	2>&1 | tee "$EVIDENCE/hook_smoke.log"
grep -Fx "PHASE4A3_Q0B_NATIVE_HOOK_SMOKE: PASS" "$EVIDENCE/hook_smoke.log"

"$PYTHON_BIN" - \
	"$EVIDENCE/context.json" "$TTY_NAME" "$REAL_TTY" \
	"$DISPLAY_MANAGER_INACTIVE" "$FORBIDDEN_ENV_CLEAR" \
	"$HEAD" "$BRANCH" "$TAG_COMMIT" "$RELEASE_IDENTITY_PASS" \
	"$PRODUCTION_IDENTITY_PASS" "$BINDING_RESTORED" \
	"$FRESH_TEST_BINDING" "$EXTENSION" "$BRIDGE" "$BENCH_CPU" <<'PY_CONTEXT'
import hashlib, json, pathlib, sys
(output, tty, real, dm, forbidden, head, branch, tag_commit, release_identity,
 production_identity, binding_restored, fresh_binding, extension, bridge, cpu) = sys.argv[1:]
b = lambda value: value.lower() == "true"
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
data = {
 "tty": tty, "real_tty": b(real), "display_manager_inactive": b(dm),
 "forbidden_environment_clear": b(forbidden), "head": head, "branch": branch,
 "release_tag_commit": tag_commit, "release_identity_pass": b(release_identity),
 "production_identity_pass": b(production_identity), "binding_restored": b(binding_restored),
 "fresh_test_binding": b(fresh_binding), "extension": extension,
 "extension_sha256": sha(extension), "bridge": bridge, "bridge_sha256": sha(bridge),
 "bench_cpu": int(cpu),
}
pathlib.Path(output).write_text(json.dumps(data, indent=2) + "\n")
PY_CONTEXT

echo
echo "===== FRESH PROCESS MATRIX ====="
INDEX_JSONL="$EVIDENCE/worker_index.jsonl"
: > "$INDEX_JSONL"
ORDERS=(AB BA AB BA)
set +e
for mode in spin auto; do
	for batch in 1 31 128; do
		for process_index in 0 1 2 3; do
			start_order="${ORDERS[$process_index]}"
			stem="${mode}_b${batch}_p${process_index}_${start_order}"
			output="$EVIDENCE/workers/${stem}.json"
			log="$EVIDENCE/workers/${stem}.log"
			echo
			echo "----- $stem -----"
			env \
				PYTHONPATH="$EVIDENCE/build/lib:$ROOT/bindings/torch" \
				PYTHONPYCACHEPREFIX="$EVIDENCE/pycache/$stem" \
				"$PYTHON_BIN" "$WORKER" \
					--contract "$CONTRACT" \
					--bridge "$BRIDGE" \
					--mode "$mode" \
					--batch "$batch" \
					--start-order "$start_order" \
					--process-index "$process_index" \
					--cpu "$BENCH_CPU" \
					--output "$output" \
				2>&1 | tee "$log"
			rc=${PIPESTATUS[0]}
			printf '{"mode":"%s","batch":%s,"process_index":%s,' "$mode" "$batch" "$process_index" >> "$INDEX_JSONL"
			printf '"start_order":"%s","returncode":%s,' "$start_order" "$rc" >> "$INDEX_JSONL"
			printf '"output":"%s","log":"%s"}\n' "$output" "$log" >> "$INDEX_JSONL"
		done
	done
done
set -e

"$PYTHON_BIN" - "$INDEX_JSONL" "$EVIDENCE/worker_index.json" <<'PY_INDEX'
import json, pathlib, sys
items = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
pathlib.Path(sys.argv[2]).write_text(json.dumps({"workers": items}, indent=2) + "\n")
PY_INDEX

echo
echo "===== FINALIZE ====="
set +e
"$PYTHON_BIN" "$FINALIZER" \
	--contract "$CONTRACT" \
	--context "$EVIDENCE/context.json" \
	--index "$EVIDENCE/worker_index.json" \
	--output-json "$EVIDENCE/phase4a3_q0b_apparatus.json" \
	--output-report "$EVIDENCE/PHASE4A3_Q0B_REPORT.md" \
	2>&1 | tee "$EVIDENCE/finalize.log"
FINAL_RC=${PIPESTATUS[0]}
set -e

(
	cd "$EVIDENCE"
	find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo
echo "Evidence: $EVIDENCE"
echo "Finalizer return code: $FINAL_RC"
exit "$FINAL_RC"
