#!/usr/bin/env bash
# TCNN_RDNA4_P4A3_Q0C_SPLIT_APPARATUS_001
# TCNN_RDNA4_P4A3_Q0C_WORKER_MANIFEST_FIX_001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT="$ROOT/contracts/phase4a3_q0c_apparatus_contract.json"
PATCHER="$ROOT/scripts/patch_phase4a3_q0c_binding.py"
WORKER="$ROOT/scripts/phase4a3_q0c_worker.py"
FINALIZER="$ROOT/scripts/finalize_phase4a3_q0c.py"
MANIFEST_TOOL="$ROOT/scripts/phase4a3_q0c_worker_manifest.py"
PROVENANCE="$ROOT/scripts/check_phase4a3_q0c_provenance.py"
OBJECT_CAPTURE="$ROOT/scripts/capture_phase4a3_q0c_build_object.py"
SMOKE="$ROOT/scripts/phase4a3_q0c_hook_smoke.py"
BRIDGE_SOURCE="$ROOT/scripts/phase4a3_q0c_hip_runtime_bridge.cpp"
P4_AUDIT="$ROOT/scripts/audit_phase4a2_p4_code_object.py"
P4_CONTRACT="$ROOT/contracts/phase4a2_p4_production_code_object_audit_contract.json"
BINDING="$ROOT/bindings/torch/tinycudann/bindings.cpp"
SETUP="$ROOT/bindings/torch/setup.py"
SOURCE="$ROOT/src/rocwmma_width64_mlp.cu"
MAPPING="$ROOT/include/tiny-cuda-nn/networks/rocwmma_width64_mapping_gfx1201.h"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
MAX_JOBS="${MAX_JOBS:-1}"
BENCH_CPU="${PHASE4A3_BENCH_CPU:-$(( $(nproc) - 1 ))}"
EXPECTED_COMMIT="cd1330a21452f7e2edab9e676567b7a040f922bc"
TAG="phase4a2-width64-production-inference-gfx1201-pass"
DRY_RUN=false
SELECT="LN,LP,TP,TD,G"
EVIDENCE=""
while (($#)); do
	case "$1" in
		--dry-run) DRY_RUN=true ;;
		--subphases) shift; SELECT="$1" ;;
		--evidence) shift; EVIDENCE="$1" ;;
		*) echo "unknown argument: $1" >&2; exit 2 ;;
	esac
	shift
done
if [[ "$DRY_RUN" == true ]]; then
	PYTHONPATH="$ROOT/scripts" "$PYTHON_BIN" - "$CONTRACT" "$SELECT" <<'PY'
import json, pathlib, sys
from phase4a3_q0c_common import load_contract, matrix
c = load_contract(pathlib.Path(sys.argv[1]))
phases = tuple(value.strip() for value in sys.argv[2].split(",") if value.strip())
items = matrix(c, phases)
print(json.dumps({"marker": c["marker"], "dry_run": True, "worker_count": len(items), "workers": items}, indent=2))
PY
	exit 0
fi

EVIDENCE_ROOT="${PHASE4A3_EVIDENCE_ROOT:-$(cd "$ROOT/../.." && pwd)/evidence}"
EVIDENCE="${EVIDENCE:-$EVIDENCE_ROOT/phase4a3_q0c_$(date -u +%Y%m%dT%H%M%SZ)}"
case "$(basename "$EVIDENCE")" in phase4a3_q0c_*) ;; *) echo "unsafe evidence path" >&2; exit 2;; esac
[[ ! -e "$EVIDENCE" ]] || { echo "evidence path already exists" >&2; exit 2; }
mkdir -p "$EVIDENCE"/{build/temp,build/lib,workers,backup,provenance,pycache}
exec > >(tee "$EVIDENCE/run.log") 2>&1
BACKUP="$EVIDENCE/backup/bindings.cpp"
PATCH_ACTIVE=false
restore_binding() {
	if [[ "$PATCH_ACTIVE" == true && -f "$BACKUP" ]]; then
		"$PYTHON_BIN" "$PATCHER" --file "$BINDING" --backup "$BACKUP" --mode restore || true
	fi
}
trap restore_binding EXIT INT TERM HUP ERR

TTY_NAME="$(tty || true)"
[[ "$TTY_NAME" =~ ^/dev/tty[0-9]+$ ]] || { echo "Q0c requires a real /dev/ttyN" >&2; exit 10; }
for service in display-manager.service lightdm.service gdm.service gdm3.service sddm.service; do
	! systemctl is-active --quiet "$service" 2>/dev/null || { echo "display manager active" >&2; exit 11; }
done
for name in HIP_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY; do
	value="${!name:-}"; [[ -z "$value" || "$value" == 0 ]] || { echo "forbidden $name=$value" >&2; exit 12; }
done
"$PYTHON_BIN" - <<'PY'
import re, subprocess
value = subprocess.run(["rocm-smi", "--showpids", "--json"], text=True, capture_output=True, check=False)
if value.returncode:
    raise SystemExit("rocm-smi process inventory failed")
pids = {int(item) for item in re.findall(r"PID[^0-9]*([0-9]+)", value.stdout, re.I)}
if pids:
    raise SystemExit("foreign GPU processes present: " + repr(sorted(pids)))
PY
[[ "$(git -C "$ROOT" rev-list -n 1 "$TAG")" == "$EXPECTED_COMMIT" ]]
git -C "$ROOT" merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD
[[ "$(sha256sum "$SOURCE" | awk '{print $1}')" == "7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4" ]]
[[ "$(sha256sum "$MAPPING" | awk '{print $1}')" == "f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517" ]]
git -C "$ROOT" show "$EXPECTED_COMMIT:bindings/torch/tinycudann/bindings.cpp" > "$EVIDENCE/backup/bindings.cpp.release"
git -C "$ROOT" show "$EXPECTED_COMMIT:bindings/torch/setup.py" > "$EVIDENCE/backup/setup.py.release"
cmp -s "$BINDING" "$EVIDENCE/backup/bindings.cpp.release"
cmp -s "$SETUP" "$EVIDENCE/backup/setup.py.release"
[[ -z "$(git -C "$ROOT" diff --name-only "$EXPECTED_COMMIT" -- src include bindings/torch/setup.py bindings/torch/tinycudann)" ]]

"$HIPCC" -std=c++17 -O2 -fPIC -shared "$BRIDGE_SOURCE" -o "$EVIDENCE/build/libphase4a3_q0c_bridge.so"
"$PYTHON_BIN" "$PATCHER" --file "$BINDING" --backup "$BACKUP" --mode apply
PATCH_ACTIVE=true
(
	cd "$ROOT/bindings/torch"
	env PYTORCH_ROCM_ARCH=gfx1201 TCNN_DEPENDENCY_ROOT="$ROOT/dependencies" TCNN_HALF_PRECISION=1 TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1 MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" setup.py build_ext --build-temp "$EVIDENCE/build/temp" --build-lib "$EVIDENCE/build/lib"
) 2>&1 | tee "$EVIDENCE/build.log"
grep -E '(^| )(/[^ ]+)?(hipcc|clang\+\+|c\+\+|g\+\+).*rocwmma_width64_mlp' "$EVIDENCE/build.log" > "$EVIDENCE/build_command.txt"
grep -E '(^| )(/[^ ]+)?(hipcc|clang\+\+|c\+\+|g\+\+).*(-shared|-o .*_C.*\\.so)' "$EVIDENCE/build.log" > "$EVIDENCE/link_command.txt"
shopt -s nullglob; extensions=("$EVIDENCE"/build/lib/tinycudann_bindings/_120_C*.so); shopt -u nullglob
[[ "${#extensions[@]}" -eq 1 ]] || { echo "expected exactly one final extension" >&2; exit 14; }
EXTENSION="${extensions[0]}"
Q0C_BUILD_OBJECT="$(
	PYTHONPATH="$ROOT/scripts" "$PYTHON_BIN" "$OBJECT_CAPTURE" \
		--contract "$CONTRACT" \
		--evidence-dir "$EVIDENCE"
)"
echo "PHASE4A3_Q0C_EXACT_BUILD_OBJECT_CAPTURED: PASS"
"$PYTHON_BIN" "$PATCHER" --file "$BINDING" --backup "$BACKUP" --mode restore
PATCH_ACTIVE=false
cmp -s "$BINDING" "$EVIDENCE/backup/bindings.cpp.release"
[[ -z "$(git -C "$ROOT" diff --name-only "$EXPECTED_COMMIT" -- src include bindings/torch/setup.py bindings/torch/tinycudann)" ]]

# Q0c-P is deliberately before smoke and every measurement worker.
PYTHONPATH="$ROOT/scripts" "$PYTHON_BIN" "$PROVENANCE" --contract "$CONTRACT" --object "$Q0C_BUILD_OBJECT" --extension "$EXTENSION" --p4-audit "$P4_AUDIT" --p4-contract "$P4_CONTRACT" --build-command-file "$EVIDENCE/build_command.txt" --link-command-file "$EVIDENCE/link_command.txt" --output-dir "$EVIDENCE/provenance"
env PYTHONPATH="$EVIDENCE/build/lib:$ROOT/bindings/torch" "$PYTHON_BIN" "$SMOKE" --output "$EVIDENCE/hook_smoke.json"

PYTHONPATH="$ROOT/scripts" "$PYTHON_BIN" - "$CONTRACT" "$SELECT" > "$EVIDENCE/matrix.tsv" <<'PY'
import pathlib, sys
from phase4a3_q0c_common import load_contract, matrix
from phase4a3_q0c_worker import argument_parser

items = matrix(load_contract(pathlib.Path(sys.argv[1])), tuple(sys.argv[2].split(",")))
identities = []
for item in items:
    label = item.get("batch", item.get("metric"))
    stem = f"{item['phase']}_{item['schedule']}_{label}_p{item['process_index']}_{item['start_order']}"
    identities.append((stem, f"workers/{stem}.json"))
    argv = [
        "--contract", sys.argv[1], "--bridge", "/preflight/bridge.so",
        "--phase", item["phase"], "--schedule", item["schedule"],
        "--process-index", str(item["process_index"]),
        "--start-order", item["start_order"], "--cpu", "0",
        "--output", f"/preflight/{stem}.json",
    ]
    argv[6:6] = (["--batch", str(item["batch"])] if "batch" in item
                  else ["--metric", item["metric"]])
    argument_parser().parse_args(argv)
if len(set(identities)) != len(items):
    raise SystemExit("Q0c matrix contains duplicate worker IDs or output paths")
for item in items:
    # A visible sentinel prevents Bash IFS whitespace from collapsing optional
    # TSV columns and shifting process_index/start_order to the left.
    print("\t".join(str(item.get(key, "-")) for key in
                    ("phase","schedule","batch","metric","process_index","start_order")))
PY
"$PYTHON_BIN" "$MANIFEST_TOOL" init \
    --matrix "$EVIDENCE/matrix.tsv" \
    --output "$EVIDENCE/worker_manifest.json"

while IFS=$'\t' read -r phase schedule batch metric process_index start_order; do
	[[ "$batch" != "-" || "$metric" != "-" ]] || { echo "matrix row has neither batch nor metric" >&2; exit 16; }
	[[ "$batch" == "-" || "$metric" == "-" ]] || { echo "matrix row has both batch and metric" >&2; exit 16; }
	label="$batch"; [[ "$label" != "-" ]] || label="$metric"
	stem="${phase}_${schedule}_${label}_p${process_index}_${start_order}"
	extra=(); [[ "$batch" == "-" ]] || extra+=(--batch "$batch"); [[ "$metric" == "-" ]] || extra+=(--metric "$metric")
	# The built extension retains the hook; the source tree must not.
	cmp -s "$BINDING" "$EVIDENCE/backup/bindings.cpp.release" || { echo "$stem: test-only source hook reappeared" >&2; exit 15; }
	set +e
	env PYTHONPATH="$EVIDENCE/build/lib:$ROOT/bindings/torch:$ROOT/scripts" \
		PYTHONPYCACHEPREFIX="$EVIDENCE/pycache/$stem" \
		"$PYTHON_BIN" "$WORKER" \
		--contract "$CONTRACT" \
		--bridge "$EVIDENCE/build/libphase4a3_q0c_bridge.so" \
		--phase "$phase" --schedule "$schedule" "${extra[@]}" \
		--process-index "$process_index" --start-order "$start_order" \
		--cpu "$BENCH_CPU" --output "$EVIDENCE/workers/$stem.json" \
		> "$EVIDENCE/workers/$stem.log" 2>&1 &
	worker_pid=$!
	wait "$worker_pid"
	worker_rc=$?
	set -e

	"$PYTHON_BIN" "$MANIFEST_TOOL" record \
		--manifest "$EVIDENCE/worker_manifest.json" \
		--workers "$EVIDENCE/workers" \
		--worker-id "$stem" \
		--pid "$worker_pid" \
		--returncode "$worker_rc"
	echo "$stem pid=$worker_pid returncode=$worker_rc"
done < "$EVIDENCE/matrix.tsv"
set +e
PYTHONPATH="$ROOT/scripts" "$PYTHON_BIN" "$FINALIZER" \
	--contract "$CONTRACT" \
	--workers "$EVIDENCE/workers" \
	--worker-manifest "$EVIDENCE/worker_manifest.json" \
	--provenance "$EVIDENCE/provenance/phase4a3_q0c_provenance.json" \
	--output "$EVIDENCE/phase4a3_q0c_apparatus.json"
FINAL_RC=$?
set -e
echo "Finalizer return code: $FINAL_RC"
exec 1>&- 2>&-
PYTHONPATH="$ROOT/scripts" "$PYTHON_BIN" "$ROOT/scripts/bundle_phase4a3_q0c_diagnostics.py" --evidence "$EVIDENCE" --output "$EVIDENCE/phase4a3_q0c_diagnostics.tar.gz"
exit "$FINAL_RC"
