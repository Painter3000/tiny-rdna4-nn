#!/usr/bin/env bash
# Canonical, versioned launcher. The workspace entry point is an identical copy.
set -Eeuo pipefail

WORKSPACE="/home/oem/therock_test/tcnn_rdna4_port/workspace"
REPO="${WORKSPACE}/repos/tiny-cuda-nn-phase2"
REPORTS="${WORKSPACE}/phase3a4_reports"
LOG="${REPORTS}/environment_qualification_run.log"
MANIFEST="${REPO}/phase3a4_reports/conditioned_bindings_v4.json"
PYTHON="/home/oem/therock_test/venv/bin/python"
RUNNER="${REPO}/scripts/run_phase3a4_environment_qualification.py"
RAW="${REPORTS}/environment_qualification_raw"
JSON_OUT="${REPORTS}/environment_qualification.json"
MD_OUT="${REPORTS}/ENVIRONMENT_QUALIFICATION.md"
DM_SERVICE=""
DM_WAS_ACTIVE=0
DM_STOPPED_BY_US=0
CONFIRM=0
TEMP_AMD_JSON=""

mkdir -p "${REPORTS}"
touch "${LOG}"

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG}"
}

privileged_systemctl() {
    if [[ ${EUID} -eq 0 ]]; then
        systemctl "$@"
    else
        sudo systemctl "$@"
    fi
}

restore_environment() {
    local rc=$?
    trap - EXIT HUP INT TERM
    if [[ -n "${TEMP_AMD_JSON}" ]]; then
        rm -f "${TEMP_AMD_JSON}"
    fi
    if [[ ${DM_STOPPED_BY_US} -eq 1 && -n "${DM_SERVICE}" ]]; then
        log "RECOVERY: starting ${DM_SERVICE}"
        if privileged_systemctl start "${DM_SERVICE}" >>"${LOG}" 2>&1; then
            log "RECOVERY_PASS: ${DM_SERVICE} started"
        else
            log "RECOVERY_FAIL: run: sudo systemctl start ${DM_SERVICE}"
        fi
    fi
    exit "${rc}"
}
trap restore_environment EXIT HUP INT TERM

invalid() {
    log "INVALID_ENVIRONMENT: $*"
    exit 3
}

usage() {
    printf 'Usage: %s [--confirm]\n' "$0"
    printf 'Without --confirm: preflight-only dry-run; display manager is not stopped.\n'
}

case "${1:-}" in
    "") ;;
    --confirm) CONFIRM=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

TTY_NAME="$(tty 2>/dev/null || true)"
[[ "${TTY_NAME}" == /dev/tty* ]] || invalid "real Linux /dev/tty required; found '${TTY_NAME:-none}'"
[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]] || invalid "DISPLAY and WAYLAND_DISPLAY must be unset"
log "PRECHECK_PASS tty=${TTY_NAME}"

if [[ -L /etc/systemd/system/display-manager.service ]]; then
    DM_SERVICE="$(basename "$(readlink -f /etc/systemd/system/display-manager.service)")"
else
    DM_SERVICE="$(systemctl show display-manager.service -p Names --value 2>/dev/null \
        | tr ' ' '\n' | awk '$0 != "display-manager.service" && /\.service$/ {print; exit}')"
fi
[[ -n "${DM_SERVICE}" ]] || invalid "could not resolve display-manager.service"
if systemctl is-active --quiet "${DM_SERVICE}"; then
    DM_WAS_ACTIVE=1
fi
log "PRECHECK_PASS display_manager=${DM_SERVICE} active=${DM_WAS_ACTIVE}"

STATUS="$(git -C "${REPO}" status --porcelain=v1 --untracked-files=all)"
[[ -z "${STATUS}" ]] || invalid "Phase-3A4 worktree is not clean: ${STATUS//$'\n'/; }"
log "PRECHECK_PASS git_worktree_clean head=$(git -C "${REPO}" rev-parse HEAD)"

declare -A EXPECTED_HASHES=(
    [scripts/benchmark_phase3a4_conditioned_metric_v4.py]=dde22b68e643f77080977bcca28d547df2eee0cd3595e6ab8dfbffa658817250
    [scripts/phase3a4_v4_native_window.patch]=77172047e889b0d56bfabda3e475684d2e8bb2883552a699ea9d4bffe974acdd
    [phase3a4_reports/conditioned_bindings_v4.json]=a6e07f76a222ed1de2cbaac48e5d42a87e489b168b7e2368e9ab9acf8de2c838
)
for relative in "${!EXPECTED_HASHES[@]}"; do
    actual="$(sha256sum "${REPO}/${relative}" | awk '{print $1}')"
    [[ "${actual}" == "${EXPECTED_HASHES[${relative}]}" ]] \
        || invalid "v4 harness hash mismatch: ${relative}"
    log "PRECHECK_PASS sha256 ${relative}=${actual}"
done

"${PYTHON}" - "${MANIFEST}" <<'PY' >>"${LOG}" 2>&1 || invalid "A3/A4 binding identity check failed"
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
for variant in ("phase3a3", "phase3a4"):
    matches = list((pathlib.Path(manifest["bindings"][variant]) / "tinycudann_bindings").glob("_120_C*.so"))
    assert len(matches) == 1, (variant, matches)
    digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
    assert digest == manifest["binding_sha256"][variant], (variant, digest)
    print(f"PRECHECK_PASS binding {variant} {matches[0]} sha256={digest}")
PY
log "PRECHECK_PASS A3/A4 bindings"

TEMP_AMD_JSON="$(mktemp /tmp/tcnn_envqual_amd_smi.XXXXXX.json)"
/opt/rocm/bin/amd-smi process --general --gpu 0 --json >"${TEMP_AMD_JSON}" 2>>"${LOG}" \
    || invalid "AMD-SMI process query failed"
"${PYTHON}" - "${TEMP_AMD_JSON}" <<'PY' >>"${LOG}" 2>&1 || invalid "foreign GPU compute process detected"
import json, pathlib, sys
doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
allowed = {"Xorg", "Xwayland", "gnome-shell", "kwin_wayland", "kwin_x11", "plasmashell"}
pids = set()
def walk(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if k.upper() in {"PID", "PROCESS_ID"}:
                try: pids.add(int(v))
                except (TypeError, ValueError): pass
            walk(v, k)
    elif isinstance(value, list):
        for item in value: walk(item, key)
walk(doc)
foreign = []
for pid in sorted(pid for pid in pids if pid > 0):
    try: comm = pathlib.Path(f"/proc/{pid}/comm").read_text().strip()
    except (FileNotFoundError, PermissionError): comm = "unknown"
    if comm not in allowed: foreign.append((pid, comm))
print("PRECHECK_GPU_PROCESSES", sorted(pids))
assert not foreign, foreign
PY
rm -f "${TEMP_AMD_JSON}"
TEMP_AMD_JSON=""
log "PRECHECK_PASS no foreign GPU compute process"

if [[ ${CONFIRM} -eq 0 ]]; then
    log "DRY_RUN_PASS: no display service was stopped and no qualification pair was run"
    log "Run again from this TTY with --confirm to execute exactly 28 pairs"
    exit 0
fi

[[ ! -e "${RAW}" && ! -e "${JSON_OUT}" && ! -e "${MD_OUT}" ]] \
    || invalid "qualification output already exists; no replacement is permitted"

if [[ ${DM_WAS_ACTIVE} -eq 1 ]]; then
    log "stopping display manager ${DM_SERVICE}"
    privileged_systemctl stop "${DM_SERVICE}" >>"${LOG}" 2>&1
    DM_STOPPED_BY_US=1
fi
if systemctl is-active --quiet "${DM_SERVICE}"; then
    invalid "${DM_SERVICE} remains active after stop"
fi
log "POSTSTOP_PASS display_manager=${DM_SERVICE} inactive"

log "starting qualification: exactly 28 alternating fresh-process A3/A4 pairs"
set +e
"${PYTHON}" "${RUNNER}" \
    --manifest "${MANIFEST}" \
    --output-dir "${RAW}" \
    --output "${JSON_OUT}" \
    --report "${MD_OUT}" \
    --display-manager "${DM_SERVICE}" 2>&1 | tee -a "${LOG}"
RUN_RC=${PIPESTATUS[0]}
set -e
log "qualification runner exit=${RUN_RC}"
exit "${RUN_RC}"
