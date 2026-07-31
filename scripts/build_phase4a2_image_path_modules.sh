#!/usr/bin/env bash
# Phase 4A2 — Build der nativen Module fuer den amd-gsplat-Bildpfad.
#
# Quellen:
#   tiny-rdna4-nn : der umgebende Public-Checkout ($ROOT)
#   amd-gsplat    : Git-Submodule unter dependencies/amd-gsplat
#   cutlass/fmt/cmrc: initialisierte und gitlink-gepruefte tcnn-Submodule
#
# Keine Tarballs, kein Clone waehrend des Builds, keine Abhaengigkeit von
# historischen Freeze-Archiven. Eine sichtbare GPU wird fuer den Build nicht benoetigt.
#
# Exit-Codes:
#   0  PASS
#   1  FAIL
#   3  BLOCKED

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

# ---------------------------------------------------------------- Pfade ----

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PHASE4A2_PYTHON:-python}"
ROCM_REAL="${PHASE4A2_ROCM_REAL:-/opt/rocm}"
ROCM_CLANGXX="${PHASE4A2_ROCM_CLANGXX:-$ROCM_REAL/lib/llvm/bin/clang++}"
ROC_OBJ_LS="${PHASE4A2_ROC_OBJ_LS:-$ROCM_REAL/bin/roc-obj-ls}"

# Aus Sicherheitsgruenden darf ein ueberschreibbares Buildziel nur innerhalb
# dieses dedizierten, ignorierten Baums liegen.
BUILD_ROOT_REQUESTED="${PHASE4A2_BUILD_ROOT:-$ROOT/.phase4a2_image_build}"

TCNN_SOURCE="$ROOT"
TCNN_DEPENDENCY_ROOT="$ROOT/dependencies"
GSPLAT_SOURCE="$ROOT/dependencies/amd-gsplat"
GLM_RELPATH="gsplat/cuda/csrc/third_party/glm"

REPORT_READY=0
REPORT_JSON=""
ENV_JSON=""
ENVIRONMENT_CLASS="UNKNOWN"

write_report() {
    local status="$1"
    local exit_code="$2"
    local reason="$3"

    (( REPORT_READY == 1 )) || return 0
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || return 0

    PHASE4A2_REPORT_JSON="$REPORT_JSON" \
    PHASE4A2_REPORT_STATUS="$status" \
    PHASE4A2_REPORT_EXIT_CODE="$exit_code" \
    PHASE4A2_REPORT_REASON="$reason" \
    PHASE4A2_REPORT_ENV_CLASS="$ENVIRONMENT_CLASS" \
    PHASE4A2_REPORT_BUILD_ROOT="$BUILD_ROOT" \
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || true
import json
import os
from pathlib import Path

path = Path(os.environ["PHASE4A2_REPORT_JSON"])
payload = {
    "schema_version": 1,
    "status": os.environ["PHASE4A2_REPORT_STATUS"],
    "exit_code": int(os.environ["PHASE4A2_REPORT_EXIT_CODE"]),
    "reason": os.environ["PHASE4A2_REPORT_REASON"] or None,
    "environment_class": os.environ["PHASE4A2_REPORT_ENV_CLASS"],
    "target_arch": "gfx1201",
    "gpu_visibility_required": False,
    "build_root": os.environ["PHASE4A2_REPORT_BUILD_ROOT"],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
PY
}

fail() {
    local reason="$*"
    write_report "FAIL" 1 "$reason"
    echo "PHASE4A2_IMAGE_PATH_BUILD: FAIL: $reason" >&2
    exit 1
}

blocked() {
    local reason="$*"
    write_report "BLOCKED" 3 "$reason"
    echo "PHASE4A2_IMAGE_PATH_BUILD: BLOCKED: $reason" >&2
    exit 3
}

on_err() {
    local rc="$1"
    local line="$2"
    trap - ERR
    fail "UNHANDLED_COMMAND_FAILED: line=$line rc=$rc"
}
trap 'on_err "$?" "$LINENO"' ERR

require_command() {
    command -v "$1" >/dev/null 2>&1 || blocked "COMMAND_NOT_FOUND: $1"
}

capture_or_blocked() {
    local outvar="$1"
    local reason="$2"
    shift 2
    local output
    if ! output="$("$@")"; then
        blocked "$reason"
    fi
    printf -v "$outvar" '%s' "$output"
}

# ------------------------------------------------- strukturelle Vorpruefung --

for cmd in git realpath tar sha256sum grep awk cmp cp install mkdir rm; do
    require_command "$cmd"
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
    blocked "PYTHON_NOT_FOUND: $PYTHON_BIN"

git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    blocked "NOT_A_GIT_CHECKOUT: $ROOT"

capture_or_blocked ROOT_TOP "GIT_TOPLEVEL_UNAVAILABLE" \
    git -C "$ROOT" rev-parse --show-toplevel
ROOT_TOP="$(realpath -e "$ROOT_TOP")"
ROOT="$(realpath -e "$ROOT")"
[[ "$ROOT_TOP" == "$ROOT" ]] ||
    blocked "SCRIPT_NOT_AT_GIT_TOPLEVEL: root=$ROOT toplevel=$ROOT_TOP"

git -C "$ROOT" rev-parse --verify HEAD >/dev/null 2>&1 ||
    blocked "GIT_HEAD_MISSING"

BUILD_ROOT="$(realpath -m "$BUILD_ROOT_REQUESTED")"
BUILD_PREFIX="$ROOT/.phase4a2_image_build"

case "$BUILD_ROOT" in
    "$BUILD_PREFIX"|"$BUILD_PREFIX"/*) ;;
    *) fail "UNSAFE_BUILD_ROOT: $BUILD_ROOT" ;;
esac

BUILD_REL="${BUILD_ROOT#"$ROOT"/}"
[[ "$BUILD_REL" != "$BUILD_ROOT" ]] ||
    fail "BUILD_ROOT_OUTSIDE_CHECKOUT: $BUILD_ROOT"

# Das abschliessende "/" ist wichtig: der Ignore-Eintrag ist ein
# Verzeichnismuster und BUILD_ROOT existiert beim ersten Lauf noch nicht.
git -C "$ROOT" check-ignore -q --no-index -- "$BUILD_REL/" ||
    blocked "BUILD_ROOT_NOT_GITIGNORED: $BUILD_REL/"

rm -rf -- "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

REPORT_JSON="$BUILD_ROOT/build_report.json"
ENV_JSON="$BUILD_ROOT/build_environment.json"
REPORT_READY=1
write_report "BLOCKED" 3 "BUILD_NOT_STARTED"

[[ -x "$ROCM_CLANGXX" ]] ||
    blocked "ROCM_CLANGXX_UNAVAILABLE: $ROCM_CLANGXX"
[[ -x "$ROC_OBJ_LS" ]] ||
    blocked "ROC_OBJ_LS_UNAVAILABLE: $ROC_OBJ_LS"
[[ -x "$ROOT/scripts/phase4a_hipcc_compat.sh" ]] ||
    blocked "HIPCC_COMPAT_SHIM_MISSING_OR_NOT_EXECUTABLE"

[[ -f "$TCNN_SOURCE/bindings/torch/setup.py" ]] ||
    blocked "TCNN_SOURCE_TREE_INCOMPLETE: bindings/torch/setup.py"
[[ -f "$TCNN_SOURCE/src/rocwmma_width64_mlp.cu" ]] ||
    blocked "TCNN_SOURCE_TREE_INCOMPLETE: src/rocwmma_width64_mlp.cu"

# --------------------------------------------- Git-/Submodule-Helfer ---------

gitlink_fields() {
    local repo="$1"
    local rel="$2"
    local line

    capture_or_blocked line "GITLINK_LOOKUP_FAILED: $rel" \
        git -C "$repo" ls-tree HEAD -- "$rel"
    [[ -n "$line" ]] || blocked "GITLINK_ENTRY_MISSING: $rel"

    # Ausgabe: mode type object path
    printf '%s\n' "$line" | awk 'NR == 1 {print $1, $2, $3}'
}

require_clean_git_tree() {
    local repo="$1"
    local label="$2"
    local status

    capture_or_blocked status "GIT_STATUS_FAILED: $label" \
        git -C "$repo" status --porcelain=v1 --untracked-files=all
    if [[ -n "$status" ]]; then
        printf '%s\n' "$status" >&2
        blocked "DIRTY_SOURCE_PRECONDITION: $label"
    fi
}

check_submodule() {
    local rel="$1"
    local path="$ROOT/$rel"
    local fields mode type expected actual

    fields="$(gitlink_fields "$ROOT" "$rel")"
    IFS=" " read -r mode type expected <<<"$fields"
    [[ "$mode" == "160000" && "$type" == "commit" ]] ||
        blocked "NOT_A_GITLINK: $rel mode=$mode type=$type"

    [[ -d "$path" && -e "$path/.git" ]] ||
        blocked "SUBMODULE_NOT_INITIALIZED: $rel — git submodule update --init --recursive"

    capture_or_blocked actual "SUBMODULE_HEAD_UNAVAILABLE: $rel" \
        git -C "$path" rev-parse HEAD
    [[ "$actual" == "$expected" ]] || {
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        blocked "SUBMODULE_GITLINK_MISMATCH: $rel"
    }

    require_clean_git_tree "$path" "$rel"
}

# Der Buildpfad braucht alle drei Abhaengigkeiten.
for dep in cutlass fmt cmrc; do
    check_submodule "dependencies/$dep"
done

check_submodule "dependencies/amd-gsplat"

capture_or_blocked GSPLAT_TREE_ENTRY "GSPLAT_GITLINK_LOOKUP_FAILED" \
    git -C "$ROOT" ls-tree HEAD -- dependencies/amd-gsplat
EXPECTED_GSPLAT_COMMIT="$(
    printf '%s\n' "$GSPLAT_TREE_ENTRY" | awk '$1 == "160000" {print $3}'
)"
[[ -n "$EXPECTED_GSPLAT_COMMIT" ]] ||
    blocked "GSPLAT_NOT_RECORDED_AS_SUBMODULE"

capture_or_blocked ACTUAL_GSPLAT_COMMIT "GSPLAT_HEAD_UNAVAILABLE" \
    git -C "$GSPLAT_SOURCE" rev-parse HEAD

[[ -f "$GSPLAT_SOURCE/setup.py" ]] ||
    blocked "GSPLAT_SETUP_PY_MISSING"

# glm darf vendort sein oder ein echtes Submodule. Im Submodule-Fall werden
# Gitlink, HEAD und Sauberkeit explizit geprueft.
capture_or_blocked GLM_ENTRY "GLM_TREE_LOOKUP_FAILED" \
    git -C "$GSPLAT_SOURCE" ls-tree HEAD -- "$GLM_RELPATH"
[[ -n "$GLM_ENTRY" ]] || blocked "GLM_TREE_ENTRY_MISSING"

IFS=$' \t' read -r GLM_MODE GLM_TYPE EXPECTED_GLM_COMMIT _ <<<"$GLM_ENTRY"
GLM_SOURCE="$GSPLAT_SOURCE/$GLM_RELPATH"

if [[ "$GLM_MODE" == "160000" ]]; then
    [[ "$GLM_TYPE" == "commit" ]] ||
        blocked "GLM_GITLINK_TYPE_INVALID: $GLM_TYPE"
    [[ -d "$GLM_SOURCE" && -e "$GLM_SOURCE/.git" ]] ||
        blocked "GLM_SUBMODULE_NOT_INITIALIZED — git submodule update --init --recursive"

    capture_or_blocked ACTUAL_GLM_COMMIT "GLM_HEAD_UNAVAILABLE" \
        git -C "$GLM_SOURCE" rev-parse HEAD
    [[ "$ACTUAL_GLM_COMMIT" == "$EXPECTED_GLM_COMMIT" ]] || {
        echo "  expected: $EXPECTED_GLM_COMMIT" >&2
        echo "  actual:   $ACTUAL_GLM_COMMIT" >&2
        blocked "GLM_GITLINK_MISMATCH"
    }
    require_clean_git_tree "$GLM_SOURCE" "$GLM_RELPATH"
else
    ACTUAL_GLM_COMMIT=""
fi

[[ -f "$GLM_SOURCE/glm/glm.hpp" ]] ||
    blocked "GLM_HEADER_MISSING: $GLM_RELPATH/glm/glm.hpp"

# Der Build muss aus exakt sauberen Quellen starten. Dadurch koennen
# Dateisystem-Staging und Gitlink-Provenienz nicht auseinanderlaufen.
require_clean_git_tree "$ROOT" "superproject"

# ------------------------------------------------------ Buildbaum anlegen ----

GSPLAT_BUILD_SOURCE="$BUILD_ROOT/amd-gsplat-source"

mkdir -p \
    "$BUILD_ROOT/rocm/bin" \
    "$BUILD_ROOT/rocm/include" \
    "$BUILD_ROOT/tcnn/temp" \
    "$BUILD_ROOT/tcnn/lib" \
    "$BUILD_ROOT/gsplat/temp" \
    "$BUILD_ROOT/gsplat/lib" \
    "$BUILD_ROOT/runtime/tiny-rdna4-nn" \
    "$BUILD_ROOT/runtime/amd-gsplat" \
    "$BUILD_ROOT/include" \
    "$BUILD_ROOT/tooling" \
    "$GSPLAT_BUILD_SOURCE"

# ------------------------------------------- Umgebung klassifizieren ---------

ROCM_CLANG_VERSION="$("$ROCM_CLANGXX" --version 2>/dev/null || printf 'unknown')"

PHASE4A2_COMPILER_PATH="$ROCM_CLANGXX" \
PHASE4A2_COMPILER_VERSION="$ROCM_CLANG_VERSION" \
PHASE4A2_ROCM_PATH="$ROCM_REAL" \
PHASE4A2_ENV_JSON="$ENV_JSON" \
"$PYTHON_BIN" - <<'PY' || blocked "ENVIRONMENT_PROBE_FAILED"
import json
import os
import platform
import sys

out = os.environ["PHASE4A2_ENV_JSON"]

def write(payload):
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

base = {
    "schema_version": 1,
    "status": "BLOCKED",
    "python_version": platform.python_version(),
    "rocm_path": os.environ["PHASE4A2_ROCM_PATH"],
    "compiler_driver": "direct-clang-via-hipcc-compat-shim",
    "compiler_path": os.environ["PHASE4A2_COMPILER_PATH"],
    "compiler_version": os.environ["PHASE4A2_COMPILER_VERSION"],
    "target_arch": "gfx1201",
    "gpu_visibility_required": False,
}

try:
    import torch
except Exception as exc:
    base.update(environment_class="UNSUPPORTED",
                reason="TORCH_IMPORT_FAILED", detail=str(exc))
    write(base)
    sys.exit(1)

base["torch_version"] = torch.__version__
base["torch_hip_version"] = torch.version.hip

if torch.version.hip is None:
    base.update(environment_class="UNSUPPORTED",
                reason="PYTORCH_NOT_ROCM_BUILD")
    write(base)
    sys.exit(1)

if sys.version_info[:2] != (3, 12):
    base.update(environment_class="UNSUPPORTED",
                reason="PYTHON_VERSION_UNSUPPORTED")
    write(base)
    sys.exit(1)

qualified = (
    torch.__version__ == "2.13.0+rocm7.2"
    and torch.version.hip == "7.2.53211"
)

base.update(
    status="PASS",
    environment_class=(
        "QUALIFIED_REFERENCE"
        if qualified
        else "UNQUALIFIED_COMPATIBILITY_BUILD"
    ),
    reason=None,
)
write(base)
print("ENVIRONMENT_CLASS:", base["environment_class"])
PY

ENVIRONMENT_CLASS="$(
    PHASE4A2_ENV_JSON="$ENV_JSON" "$PYTHON_BIN" - <<'PY'
import json
import os
with open(os.environ["PHASE4A2_ENV_JSON"], encoding="utf-8") as fh:
    print(json.load(fh)["environment_class"])
PY
)"
write_report "BLOCKED" 3 "BUILD_IN_PROGRESS"

# --------------------------------------- Sauberkeit des Quellbaums (vorher) --

snapshot_worktrees() {
    git -C "$ROOT" status --porcelain=v1 --untracked-files=all
    git -C "$ROOT" submodule foreach --quiet --recursive '
        dirty="$(git status --porcelain=v1 --untracked-files=all)"
        if [ -n "$dirty" ]; then
            printf "SUBMODULE:%s\n%s\n" "$displaypath" "$dirty"
        fi
    '
}

snapshot_worktrees > "$BUILD_ROOT/git_status_before.txt"
[[ ! -s "$BUILD_ROOT/git_status_before.txt" ]] ||
    blocked "SOURCE_TREE_DIRTY_BEFORE_BUILD"

# ------------------------------------------------------------- ROCm-Shim ----

cp -a "$ROCM_REAL/include/." "$BUILD_ROOT/rocm/include/"
ln -s "$ROCM_REAL/lib" "$BUILD_ROOT/rocm/lib"
install -m 0755 \
    "$ROOT/scripts/phase4a_hipcc_compat.sh" \
    "$BUILD_ROOT/tooling/phase4a_hipcc_compat.sh"
ln -s "$BUILD_ROOT/tooling/phase4a_hipcc_compat.sh" \
    "$BUILD_ROOT/rocm/bin/hipcc"

export PHASE4A_ROCM_REAL="$ROCM_REAL"
export PHASE4A_ROCM_CLANGXX="$ROCM_CLANGXX"
export ROCM_HOME="$BUILD_ROOT/rocm"
export ROCM_PATH="$BUILD_ROOT/rocm"
export PYTORCH_ROCM_ARCH=gfx1201
export MAX_JOBS="${MAX_JOBS:-8}"
export TCNN_DEPENDENCY_ROOT
export TCNN_HALF_PRECISION=1
export TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1
export TORCH_EXTENSIONS_DIR="$BUILD_ROOT/torch_extensions"

# ------------------------------------------------------ tiny-rdna4-nn bauen --

(
    cd "$TCNN_SOURCE/bindings/torch"
    "$PYTHON_BIN" setup.py build_ext \
        --build-temp "$BUILD_ROOT/tcnn/temp" \
        --build-lib "$BUILD_ROOT/tcnn/lib"
) || fail "TCNN_EXTENSION_BUILD_FAILED"

# --------------------------------------- amd-gsplat stagen, patchen, bauen ---

if ! tar --exclude-vcs -C "$GSPLAT_SOURCE" -cf - . |
    tar -C "$GSPLAT_BUILD_SOURCE" -xf -
then
    fail "GSPLAT_STAGING_FAILED"
fi

cp -a "$GSPLAT_BUILD_SOURCE/$GLM_RELPATH/glm" "$BUILD_ROOT/include/"
export CPLUS_INCLUDE_PATH="$BUILD_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PHASE4A_AMD_GSPLAT_SOURCE_REV="${ACTUAL_GSPLAT_COMMIT:0:7}"
export PHASE4A2_AMD_GSPLAT_ARCH=gfx1201

replace_exact_once() {
    local file="$1"
    local old="$2"
    local new="$3"
    local reason="$4"

    "$PYTHON_BIN" - "$file" "$old" "$new" <<'PY' || fail "$reason"
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]

text = path.read_text(encoding="utf-8")
old_count = text.count(old)
new_count = text.count(new)

if old_count != 1 or new_count != 0:
    raise SystemExit(
        f"precondition failed: old_count={old_count}, new_count={new_count}"
    )

updated = text.replace(old, new, 1)
if updated.count(old) != 0 or updated.count(new) != 1:
    raise SystemExit("postcondition failed")

path.write_text(updated, encoding="utf-8")
PY
}

replace_exact_once \
    "$GSPLAT_BUILD_SOURCE/setup.py" \
    'git_rev = get_git_rev(os.getcwd())' \
    'git_rev = os.environ["PHASE4A_AMD_GSPLAT_SOURCE_REV"]' \
    "GSPLAT_SETUP_PATCH_GIT_REV_NOT_APPLIED_EXACTLY_ONCE"

replace_exact_once \
    "$GSPLAT_BUILD_SOURCE/setup.py" \
    'gpu_arch = get_rocm_arch()' \
    'gpu_arch = os.environ["PHASE4A2_AMD_GSPLAT_ARCH"]' \
    "GSPLAT_SETUP_PATCH_ARCH_NOT_APPLIED_EXACTLY_ONCE"

(
    cd "$GSPLAT_BUILD_SOURCE"
    "$PYTHON_BIN" setup.py build_ext \
        --build-temp "$BUILD_ROOT/gsplat/temp" \
        --build-lib "$BUILD_ROOT/gsplat/lib"
) || fail "GSPLAT_EXTENSION_BUILD_FAILED"

# ------------------------------------------------------- Runtime montieren ---

cp -a "$BUILD_ROOT/tcnn/lib/." "$BUILD_ROOT/runtime/tiny-rdna4-nn/"
cp -a "$TCNN_SOURCE/bindings/torch/tinycudann" \
    "$BUILD_ROOT/runtime/tiny-rdna4-nn/"
cp -a "$BUILD_ROOT/gsplat/lib/." "$BUILD_ROOT/runtime/amd-gsplat/"
cp -a "$GSPLAT_BUILD_SOURCE/gsplat/." \
    "$BUILD_ROOT/runtime/amd-gsplat/gsplat/"

TCNN_BINARY="$BUILD_ROOT/runtime/tiny-rdna4-nn/tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so"
GSPLAT_BINARY="$BUILD_ROOT/runtime/amd-gsplat/gsplat/csrc.so"

[[ -f "$TCNN_BINARY" ]] || fail "TCNN_NATIVE_MODULE_MISSING"
[[ -f "$GSPLAT_BINARY" ]] || fail "GSPLAT_NATIVE_MODULE_MISSING"

grep -a -q gfx1201 "$TCNN_BINARY" ||
    fail "TCNN_LACKS_GFX1201_CODE_OBJECT"
grep -a -q gfx1201 "$GSPLAT_BINARY" ||
    fail "GSPLAT_LACKS_GFX1201_CODE_OBJECT"

"$ROC_OBJ_LS" "$TCNN_BINARY" \
    > "$BUILD_ROOT/tcnn_roc_obj_ls.txt" ||
    fail "TCNN_ROC_OBJ_LS_FAILED"
grep -q -- '--gfx1201' "$BUILD_ROOT/tcnn_roc_obj_ls.txt" ||
    fail "TCNN_CODE_OBJECT_AUDIT_REJECTED"

"$ROC_OBJ_LS" "$GSPLAT_BINARY" \
    > "$BUILD_ROOT/gsplat_roc_obj_ls.txt" ||
    fail "GSPLAT_ROC_OBJ_LS_FAILED"
grep -q -- '--gfx1201' "$BUILD_ROOT/gsplat_roc_obj_ls.txt" ||
    fail "GSPLAT_CODE_OBJECT_AUDIT_REJECTED"

# ---------------------------------------- Sauberkeit des Quellbaums (nachher) -

snapshot_worktrees > "$BUILD_ROOT/git_status_after.txt"

if ! cmp -s \
    "$BUILD_ROOT/git_status_before.txt" \
    "$BUILD_ROOT/git_status_after.txt"
then
    echo "--- Veraenderungen am Quellbaum ---" >&2
    diff -u \
        "$BUILD_ROOT/git_status_before.txt" \
        "$BUILD_ROOT/git_status_after.txt" >&2 || true
    fail "SOURCE_TREE_DIRTY_AFTER_BUILD"
fi

# ------------------------------------------------------------- Ergebnis ------

capture_or_blocked SUPERPROJECT_COMMIT "SUPERPROJECT_HEAD_UNAVAILABLE" \
    git -C "$ROOT" rev-parse HEAD

PHASE4A2_ENV_JSON="$ENV_JSON" \
PHASE4A2_OUT_JSON="$BUILD_ROOT/binary_hashes.json" \
PHASE4A2_TCNN_SHA="$(sha256sum "$TCNN_BINARY" | awk '{print $1}')" \
PHASE4A2_GSPLAT_SHA="$(sha256sum "$GSPLAT_BINARY" | awk '{print $1}')" \
PHASE4A2_SUPERPROJECT_COMMIT="$SUPERPROJECT_COMMIT" \
PHASE4A2_GSPLAT_COMMIT="$ACTUAL_GSPLAT_COMMIT" \
PHASE4A2_GLM_COMMIT="$ACTUAL_GLM_COMMIT" \
"$PYTHON_BIN" - <<'PY' || fail "RESULT_MANIFEST_WRITE_FAILED"
import json
import os

with open(os.environ["PHASE4A2_ENV_JSON"], encoding="utf-8") as fh:
    env = json.load(fh)

payload = {
    "schema_version": 1,
    "status": "PASS",
    "exit_code": 0,
    "amd_gsplat": os.environ["PHASE4A2_GSPLAT_SHA"],
    "tiny_rdna4_nn": os.environ["PHASE4A2_TCNN_SHA"],
    "tiny_rdna4_nn_commit": os.environ["PHASE4A2_SUPERPROJECT_COMMIT"],
    "amd_gsplat_commit": os.environ["PHASE4A2_GSPLAT_COMMIT"],
    "glm_commit": os.environ["PHASE4A2_GLM_COMMIT"] or None,
    "environment_class": env["environment_class"],
    "python_version": env["python_version"],
    "torch_version": env["torch_version"],
    "torch_hip_version": env["torch_hip_version"],
    "compiler_driver": env["compiler_driver"],
    "compiler_path": env["compiler_path"],
    "compiler_version": env["compiler_version"],
    "target_arch": env["target_arch"],
    "gpu_visibility_required": False,
}

with open(os.environ["PHASE4A2_OUT_JSON"], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

write_report "PASS" 0 ""
echo "PHASE4A2_IMAGE_PATH_BUILD: PASS"
