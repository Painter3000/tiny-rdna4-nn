#!/usr/bin/env python3
"""Rebuild persistent, isolated Protocol-v4 qualification bindings."""

import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PATCH = ROOT / "scripts/phase3a4_v4_native_window.patch"
ARTIFACT_ROOT = WORKSPACE / "artifacts/environment_qualification_bindings"
BUILD_ROOT = ARTIFACT_ROOT / "build"
MANIFEST = ROOT / "phase3a4_reports/environment_qualification_bindings.json"
VALIDATION = ARTIFACT_ROOT / "binding_validation.json"
COMMITS = {
    "phase3a3": "a26a0c1218d7ddeaad174c86a33255189ca5c2cc",
    "phase3a4": "6258184d8d9d032ef423b75eddeeaf8168c7e45a",
}
TAGS = {"phase3a3": ["phase3a3-capability-blocked-gfx1201-rocm72"], "phase3a4": []}
DEPENDENCIES = ("cmrc", "cutlass", "fmt")
COUNTERS = (
    "_hipblaslt_execution_handle_creations",
    "_hipblaslt_mlp_cache_misses",
    "_hipblaslt_fused_partial_bytes_live",
    "_hipblaslt_fused_partial_bytes_peak",
)


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_commit(commit, destination):
    destination.mkdir(parents=True)
    archive = subprocess.Popen(["git", "-C", str(ROOT), "archive", commit], stdout=subprocess.PIPE)
    try:
        extracted = subprocess.run(["tar", "-x", "-C", str(destination)], stdin=archive.stdout, check=False)
    finally:
        if archive.stdout:
            archive.stdout.close()
    archive_rc = archive.wait()
    if archive_rc or extracted.returncode:
        raise RuntimeError(f"git archive extraction failed for {commit}: git={archive_rc}, tar={extracted.returncode}")


def materialize_dependencies(source):
    for dependency in DEPENDENCIES:
        target = source / "dependencies" / dependency
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                target.rmdir()
            else:
                target.unlink()
        target.symlink_to(ROOT / "dependencies" / dependency, target_is_directory=True)


def build_variant(variant, commit):
    source = BUILD_ROOT / variant / "source"
    # setup.py places selected objects at ../../src relative to build-temp.
    # Two levels here keep those objects inside the respective variant root.
    build_temp = BUILD_ROOT / variant / "temp/build"
    build_lib = BUILD_ROOT / variant / "build_lib"
    package = ARTIFACT_ROOT / variant
    extract_commit(commit, source)
    materialize_dependencies(source)
    run(["git", "apply", "--check", str(PATCH)], cwd=source)
    run(["git", "apply", str(PATCH)], cwd=source)
    setup = source / "bindings/torch/setup.py"
    environment = os.environ.copy()
    environment.update({
        "MAX_JOBS": "8",
        "PYTORCH_ROCM_ARCH": "gfx1201",
        "TCNN_CUDA_ARCHITECTURES": "120",
        "ROCM_PATH": "/opt/rocm",
    })
    log_path = ARTIFACT_ROOT / f"build_{variant}.log"
    with log_path.open("w") as log:
        completed = subprocess.run([
            sys.executable,
            str(setup),
            "build_ext",
            "--force",
            "--build-temp",
            str(build_temp),
            "--build-lib",
            str(build_lib),
        ], cwd=setup.parent, env=environment, text=True, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{variant} build failed with rc={completed.returncode}; see {log_path}")
    libraries = list(build_lib.glob("tinycudann_bindings/_120_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"{variant} produced {len(libraries)} native libraries")
    (package / "tinycudann_bindings").mkdir(parents=True)
    shutil.copytree(source / "bindings/torch/tinycudann", package / "tinycudann")
    target = package / "tinycudann_bindings" / libraries[0].name
    shutil.copy2(libraries[0], target)
    provenance = {
        "variant": variant,
        "source_commit": commit,
        "source_tags": TAGS[variant],
        "patch_sha256": sha256(PATCH),
    }
    (package / "qualification_binding_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return package, target


def validate_variant(variant, package, library):
    smoke = r'''
import importlib
import json
import pathlib
import sys

import torch

variant = sys.argv[1]
binding_root = pathlib.Path(sys.argv[2]).resolve()
expected_library = pathlib.Path(sys.argv[3]).resolve()
sys.path.insert(0, str(binding_root))
import tinycudann as tcnn
native = importlib.import_module("tinycudann_bindings._120_C")
loaded = pathlib.Path(native.__file__).resolve()
assert loaded == expected_library, (loaded, expected_library)
assert binding_root in loaded.parents
provenance = json.loads((binding_root / "qualification_binding_provenance.json").read_text())
assert provenance["variant"] == variant
assert torch.version.hip is not None
assert torch.cuda.is_available()
properties = torch.cuda.get_device_properties(0)
assert getattr(properties, "gcnArchName", None) == "gfx1201"
model = tcnn.Network(8, 8, {"otype": "HipBLASLtMLP", "n_hidden_layers": 4,
    "n_neurons": 128, "activation": "ReLU", "output_activation": "None"}, seed=20260722)
assert hasattr(model.native_tcnn_module, "_benchmark_forward_window_128")
counters = {}
for name in ("_hipblaslt_execution_handle_creations", "_hipblaslt_mlp_cache_misses",
             "_hipblaslt_fused_partial_bytes_live", "_hipblaslt_fused_partial_bytes_peak"):
    function = getattr(native, name, None)
    # This is the exact Protocol-v4 snapshot contract: A3 predates the two
    # scratch exports, for which the unchanged harness supplies zero.
    counters[name] = {"value": int(function()) if function else 0,
        "availability": "native" if callable(function) else "v4_fallback_zero"}
assert counters["_hipblaslt_execution_handle_creations"]["availability"] == "native"
assert counters["_hipblaslt_mlp_cache_misses"]["availability"] == "native"
x = (torch.randn(16, 8, device="cuda") * 0.2).contiguous()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
assert tuple(y.shape) == (16, 8)
assert torch.isfinite(y).all().item()
print(json.dumps({"variant": variant, "binding_root": str(binding_root),
    "loaded_module": str(loaded), "native_window_present": True,
    "forward_smoke": "PASS", "counter_values": counters,
    "python": sys.version.split()[0], "pytorch": torch.__version__,
    "hip": torch.version.hip, "device": torch.cuda.get_device_name(0),
    "gcn_arch": getattr(properties, "gcnArchName", None)}, sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="tcnn_envqual_neutral_", dir="/tmp") as neutral:
        completed = subprocess.run(
            [sys.executable, "-c", smoke, variant, str(package), str(library)],
            cwd=neutral,
            text=True,
            capture_output=True,
        )
    log_path = ARTIFACT_ROOT / f"validate_{variant}.log"
    log_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"{variant} neutral-directory validation failed; see {log_path}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError(f"{variant} validation did not emit exactly one JSON result")
    result = json.loads(lines[0])
    result["library_sha256"] = sha256(library)
    result["library_inode"] = library.stat().st_ino
    return result


def main():
    if ARTIFACT_ROOT.exists():
        raise SystemExit(f"refusing to overwrite {ARTIFACT_ROOT}")
    if MANIFEST.exists():
        raise SystemExit(f"refusing to overwrite {MANIFEST}")
    ARTIFACT_ROOT.mkdir(parents=True)
    patch_digest = sha256(PATCH)
    built = {}
    results = {}
    for variant, commit in COMMITS.items():
        resolved = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"{commit}^{{commit}}"], text=True
        ).strip()
        if resolved != commit:
            raise RuntimeError(f"source identity mismatch for {variant}: {resolved}")
        package, library = build_variant(variant, commit)
        built[variant] = (package, library)
        results[variant] = validate_variant(variant, package, library)
    a3_package, a3_library = built["phase3a3"]
    a4_package, a4_library = built["phase3a4"]
    identity_checks = {
        "binding_paths_distinct": a3_package.resolve() != a4_package.resolve(),
        "library_inodes_distinct": a3_library.stat().st_ino != a4_library.stat().st_ino,
        "library_sha256_distinct": sha256(a3_library) != sha256(a4_library),
        "patch_sha256_matches_predeclared": patch_digest == "77172047e889b0d56bfabda3e475684d2e8bb2883552a699ea9d4bffe974acdd",
        "source_commits_resolved": all(results[v]["variant"] == v for v in COMMITS),
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"cross-binding identity check failed: {identity_checks}")
    validation_document = {"identity_checks": identity_checks, "variants": results}
    VALIDATION.write_text(json.dumps(validation_document, indent=2, sort_keys=True) + "\n")
    environment = {
        key: results["phase3a3"][key]
        for key in ("python", "pytorch", "hip", "device", "gcn_arch")
    }
    manifest = {
        "schema": 1,
        "purpose": "phase3a4_environment_qualification",
        "protocol": "conditioning_v4_native_window",
        "build_timestamp_utc_metadata_only": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bindings": {variant: str(built[variant][0].resolve()) for variant in COMMITS},
        "binding_libraries": {variant: str(built[variant][1].resolve()) for variant in COMMITS},
        "binding_sha256": {variant: sha256(built[variant][1]) for variant in COMMITS},
        "source_commits": COMMITS,
        "source_tags": TAGS,
        "test_only_native_patch": str(PATCH),
        "test_only_native_patch_sha256": patch_digest,
        "environment": environment,
        "validation": validation_document,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(MANIFEST)
    print(VALIDATION)


if __name__ == "__main__":
    main()
