#!/usr/bin/env python3
"""Build A3/A4 bindings with the test-only native Protocol-v4 window method."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts/phase3a4_v4_native_window.patch"
OUT = pathlib.Path("/tmp/tcnn_phase3a4_conditioned_v4")
COMMITS = {"phase3a3": "a26a0c1218d7ddeaad174c86a33255189ca5c2cc", "phase3a4": "6258184"}


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    bindings, hashes = {}, {}
    for variant, commit in COMMITS.items():
        worktree = OUT / f"worktree_{variant}"
        run(["git", "worktree", "add", "--detach", str(worktree), commit], cwd=ROOT)
        # Git worktrees contain gitlinks but not populated submodule contents.
        # Reuse the already verified local dependency checkouts without copying
        # or modifying them; only the temporary worktree receives symlinks.
        for dependency in ("cmrc", "cutlass", "fmt"):
            target = worktree / "dependencies" / dependency
            if target.exists():
                target.rmdir()
            target.symlink_to(ROOT / "dependencies" / dependency, target_is_directory=True)
        run(["git", "apply", str(PATCH)], cwd=worktree)
        setup = worktree / "bindings/torch/setup.py"
        build_temp = OUT / variant / "build_temp"
        build_lib = OUT / variant / "build_lib"
        package = OUT / variant / "binding"
        environment = os.environ.copy()
        environment.update({"MAX_JOBS": "8", "PYTORCH_ROCM_ARCH": "gfx1201", "TCNN_CUDA_ARCHITECTURES": "120"})
        run([sys.executable, str(setup), "build_ext", "--force", "--build-temp", str(build_temp),
            "--build-lib", str(build_lib)], cwd=setup.parent, env=environment)
        library = next(build_lib.glob("tinycudann_bindings/_120_C*.so"))
        (package / "tinycudann_bindings").mkdir(parents=True)
        target = package / "tinycudann_bindings" / library.name
        shutil.copy2(library, target)
        (package / "tinycudann").symlink_to(worktree / "bindings/torch/tinycudann", target_is_directory=True)
        bindings[variant] = str(package)
        hashes[variant] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"schema": 4, "protocol": "conditioning_v4", "test_only_native_patch": str(PATCH),
        "commits": COMMITS, "bindings": bindings, "binding_sha256": hashes}
    destination = ROOT / "phase3a4_reports/conditioned_bindings_v4.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
