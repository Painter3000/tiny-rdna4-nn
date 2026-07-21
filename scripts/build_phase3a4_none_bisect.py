#!/usr/bin/env python3
"""Build isolated Phase-3A4 None-regression bisection bindings (B through E)."""
import json, os, pathlib, shutil, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
SETUP=ROOT/"bindings/torch/setup.py"
OUT=pathlib.Path("/tmp/tcnn_phase3a4_none_bisect")
VARIANTS={
    "B":{"TCNN_P3A4_KERNELS":"1","TCNN_P3A4_DISPATCH":"0","TCNN_P3A4_COUNTERS":"0"},
    "C":{"TCNN_P3A4_KERNELS":"1","TCNN_P3A4_DISPATCH":"0","TCNN_P3A4_COUNTERS":"1"},
    "D":{"TCNN_P3A4_KERNELS":"1","TCNN_P3A4_DISPATCH":"1","TCNN_P3A4_COUNTERS":"0"},
    "E":{"TCNN_P3A4_KERNELS":"1","TCNN_P3A4_DISPATCH":"1","TCNN_P3A4_COUNTERS":"1"},
}

def main():
    OUT.mkdir(parents=True,exist_ok=True); manifest={"variants":{"A":"/tmp/tcnn_phase3a3_rebuilt_binding"},"definitions":{"A":"phase3a3 tag a26a0c1"}}
    for name,flags in VARIANTS.items():
        base=OUT/name; temp=base/"build_temp"; lib=base/"build_lib"; package=base/"binding"
        shutil.rmtree(temp,ignore_errors=True); shutil.rmtree(lib,ignore_errors=True); shutil.rmtree(package,ignore_errors=True)
        env=os.environ.copy(); env.update(flags); env.update({"MAX_JOBS":"8","PYTORCH_ROCM_ARCH":"gfx1201","TCNN_CUDA_ARCHITECTURES":"120"})
        subprocess.run([sys.executable,str(SETUP),"build_ext","--force","--build-temp",str(temp),"--build-lib",str(lib)],
            cwd=SETUP.parent,env=env,check=True)
        so=next(lib.glob("tinycudann_bindings/_120_C*.so")); (package/"tinycudann_bindings").mkdir(parents=True)
        shutil.copy2(so,package/"tinycudann_bindings"/so.name)
        (package/"tinycudann").symlink_to(ROOT/"bindings/torch/tinycudann",target_is_directory=True)
        manifest["variants"][name]=str(package); manifest["definitions"][name]=flags
    target=ROOT/"phase3a4_reports/none_bisect_bindings.json"; target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(target)
if __name__=="__main__": main()
