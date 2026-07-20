# Phase 3A1 preflight

- Base tag dereference: `d80b7398d77912d47484e7f544f6ab73da874201`.
- Starting HEAD: `d80b7398d77912d47484e7f544f6ab73da874201`.
- Starting branch: `rdna4-phase2e-addendum`; Phase 3A1 branch: `rdna4-phase3a1-hipblaslt-mlp`.
- Starting Git status: clean.
- Phase 3A0 summary SHA-256: `8740f6734c878e8e26f77593dac6eb600629a650151ebde6f0c2fd7687b37f86`; result PASS, 192/192 supported and correct, all selected workspaces zero bytes, 20/20 fresh processes.
- ROCm/hipcc: 7.2.26015; hipBLASLt: 1.2.1.70200; PyTorch: 2.13.0+rocm7.2; HIP runtime: 7.2.53211.
- GPU: AMD Radeon AI PRO R9700 reported as AMD Radeon Graphics, `gfx1201`.
- Validated binding before changes: Phase-2E Addendum `_120_C` build for gfx1201.
- Dependency root: `/home/oem/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-phase1/dependencies`.
- Build contract: `TCNN_DEPENDENCY_ROOT=... PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=1 python setup.py build_ext --inplace`; link `amdhip64` and `hipblaslt`.
- Existing permanent markers are Phase 1/2/2D/2E markers; no Phase-3A1 DEBUG or ISOLATION marker exists.
- Relevant baseline files and SHA-256: `portable_mlp.h` `140de30e...edc83d`; `portable_mlp.cu` `9fd869e9...a242f4`; `portable_network.cu` `42e17b32...98c9d`; `setup.py` `d2646f41...a50bfc`.
