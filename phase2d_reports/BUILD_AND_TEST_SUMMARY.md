# Phase 2D build and test summary

- Clean ROCm build: PASS, `MAX_JOBS=1`, `PYTORCH_ROCM_ARCH=gfx1201`, ROCm 7.2
- Phase 2D required matrix: 8/8 PASS against full PyTorch FP32 references
- Additional explicit-stream Phase 2D cases: 2/2 PASS
- Phase 2D fresh processes: 8/8 PASS
- NetworkWithInputEncoding: Identity/Frequency/OneBlob/HashGrid PASS
- Encoding FP16/FP32 regression: 8/8 PASS
- Phase 2C in-process stream cases: 12/12 PASS
- Phase 2C fresh processes: 12/12 PASS
- Training: 200/200 finite steps; loss 0.07817384 to 0.0000232811 (ratio 0.0002978)
- Checkpoint: `state_dict()` roundtrip, exact output (`max_abs=0`)
- Temporary DEBUG/ISOLATION markers: 0

Machine-readable detailed results are in `phase2d_validation.json` and the
`phase2c_regression` JSON files.
