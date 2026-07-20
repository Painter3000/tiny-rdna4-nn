# C++ API source compatibility

## Pre-change failure

At the unchanged Phase 2E commit `c9d385ca54e9d27c904f7cdfcfd52ae6cdc0c4e7`, the legacy call through `tcnn::cpp::Module*` failed to compile because Phase 2E exposed only the new ten-argument virtual `backward()` method.

Exact compiler command:

```text
/opt/rocm/bin/hipcc -std=c++14 -fsyntax-only -I/home/oem/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-phase2/include -I/home/oem/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-phase2/dependencies /home/oem/therock_test/tcnn_rdna4_port/workspace/tcnn_phase2e_addendum_workspace/cpp_api_legacy_call_smoke.cpp
```

The compiler returned `1`. Its decisive diagnostic was:

```text
cpp_api_legacy_call_smoke.cpp:20:2: error: too few arguments to function call, expected 10, have 9
cpp_api.h:101:15: note: 'backward' declared here
CPP_API_LEGACY_CALL_SMOKE=FAIL
```

The captured pre-change compiler log is preserved in `cpp_api_legacy_call_smoke_prechange.log`; its command, return code, repository and dependency paths are preserved in `cpp_api_legacy_call_smoke_prechange_status.txt`.

## Compatibility repair

`TCNN_RDNA4_P2E_ADDENDUM_FIX_001` adds the old nine-argument signature as a non-virtual inline overload. It delegates to the unchanged mode-aware virtual ten-argument method with `GradientMode::Overwrite`. Thus calls made through `tcnn::cpp::Module*` retain their prior source-level behavior while the Phase 2E mode-aware path remains available.

This is evidence of C++ source compatibility for calls through `tcnn::cpp::Module*`. No ABI compatibility is claimed.

## Post-change verification

The same compile smoke and exact command were rerun after the repair. It returned `0` and printed `CPP_API_LEGACY_CALL_SMOKE=PASS`. The final compiler log is recorded in `cpp_api_legacy_call_smoke.log`.
