# Fresh-clone smoke

- Clone source: local canonical Phase 2 repository.
- Checkout: annotated tag `phase2e-addendum-portable-mlp-robustness-gfx1201-rocm72-pass`.
- Dependencies: populated immutable local Phase 1 dependency checkout (no network access).
- Native build: all 9 compilation units and the gfx1201 shared-library link completed successfully. The legacy setup's final `--inplace` copy reported a missing namespace-package destination; testing deliberately loaded the successfully linked binary from `build/lib.linux-x86_64-cpython-312`.
- Legacy C++ compile smoke: `CPP_API_LEGACY_CALL_SMOKE=PASS`.
- Numerical addendum smoke: `PHASE2E_ADDENDUM_NUMERICAL_PASS`, covering A1-A4 against the freshly linked clone binary.
- Clone Git status after smoke: clean; generated build outputs are ignored.

Full fresh-clone build and test logs are included in the external release handoff directory.
