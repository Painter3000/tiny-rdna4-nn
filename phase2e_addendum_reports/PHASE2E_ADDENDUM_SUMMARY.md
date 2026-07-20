# Phase 2E addendum summary

- A1: both PortableMLP cases pass against the independent PyTorch FP32 reference at `atol=8e-5`, `rtol=5e-4`, including output, input gradient, flat parameter gradient, and per-layer weights and biases.
- A2: HashGrid Overwrite, first Accumulate (`G`), and second Accumulate (`2G`) pass for separate network and encoding parameter regions. Public autograd is only an integration reference for HashGrid, not an independent mathematical reimplementation.
- A3: the three-stream HIP event chain passes at `G`, `2G`, and `3G` without intermediate global synchronization.
- A4: the 200-step run and 100+fresh-process+100 resume are bitwise exact for parameters, probe output, final loss, complete `optimizer.state`, and `optimizer.param_groups`.
- A5: the pre-change nine-argument C++ call failed; `TCNN_RDNA4_P2E_ADDENDUM_FIX_001` restores it as a non-virtual inline overload delegating to the unchanged virtual mode-aware overload with `GradientMode::Overwrite`. This proves source compatibility for calls through `tcnn::cpp::Module*`; no ABI compatibility is claimed.
- Clean native ROCm build: 9/9 compilation units and link pass for `gfx1201`; `pip check` and neutral-directory import pass.
- Full Phase 2E regression: PASS, including 20/20 fresh processes.
- Full Phase 2D regression: PASS, 10 network cases and 8/8 fresh processes.
- Full Phase 2C regression: PASS, 12 stream cases, 12/12 fresh processes, 4 Identity compositions, 8 encoding regressions, and 3 encoded networks.
- Fresh-clone smoke: PASS at the annotated Addendum tag; native link, legacy C++ compile smoke, all A1-A4 numerical addendum checks, and clean clone status confirmed.

Machine-readable evidence and full logs are adjacent to this report.
