# Phase 4A3-Q0b — Apparatus Redesign, Tool Repair 003, implementation revision 2

Marker: `TCNN_RDNA4_P4A3_Q0B_APPARATUS_REDESIGN_001`

Q0b is an apparatus qualification, not a performance comparison. It uses the
unchanged Phase-4A2 production kernel and a temporary test-only binding hook.
The binding source is restored byte-for-byte **before any worker executes**.

## Frozen before data

The complete analysis pipeline is in
`contracts/phase4a3_q0b_apparatus_contract.json`. In particular:

- no expected ratio direction;
- the known Q0 diagnostic value near `1.59×` is not a gate;
- no outlier deletion;
- no post-hoc block dropping;
- no adaptive extension;
- exactly four paired rounds per fresh process;
- convergence values are never scored;
- Student-t interval from process-level log ratios;
- no floor subtraction and no assumed common additive `S`.

The accepted Q0a forensic summary is bundled with SHA256:

`517a831ed778424133b6f0721ccf393e8e1091db15346d63e5cc38a25e639ae8`


## Implementation repair 2

The first Q0b execution built and linked successfully, but all 24 workers
failed before their first timing window. The test-only native hook received
`Module.params`, which is the public FP32 master tensor. The ordinary public
forward path converts that tensor to the backend parameter precision before
calling the native module; the direct hook bypassed that conversion and
correctly rejected FP32.

Revision 2:

- converts and retains one FP16 native parameter tensor per backend outside
  every timing window;
- leaves the public FP32 master tensor and product path unchanged;
- adds a fail-fast hook smoke before the 24-worker matrix;
- changes neither production code nor the frozen Q0b analysis pipeline.

## Tool repair 003

Revision 2 correctly converted the FP32 public parameter master to FP16, but
its direct native input adapter incorrectly padded to a hard-coded multiple of
16. The frozen public C++ API requires `BATCH_SIZE_GRANULARITY = 256`. The
public Python wrapper obtains that value from `_C.batch_size_granularity()`.

Revision 3 therefore:

- obtains the granularity dynamically from the loaded test binding;
- requires the frozen release value `256`;
- pads public batches `1`, `31`, and `128` to `256`;
- records the effective granularity and padded batch in every worker;
- smoke-tests both backends at all three contracted public batches before any
  performance worker starts.

The failed Revision-2 smoke produced no Q0b measurement matrix and does not
change the frozen analysis pipeline.

## What changed from Q0

1. No telemetry is executed between final conditioning and timing.
2. Each backend and paired round must converge before scoring.
3. Native synchronized single-shot diagnostics avoid per-sample Python calls.
4. Public Python single-shot remains a separate product metric.
5. Native queued loops provide real submission headroom.
6. Gap sweeps replace the arbitrary choice of one idle interval:

`0, 0.5, 1, 2, 4, 8, 16.666667, 50 ms`

The common wake-up increment is a falsifiable diagnostic, not an apparatus
gate.

## Run

Switch to a real TTY with `Ctrl+Alt+F3`, log in, then:

```bash
source ~/therock_test/venv/bin/activate
sudo systemctl stop display-manager

cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe
unzip -o ~/Downloads/phase4a3_q0b_apparatus_redesign_bundle.zip

chmod +x \
  scripts/patch_phase4a3_q0b_binding.py \
  scripts/phase4a3_q0b_worker.py \
  scripts/phase4a3_q0b_hook_smoke.py \
  scripts/finalize_phase4a3_q0b.py \
  scripts/run_phase4a3_q0b_apparatus.sh

unset HIP_LAUNCH_BLOCKING
unset AMD_SERIALIZE_KERNEL
unset AMD_SERIALIZE_COPY
unset PHASE4A3_Q0B_EVIDENCE

export MAX_JOBS=1
scripts/run_phase4a3_q0b_apparatus.sh
```

Restore the desktop afterward:

```bash
sudo systemctl start display-manager
```

## Decisions

- `PROCEED_TO_PHASE4A3_P0_PROTOCOL_FREEZE`
- `PHASE4A3_Q0B_BLOCKED`

Neither outcome is a performance PASS or FAIL.
