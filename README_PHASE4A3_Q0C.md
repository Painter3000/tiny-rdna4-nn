# Phase 4A3-Q0c — split, fail-closed apparatus qualification

This bundle qualifies measurement apparatus only. It never emits a
performance PASS/FAIL and does not use Q0/Q0b ratio direction as a threshold,
expectation, or control-flow input.

Q0c is split into independently retained results:

- `Q0c-P`: exact measurement-object and loaded-extension provenance
- `Q0c-LN`: native synchronized latency diagnostic, actual batch 256
- `Q0c-LP`: synchronized public product latency, batches 1/31/128
- `Q0c-TP`: queued public product throughput, including batch 257
- `Q0c-TD`: queued native device throughput with per-batch headroom
- `Q0c-G`: independent Latin-square gap diagnostic; never a gate

The contract freezes the P4 reference kernel-ISA SHA256 before Q0c execution:
`2c03a92c6c4e3c7986411c4dd40fb8b9d8745f05681c046562190ffc029886ec`.
It comes from the audited P4 reference artifact
`workspace/evidence/phase4a2_p4_20260726T030952Z/code_object/production_kernel.isa.txt`.
The final extension SHA is recorded but is not accepted as kernel identity by
itself.

Immediately after the build, the runner searches the complete fresh evidence
directory for the contract-frozen object basename and requires exactly one
match. It records that object's absolute path and SHA256 under `provenance/`,
then passes the exact path to Q0c-P via `--object`. The checker never searches
for a substitute object. This deliberately covers Setuptools layouts where
the object is written to `src/` beside, rather than below, `build/temp/`.

## Static self-test and dry-run

From the repository root:

```bash
python3 -m py_compile scripts/*phase4a3_q0c*.py
bash -n scripts/run_phase4a3_q0c_apparatus.sh
python3 -m unittest -v tests/test_phase4a3_q0c_static.py
PYTHON_BIN=python3 scripts/run_phase4a3_q0c_apparatus.sh --dry-run > /tmp/q0c-matrix.json
```

The dry-run contains exactly 100 fresh-process workers and performs no GPU
initialization. Subsets can be inspected with `--subphases LN,LP`.

## Measurement execution

Use a real Linux virtual console (`Ctrl+Alt+F3`, login, no graphical display
manager), with no other GPU process. Do not run this through an IDE terminal,
SSH pseudo-terminal, `tmux`, or `screen`.

```bash
cd /home/oem/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe
PYTHON_BIN=python3 scripts/run_phase4a3_q0c_apparatus.sh
```

The runner verifies the frozen release, builds a uniquely marked test-only
binding, captures the unique exact build object, restores and byte-checks the
production source, then runs Q0c-P before any smoke or measurement worker.
Any missing, duplicate, or wrongly named object and every Q0c-P failure stops
execution.
Later worker failures are recorded without deleting valid independent groups.

The default evidence directory is
`workspace/evidence/phase4a3_q0c_<UTC timestamp>`. The diagnostic bundler runs
only after `run.log` has closed, excludes its own checksum file while hashing,
and produces `phase4a3_q0c_diagnostics.tar.gz`.

## Decision boundary

`PROCEED_TO_PHASE4A3_P0_PROTOCOL_FREEZE` means only that the apparatus
prerequisites are qualified. It is not a speedup claim. Q0c-G may be
incomplete without invalidating independently valid latency or throughput
groups.
