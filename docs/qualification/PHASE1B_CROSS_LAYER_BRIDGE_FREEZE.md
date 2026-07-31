# Phase 1B gfx1201 cross-layer bridge freeze

Marker: `P1B-GATE-FREEZE-001`

Status: `RDNA4_ROCWMMA_CROSS_LAYER_BRIDGE: PASS`

The immutable qualification evidence is stored at:

`<local-evidence-path-redacted>`

Evidence anchors:

- `SHA256SUMS`: `f958d58df64cd8118ab88a61f8ee353b01456b44df80a2bcfee6980eec8e2104`
- `verification.json`: `b962651a67f9c2636c659e38b4d9cfe89e3059650d1de186b154c42d77b3ae8e`
- `provenance.json`: `14781f2a3764ab563838fbf09f977e77a5a3d8ede8eb7b7ad3d47345d203e398`
- native bridge source:
  `075c038638ad0d738d2ca97c471c3b6648dae9717a3eea1bc088c5b865017938`
- gfx1201 ISA evidence:
  `8e47c0919f2274d672fd03ec7a1271213c0ff33f4487b9f4894eddc1dd4c660f`
- probe driver:
  `c7d6dc3539f34384a7de6a6ccda4c5f806825c88e42e660a6479bd94cf9b3983`
- ROCm 7.2 ISA verifier:
  `7149855c175093576aad5ad8b46842312d3057283861cf414756308e5c4f6ae0`

Qualification summary:

- four required bridge cases passed;
- three bit-identical fresh hardware processes per case;
- 1024 Phase-1A-oracle reload entries per case;
- canonical FP16 LDS, canaries, and partial-M13 positive-zero rows passed;
- source and gfx1201 ISA gates passed;
- Phase-1A tag is an ancestor and its evidence remains unchanged;
- the Phase-4A2 mapping header remains unchanged.

The Phase-1A and Phase-1B evidence directories are read-only inputs to any
subsequent phase.
