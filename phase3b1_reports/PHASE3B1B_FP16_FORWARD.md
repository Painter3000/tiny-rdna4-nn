# Phase 3B1-B – FP16 Forward-Epilogues und Forward-Basispfad

Marker: `TCNN_RDNA4_P3B1B_FP16_FORWARD_001`

**Decision: `PROCEED_TO_3B1C`**

## Identität und Vertrag

- Branch: `phase3b1b-fp16-forward-baseline`
- Ausgangscommit: `22364010853d872702bdf8f63cad26b890b6f47b`
- eingefrorene Phase-3A4-Identität: `6258184d8d9d032ef423b75eddeeaf8168c7e45a`
- Phase-3A4-Kernel-/FP32-Backend-Diff: `none`
- Integrationsdiff ab Capability-Commit: `bindings/torch/setup.py, bindings/torch/tinycudann/bindings.cpp, include/tiny-cuda-nn/networks/hipblaslt_mlp_fp16.h, src/cpp_api.cu, src/hipblaslt_mlp_fp16.cu, src/portable_network.cu`
- Backend: explizit `HipBLASLtMLPFP16` mit `precision=Fp16`; keine automatische Umschaltung
- Operanden/Gewichte/Bias: FP16; Compute: FP32; Hidden und finale Ausgabe: FP16
- FP32-Endausgabe: nicht angeboten, da der bestehende Module-Vertrag Parameter- und Ausgabepräzision koppelt
- Backward: explizit und empirisch abgewiesen

## Epilogue-Checkpoint

- BIAS: `True` (32 Signaturen)
- RELU_BIAS: `True` (32 Signaturen)
- Gesamt: 64 Signaturen, 128 frische Prozesse
- NN, D=FP16/FP32, Bias=FP16, Compute=FP32; Sentinels/Guards und Bias-Achse bestätigt

## Funktionale und numerische Gates

- reguläre Fälle: 120; adversariale Fälle: 8; bestanden: 128
- Fresh Processes: 12, bestanden: True
- CPU64 quantisiert: max_abs=3.0517578125e-05, max_rel=0.0009689922480620155, nL2=2.0388297743948432e-05, Maskenfehler=0
- PyTorch FP32-Compute: max_abs=3.0517578125e-05, max_rel=0.0009442870632672333, nL2=3.999985048464087e-05, Maskenfehler=0
- eingefrorener tcnn-FP32-Crosscheck: max_abs=0.0004687309265136719, max_rel=0.0007001440103734372, nL2=0.00046827265957398185, Near-zero-Maskendifferenzen=13 (informativ; keine Hidden-FP16-Quantisierung)

## Cache, Handles, Streams und Speicher

- zwei verschiedene Streams: `True`
- Warm-Cache: `True`; Misses 24→24; Heuristiken 24→24; Handle-Erzeugungen 2→2
- Scratch live/peak: 0/0 Byte
- Speicherwachstum über 20→100 Läufe: 0 Byte

## Regression und Audits

- Phase-3A1-FP32: `PASS`
- Phase-3A4-FP32: `PASS`
- Marker-Audit: `True`
- Binding: `/home/oem/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-phase2/bindings/torch/build/lib.linux-x86_64-cpython-312/tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so`

Die vollständigen Einzelfälle, vorab fixierten Toleranzen, Layer-Crosschecks, Counter-Snapshots, Fresh-Process-Ausgaben und Hashes stehen im JSON-Report.
