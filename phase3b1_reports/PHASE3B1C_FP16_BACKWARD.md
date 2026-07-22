# Phase 3B1-C – FP16 Backward-Basispfad

Marker: `TCNN_RDNA4_P3B1C_FP16_BACKWARD_001`

**Entscheidung: `PROCEED_TO_3B1D`**

## Vertrag

- FP16: Operanden, gespeicherte Aktivierungen, dZ, internes dX und nativer Parametergradientpuffer.
- FP32: hipBLASLt-Akkumulation, dW/db-Scratch und deterministische Biasreduktion.
- Python/PyTorch: FP32-Masterparameter und FP32-Gradienten; einmalige Konvertierung des nativen FP16-Gradienten.
- Overwrite, Accumulate und Ignore bestanden; keine FP16-Atomics und kein dynamisches Loss Scaling.

## Capability und Numerik

- Capability: `BACKWARD_GEMM_CAPABILITY_PASS`; 180 Signaturen in 360 Fresh Processes.
- Ausgewählte Algorithmen: 37 eindeutige Lösungen; Workspace stets 0 Byte (vollständig im JSON).
- Funktionale Fälle: 291/291 bestanden.
- dX max_abs=3.814697265625e-06, nL2=3.6547735504655066e-05, ULP außerhalb Near-zero=1
- dW max_abs=0.00048828125, nL2=0.00019231222490689336, ULP außerhalb Near-zero=14
- db max_abs=0.0009765625, nL2=0.00031327507686208447, ULP außerhalb Near-zero=16
- dZ max_abs=0.0, Maskenabweichungen=0

## Invarianten

- Echter Multistream-Test: `True` (2 Streams, 64+64 asynchrone Runden).
- Dynamische Batchfolge: `True`; zweiter Durchlauf ohne neue Pläne/Heuristiken/Handles/Deskriptoren.
- Scratch live/peak: 0/6662656 Byte.
- Phase 3A1 / 3A4: `PASS` / `PASS`.
- Historische A/B/B1-Berichte bytegleich: `True`.
- Marker-Audit: `True`.
- Eingefrorene Phase-3A4-/FP32-Kerneldateien unverändert: `True`.

Alle Einzelfälle, Algorithmen, Counter, Toleranzen, Hashes und Diffpfade stehen im JSON-Bericht.
