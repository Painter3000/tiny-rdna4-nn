# Phase 3B1-D – FP16 Training, Loss Scaling und Checkpoint/Resume

Marker: `TCNN_RDNA4_P3B1D_FP16_TRAINING_001`

**Entscheidung: `PROCEED_TO_3B1E`**

## Ergebnis

- Fresh-GPU-Baseline: 296/296.
- Kurztrainings: 24 × 50 Schritte; SGD und Adam.
- Langläufe: 3 × 1000 = 3000 Schritte.
- Gradient Accumulation: `True`.
- Statisches Loss Scaling 1/8/128/1024/8192: `True`.
- Dynamisches Loss Scaling, Overflow-Skip und Recovery: `True`.
- Checkpoint/Resume: 4/4 bitidentisch in frischen Prozessen.
- Zwei-Stream-Training: `True`.

## Regressionen

- Phase 3A1 / 3A4: PASS / PASS.
- Phase 3B1-B Forward: PASS; der historische Gesamtstatus `PHASE3B1B_BLOCKED` stammt ausschließlich vom heute überholten Vertrag 'Backward muss abgewiesen werden'.
- Phase 3B1-B1: PASS; C/C1/C1a-Hülle: 296/296.
- Historische Reports bytegleich: `True`; Produktionscode unverändert: `True`.

Der große Rohreport bleibt lokal und ist im kompakten JSON mit absolutem Pfad, Größe und SHA256 referenziert.
