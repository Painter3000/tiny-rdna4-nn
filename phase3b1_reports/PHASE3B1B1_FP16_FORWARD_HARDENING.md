# Phase 3B1-B1 – FP16 Forward Hardening

Marker: `TCNN_RDNA4_P3B1B1_FP16_FORWARD_HARDENING_001`

**Decision: `PROCEED_TO_3B1C`**

## Scope und Identität

- Branch: `phase3b1b-fp16-forward-baseline`
- Ausgangscommit: `576bdd8cafd011254538cdd33aa5c6d3cb9b6091`
- Keine Backward-GEMMs, Gradienten- oder Biasgradienten-Implementierung.
- Kein PASS-Tag.

Die eingefrorenen Phase-3A4- und FP32-Kerneldateien blieben unverändert. Die gemeinsame Factory-/API-/Binding-Schicht wurde für den expliziten FP16-Pfad erweitert; bestehende FP32-Regressionen bestanden.

## Aktivierungsverträge

- Hidden=None,Output=None: 72/72 bestanden
- Hidden=None,Output=ReLU: 72/72 bestanden
- Hidden=ReLU,Output=None: 72/72 bestanden
- Hidden=ReLU,Output=ReLU: 72/72 bestanden
- Gesamt: 288/288
- CPU64: max_abs=6.103515625e-05, max_rel=0.0009699321047526673, nL2=2.1460506112307237e-05, ReLU-Maskenfehler=0
- PyTorch: max_abs=6.103515625e-05, max_rel=0.000975609756097561, nL2=2.146994136989351e-05, ReLU-Maskenfehler=0
- Aktivierte Hidden- und Output-Schichten wurden jeweils separat gegen beide Oracles geprüft.

## Rechteckige Formen

- 16->64->32: 6/6 bestanden
- 128->32->16: 6/6 bestanden
- 32->128->64: 6/6 bestanden
- 64->16->128: 6/6 bestanden
- Gesamt: 24/24; Parameteranzahl, Offsets, Achsen und Ausgabeformen geprüft.

## Echter Multistream-Test

- Runden/Submissions: 64/128; nur eine terminale Synchronisation
- verschiedene Streams/Modelle/Inputs/Parameteridentitäten: PASS
- Pläne 24→24, Heuristiken 24→24, Handles 3→3, Descriptoren 4→4
- Speicherwachstum nach Freigabe: 0 Byte; Ergebnis: True

## Dynamische Batchfolge

- Folge: 1 → 16 → 128 → 1024 → 4096 → 128 → 16 → 1, zweimal auf demselben Modell
- Pläne 26→26, Heuristiken 26→26, Handles 4→4, Descriptoren 6→6
- Speicher: 79775232→79775232 Byte; Ergebnis: True

## Guards, Counter und Regression

- Nullparameter-Guard: True
- Invalid-Descriptor ohne Counteränderung: True (0→0)
- RAII-`counted`-Audit: True
- Phase-3A1: PASS; Phase-3A4: PASS
- Historische 3B1-B-Reports bytegleich: True
