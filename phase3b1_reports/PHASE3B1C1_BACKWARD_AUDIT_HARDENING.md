# Phase 3B1-C1 – FP16 Backward Audit Hardening

Marker: `TCNN_RDNA4_P3B1C1_BACKWARD_AUDIT_HARDENING_001`

**Entscheidung: `PROCEED_TO_3B1D_AUDITED`**

## Fail-closed Abschlussgate

- Pflichtchecks: 19/19 bestanden.
- Manipulationstests: 3/3 lieferten jeweils `PHASE3B1C1_BLOCKED`.
- Historische Phase-3B1-C-Berichte bytegleich: `True`.
- Mathematische FP16-Backward-Produktion unverändert: `True`.

## Korrigierte Semantik

- Direkte dZ-Maskenabweichungen: `0`.
- Integrierte ReLU-Maskenvalidierung: `indirect_through_dx_dw_db_oracles`.
- Scratch-Zähler sind Host-Scope-Schätzungen, nicht eventgebunden, kein asynchrones Allocator-Peak und nicht Multi-GPU-fähig.
- Reale PyTorch/HIP allocated/reserved Beobachtungen vor Warm-up und nach beiden Folgen stehen im JSON.
- Range-Tests trennen Inf-Propagation/FP16-Unterlauf von echtem finite-to-FP16-Overflow; ein normaler Folgeaufruf bestand.

## Prospektive Regression

Die feste Hülle für Phase 3B1-D ist gegen `fc2432d09a344624a8ecdf1cc0065d879fb5db31` vorab im JSON dokumentiert. Historische 3B1-C-Gates wurden nicht verändert.
