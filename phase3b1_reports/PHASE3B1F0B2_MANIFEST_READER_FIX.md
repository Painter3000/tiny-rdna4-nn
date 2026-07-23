# Phase 3B1-F0b2 – Manifest Reader Fix

Marker: `TCNN_RDNA4_P3B1F0B2_MANIFEST_READER_FIX_001`

Basiscommit: `56f961dda410def4dddd0ff69173c38dec2b01d2`

## Entscheidung

`PROCEED_TO_3B1F1_72_PROCESS_MEASUREMENT`

Der deterministische F1-Abbruch wurde auf einen einzelnen Schema-Lesefehler
zurückgeführt. Der aktive F0b-Vertrag definiert
`paired_rounds_per_process` top-level; der Worker las den Wert fälschlich aus
`manifest["measurement"]`. Der Worker verwendet nun ausschließlich den
verbindlichen Top-Level-Wert und besitzt dafür keinen stillen Legacy-Fallback.

## Regressionstest

`test_phase3b1f0b_contract.py --reader-only` startet den realen Worker-Leser in
einem isolierten Subprozess, lädt das aktive Manifest und verlangt:

- erfolgreicher Reader-Prozess ohne `KeyError`;
- `manifest_paired_rounds_per_process == 5`;
- keine alternative Schemaquelle.

Die Subprozessisolation verhindert, dass ein im Test-Elternprozess gehaltener
GPU-Kontext den nachfolgenden Fresh-Process-Smoke als Fremdlast kontaminiert.

## Echter Pfad-Smoke

`network.N32-L2.b1024.forward` bestand mit einem Fresh Process, einer gepaarten
Runde und drei festen Iterationen über denselben
Orchestrator→Worker-Aufrufpfad wie F1. Correctness Pre/Post sowie der bestehende
lokale 2/3-/Replacement-Vertrag bestanden. Der Smoke liegt außerhalb der
späteren F1-Statistik in:

`/tmp/phase3b1f0b2_single_worker_smoke.json`

Der anschließende nicht messende Dry-Run bestätigte:

- 24 Primärfälle;
- 3 Fresh Processes je Fall;
- 72 reguläre Primärprozesse;
- 5 gepaarte Runden;
- aktiver Manifest-SHA256
  `98bd1a1c2d447bfeaba419b9f8cd705320ff0a92f31ef1dca71adf4a3d272708`.

## Erhaltener Fehlrun

Der abgebrochene Lauf `20260724T000728_179565` wurde weder verändert noch
resumed und bleibt von F1-Statistiken ausgeschlossen:

- Rohdaten:
  `/tmp/phase3b1f1_reproducible_runs/20260724T000728_179565`
- Index-SHA256:
  `370c935577850a5acbf522ad2c3298abb4132fa848255b9f37d68cb1d12f0912`

Das aktive Manifest blieb bytegleich. Produktionscode wurde nicht geändert.
Eine neue F1-Messung muss mit einer neuen Run-ID beginnen.
