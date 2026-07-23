# Phase 3B1-F0b – Reproducible Benchmark Simplification

Marker: `TCNN_RDNA4_P3B1F0B_REPRODUCIBLE_BENCHMARK_001`

Basiscommit: `c2921ea14c82cf2222487b95f8514f678afc2f2c`

## Entscheidung

`PROCEED_TO_3B1F1_72_PROCESS_MEASUREMENT`

Der aktive, nie ausgeführte 123-Fälle-/861-Prozesse-Vertrag wurde durch einen reproduzierbarkeitsorientierten F1-Vertrag mit 24 Fällen und 72 Primärprozessen ersetzt. Das historische Manifest und die historischen F/F0a/F0a1-Berichte blieben bytegleich. Es wurden keine Produktionsdateien und keine Performancegrenzen geändert, keine F1-Vollmessung ausgeführt und kein PASS-Tag erzeugt.

## Aktiver F1-Vertrag

- 12 Network-only-Fälle: N32-L2 und N128-L4; Batch 1024 und 16384; Forward, Forward+Backward und Adam-Schritt.
- 12 NetworkWithInputEncoding-Fälle: Identity, Frequency und HashGrid3D; Batch 1024 und 16384; Forward+Backward und Adam-Schritt.
- Drei Fresh Processes je Fall, fünf gepaarte Runden und wechselnde FP16/FP32-Reihenfolge.
- Mindestens 2/3 gültige Primärprozesse; höchstens ein dokumentierter Ersatzversuch, wobei der ursprüngliche Fehllauf erhalten bleibt.
- Correctness Pre/Post, vier per-Prozess-Numerikgruppen, NaN/Inf, Median, MAD, gepaartes 95-%-Konfidenzintervall sowie getrennte FP16-/FP32-Speicher-Peaks.
- Algorithmus-ID, GEMM-Logforensik, Profiler, Manipulationstests und Contract-Verkettung sind keine aktiven PASS-Gates.
- Fehlender Speedup blockiert nicht. Zeit und Speicher werden im Fazit gleichberechtigt bewertet.
- Batch 1024 ist als mögliches `latency-bound / launch-overhead-dominated regime` markiert; Batch 16384 bildet das Durchsatzregime.

Für Frequency und Identity unterscheiden sich die nativen FP16-/FP32-Puffer wegen der gepaddeten Encodingbreite. Der gemeinsame logische FP32-Master wird deshalb explizit auf beide Layouts abgebildet: logische First-Layer-Gewichte bleiben identisch, FP16-Padding ist separat nullgefüllt.

## F0b-Smoke

Die folgenden vier Fresh-Process-Smokes bestanden mit je einer gepaarten Runde und drei festen Iterationen:

1. `network.N32-L2.b1024.forward`
2. `network.N128-L4.b16384.adam_training_step`
3. `nwe.Frequency.b1024.forward_backward`
4. `nwe.HashGrid3D.b16384.adam_training_step`

Correctness Pre/Post bestanden. Der lokale Vertragstest bestätigte:

- 2/3 gültig: akzeptiert;
- 1/3 gültig: blockiert;
- genau ein Ersatzversuch: zulässig;
- mehr als ein Ersatzversuch: blockiert.

Die Smoke-Zeiten sind keine F1-Ergebnisse und fließen nicht in die spätere 72-Prozess-Auswertung ein. Rohpfade, Prüfsummen und Einzelgates stehen in `phase3b1f0b_smoke.json`.
