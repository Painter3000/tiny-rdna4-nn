# Phase 3B1-F0 – FP16 Performance Protocol Freeze

Marker: `TCNN_RDNA4_P3B1F_FP16_PERFORMANCE_001`

## Entscheidung

`PHASE3B1F0_PROTOCOL_READY`

Basis ist Commit `3265070edbef35969f569972eaf0731d9dab2fe3`. Phase 3B1-F0 verändert keinen Produktionscode, kein Parameterlayout und keine numerische Toleranz. Es enthält keine Performanceergebnisse. Die F1-Vollmessung wurde nicht ausgeführt.

## Primärmatrix

- 48 Network-only-Fälle: vier qualifizierte Topologien, vier Batches und drei Operationen.
- 75 Encoding-Fälle: fünf Encodings, drei Batches und fünf Operationen.
- Insgesamt 123 unterstützte Primärfälle.
- Sieben Fresh Processes je Fall, insgesamt 861 Primärprozesse.
- Jeder Fresh Process misst exakt einen Primärfall.
- Fünf gepaarte Runden mit insgesamt 40 Timingblöcken pro Prozess.
- Reihenfolgen: `FP16, FP32, FP32, FP16` und invertiert.
- Mindestens 35 getrennte Cold-Start-Prozesse und vier separate Profilingfälle.

Die vollständige Fallliste einschließlich Seeds, Datenvertrag, Warm-up, Kalibrierung, Statistik, Telemetrie und Gates steht ausschließlich im unveränderlichen Manifest.

## Messvertrag

Primärtimer sind GPU-Events auf demselben Stream; `perf_counter_ns` wird parallel protokolliert. Ein Timingblock zeichnet Start und Ende um N unsynchronisierte Iterationen auf und synchronisiert nur das Endevent. Die Kalibrierung zielt auf 250–1500 ms bei 50–5000 Iterationen. Kandidat und Referenz verwenden innerhalb eines Falls dieselbe Iterationszahl.

Korrektheit wird außerhalb der Timingregion geprüft. Parameter, Inputs, Targets und Upstream-Gradienten sind innerhalb jedes Paars identisch; der FP16-Kandidat erhält genau eine Quantisierung der gemeinsamen FP32-Masterparameter.

Es gibt keine zahlenwertbasierte Ausreißerentfernung und keine stillen Wiederholungen. Jeder ungültige Prozess bleibt in den Rohdaten. Höchstens einer von sieben Prozessen darf ungültig sein.

## Feste Performancegrenzen

Für Network-only, Batch ≥1024:

| Operation | Geometrisches Mittel | Untere 95-%-CI-Grenze |
|---|---:|---:|
| Forward | 1,20× | 1,05× |
| Forward + Backward | 1,15× | 1,02× |
| Adam-Schritt | 1,10× | 1,00× |

Für `NetworkWithInputEncoding`, Batch ≥1024:

| Operation | Geometrisches Mittel |
|---|---:|
| Forward + Backward | 1,05× |
| Adam-Schritt | 1,03× |

Ein großer Fall blockiert bei Median unter 0,90× und oberer 95-%-CI-Grenze unter 1,00×. PyTorch-Vergleiche sind ausschließlich Kontext.

## Feste Numerikgrenzen

- Network output max_abs ≤ 0,03
- dL/dinput max_abs ≤ 0,04
- Network gradient max_abs ≤ 0,06
- Encoding gradient max_abs ≤ 0,06
- NaN/Inf: jeweils null

Die deterministische E1a-Ausgangsnumerik darf nicht überschritten werden:

- dL/dinput: Frequency, dims=2, variant=3, batch=1024; max_abs `0.02745274268090725`, normalized L2 `0.028579029471602468`.
- Network gradient: HashGrid, dims=2, Smoothstep, 16 Levels, 4 Features, batch=1024; max_abs `0.0007190704345703125`, normalized L2 `0.10276123438550208`.

Diese Werte sind Ausgangsnumerik, keine Performanceverbesserung.

## Ausführungsschutz

Der Standardaufruf führt ausschließlich den F0-Protokollaudit aus:

```bash
scripts/test_phase3b1f_performance.sh
```

Die F1-Matrix erfordert nach externem Harness-Audit die doppelte explizite Freigabe:

```bash
scripts/test_phase3b1f_performance.sh --execute-full \
  --confirm RUN_PHASE3B1F1_FULL_MEASUREMENT
```

F1 schreibt Rohdaten nach `/tmp/phase3b1f_fp16_performance_raw.json`. F1-Berichte werden erst durch den fail-closed Finalizer nach vollständigen Rohdaten erzeugt.
