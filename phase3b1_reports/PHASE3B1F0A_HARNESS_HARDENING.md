# Phase 3B1-F0a – Benchmark Harness Hardening

Marker: `TCNN_RDNA4_P3B1F0A_HARNESS_HARDENING_001`

Basis: `4cdfb249ceff6fd920b20d5d92e4d46865819056`

## Ergebnis

Der F0a-Harness wurde für einen späteren, ausdrücklich bestätigten F1-Lauf gehärtet. Das eingefrorene Protokollmanifest und sämtliche Performancegrenzen blieben unverändert. In F0a wurden keine vollständige Performance-Matrix und keine für F1 verwertbaren Performancewerte erzeugt.

## Implementierte Verträge

- Correctness-Pre und -Post führen den E1a1-Finalizer, zwei numerische Baselinefälle, historische Hashprüfungen und den Produktionsdiff real aus. Der F1-Finalizer prüft die einzelnen Messfelder gegen die eingefrorenen Grenzen.
- Jeder Primärprozess besitzt einen eindeutigen Index. Der F1-Finalizer verlangt je Fall exakt `0..6`, ohne Duplikate, und wertet nur boolesche Integritätsfelder aus.
- Unvollständige oder falsch typisierte Rohdaten führen fail-closed zu `PHASE3B1F_BLOCKED`.
- Encoding-only-Fälle werden als `encoding_kernel` mit `gemm_expected=false`, `algorithm_ids=not_applicable`, `fallback=not_applicable` und gemessenem Scratch/Workspacewert klassifiziert.
- GEMM-Fälle erhalten konkrete hipBLASLt-Solution-IDs aus nach Prozessende geparsten Logs. Der native ausgewählte Plan weist Zero-Workspace aus; ein fehlender ergänzender Logwert überschreibt diesen Messwert nicht.
- Adam-Kalibrierung verwendet Wegwerfmodelle; die Timingmodelle beginnen mit frischem Optimizerzustand und identischen logischen Masterparametern.
- Paaridentität, Telemetrie, Warm-up-Zähler, Scratch-, Descriptor- und Prozessende-Nachweise werden pro Fresh Process erfasst.
- Separate Profilingprozesse werden nicht in Timingstatistiken aufgenommen. Der Parser aggregiert Kernelnamen, Aufrufzahlen, Laufzeiten, Kategorien, Kopien und Host-Synchronisationen.
- Matrix-Konfidenzintervalle werden gemeinsam über Fälle und Fresh-Process-Speedups gebootstrapped.
- Prozessdateien entstehen in eindeutigen Verzeichnissen unter `/tmp/phase3b1f_runs`; stale Outputs werden verworfen, Timeouts sind fest, und der Index ist append-only.

## Smoke

Der Abschluss-Smoke umfasst ausschließlich:

1. `network.N32-L2.b1024.forward`
2. `network.N64-L2.b1024.adam_training_step`
3. `encoding.HashGrid3D.b1024.network_with_encoding_forward_backward`
4. `encoding.Frequency.b1024.encoding_forward`

Jeder Fall läuft in genau einem frischen Prozess mit einer Runde und fünf festen Iterationen. Die Werte sind ausdrücklich nicht Teil späterer F1-Statistiken. Die maschinenlesbaren Pfade, Hashes, Einzelgates, Correctness-Snapshots und die 40 tatsächlich ausgeführten fail-closed Manipulationstests stehen in `phase3b1f0a_harness_smoke.json`.

## Grenzen

F0a ist eine Harness-Qualification, keine Performanceaussage. Die vollständige F1-Matrix, Cold-start-Serie und separaten Profilingfälle wurden nicht ausgeführt. Es wurde kein Performance-PASS vergeben und kein Tag erzeugt.

Abschlussentscheidung: `PHASE3B1F0A_HARNESS_READY`
