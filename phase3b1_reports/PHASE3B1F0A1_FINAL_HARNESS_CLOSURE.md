# Phase 3B1-F0a1 – Final Harness Closure

Marker: `TCNN_RDNA4_P3B1F0A1_FINAL_HARNESS_CLOSURE_001`

Basiscommit: `8b1c785d2b250344521f33c95339ea7b8f92269e`

## Entscheidung

`PHASE3B1F0A1_HARNESS_READY`

Das eingefrorene Phase-3B1-F-Manifest blieb bytegleich. Performancegrenzen und Produktionsdateien wurden nicht geändert. Es wurde keine F1-Vollmessung ausgeführt und keine Performanceaussage abgeleitet.

## Geschlossene Verträge

- Worker, Orchestrator und Finalizer verwenden ein gemeinsames `backend_evidence`-Schema. Algorithmus-IDs, Fallback, Workspace, GEMM-Anzahl und Custom-Backward-Nachweis werden an denselben Feldern erfasst und geprüft.
- Das Harness-Contract-JSON ist per SHA256 an Workerargumente, Prozessrohberichte, Orchestratoridentität und Finalizer gebunden.
- Ein FP32-Masterparametersatz wird in der nativen logischen Kandidatenreihenfolge materialisiert. Erwartete FP16-Quantisierung, tatsächlicher Kandidatenpuffer und tatsächliche Kandidatenquantisierung besitzen getrennte Hashes.
- Parametergradienten werden für Network-only, Encoding-only und NetworkWithInputEncoding über explizite Offsets und Counts getrennt.
- Der Finalizer prüft pro Prozess alle vier `max_abs`-Werte, NaN/Inf-Zahlen sowie endliche normierte L2-Werte.
- Fallback kann nicht über eine leere Sammlung bestehen. Workspace stammt aus dem nativen selektierten Plan beziehungsweise der Encoding-Scratch-Beobachtung. Custom ReLU/Backward/Biasgrad wird bei relevanten Fällen aus nativen Launch-Counter-Deltas abgeleitet.
- Adam-State wird außerhalb der Timingregion real angelegt und anschließend auf Schritt null zurückgesetzt. Parameter-, Schritt- und Optimizer-State-Hashes werden vor Timing dokumentiert.
- Die gemeinsame Kalibrierung prüft, ob beide Backends mindestens 250 ms erreichen können, ohne dass der langsamere Block ungefähr 1500 ms überschreitet. Unlösbare Fälle werden als nicht feasible blockiert.
- Resume übernimmt Primärprozessdateien nur nach Prüfung von Run-Pfad, Index-SHA256, Marker, Fall-ID, Prozessindex sowie Manifest- und Contract-Hash.
- Primär-, Cold-start-, Correctness- und Profilingprozesse besitzen Timeout- und Freshness-Prüfungen.

## Abschluss-Smoke

Alle vier vorgesehenen Fresh-Process-Fälle bestanden:

1. `network.N32-L2.b1024.forward`
2. `network.N64-L2.b1024.adam_training_step`
3. `encoding.HashGrid3D.b1024.network_with_encoding_forward_backward`
4. `encoding.Frequency.b1024.encoding_forward`

Correctness Pre/Post bestanden. Die reale Smoke-Rohdatenbasis bestand sämtliche strukturellen Gates von `derive()` und `evaluate()`. Anschließend wurden 40 unterschiedliche Rohdatenmanipulationen ausgeführt; 40/40 führten über denselben Finalizerpfad fail-closed zu einer Blockierung.

Die Smoke-Zeiten sind ausschließlich Harness-Evidenz und fließen nicht in F1-Statistiken ein. Vollständige Rohpfade, Hashes, Einzelgates und Manipulationsergebnisse stehen in `phase3b1f0a1_harness_closure.json`.
