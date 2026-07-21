# Phase 3A4 – Activation=None regression bisection

## Ergebnis

Die reproduzierbare Abweichung ist kein ausgeführter Phase-3A4-None-Pfad und keine GEMM-/Cache-Mutation. Die durch die Messungen am stärksten gestützte Erklärung ist eine reihenfolgeabhängige GPU-Konditionierung: Der stark verkürzte fusionierte ReLU-Block wärmt das Gerät vor dem unveränderten None-Kontrollfall weniger auf. Direkte Takttelemetrie wurde nicht aufgezeichnet, daher ist dies als starke Inferenz und nicht als direkt gemessener Taktbefund markiert.

Der bestehende Performance-Gate bleibt dennoch unverändert bindend. Es wurden keine offiziellen Läufe gestartet und kein PASS-Tag gesetzt.

## Varianten und Rohmessungen

| Variante | None-only Median | ReLU→None Median | None→ReLU→None: vorher | None→ReLU→None: nachher |
|---|---:|---:|---:|---:|
| A | 0.682200× | 1.034127× | 0.682045× | 1.034069× |
| B | 0.682214× | 1.035332× | 0.682602× | 1.034002× |
| C | 0.681664× | 1.035490× | 0.681759× | 1.034482× |
| D | 0.681820× | 0.913853× | 0.681746× | 1.018702× |
| E | 0.680943× | 0.914656× | 0.681078× | 1.014252× |

Jeder Tabellenwert umfasst alle vorab festgelegten zehn Fresh-Process-Läufe; beim Doppel-None-Szenario je zehn Messungen vor und nach ReLU. Es erfolgte keine Bestwertauswahl.

Die erste eindeutig schlechte beobachtete Änderung liegt zwischen C und D: Mit aktiviertem Dispatch, aber weiterhin vollständig herauskompilierten Zählern, sinken alle zehn ReLU→None-Läufe unter 0,99×. Das isoliert den Reihenfolge-/Konditionierungseffekt auf die stark verkürzte ReLU-Arbeit; es beweist keine Ausführung des Dispatchs im None-Modell.

## Harte None-Invarianten

Alle 200 None-Messblöcke bestanden:

- fused Stage-1 delta = 0
- fused ReLU-only delta = 0
- Biasgrad-Finalize delta = 0
- Partial-Scratch live vor und nach dem Block = 0
- Scratch-Allokationsdelta = 0
- Fusionsfallbackdelta = 0

## GEMM- und Cachevergleich

Alle neun normalisierten None-GEMMs sind zwischen Phase 3A3 und Phase 3A4 identisch (Trace-Hash `cbdfa036ac6fb173`):

- identische M/N/K, Transpositionen, Leading Dimensions und nicht-batched Strides
- identische Epilogues: drei Bias-Forward-GEMMs, sechs Default-Backward-GEMMs
- Workspace immer 0 Byte
- Stream in allen Traces: Default-Stream 0
- identische Cache-Keys
- identische Algorithmus-IDs: 91217 (Forward), 91206 (Weight Gradient), 91207 (Input Gradient)
- identische Tiles: 8×8×16, 16×16×16 beziehungsweise 8×8×8
- Split-K/GSU = 1 und WGM = 1; ein separates Swizzle-Attribut wird von der Lösung nicht ausgewiesen
- identische Lösungs- und Kernelnamen

Nach ReLU sind die sechs Default-Backward-Keys legitime Cache-Hits; die aktivierungsabhängigen Forward-Epilogues besitzen getrennte Keys. Es gibt weder eine Key-Kollision noch Deskriptormutation.

## Schlussfolgerung

A bis E sind im kalten None-only-Prozess praktisch gleich langsam (Median etwa 0,681–0,682×). A/B/C erreichen nach der längeren Legacy-ReLU-Arbeit etwa 1,034–1,035×. D/E erreichen nach der kürzeren fusionierten ReLU-Arbeit zunächst nur etwa 0,914×; ein vorheriger None-Block hebt D/E danach wieder auf etwa 1,014–1,019×. Das ist mit einem Takt-/Power-/Warm-up-Zustand vereinbar und mit einer None-Codeänderung unvereinbar.

Status: weiterhin BLOCKED. Keine Grenzwerte wurden gelockert. Vor vier offiziellen Läufen muss ein gleicher, aktivierungsunabhängiger Konditionierungszustand definiert und gegen die unveränderte 0,99×-Grenze nachgewiesen werden.

Vollständige Rohdaten stehen in `none_regression_bisect.json` und den Einzeldateien unter `phase3a4_reports/none_regression_bisect/`.
