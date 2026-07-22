# Phase 3B1-C – Gradientenvertrag und API-Audit

Marker: `TCNN_RDNA4_P3B1C_FP16_BACKWARD_001`

Ausgangscommit: `b59d569e5c662e8738f6af929e122c954ec68d7c`

Dieses Dokument ist der Stufe-0-Checkpoint vor jeder Backward-Produktionsänderung.

## Tatsächlicher Typvertrag

- Der explizite FP16-Pfad wird als `DifferentiableObject<float, __half, __half>` exportiert.
- `dL_doutput` hat daher am nativen Modul die Ausgabepräzision `__half` / FP16.
- Der MLP-interne `dL_dinput` ist wegen `Network<__half>` FP16. `NetworkWithInputEncoding<__half>`
  übergibt ihn an das Identity-Encoding, das den extern sichtbaren Eingabegradienten einmal nach
  FP32 konvertiert. Der Python-/Trainer-Vertrag für `dL_dinput` ist somit FP32.
- Der native Parametergradientenpuffer ist durch `DifferentiableObject<float, __half, __half>`
  und `c10_param_precision()` FP16.
- Das Python-Modul hält `self.params` als FP32-Masterparameter. Für den nativen Forward werden
  diese explizit nach FP16 konvertiert. Autograd propagiert den nativen FP16-Parametergradienten
  durch diese Konvertierung zurück zum FP32-Masterparameter; Standard-PyTorch-Optimizer arbeiten
  anschließend auf diesem FP32-Parameter.

## Festgelegter Backward-Vertrag

- Alle `dW`- und `db`-Ergebnisse werden intern vollständig in FP32 berechnet.
- Der externe Parametergradient bleibt aus API-Gründen FP16. Jedes vollständige FP32-Ergebnis
  wird genau einmal beim finalen Schreiben nach FP16 konvertiert.
- `Overwrite` ersetzt jedes Ziel vollständig und liest keinen Altwert.
- `Accumulate` lädt den alten FP16-Zielwert einmal nach FP32, addiert dort das neue vollständige
  FP32-Ergebnis und schreibt genau einmal FP16 zurück.
- `Ignore` lässt den Parametergradientenpuffer bytegleich, erzeugt weder `dW` noch `db`, berechnet
  aber `dL_dinput`, falls dieser angefordert ist.
- Es werden keine FP16-Atomics und keine schrittweisen FP16-Reduktionen verwendet.
- Dynamisches Loss Scaling wird nicht implementiert. Der bestehende statische `loss_scale`-Pfad
  des Python-Wrappers bleibt unverändert.

## Abgrenzung

Der eingefrorene FP32-/Phase-3A4-Code wird nicht geändert. BF16, FP8, rocWMMA,
Fully-Fused-Backends und HIPRTC sind ausgeschlossen.
