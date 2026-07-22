# Phase 3B1-D – Trainings- und Optimizer-Vertrag

Marker: `TCNN_RDNA4_P3B1D_FP16_TRAINING_001`

- Masterparameter: FP32 (`torch.nn.Parameter`).
- PyTorch-Gradient: FP32. Der native FP16-Gradient wird an der Autograd-Grenze einmal nach FP32 übertragen.
- Native Parameter und Forward-/Backward-Operanden: FP16; `dW` und `db` werden intern in FP32 akkumuliert und einmal final nach FP16 geschrieben.
- FP32→FP16-Parameterkonvertierung: unmittelbar in `Module.forward()` vor jedem nativen Forward.
- FP16→FP32-Gradientenpropagation: Rückgabe des nativen Backward an PyTorch; danach akkumuliert Autograd ausschließlich in FP32.
- SGD- und Adam-Optimizer-State: FP32. FP16-Optimizer-State ist unzulässig.
- `zero_grad(set_to_none=True)` entfernt den Gradientenpuffer; `False` erhält ihn und setzt ihn auf null.
- Microbatch-Akkumulation erfolgt im FP32-PyTorch-Gradientenpuffer. Der native `Accumulate`-Vertrag wird separat geprüft, aber nicht zur schrittweisen FP16-Microbatch-Akkumulation missbraucht.
- Statisches Loss Scaling wird über `Module.loss_scale` in der bestehenden Autograd-Brücke ausgeführt: vor dem nativen Backward skalieren, nach dessen Rückkehr einmal unscalen.
- Der dynamische Scaler liegt ausschließlich im Python-Trainingslayer. Sein serialisierter Zustand enthält Scale, Growth-/Backoff-Faktor, Growth-Intervall, erfolgreiche Schritte, Growth-Tracker, Overflow- und Skip-Zähler.
- Ein Overflow überspringt den Optimizer-Step vollständig, lässt Masterparameter und Optimizer-State bytegleich, reduziert die Scale und setzt Gradienten sicher zurück.

Kein BF16, FP8, rocWMMA, Fully-Fused, HIPRTC oder allgemeiner Encoding-Ausbau wird eingeführt.
