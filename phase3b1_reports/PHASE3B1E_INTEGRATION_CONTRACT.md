# Phase 3B1-E – Integrationsvertrag

Marker: `TCNN_RDNA4_P3B1E_FP16_ENCODING_INTEGRATION_001`

FP32 Input → FP16 Encoding-Aktivierung → FP16 HipBLASLtMLP → FP32 externes dInput und FP32 Python-Gradienten/Master/Optimizer-State. Native MLP- und Encodingparameter sind FP16. Netzwerkparameter liegen zuerst, Encodingparameter danach. Encoding-Ausgaben werden auf die kleinste unterstützte Breite 16/32/64/128 gepolstert und für den expliziten FP16-Backendpfad contiguous ColumnMajor verkettet.
