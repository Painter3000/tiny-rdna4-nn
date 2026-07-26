# Phase 4A0-P2 — Device-derived rocWMMA lane/register-file map

Marker:

```text
TCNN_RDNA4_P4A0_P2_RAW_LANE_FRAGMENT_MAP_003
```

The previous host-side geometry probes were invalid because rocWMMA deliberately
uses a Wave64 fallback during host compilation. Host-side `sizeof(...)` and
`num_elements` therefore do not describe the compiled `gfx1201` Wave32 device
fragment.

This version records all authoritative geometry inside the device kernel and
uses fixed-capacity diagnostic buffers on the host.

It captures:

- FP16 `matrix_a`, row-major;
- FP16 `matrix_b`, column-major;
- FP32 accumulator after `identity × uniquely marked matrix`;
- raw FP16/FP32 bit patterns;
- one write count per `(lane, register-file row)`;
- untouched inactive slots and guard regions;
- regular `store_matrix_sync()` output;
- two fresh-process maps that must be identical.

Expected device geometry for this qualified case:

```text
8 register-file elements/lane × 32 lanes = 256 elements per role
```

The output is version-bound to the recorded rocWMMA, compiler, architecture,
tile, datatype, and layouts. It is not a stable ABI.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o ~/Downloads/phase4a0_p2_device_geometry_bundle.zip

chmod +x \
  scripts/finalize_phase4a0_p2.py \
  scripts/run_phase4a0_p2_raw_lane_fragment_map.sh

scripts/run_phase4a0_p2_raw_lane_fragment_map.sh
```

Expected final markers:

```text
ROCWMMA_P2_DEVICE_GEOMETRY_CAPTURE: PASS
ROCWMMA_P2_FRESH_PROCESS_REPRODUCIBILITY: PASS
ROCWMMA_P2_MAP_CONTEXT: RECORDED
ROCWMMA_RAW_LANE_FRAGMENT_MAP: CAPTURED
PHASE4A0_P2_RAW_LANE_FRAGMENT_MAP: PASS
PHASE4A0_P2_JSON_AUDIT: PASS
```
