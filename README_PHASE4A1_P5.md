# Phase 4A1-P5 — Width-64 ISA, resource and global-traffic audit

Marker:

```text
TCNN_RDNA4_P4A1_P5_ISA_RESOURCE_GLOBAL_TRAFFIC_001
```

This checkpoint audits the exact committed Phase 4A1-P4 kernel rather than
introducing another mathematical kernel.

## What is checked

- exact P4 source and generated-header hashes;
- two fresh `gfx1201` device-only assembly compilations;
- normalized kernel ISA reproducibility;
- extraction of the linked AMDGPU code object from the rebuilt executable;
- exactly one device kernel in that extracted code object;
- Wave32 code-object metadata;
- VGPR and SGPR allocation boundaries;
- exactly `2048` bytes of LDS/group segment;
- zero private/scratch segment;
- no scratch load/store instructions;
- matrix-core instructions present;
- LDS reads, LDS writes and block barriers present;
- optimized LLVM IR contains LDS address space 3;
- external global traffic is classified;
- no explicit hidden global output path and no compiler scratch path;
- the rebuilt audited executable reproduces the P4 JSON and CSV exactly in
  two fresh processes.

## Important claim boundary

The audit does **not** claim formal pointer provenance for every global ISA
instruction. The kernel deliberately performs global loads for input, weights,
biases and two diagnostic hidden-oracle tensors, plus global stores for final
output and diagnostics.

The stronger supported conclusion is:

```text
hidden 1 and hidden 2 use LDS;
there is no hidden global output argument;
the ISA contains LDS traffic;
the private segment is zero;
no scratch load/store instruction exists.
```

That closes the compiler-spill/global-round-trip gap without pretending that
all global traffic should disappear.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a1_p5_width64_isa_resource_global_traffic_audit_bundle.zip

chmod +x \
  scripts/prepare_phase4a1_p5.py \
  scripts/audit_phase4a1_p5_isa.py \
  scripts/finalize_phase4a1_p5.py \
  scripts/run_phase4a1_p5_isa_resource_and_global_traffic_audit.sh

export P0_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p0_20260725T080946Z"

export PHASE4A1_P4_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a1_p4_20260725T150928Z"

unset PHASE4A1_P5_EVIDENCE

scripts/run_phase4a1_p5_isa_resource_and_global_traffic_audit.sh
```

Expected final markers:

```text
PHASE4A1_P4_PREREQUISITE: PASS
PHASE4A1_P5_SOURCE_CONTRACT: PASS
PHASE4A1_P5_PREPARATION: PASS

WIDTH64_SINGLE_DEVICE_KERNEL_OBJECT: PASS
WIDTH64_WAVE32_CODE_OBJECT: PASS
WIDTH64_ISA_FRESH_BUILD_REPRODUCIBILITY: PASS
WIDTH64_LDS_RESOURCE_2048_BYTES: PASS
WIDTH64_PRIVATE_SEGMENT_ZERO: PASS
WIDTH64_NO_SCRATCH_INSTRUCTIONS: PASS
WIDTH64_MATRIX_CORE_INSTRUCTIONS_PRESENT: PASS
WIDTH64_LDS_READ_WRITE_PRESENT: PASS
WIDTH64_BLOCK_BARRIERS_PRESENT: PASS
WIDTH64_GLOBAL_TRAFFIC_CLASSIFIED: PASS
WIDTH64_NO_HIDDEN_INTERMEDIATE_GLOBAL_TRAFFIC: PASS

WIDTH64_AUDITED_BINARY_FUNCTIONAL_REPLAY: PASS
WIDTH64_AUDITED_BINARY_EXACT_P4_REPLAY: PASS
PHASE4A1_P5_JSON_AUDIT: PASS
PHASE4A1_P5_MAP_CONTEXT: RECORDED
PHASE4A1_P5_WIDTH64_ISA_RESOURCE_GLOBAL_TRAFFIC_AUDIT: PASS
```


## v2 correction

The first runner attempted to pass the output of:

```text
hipcc --offload-device-only -c
```

directly to `llvm-objdump -d`. With the tested ROCm 7.2 driver this output is
a HIP offload fat binary/container, not a standalone ELF AMDGPU code object,
so direct disassembly correctly failed with “file was not recognized”.

The corrected runner follows the linked executable path:

```text
exact P4 executable
→ llvm-objdump --offloading
→ exactly one extracted EM_AMDGPU code object
→ llvm-objdump / llvm-readobj on that extracted HSACO
```

The failed first run made no P5 correctness claim and should not be committed
as evidence.


## v3 correction

`llvm-objdump --offloading` reports offload content but does not materialize
the embedded files in the working directory on this ROCm/LLVM build. The
officially documented HIP extraction route is now used:

```text
linked executable
-> llvm-objcopy --dump-section .hip_fatbin
-> clang-offload-bundler --list
-> exact gfx1201 HIP bundle ID
-> clang-offload-bundler --unbundle
-> standalone EM_AMDGPU HSACO
```

The runner accepts both the plural (`--inputs/--outputs`) and singular
(`--input/--output`) command-line spellings used across Clang versions. It
requires exactly one `gfx1201` HIP bundle ID and records the fatbin,
bundle-ID list, selected target and extracted code object.


## v4 correction

The first successful GFX12 code-object extraction exposed an ISA naming
difference in the audit parser. The parser searched only for the older
`ds_read_*` / `ds_write_*` spellings. The extracted `gfx1201` ISA uses
`ds_load_*` / `ds_store_*`.

The audit now accepts both families:

```text
LDS read  = ds_read_*  or ds_load_*
LDS write = ds_write_* or ds_store_*
```

It also records the complete set of observed `ds_*` mnemonics in JSON, the
report and the console log. No kernel source or mathematical path changed.


## v5 correction

The v4 run exposed two remaining audit-parser defects after all visible core
gates had passed:

1. the extracted-code-object check still searched only for
   `ds_read_*` / `ds_write_*`, even though the main instruction inventory had
   already been updated for GFX12 `ds_load_*` / `ds_store_*`;
2. the optimized LLVM-IR gate incorrectly required the source-level symbol
   name `hidden_lds` to survive optimization. LLVM may rename or eliminate
   that debug/source name while preserving the required `addrspace(3)` LDS
   allocation.

The corrected gates are:

```text
objdump LDS evidence:
    ds_read_* | ds_write_* | ds_load_* | ds_store_*

LLVM LDS evidence:
    kernel identity present AND addrspace(3) present
```

`hidden_lds_name_present` remains recorded as diagnostic information, but is
no longer treated as a correctness requirement. All individual LLVM-IR and
code-object checks are now printed, so a future final FAIL cannot be hidden
behind unreported sub-gates.
