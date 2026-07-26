# Phase 4A2-P4 — Production code-object and resource audit

Marker:

```text
TCNN_RDNA4_P4A2_P4_PRODUCTION_CODE_OBJECT_AUDIT_001
```

P4 performs a fresh explicit-on production build, extracts the embedded
`gfx1201` HIP code object from the real `rocwmma_width64_mlp.o`, identifies the
exact production kernel, inventories its ISA and resource metadata, and then
replays the full P3 runtime matrix twice against the fresh extension.

## Bound baseline

```text
Phase 4A2-P3 commit:
de76469

Production source SHA-256:
7b8736534fd94a3d8135a2573a72285dc1e75015794adeeef222e0fd8b5bd6f4

Mapping-header SHA-256:
f7e25b69d3f55c63208e18cece9034bcda54b1114e65a68895c7f8b060ffa517
```

No file under `src/`, `include/`, or `bindings/` is modified by P4.

## Fresh production build

```text
PYTORCH_ROCM_ARCH=gfx1201
TCNN_ENABLE_ROCWMMA_WIDTH64_MLP=1
TCNN_HALF_PRECISION=1
```

The runner records the exact object and linked extension hashes.

## Code-object extraction

The audit supports the HIP layouts used by current ROCm toolchains:

```text
.hip_fatbin
.llvm.offloading
direct clang offload bundle
```

It first uses `clang-offload-bundler` and also contains a parser for the
documented binary clang offload-bundle layout. Exactly one HIP/hipv4 bundle
entry containing `gfx1201` must be extracted as an AMDGPU ELF code object.

## Exact kernel and resource gates

Kernel token:

```text
rocwmma_width64_inference_kernel
```

Required production code-object facts:

```text
exact kernel symbols:            1
group segment fixed size:     2048 bytes
private segment fixed size:      0 bytes
MFMA or WMMA instructions:      12
static LDS load instructions:      8
static LDS store instructions:     2
static ds_bpermute_b32:           192
barrier instructions:             6
scratch instructions:             0
```

Required DS mnemonic inventory:

```text
ds_bpermute_b32
ds_load_b128
ds_store_b128
```

VGPR and SGPR fields are recorded without converting them into an occupancy
claim. Global-memory mnemonics are inventoried without claiming formal
per-instruction pointer provenance.

## Fresh runtime replay

The committed P3 probe runs twice against the new P4 extension:

```text
20 public batch sizes
internal padding 256 through 1024
64-launch bitwise replay
tile-prefix invariance
parameter hot swap A -> B -> A
two models on two non-default streams
existing factory construction
training-forward fail-closed behavior
```

Both JSON results must be byte-identical.

## Run

```bash
cd ~/therock_test/tcnn_rdna4_port/workspace/repos/tiny-cuda-nn-rocwmma-probe

unzip -o \
  ~/Downloads/phase4a2_p4_width64_production_code_object_resource_audit_bundle.zip

chmod +x \
  scripts/audit_phase4a2_p4_code_object.py \
  scripts/finalize_phase4a2_p4.py \
  scripts/run_phase4a2_p4_production_code_object_audit.sh

export PHASE4A2_P3_SOURCE_EVIDENCE="$HOME/therock_test/tcnn_rdna4_port/workspace/evidence/phase4a2_p3_20260726T025720Z"

unset PHASE4A2_P4_EVIDENCE

scripts/run_phase4a2_p4_production_code_object_audit.sh
```

Optional compiler parallelism:

```bash
export MAX_JOBS=4
```

## Expected final markers

```text
WIDTH64_FRESH_PRODUCTION_BUILD: PASS
WIDTH64_GFX1201_CODE_OBJECT_BOUND: PASS
WIDTH64_EXACT_PRODUCTION_KERNEL_BOUND: PASS
WIDTH64_RESOURCE_FIELDS_RECORDED: PASS
WIDTH64_GROUP_2048_PRIVATE_ZERO: PASS
WIDTH64_MFMA12_LDSLOAD8_STORE2_BARRIER6: PASS
WIDTH64_DS_BPERMUTE_B32_192: PASS
WIDTH64_SCRATCH_ZERO: PASS
WIDTH64_FRESH_RUNTIME_REPLAY_TWICE: PASS
WIDTH64_P4_NO_PRODUCTION_CODE_CHANGE: PASS
PHASE4A2_P4_JSON_AUDIT: PASS
PHASE4A2_P4_MAP_CONTEXT: RECORDED
PHASE4A2_P4_WIDTH64_PRODUCTION_CODE_OBJECT_RESOURCE_AUDIT: PASS
```

## Claim boundary

P4 proves the static facts recorded in the extracted production code object and
the runtime behavior covered by the P3 replay. Register counts alone do not
prove occupancy. A mnemonic inventory does not establish formal
per-instruction pointer provenance. No timing or performance claim is made.


## v2 symbol-tool fallback

The first P4 build completed successfully, but the audit stopped before code
object extraction because this ROCm installation does not ship
`/opt/rocm/llvm/bin/llvm-nm`.

`llvm-nm` is not required for the audit. The corrected implementation uses:

```text
llvm-readelf --symbols --wide
```

for the raw defined-symbol inventory and:

```text
llvm-objdump --syms --demangle
```

for the human-readable symbol view. The exact production-kernel symbol gate is
unchanged.

The included resume script reuses the completed fresh P4 object and extension.
It does not rebuild.


## v3 evidence object-path correction

The v2 resume script searched only below:

```text
$EVIDENCE/build/temp
```

The setuptools build command passes source-relative object paths containing
`../../src`. Once normalized, the completed production object is stored below
the evidence root, typically as:

```text
$EVIDENCE/src/rocwmma_width64_mlp.o
```

The original P4 runner already searched the complete evidence tree. The v3
resume script now does the same:

```bash
find "$EVIDENCE" -type f -name 'rocwmma_width64_mlp.o'
```

The successful fresh build is still reused byte-for-byte; no rebuild occurs.


## v4 kernel metadata-companion classification

`llvm-readelf --symbols --wide` reports the executable kernel symbol together
with compiler-generated scalar metadata companions:

```text
<kernel>.has_dyn_sized_stack
<kernel>.has_recursion
<kernel>.num_agpr
<kernel>.num_vgpr
<kernel>.private_seg_size
```

Those suffix symbols are not additional kernels. The corrected audit now:

1. collects all symbols containing the exact production-kernel token;
2. classifies dot-suffixed entries as metadata companions;
3. requires every companion to share the exact raw-kernel prefix;
4. requires exactly one remaining executable raw kernel symbol;
5. still requires exactly one matching disassembly label.

The completed fresh build is reused; no rebuild occurs.


## v5 AMDGPU llvm-objdump syntax correction

The v4 audit successfully extracted the code object, found the exact kernel,
classified its metadata companion symbols, and read the resource metadata.
Every instruction-count gate then reported zero.

The reason is a formatting difference in ROCm's `llvm-objdump`. AMDGPU output
commonly places the mnemonic first and the address/encoding in a trailing
comment:

```text
s_load_dword ... // 0000000000000000: C002...
v_mfma_f32_16x16x16_f16 ...
ds_load_b128 ...
```

The previous parser accepted only address-prefixed lines such as:

```text
0000000000000000: ... s_load_dword ...
```

The v5 parser accepts both formats, strips only trailing encoding comments, and
requires a nonzero parsed instruction count before evaluating the exact ISA
gates.

The completed P4 build and extracted production object are reused. No rebuild
or production-source change occurs.


## v6 production DS-inventory correction

The exact production-kernel ISA reports:

```text
parsed instructions:       943
MFMA/WMMA:                   12
ds_load_b128:                 8
ds_store_b128:                2
ds_bpermute_b32:            192
barriers:                      6
scratch instructions:          0
```

The earlier expectation of 24 LDS-read instructions and an additional
16-bit DS-load mnemonic came from the standalone Phase 4A1 probe kernel. It is
not the instruction shape emitted for the integrated production kernel.

`ds_bpermute_b32` is a cross-lane DS permutation instruction. It is recorded
separately and is not counted as an LDS memory-load instruction.

The corrected production gates require the exact observed static inventory:

```text
8   ds_load_b128
2   ds_store_b128
192 ds_bpermute_b32
```

No production source, mapping, parameter ABI, build artifact, or runtime
behavior is changed. The completed fresh P4 build is reused.


## v7 release reproducibility and prior-evidence equivalence

Two release-closure issues are corrected before publication.

### Committed-tag execution

The original P4 runner was designed to be applied as an uncommitted bundle on
top of the P3 commit. It therefore required `HEAD` to be the P3 commit and the
finalizer obtained the six P4 files from `git status`.

After P4 is committed and tagged, those assumptions are no longer true. The
v7 runner and finalizer support both contexts:

```text
p3_bundle_precommit
p4_release_commit
```

In release mode the finalizer requires:

```text
HEAD subject  = P4 audit commit subject
HEAD^         = P3 commit de76469...
HEAD^ subject = P3 runtime-closure subject
worktree      = clean
HEAD^..HEAD   = exactly the six P4 audit files
```

This makes the committed/tagged audit reproducible from the release commit.

### Byte identity across the audit correction

The release resume preserves the prior successful P4 JSON and both replay JSON
files before regenerating the audit. It then requires byte identity for:

```text
production object
linked Python extension
extracted gfx1201 code object
production source
mapping header
network bridge
runtime replay 1 JSON
runtime replay 2 JSON
```

The contract and audit reports are expected to change because their incorrect
DS classification was corrected. Production artifacts and runtime results are
not allowed to change.
