# Public-release preflight

No tag or artifact was published by this preflight.

## Gates

```text
PUBLIC_RELEASE_VALIDATION_TAXONOMY: PASS
PUBLIC_RELEASE_LOCAL_PATH_AUDIT: PASS
PUBLIC_RELEASE_BINARY_SCOPE_ATTESTATION: PASS
PUBLIC_RELEASE_COMMAND_SURFACE: BLOCKED
PUBLIC_RELEASE_CLAIM_SCOPE_VALIDATION: PASS
PUBLIC_RELEASE_TIER3_REPORT_CONTRACT: PASS
PUBLIC_RELEASE_ARTIFACT_INVENTORY: PASS
TINY_RDNA4_NN_PUBLIC_RELEASE_PREFLIGHT: BLOCKED
```

## Local-path audit

The public entry points use checkout-relative paths and no workspace sibling.
The fixed developer virtual-environment path in the image-path orchestrator
was replaced with a portable active-venv invariant. Phase-4A0/4A1/4A2/4A3
research READMEs are marked at their beginning:

```text
HISTORICAL_INTERNAL_DOCUMENT
NOT A PUBLIC INSTALLATION GUIDE
```

Remaining local paths in tracked raw evidence, frozen reports, logs, and
machine-readable provenance are historical observations and are not public
installation instructions. They are intentionally preserved byte-for-byte.
Runtime-created temporary directories are portable scratch storage, not
external prerequisites. Archive filenames are inventory identifiers.

## Frozen release inventory

| Component | Tag | Commit | Archive SHA-256 |
|---|---|---|---|
| Phase 3C V2 | `phase3c-portable-smoke-gfx1201-pass-v2` | `884b82f1a83bed98533c51c7ba2ba7bf23cc2c92` | `e21d3e01b8b13b4e1d3ae6d9137ad09df62b5c2ccf44eb972ede73198fe3b3ae` |
| Phase 4A HipBLASLt | `phase4a-hipblaslt-amd-gsplat-integration-gfx1201-pass` | `158da07cda09104b2888334dc66bd5994cd74ddd` | `9629f157ca8f27e6d745936a2d40a3c868fe9df5b9f0e19d5df9733f224c3ea6` |
| Phase 4A2 fused rocWMMA | `phase4a2-fused-rocwmma-amd-gsplat-gfx1201-pass` | `1a9b166c79649ccc494d26b709dccf7b3e3310c7` | `ab9b2c2c8a99c503d630833ce240233f8b8b8183ed40b42831526d777b0e09f8` |

The three SHA-256 sidecars are staged under `release/sha256/`. Phase-4A2's
archive contains `repository.bundle`; the Phase-3C and Phase-4A archives retain
their own frozen contents. Phase-3C V1 remains explicitly historical and
failed; V2 is the release candidate and is not presented as a silent repair.

## Publishing plan

1. Review this commit and command-test evidence.
2. Attach each already-frozen archive with its matching staged sidecar.
3. Publish the three existing annotated tags without moving them.
4. Present README, archive scope, binary manifest, validation contract, and
   Tier-3 template together.
5. Do not describe local validation as Tier 3; populate the compatibility
   matrix only from complete independent reports.

## Clean-clone command test

The native-module command passed from a clean local clone:

```text
PHASE4A2_IMAGE_PATH_BUILD: PASS
```

The Tier-1 command then failed closed before qualification because the execution
environment exposed no ROCm device:

```text
P3B-PROBE-HIP-ERROR: no ROCm-capable device is detected
```

The demo was not run after that environmental precondition failure. Direct GPU
execution could not be retried because elevated execution was unavailable.
Consequently the command-surface and overall gates remain BLOCKED; no technical
qualification failure is inferred. Publishing and pushing remain separate,
explicitly authorized operations.
