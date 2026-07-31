# Public Model-B cleanup report

Generated/resumed: 20260731T101433Z

## Scope

Cleanup was applied or resumed only inside the curated Model-B target. The qualified
development repository was not modified. Files removed from the public view
were moved to the ignored local quarantine:

`.model-b-private/cleanup-quarantine/20260731T101433Z/`

## Results

- Recovery-aware continuation after any earlier partial cleanup: enabled
- Restored required helpers from quarantine: 0
- Runtime/test closure under `scripts/` and `tools/`: 27 files
- Quarantined files: 176
- Remaining files under `scripts/`: 14
- Remaining files under `tools/`: 13
- Public file count, excluding local private state: 326
- Public tree size, excluding local private state: 19M
- Private/local-path findings: 0
- Forbidden runtime-path findings: 0

## Gates

- `PUBLIC_MODEL_B_CLEANUP: PASS`
- `PUBLIC_MODEL_B_LOCAL_PATH_AUDIT: PASS`
- `PUBLIC_MODEL_B_BUILD_SCRIPT_CONVERSION: PASS`
- `PUBLIC_MODEL_B_COMMAND_SURFACE: PASS`

## Remaining intentional blocker

The public Phase-4A2 module build script is not rewritten automatically unless
its old freeze-tarball implementation has already been replaced. The required
future design is:

- current tiny-rdna4-nn checkout as the tcnn source;
- `dependencies/amd-gsplat` as a real submodule;
- amd-gsplat pinned to `2c62b22552c0ad4ed120aae304ce66ae27bc5d08`;
- no tar extraction and no network clone inside the build script.

## Dependency anchor

- amd-gsplat upstream: `https://github.com/Painter3000/amd-gsplat-rocm72-gfx1201`
- required commit: `2c62b22552c0ad4ed120aae304ce66ae27bc5d08`
- Phase-4A2 archive SHA-256:
  `ab9b2c2c8a99c503d630833ce240233f8b8b8183ed40b42831526d777b0e09f8`

## Next step

Convert and review the build script, then initialize the independent Git
repository, add the exact submodules, build, and run the real Fresh-Clone gate.
