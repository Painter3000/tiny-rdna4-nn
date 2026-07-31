# Phase 3C portable smoke

Run the complete local validation from the repository root:

```bash
./scripts/fresh_clone_user_smoke.sh
```

The command first verifies the platform-independent counter-stream hashes. It
then fresh-builds the gfx1201 production driver and independent hipBLASLt
crosscheck, runs four cases with three 100-step replays, checks the CPU oracle,
local replay determinism, S50-to-S66 resume equivalence, finite values, and
effective updates.

The result is written to `phase3c_smoke_result/`. It contains six compact files,
no binaries and no per-step tensor files. Temporary raw state is deleted before
the command exits.

Tier 1 is the portable gate for the locally built binaries. Tier 2 compares the
four local S100 state hashes with the frozen Phase-3D-A3 reference build. Tier 2
is explicitly informational: `MISMATCH` does not change Tier 1 or the process
return code.

The frozen Phase-3D-A3 qualification applies only to its reference build and is
not transitive to newly built local binaries. A foreign build is validated for
its own environment solely by its Tier-1 smoke result.
