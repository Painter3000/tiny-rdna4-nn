# Phase 4A3 Q0c worker-manifest finalizer repair

Marker: `TCNN_RDNA4_P4A3_Q0C_WORKER_MANIFEST_FIX_001`

## Purpose

This repair closes the fail-closed gap observed in the Q0c run where a worker wrote a result JSON and then terminated with `SIGABRT` / return code 134. A result file alone is no longer sufficient for acceptance.

## Acceptance rule

Every one of the exact 100 frozen Q0c workers must satisfy all of the following:

1. The worker ID is present exactly once in the initialized manifest.
2. The worker process reached a recorded terminal state.
3. The recorded process return code is exactly `0`.
4. The expected result JSON exists.
5. The result JSON SHA-256 matches the hash captured when the process ended.
6. The result JSON marker and worker identity match the manifest entry.
7. No unexpected worker JSON exists.

The overall Q0c decision is blocked whenever the manifest gate fails, even when enough remaining JSON files would otherwise satisfy a subgroup's statistical minimum.

## Files

- `scripts/phase4a3_q0c_worker_manifest.py`
  - Initializes the exact worker matrix before execution.
  - Records PID, return code, JSON presence and JSON SHA-256 atomically after every process.
  - Rejects duplicate terminal assignment.
- `scripts/run_phase4a3_q0c_apparatus.sh`
  - Runs each worker as a separately tracked child process.
  - Waits for the child, captures its real PID and return code, and records both in the manifest.
  - Passes the manifest to the finalizer.
- `scripts/finalize_phase4a3_q0c.py`
  - Reconstructs the frozen 100-worker matrix from the contract.
  - Validates the manifest and only admits RC-0, identity-matching, hash-matching worker results.
  - Includes a `worker_manifest` audit block in the final apparatus JSON.
- `tests/test_phase4a3_q0c_finalizer_manifest.py`
  - Includes the required negative regression: a JSON exists, but the process return code is 134; the finalizer must reject it.

## Scope

This is an apparatus-only repair. It does not change the production rocWMMA kernel, the HipBLASLt reference implementation, the Q0c statistical thresholds, or any previously captured evidence.
