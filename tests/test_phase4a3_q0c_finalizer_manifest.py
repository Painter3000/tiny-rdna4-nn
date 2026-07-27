#!/usr/bin/env python3
"""Regression tests for the Q0c worker manifest and strict finalizer gate."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from finalize_phase4a3_q0c import validate_worker_manifest  # noqa: E402
from phase4a3_q0c_common import MARKER, load_contract, matrix  # noqa: E402
from phase4a3_q0c_worker_manifest import (  # noqa: E402
    initialize,
    load_manifest,
    record,
    worker_id,
)


class Q0cFinalizerManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract_path = ROOT / "contracts" / "phase4a3_q0c_apparatus_contract.json"
        self.contract = load_contract(self.contract_path)
        self.temporary = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temporary.name)
        self.workers = self.base / "workers"
        self.workers.mkdir()
        self.matrix_path = self.base / "matrix.tsv"
        rows = []
        for item in matrix(self.contract):
            rows.append(
                "\t".join(
                    str(item.get(key, "-"))
                    for key in (
                        "phase",
                        "schedule",
                        "batch",
                        "metric",
                        "process_index",
                        "start_order",
                    )
                )
            )
        self.matrix_path.write_text("\n".join(rows) + "\n")
        self.manifest = self.base / "worker_manifest.json"
        initialize(self.matrix_path, self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_result(self, item: dict[str, object]) -> str:
        ident = worker_id(item)
        value = {"marker": MARKER, **item}
        (self.workers / f"{ident}.json").write_text(json.dumps(value) + "\n")
        (self.workers / f"{ident}.log").write_text("synthetic test worker\n")
        return ident

    def populate(self, failing_id: str | None = None, failing_rc: int = 134) -> None:
        for pid, item in enumerate(matrix(self.contract), 10000):
            ident = self.write_result(item)
            record(
                self.manifest,
                self.workers,
                ident,
                pid,
                failing_rc if ident == failing_id else 0,
            )

    def test_exact_100_rc0_workers_pass_manifest_gate(self) -> None:
        self.populate()
        audit, values = validate_worker_manifest(
            self.manifest, self.workers, self.contract
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["expected_count"], 100)
        self.assertEqual(audit["returncode_zero_count"], 100)
        self.assertEqual(audit["accepted_count"], 100)
        self.assertEqual(len(values), 100)

    def test_rc134_is_rejected_even_when_json_exists(self) -> None:
        target = "G_spin_native256_p1_BA"
        self.populate(failing_id=target, failing_rc=134)
        self.assertTrue((self.workers / f"{target}.json").is_file())

        audit, values = validate_worker_manifest(
            self.manifest, self.workers, self.contract
        )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["expected_count"], 100)
        self.assertEqual(audit["manifest_count"], 100)
        self.assertEqual(audit["returncode_zero_count"], 99)
        self.assertEqual(audit["json_recorded_present_count"], 100)
        self.assertEqual(audit["accepted_count"], 99)
        self.assertEqual(len(values), 99)
        failure = next(
            item for item in audit["failures"] if item.get("worker_id") == target
        )
        self.assertEqual(failure["returncode"], 134)
        self.assertIn("returncode_not_zero", failure["reasons"])

    def test_missing_json_is_rejected(self) -> None:
        items = matrix(self.contract)
        for pid, item in enumerate(items, 20000):
            ident = worker_id(item)
            if ident != "G_spin_native256_p6_AB":
                self.write_result(item)
            else:
                (self.workers / f"{ident}.log").write_text("aborted\n")
            record(self.manifest, self.workers, ident, pid, 0)
        audit, values = validate_worker_manifest(
            self.manifest, self.workers, self.contract
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["accepted_count"], 99)
        self.assertEqual(len(values), 99)

    def test_g_result_identity_prefers_metric_over_internal_batch(self) -> None:
        value = {
            "phase": "G",
            "schedule": "spin",
            "batch": 256,
            "metric": "native256",
            "process_index": 1,
            "start_order": "BA",
        }
        self.assertEqual(worker_id(value), "G_spin_native256_p1_BA")

    def test_manifest_record_is_single_assignment(self) -> None:
        first = matrix(self.contract)[0]
        ident = self.write_result(first)
        record(self.manifest, self.workers, ident, 30000, 0)
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            record(self.manifest, self.workers, ident, 30000, 0)
        loaded = load_manifest(self.manifest)
        entry = next(item for item in loaded["workers"] if item["worker_id"] == ident)
        self.assertEqual(entry["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
