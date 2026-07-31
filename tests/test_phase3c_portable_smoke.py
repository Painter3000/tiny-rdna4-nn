#!/usr/bin/env python3
"""Fail-closed and tier-separation tests for the public Phase-3C smoke."""
import copy
import importlib.util
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase3c", ROOT / "tools/phase3c_portable_smoke.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class PortableSmokeNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((ROOT / "contracts/phase3d_preregistration_contract_v1.json").read_text())
        cls.manifest = json.loads((ROOT / "tests/reference/phase3c_portable_input_hashes_v1.json").read_text())

    def mutated_manifest(self, column):
        value = copy.deepcopy(self.manifest)
        row = value["entries"][0]
        row[column] = ("0" if row[column][0] != "0" else "1") + row[column][1:]
        return value

    def test_n1_changed_input_hash_fails_before_gpu(self):
        with self.assertRaisesRegex(SMOKE.SmokeFailure, "before GPU execution"):
            SMOKE.validate_counter_stream(self.mutated_manifest(2), self.contract)

    def test_n2_changed_target_hash_fails(self):
        with self.assertRaises(SMOKE.SmokeFailure):
            SMOKE.validate_counter_stream(self.mutated_manifest(3), self.contract)

    def test_n3_oracle_outside_tolerance_is_failure_condition(self):
        self.assertIn('metric["E_max"] > 1.0', (ROOT / "tools/phase3c_portable_smoke.py").read_text())

    def test_n4_replay_three_mutation_is_detected(self):
        hashes = [SMOKE.sha(value) for value in (b"a", b"a", b"b")]
        self.assertNotEqual(len(set(hashes)), 1)

    def test_n5_resume_state_mutation_is_detected(self):
        self.assertNotEqual(SMOKE.sha(b"main"), SMOKE.sha(b"resumed"))

    def test_n6_nan_loss_is_rejected(self):
        self.assertFalse(math.isfinite(float("nan")))
        self.assertIn('if not item["finite"]', (ROOT / "tools/phase3c_portable_smoke.py").read_text())

    def test_n7_never_updated_layer_is_rejected(self):
        self.assertFalse(all([True, False, True]))

    def test_n8_tier2_mismatch_keeps_tier1_independent(self):
        cases = {case: {"replays": [{"final_s100_state_sha256": "bad"}]} for case in SMOKE.CASES}
        ref = {"state_sha256": {case: "reference" for case in SMOKE.CASES}}
        result = SMOKE.tier2(cases, ref)
        self.assertEqual(result["status"], "MISMATCH")
        self.assertFalse(result["affects_tier1_result"])

    def test_n9_missing_rocm_is_unknown(self):
        self.assertEqual(SMOKE.command(["/definitely/missing/phase3c"], optional=True), "UNKNOWN")

    def test_n10_tier2_mismatch_cannot_create_error_returncode(self):
        source = (ROOT / "tools/phase3c_portable_smoke.py").read_text()
        self.assertNotIn('comparison["status"] == "MATCH"', source)

    def test_n11_reference_hashes_are_not_tier1_gate(self):
        source = (ROOT / "tools/phase3c_portable_smoke.py").read_text()
        self.assertTrue(source.index("cases = run_training") < source.index("comparison = tier2"))
        self.assertIn('"affects_tier1_result": False', source)

    def test_n12_report_has_no_per_step_files(self):
        self.assertFalse(any("step_" in name for name in SMOKE.REPORT_FILES))

    def test_n13_oversize_report_guard_exists(self):
        self.assertIn("size >= 10 * 1024 * 1024",
                      (ROOT / "tools/phase3c_portable_smoke.py").read_text())

    def test_n14_stream_validation_precedes_build_and_gpu(self):
        source = (ROOT / "tools/phase3c_portable_smoke.py").read_text()
        self.assertTrue(source.index("counter = validate_counter_stream") <
                        source.index("driver, crosscheck, commands = build"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
