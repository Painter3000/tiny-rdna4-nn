#!/usr/bin/env python3
import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/validate_phase3da2_100_step_execution_contract_v1.py"
SPEC = importlib.util.spec_from_file_location("phase3da2_validator", MODULE_PATH)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = V.load_contract()

    def test_positive_contract_and_production_anchors(self):
        V.validate(self.contract)
        V.validate_production_immutability(self.contract)

    def _rejects(self, name):
        candidate = copy.deepcopy(self.contract)
        V.mutations()[name](candidate)
        with self.assertRaises(ValueError):
            V.validate(candidate)

    def test_N1_historical_primary(self): self._rejects("N1_historical_primary")
    def test_N2_two_replays(self): self._rejects("N2_two_replays")
    def test_N3_old_run_id(self): self._rejects("N3_old_run_id")
    def test_N4_old_harness_hash(self): self._rejects("N4_old_harness_hash")
    def test_N5_wrong_transpose(self): self._rejects("N5_wrong_transpose")
    def test_N6_D_as_pass(self): self._rejects("N6_D_as_pass")
    def test_N7_E_fail_as_review(self): self._rejects("N7_E_fail_as_review")
    def test_N8_review_reclassified(self): self._rejects("N8_review_reclassified")
    def test_N9_remove_oracle_50(self): self._rejects("N9_remove_oracle_50")
    def test_N10_resume_50(self): self._rejects("N10_resume_50")
    def test_N11_resume_52(self): self._rejects("N11_resume_52")
    def test_N12_four_trend_points(self): self._rejects("N12_four_trend_points")
    def test_N13_activity_change(self): self._rejects("N13_activity_change")
    def test_N14_replacement_runs(self): self._rejects("N14_replacement_runs")
    def test_N15_reuse_old_D(self): self._rejects("N15_reuse_old_D")
    def test_N16_production_hash(self): self._rejects("N16_production_hash")

    def test_trend_series_and_invalid_point_sets(self):
        V.run_negative_tests(self.contract)


if __name__ == "__main__":
    unittest.main()
