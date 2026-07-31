#!/usr/bin/env python3
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "metric_trajectory_validator",
    ROOT / "tools/validate_phase3da3_metric_trajectory_addendum_v1.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class MetricTrajectoryAddendumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addendum = V.load()
        cls.manifest = json.loads(V.MANIFEST_PATH.read_text())

    def test_positive_contract_and_files(self):
        V.validate(self.addendum, self.manifest)

    def test_all_mandatory_negative_mutations(self):
        self.assertEqual(set(V.mutations()), {
            "M1_flip_historical_hash_bit", "M2_only_replay_1", "M3_update_after_run",
            "M4_blocked_as_primary", "M5_merge_backend_roles", "M6_omit_global_backend_role",
            "M7_combined_bound_pass_gate", "M8_pair_different_indices", "M9_one_backend_trend",
            "M10_four_point_trend", "M11_accept_0_94", "M12_256_pass",
            "M13_recovery_pass", "M14_crosscheck_write",
        })
        V.negative_tests(self.addendum, self.manifest)

    def test_each_negative_mutation_is_rejected_independently(self):
        for name, mutate in V.mutations().items():
            with self.subTest(name=name):
                addendum = copy.deepcopy(self.addendum)
                manifest = copy.deepcopy(self.manifest)
                mutate(addendum, manifest)
                with self.assertRaises(ValueError):
                    V.validate(addendum, manifest, verify_files=False)


if __name__ == "__main__":
    unittest.main()
