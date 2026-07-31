#!/usr/bin/env python3
import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "triage_validator", ROOT / "tools/validate_phase3da3_trajectory_triage_v1.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class TrajectoryTriageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.attestation = V.load()

    def test_positive_anchors_and_attestation(self):
        V.validate_anchors(self.manifest)
        V.validate_attestation(self.attestation)

    def test_mandatory_negative_test_inventory(self):
        self.assertEqual(len(V.anchor_mutations()), 9)
        self.assertEqual(len(V.attestation_mutations()), 5)
        V.negative_tests(self.manifest, self.attestation)

    def test_each_anchor_and_triage_mutation_rejected(self):
        for name, mutate in V.anchor_mutations().items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                with self.assertRaises((KeyError, ValueError)):
                    V.validate_anchors(candidate, verify_files=True)

    def test_each_metric_attestation_mutation_rejected(self):
        for name, mutate in V.attestation_mutations().items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.attestation)
                mutate(candidate)
                with self.assertRaises((KeyError, ValueError)):
                    V.validate_attestation(candidate, verify_files=False)


if __name__ == "__main__":
    unittest.main()
