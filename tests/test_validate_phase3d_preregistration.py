#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("validator", ROOT / "tools" / "validate_phase3d_preregistration.py")
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)

with open(ROOT / "contracts" / "phase3d_preregistration_contract_v1.json", encoding="utf-8") as f:
    BASE = json.load(f)
with open(ROOT / "contracts" / "phase3d_preregistration_schema_v1.json", encoding="utf-8") as f:
    SCHEMA = json.load(f)


class ContractTests(unittest.TestCase):
    def assertRejected(self, mutate):
        doc = copy.deepcopy(BASE)
        mutate(doc)
        with self.assertRaises(ValueError):
            validator.validate(doc, SCHEMA)

    def test_valid(self):
        validator.validate(copy.deepcopy(BASE), SCHEMA)

    def test_missing_teacher_forced(self):
        self.assertRejected(lambda d: d["oracle"].__setitem__("mode", "free_running"))

    def test_missing_850(self):
        self.assertRejected(lambda d: d["phase3db"].__setitem__("oracle_points", [x for x in d["phase3db"]["oracle_points"] if x != 850]))

    def test_wrong_horizon(self):
        self.assertRejected(lambda d: d["phase3db"].__setitem__("steps", 10000))

    def test_threshold_relaxed(self):
        self.assertRejected(lambda d: d["drift"].__setitem__("hard_fail_e_max_gt", 1.1))

    def test_review_not_blocked(self):
        self.assertRejected(lambda d: d["drift"].__setitem__("review_result", "PASS"))

    def test_start_long_runs_in_phase3d0(self):
        self.assertRejected(lambda d: d["phase3d0"].__setitem__("start_long_runs", True))

    def test_wrong_case_order(self):
        self.assertRejected(lambda d: d["cases"].reverse())

    def test_duplicate_case(self):
        self.assertRejected(lambda d: d["cases"].__setitem__(1, copy.deepcopy(d["cases"][0])))

    def test_bad_commit(self):
        self.assertRejected(lambda d: d["base"].__setitem__("commit", "abc"))

    def test_stateful_stream(self):
        self.assertRejected(lambda d: d["data_stream"].__setitem__("stateful", True))

    def test_allow_threshold_changes(self):
        self.assertRejected(lambda d: d["decisions"].__setitem__("threshold_changes_after_lock_allowed", True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
