#!/usr/bin/env python3
import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "a3validator", ROOT / "tools/validate_phase3da3_100_step_execution_contract_v1.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
LAUNCHER = ROOT / "tools/phase3da3_driver_launcher.py"
S0 = ROOT / "evidence_phase3da_100_step_work/inputs/dense_a_m32/step_0"


def valid_args(output):
    return [
        "python3", str(LAUNCHER), "--validate-arguments-only", "--steps", "100",
        "--case", "dense_a_m32", "--case-id", "train_dense_set_a_m32",
        "--replay-id", "1", "--run-id", "dense_a_m32_replay_1_phase3da3_100",
        "--output-dir", str(output), "--checkpoint-step", "50",
        "--resume-start", "51", "--resume-end", "66",
        "--crosscheck-hash", "e9416fdde6394b0944e7d858332dd52a7826140dfad1c4405e6113a32f1a5748",
        "--output-format", "phase3da3-v1", "--s0-dir", str(S0)
    ]


class ContractAndCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = V.load()

    def test_contract_positive_and_files(self):
        V.validate(self.contract)

    def test_contract_negative_mutations(self):
        V.negative_tests(self.contract)

    def run_invalid(self, mutate):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            args = valid_args(output)
            mutate(args, output)
            before = sorted((path.name, path.read_bytes()) for path in output.iterdir()) if output.exists() else []
            result = subprocess.run(args, cwd=ROOT, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode, 0)
            after = sorted((path.name, path.read_bytes()) for path in output.iterdir()) if output.exists() else []
            self.assertEqual(before, after)

    def test_A1_old_driver_100_returns_3(self):
        value = int((ROOT / "evidence_phase3da3_driver_authorization_work/capability/old_driver_returncode.txt").read_text())
        self.assertEqual(value, 3)

    def test_A2_wrong_driver_hash_contract(self):
        candidate = copy.deepcopy(self.contract)
        V.mutations()["wrong_driver_hash"](candidate)
        with self.assertRaises(ValueError):
            V.validate(candidate, verify_files=False)

    def test_A3_step_zero(self):
        self.run_invalid(lambda a, o: a.__setitem__(a.index("100"), "0"))

    def test_A4_missing_steps(self):
        def mutate(args, output):
            index = args.index("--steps")
            del args[index:index + 2]
        self.run_invalid(mutate)

    def test_A5_wrong_run_id(self):
        def mutate(args, output):
            args[args.index("dense_a_m32_replay_1_phase3da3_100")] = "dense_a_m32_replay_1_100"
        self.run_invalid(mutate)

    def test_A6_wrong_crosscheck_hash(self):
        def mutate(args, output):
            index = args.index("--crosscheck-hash")
            args[index + 1] = "0" * 64
        self.run_invalid(mutate)

    def test_A7_nonempty_output(self):
        def mutate(args, output):
            output.mkdir()
            (output / "occupied").write_text("x")
        self.run_invalid(mutate)

    def test_A8_unknown_output_format(self):
        def mutate(args, output):
            index = args.index("--output-format")
            args[index + 1] = "unknown"
        self.run_invalid(mutate)

    def test_unknown_option(self):
        self.run_invalid(lambda args, output: args.extend(["--unknown-option", "x"]))

    def test_capability_does_not_reserve_or_mutate_S0(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            before = V.sha(S0 / "W_master.fp32.bin")
            result = subprocess.run(valid_args(output), cwd=ROOT, check=True, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            row = json.loads(result.stdout)
            self.assertEqual(row["training_steps_executed"], 0)
            self.assertFalse(row["run_id_reserved"])
            self.assertTrue(row["S0_unchanged"])
            self.assertFalse(output.exists())
            self.assertEqual(before, V.sha(S0 / "W_master.fp32.bin"))


if __name__ == "__main__":
    unittest.main()
