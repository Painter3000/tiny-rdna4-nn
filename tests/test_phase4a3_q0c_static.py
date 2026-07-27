from __future__ import annotations
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase4a3_q0c_common import MAGIC, calibrate_count, enumerate_bundles, load_contract, matrix, padded_batch, symbol_lines
from capture_phase4a3_q0c_build_object import capture
from check_phase4a3_q0c_provenance import validate_build_object


class Q0cTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_path = ROOT / "contracts/phase4a3_q0c_apparatus_contract.json"
        cls.contract = load_contract(cls.contract_path)

    def make_bundle(self, prefix: bytes, ident: bytes, payload: bytes) -> bytes:
        header_size = len(MAGIC) + 8 + 24 + len(ident)
        offset = len(prefix) + header_size
        return prefix + MAGIC + struct.pack("<Q", 1) + struct.pack("<QQQ", offset, len(payload), len(ident)) + ident + payload

    def test_multibundle_and_kernel_not_first(self):
        first = self.make_bundle(b"", b"hip-amdgcn-gfx1201", b"\x7fELF-no-target")
        second = self.make_bundle(first + b"padding", b"hip-amdgcn-gfx1201", b"\x7fELF-rocwmma_width64_inference_kernel")
        bundles = enumerate_bundles(second)
        self.assertEqual(len(bundles), 2)
        self.assertNotIn(b"rocwmma_width64_inference_kernel", bundles[0]["entries"][0]["payload"])
        self.assertIn(b"rocwmma_width64_inference_kernel", bundles[1]["entries"][0]["payload"])

    def test_local_symbol(self):
        fixture = "12: 0000 120 FUNC LOCAL DEFAULT 7 _GLOBAL__N_1foo_rocwmma_width64_inference_kernel"
        self.assertEqual(len(symbol_lines(fixture, "rocwmma_width64_inference_kernel")), 1)

    def test_padding(self):
        self.assertEqual([(x, padded_batch(x)) for x in (1,31,128,257)], [(1,256),(31,256),(128,256),(257,512)])

    def test_calibration_bounds_and_freeze_value(self):
        self.assertEqual(calibrate_count(0.01, 300, 4096, 65536), 30000)
        self.assertEqual(calibrate_count(100, 300, 4096, 65536), 4096)
        self.assertEqual(calibrate_count(0.0001, 300, 4096, 65536), 65536)

    def test_patch_build_restore_cycle_preserves_production_file(self):
        source = ROOT / "bindings/torch/tinycudann/bindings.cpp"
        patcher = ROOT / "scripts/patch_phase4a3_q0c_binding.py"
        with tempfile.TemporaryDirectory() as raw:
            temp = pathlib.Path(raw)
            fixture, backup = temp / "bindings.cpp", temp / "backup.cpp"
            fixture.write_bytes(source.read_bytes())
            original = fixture.read_bytes()
            subprocess.run([sys.executable, str(patcher), "--file", str(fixture), "--backup", str(backup), "--mode", "apply"], check=True)
            self.assertNotEqual(fixture.read_bytes(), original)
            subprocess.run([sys.executable, str(patcher), "--file", str(fixture), "--backup", str(backup), "--mode", "restore"], check=True)
            self.assertEqual(fixture.read_bytes(), original)

    def test_complete_dry_matrix(self):
        items = matrix(self.contract)
        self.assertEqual(len(items), 8 + 24 + 32 + 20 + 16)
        self.assertIn({"phase":"TP","schedule":"spin","batch":257,"process_index":0,"start_order":"AB"}, items)

    def test_setuptools_sibling_object_exact_path_handshake(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = pathlib.Path(raw)
            (fixture / "build/temp").mkdir(parents=True)
            obj = fixture / "src/rocwmma_width64_mlp.o"
            obj.parent.mkdir()
            obj.write_bytes(b"exact-q0c-object")

            captured = capture(fixture, self.contract_path)

            self.assertEqual(captured, obj.resolve())
            self.assertFalse(captured.is_relative_to((fixture / "build/temp").resolve()))
            self.assertEqual(
                (fixture / "provenance/build_object_path.txt").read_text().strip(),
                str(obj.resolve()),
            )
            self.assertEqual(
                validate_build_object(captured, "rocwmma_width64_mlp.o"),
                obj.resolve(),
            )

    def test_build_object_capture_rejects_duplicate_basename(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = pathlib.Path(raw)
            for relative in ("src/rocwmma_width64_mlp.o", "build/temp/rocwmma_width64_mlp.o"):
                path = fixture / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"duplicate")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                capture(fixture, self.contract_path)

    def test_checker_rejects_wrong_object_basename(self):
        with tempfile.TemporaryDirectory() as raw:
            obj = pathlib.Path(raw) / "wrong.o"
            obj.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "unexpected Q0c build object basename"):
                validate_build_object(obj, "rocwmma_width64_mlp.o")

    def test_finalizer_preserves_independent_lp_when_td256_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            temp = pathlib.Path(raw); workers = temp / "workers"; workers.mkdir()
            provenance = temp / "p.json"; output = temp / "out.json"
            provenance.write_text(json.dumps({"marker": self.contract["marker"], "decision": self.contract["decisions"]["P_pass"]}))
            for index, order in enumerate(("AB","BA","AB","BA")):
                score = {"passed": True, "round_median_ns": 10}
                conv = {"passed": True}
                rounds = [{"ratio": 1.0, "backends": {"candidate":{"convergence":conv,"score":score},"reference":{"convergence":conv,"score":score}}} for _ in range(4)]
                value = {"marker":self.contract["marker"],"phase":"LP","schedule":"spin","batch":1,"metric":None,"process_index":index,"start_order":order,"pre_correctness":{"passed":True},"post_correctness":{"passed":True},"rounds":rounds}
                (workers / f"lp{index}.json").write_text(json.dumps(value))
            blocks = [{"iterations":64,"event_ms":1,"host_total_ns":1_000_000,"submission_over_gpu":0.9}]
            for index, order in enumerate(("AB","BA","AB","BA")):
                rounds = [{"backends":{"candidate":{"blocks":blocks},"reference":{"blocks":blocks}}} for _ in range(4)]
                value = {"marker":self.contract["marker"],"phase":"TD","schedule":"spin","batch":256,"metric":None,"process_index":index,"start_order":order,"pre_correctness":{"passed":True},"post_correctness":{"passed":True},"rounds":rounds}
                (workers / f"td{index}.json").write_text(json.dumps(value))
            subprocess.run([sys.executable, str(ROOT/"scripts/finalize_phase4a3_q0c.py"), "--contract", str(self.contract_path), "--workers", str(workers), "--provenance", str(provenance), "--output", str(output)], env={"PYTHONPATH":str(ROOT/"scripts")}, check=False)
            result = json.loads(output.read_text())
            self.assertTrue(result["groups"]["LP/spin/batch1"]["passed"])
            self.assertFalse(result["groups"]["TD/spin/batch256"]["passed"])


if __name__ == "__main__":
    unittest.main()
