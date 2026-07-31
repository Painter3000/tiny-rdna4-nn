import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class StaticTests(unittest.TestCase):
 def test_frozen_sources_immutable(self):
  b=json.loads((ROOT/"evidence_phase3d0_work/source_baseline.json").read_text())
  for x in b["sources"]:
   self.assertEqual(x["expected_sha256"],hashlib.sha256((ROOT/x["path"]).read_bytes()).hexdigest())
 def test_markers(self):
  text="".join((ROOT/p).read_text() for p in ("src/impl/phase3d_inprocess_loop.cpp","src/impl/phase3d_inprocess_driver.cpp"))
  self.assertIn("P3D0-LOOP-",text);self.assertIn("P3D0-BRIDGE-",text)
 def test_new_driver_only_orchestrates_frozen_kernels(self):
  text=(ROOT/"src/impl/phase3d_inprocess_driver.cpp").read_text()
  self.assertIn("phase3a_fused_backward.hip",text);self.assertIn("phase3b_adam_update.hip",text)
  self.assertNotIn("__global__ void",text)
