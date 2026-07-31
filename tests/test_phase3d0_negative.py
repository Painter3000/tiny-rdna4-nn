import copy,shutil,tempfile,unittest
from pathlib import Path
from tools.validate_phase3d0_bridge import SEQUENCE,audit_ledger,validate_files
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT.parent/"tests/reference/tier1_initial_states/replays/dense_a_m32_r1/four_step/raw_states/train_dense_set_a_m32"
ACT=ROOT/"evidence_phase3d0_work/bridge_runs/full/train_dense_set_a_m32/replay_1"
BASE={"events":SEQUENCE,"adam_count":1,"cast_complete":True,"state_hash_event_index":6,
      "optimizer_step":2,"step":2,"beta1_power_advanced":True,"input_hash_match":True,
      "fresh_replay_initialization_count":1,"expected_hash_match":True}
class Negatives(unittest.TestCase):
 def mutate_file(self,name,source):
  td=Path(tempfile.mkdtemp());shutil.copytree(ACT/"step_2",td/"step")
  shutil.copy2(source,td/"step"/name);self.assertTrue(validate_files(REF/"step_2",td/"step"));shutil.rmtree(td)
 def ledger(self,fn,reason):
  x=copy.deepcopy(BASE);fn(x);self.assertIn(reason,audit_ledger(x))
 def test_n01_reset_compute(self):self.mutate_file("W_compute.fp16.bin",REF/"step_0/W_compute.fp16.bin")
 def test_n02_reset_master(self):self.mutate_file("W_master.fp32.bin",REF/"step_0/W_master.fp32.bin")
 def test_n03_double_adam(self):self.ledger(lambda x:x.update(adam_count=2),"adam_count")
 def test_n04_skip_cast(self):self.ledger(lambda x:x.update(cast_complete=False),"cast_missing")
 def test_n05_wrong_input(self):self.ledger(lambda x:x.update(input_hash_match=False),"input_hash")
 def test_n06_swap_dw_layers(self):
  td=Path(tempfile.mkdtemp());shutil.copytree(ACT/"step_2",td/"step");p=td/"step/dW.fp32.bin";b=p.read_bytes();p.write_bytes(b[16384:32768]+b[:16384]+b[32768:]);self.assertTrue(validate_files(REF/"step_2",td/"step"));shutil.rmtree(td)
 def test_n07_beta_not_advanced(self):self.ledger(lambda x:x.update(beta1_power_advanced=False),"beta1_power")
 def test_n08_optimizer_not_advanced(self):self.ledger(lambda x:x.update(optimizer_step=1),"optimizer_step")
 def test_n09_stale_forward(self):self.mutate_file("forward.fp16.bin",ACT/"step_1/forward.fp16.bin")
 def test_n10_hash_before_cast(self):self.ledger(lambda x:x.update(state_hash_event_index=4),"hash_before_cast")
 def test_n11_reuse_process(self):self.ledger(lambda x:x.update(fresh_replay_initialization_count=0),"replay_initialization")
 def test_n12_expected_hash(self):self.ledger(lambda x:x.update(expected_hash_match=False),"expectation_hash")
