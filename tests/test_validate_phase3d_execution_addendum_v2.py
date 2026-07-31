import copy,importlib.util,json,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("addendum_validator",ROOT/"tools/validate_phase3d_execution_addendum_v2.py")
V=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(V)
BASE=json.loads((ROOT/"contracts/phase3d_execution_contract_addendum_v2.json").read_text())
SCHEMA=json.loads((ROOT/"contracts/phase3d_execution_contract_addendum_v2.schema.json").read_text())
class Addendum(unittest.TestCase):
 def reject(self,fn):
  x=copy.deepcopy(BASE);fn(x)
  with self.assertRaises(ValueError):V.validate(x,SCHEMA,ROOT)
 def test_valid(self):V.validate(copy.deepcopy(BASE),SCHEMA,ROOT)
 def test_d_cannot_be_fail(self):self.reject(lambda x:x["drift"].update(d_max_ge_result="FAIL"))
 def test_d_threshold(self):self.reject(lambda x:x["drift"].update(d_max_ge=1.01))
 def test_combined_bound(self):self.reject(lambda x:x["drift"].update(d_combined_bound=2.0))
 def test_e_remains_fail(self):self.reject(lambda x:x["drift"].update(e_max_gt_result="MANUAL_REVIEW_BLOCKED"))
 def test_replay_count_a(self):self.reject(lambda x:x["phase3da"].update(fresh_complete_replays_per_case=2))
 def test_replay_count_b(self):self.reject(lambda x:x["phase3db"].update(fresh_complete_replays_per_case=2))
 def test_noop_rule(self):self.reject(lambda x:x["activity_rules"].update(layer_consecutive_no_op_steps_blocked_at=255))
 def test_effective_fraction(self):self.reject(lambda x:x["activity_rules"].update(global_effective_step_fraction_minimum=.9))
 def test_beta_semantics(self):self.reject(lambda x:x["beta_power"].update(evaluation="direct_pow"))
 def test_beta_fixed_point(self):self.reject(lambda x:x["beta_power"].update(beta1_fixed_point_first_step=965))
 def test_parent_mutation(self):self.reject(lambda x:x["parent_contract"].update(mutation_allowed=True))
 def test_review_not_pass(self):self.reject(lambda x:x["review_policy"].update(review_may_be_reclassified_to_pass_in_same_run=True))
 def test_missing_trend_series(self):self.reject(lambda x:x.update(trend_validator_required_series=["flat","falling","rising"]))
 def test_flat_trend(self):self.assertFalse(V.trend_review([.2,.2,.2,.2,.2],.6))
 def test_falling_trend(self):self.assertFalse(V.trend_review([.8,.6,.4,.2,.1],.6))
 def test_rising_trend(self):self.assertTrue(V.trend_review([.1,.2,.3,.4,.5],.5))
 def test_noisy_trend(self):self.assertTrue(V.trend_review([.1,.21,.19,.4,.6],.6))
 def test_four_points_rejected(self):
  with self.assertRaises(ValueError):V.trend_review([.1,.2,.3,.4],.6)
if __name__=="__main__":unittest.main(verbosity=2)
