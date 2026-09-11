import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from experiments.midge_experiments.config import load_config,run_id
from experiments.midge_experiments.data import contexts,load_split,training_rows
from experiments.midge_experiments.scoring import score_rows

ROOT=Path(__file__).resolve().parents[1];CFG=ROOT/"experiments/configs/smoke.toml"
class ExperimentTests(unittest.TestCase):
 def setUp(self):self.cfg=load_config(CFG)
 def test_run_id_is_deterministic(self):
  self.assertEqual(run_id(self.cfg,"train"),run_id(load_config(CFG),"train"))
 def test_training_loader_refuses_test_and_gold(self):
  with self.assertRaises(ValueError):training_rows(self.cfg,"test")
  with self.assertRaises(ValueError):training_rows(self.cfg,"gold")
 def test_cosmetic_rule_fields_do_not_change_semantic_score(self):
  row=next(r for r in load_split(self.cfg,"gold") if r.get("target",{}).get("action")=="install_rule")
  action=deepcopy(row["target"]);action["rule"]["id"]="different-but-valid";action["rule"]["name"]="different label"
  metrics,details=score_rows([row],[{"id":row["id"],"raw_output":json.dumps(action)}],contexts(self.cfg))
  self.assertFalse(details[0]["strict_exact"]);self.assertTrue(details[0]["semantic_correct"])
 def test_negative_requires_explicit_safe_behavior(self):
  row=next(r for r in load_split(self.cfg,"gold") if r["expected_behavior"]=="clarify")
  _,bad=score_rows([row],[{"id":row["id"],"raw_output":"not json"}],contexts(self.cfg))
  _,good=score_rows([row],[{"id":row["id"],"predicted_behavior":"clarify","raw_output":""}],contexts(self.cfg))
  self.assertFalse(bad[0]["semantic_correct"]);self.assertTrue(good[0]["semantic_correct"])
if __name__=="__main__":unittest.main()
