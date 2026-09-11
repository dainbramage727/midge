import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_committed_dataset_validates(self):
        result = subprocess.run([sys.executable, "tools/validate_dataset.py"], cwd=ROOT)
        self.assertEqual(result.returncode, 0)

    def test_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "tools/generate_dataset.py", "--output", tmp], cwd=ROOT)
            self.assertEqual(result.returncode, 0)
            for relative in ["contexts.json", "gold_benchmark.jsonl", "splits/train.jsonl", "splits/validation.jsonl", "splits/test.jsonl"]:
                self.assertEqual((ROOT/"data"/relative).read_bytes(), (Path(tmp)/relative).read_bytes())

    def test_all_executable_targets_are_json_objects(self):
        paths=list((ROOT/"data"/"splits").glob("*.jsonl"))+[ROOT/"data"/"gold_benchmark.jsonl"]
        for path in paths:
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row["expected_behavior"]=="execute": self.assertIsInstance(row["target"],dict)


if __name__ == "__main__": unittest.main()
