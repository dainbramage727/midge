#!/bin/bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python tools/validate_dataset.py
python -m experiments.midge_experiments.train --config experiments/configs/smoke.toml --dry-run
python -m experiments.midge_experiments.evaluate --config experiments/configs/smoke.toml --split validation --label base --smoke
python -m experiments.midge_experiments.evaluate --config experiments/configs/smoke.toml --split test --label base --smoke
python -m experiments.midge_experiments.evaluate --config experiments/configs/smoke.toml --split gold --label base --smoke
echo "Smoke pipeline passed. Outputs are fixtures, not experimental results."
