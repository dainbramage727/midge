# Midge experiment harness

This package trains and evaluates configurable Hugging Face causal/instruction models without changing Midge's JSON action contract. Heavy ML dependencies are imported only for real model runs; dataset, scoring, comparison, export, and smoke checks remain usable on a CPU-only development machine.

## Commands

```bash
# No model download or training
bash experiments/run_smoke.sh

# Train; train split only, validation for checkpoint selection
python -m experiments.midge_experiments.train \
  --config experiments/configs/qlora_example.toml

# Resume an interrupted run
python -m experiments.midge_experiments.train \
  --config experiments/configs/qlora_example.toml \
  --resume-from-checkpoint /verified/path/to/checkpoint-N

# Evaluate each split independently
python -m experiments.midge_experiments.evaluate --config CONFIG.toml --split validation --label base
python -m experiments.midge_experiments.evaluate --config CONFIG.toml --split test --label base
python -m experiments.midge_experiments.evaluate --config CONFIG.toml --split gold --label base
python -m experiments.midge_experiments.evaluate --config CONFIG.toml --split test --label lora --adapter ADAPTER_DIR

# Compare aligned base and tuned scored records
python -m experiments.midge_experiments.compare \
  --base BASE_RUN/scored_records.jsonl --tuned LORA_RUN/scored_records.jsonl \
  --output COMPARISON_DIR

# Export a complete run and optional scheduler log
python -m experiments.midge_experiments.export_artifacts \
  --run-dir RUN_DIR --destination SAFE_DESTINATION --include VERIFIED_SLURM_LOG
```

Every real evaluation writes `predictions.jsonl`, `scored_records.jsonl`, `metrics.json`, and `environment.json`. Training writes checkpoints, the selected adapter/tokenizer, trainer state, embedded configuration/environment metadata, and a run manifest. Export creates a tarball, per-file manifest, and archive SHA-256 checksum.

Smoke outputs carry `not_an_experimental_result: true` and never load a model. They verify configuration, split loading, prompt construction, JSON parsing, semantic scoring, metadata, and artifact creation—not model quality.

The example model and LoRA target modules are configuration values, not code assumptions. Pin the model revision and verify module names, dtype support, and tokenizer chat template for every candidate.
