# Midge Dataset v0.1 — base vs. LoRA/QLoRA experiment plan

## Objective and guardrails

Measure whether parameter-efficient adaptation improves translation of musician language into Midge's validated JSON actions without hiding malformed output, negative-input hallucination, or regressions. Dataset v0.1 is frozen at commit `9383d48`. Training reads only `train`; validation loss and validation task metrics select checkpoints/configurations. Test is run after the choice is frozen. Gold is run once for the final comparison and must never influence prompts, hyperparameters, epochs, or checkpoint selection.

No smoke fixture is an experimental result. A result is reportable only when it includes the pinned model revision, dataset hash, code commit, complete TOML configuration, environment lock, scheduler metadata, raw predictions, scored records, and adapter/checkpoint identity.

## Protocol

1. Validate the frozen corpus and archive its SHA-256 from `data/dataset_stats.json`.
2. For each candidate model, run the untuned base model on validation with the exact Midge system prompt and deterministic generation settings.
3. Train LoRA or QLoRA on the 424 SFT-eligible training records. Use only the 69 SFT-eligible validation records for loss, early stopping decisions, and checkpoint selection. Negative validation cases remain evaluation-only because the runtime has no abstention action.
4. Select one adapter/checkpoint per base model using predeclared validation criteria. Prefer semantic accuracy; use malformed-output rate and executable false-action behavior as safety tie-breakers.
5. Freeze all choices, then evaluate base and selected adapter independently on held-out test.
6. Only after the test comparison is complete, evaluate both once on gold. Do not revise from gold errors.
7. Compare aligned scored records to report improvements and base-correct/tuned-wrong regressions.
8. Export every run before cluster access ends and verify the exported archive checksum off-cluster.

## Metrics

- Strict JSON exact match against the canonical wire target.
- Semantic/execution-field accuracy after Midge validation, excluding cosmetic rule `id`, `name`, and `version`.
- Mean execution-field score for partially correct actions.
- Validator-pass and malformed-output rates.
- Results by capability category and by executable/clarify/reject expectation.
- Improvement and regression examples from aligned base/tuned predictions.

Negative examples require an explicit router decision of `clarify` or `reject`. A parse failure is malformed, not a successful abstention. Until Midge gains such a router, base/LoRA generation alone is expected to expose—not solve—the negative-input gap.

## Candidate matrix

Start small enough to debug cheaply, then scale only if the pipeline and allocation support it.

| Axis | Initial candidates |
|---|---|
| Base model | One approximately 1–2B instruct model; optionally one 3–4B and one 7–8B instruct model after resource verification |
| Adaptation | Base/no tuning; LoRA in native precision; 4-bit NF4 QLoRA |
| Rank / alpha | 8/16 and 16/32 |
| Dropout | 0.0 and 0.05 |
| Learning rate | 1e-4 and 2e-4 |
| Epoch cap | 3 initially; at most 5 if validation is still improving |
| Seeds | 5636 for pipeline/debug; at least three declared seeds for any stability claim resources permit |
| Generation | Greedy, temperature 0, one beam; identical settings for base and tuned |

This is a candidate matrix, not authorization for a full grid. Run a single small pilot, inspect memory/runtime and learning curves, then prune combinations before spending cluster allocation.

## Stopping and selection criteria

- Stop a run on non-finite loss, repeated OOM, corrupt checkpoints, data-hash mismatch, or validator/prompt construction failure.
- Do not rescue OOM by silently changing configuration; create a new config/run ID.
- Use validation loss/checkpoints during training. For final adapter selection, require improved validation semantic accuracy over base without a material increase in malformed rate.
- Treat any rise in base-correct/tuned-wrong cases as a regression budget to inspect, not something averaged away.
- Stop scaling model size when improvement is negligible relative to run-to-run variation or operational cost.

## Reproducibility requirements

- Pin model repository revision SHA; never report a run using `main` or a placeholder.
- Record code commit, clean/dirty state, dataset hash, seed, tokenizer/chat template, package lock, Python/platform, precision, quantization, adapter targets, generation settings, and checkpoint.
- Keep raw outputs. Re-scoring must not require model inference.
- Never overwrite a run with changed settings; configuration plus dataset/code identity determines the run ID.
- Archive adapters, tokenizer files, checkpoints retained for resumption, trainer state, logs, predictions, metrics, comparisons, environment metadata, and scheduler output.

## Verify on Newton/Stokes before the first real job

- Correct login/transfer workflow and whether compute jobs may access Hugging Face; pre-stage model weights if not.
- Approved Slurm account/allocation, partition, GPU request syntax and available GPU types, maximum wall time, CPU/memory syntax, and job-output location.
- Available CUDA/driver version and which PyTorch/bitsandbytes builds are compatible.
- Whether `bf16` is supported; otherwise create a reviewed `fp16` config.
- Python/conda/module procedure. The templates intentionally contain no invented module names.
- Scratch/project storage paths, quotas, purge policy, checkpoint frequency, and the durable destination for exports.
- Model license/access requirements and Hugging Face cache placement.
- Selected architecture's actual LoRA target-module names and tokenizer chat template.
- A smoke job, then one short real training job with a deliberately tiny step limit before the full pilot.
- Off-cluster copy route and checksum verification while access is still active.

Cluster-specific placeholders in `experiments/slurm/*.template` must be replaced only with verified values.
