# Midge NL→JSON corpus v1.0

This directory contains a reproducible corpus for adapting a small language model to translate a performer's natural-language request into **Midge's existing JSON action contract**. It does not introduce a replacement DSL.

## Contents

| File | Purpose |
|---|---|
| `splits/train.jsonl` | 480 development examples |
| `splits/validation.jsonl` | 80 examples for tuning/early stopping |
| `splits/test.jsonl` | 80 locked development-test examples |
| `gold_benchmark.jsonl` | 48 separately authored, final-report cases |
| `contexts.json` | Frozen aliases, groups, sources, and installed-rule contexts |
| `schema.json` | Machine-readable record schema |
| `dataset_stats.json` | Recomputed counts and content hash |
| `QA_REPORT.md` | Pre-commit adversarial review and remaining risks |

Generate and validate from the repository root:

```bash
python tools/generate_dataset.py
python tools/validate_dataset.py
python -m unittest tests/test_dataset.py
```

The generator seed is `5636`. Committed artifacts are intentional: they make experiments auditable without requiring corpus regeneration on Newton/Stokes.

## Record schema

Each JSONL record includes an immutable ID, split, capability labels, natural-language `utterance`, a `context_ref`, and a `target`. `target` is exactly the JSON object the language model should emit. It is accepted by `llm_parser.validate_action`; for `update_rule`, this correctly retains the public `changes` patch rather than the validator's post-merge runtime representation.

`expected_behavior` is one of:

- `execute`: one supported, unambiguous action; `target` is non-null and `sft_eligible` is true.
- `clarify`: ambiguity or more than one ordered action; `target` is null.
- `reject`: outside Midge's supported MIDI control domain; `target` is null.

Negative examples are benchmark data but are deliberately not direct SFT examples in v1. Midge currently requires exactly one JSON action and defines no safe abstention/clarification action. Inventing a sentinel would be incompatible with the runtime. A future classifier/router or explicit protocol extension can make these trainable.

`semantic_group` is the leakage boundary: paraphrases of the same intent stay in one development split. `template_id` supports analysis of generation families. `tags` identify phenomena such as noisy language, exceptions, compositional rules, and ordering dependencies.

## Design coverage

The corpus covers alias/group/source routing, passthrough, route clearing, note/chord capture, mute/unmute, fixed-note rules, active-chord selection, transposition, every-N conditions, velocity thresholds, active-context exceptions, lifecycle operations, patch updates, noisy speech-like language, unresolved references, multiple ordered actions, and out-of-domain requests.

“Multi-operation” means a single valid runtime rule composed from trigger + condition + derivation + transform + MIDI output. Requests that truly require two state mutations are marked `clarify`, because the current parser's system prompt and validator accept exactly one action object.

The fixed contexts prevent a misleading form of evaluation: a target is only valid if every referenced alias, group, source, and installed rule exists in the supplied context. All executable targets are checked by the production validator.

## Split and leakage policy

The development corpus is 480/80/80. Exact normalized utterance duplicates are forbidden globally. Semantic groups may occur in only one of train/validation/test. Gold IDs and semantic groups are separate from all development data. The validator fails closed for malformed rows, duplicate IDs, normalized duplicate utterances, invalid actions, wrong counts, unknown contexts, or leakage.

The gold set is handwritten and adversarially phrased. Do not use it for prompt selection, epoch selection, hyperparameter tuning, or LoRA training. Run it only for final comparisons. Version or replace the benchmark if its contents become part of iterative development.

## LoRA workflow and evaluation

Export executable chat records with the actual Midge system prompt:

```bash
python tools/export_sft.py --split train --output runs/sft_train.jsonl
python tools/export_sft.py --split validation --output runs/sft_validation.jsonl
```

Recommended experiment:

1. Freeze a base model revision, tokenizer, prompt, decoding settings, and context length.
2. Record zero-shot base predictions on validation/test; tune the LoRA only with train/validation.
3. Merge or load the adapter and rerun the identical test inputs with temperature 0.
4. After choices are frozen, run base and LoRA once on gold.
5. Report exact JSON/action accuracy, action-type accuracy, field-level micro F1, validator pass rate, false-execution rate on negative inputs, latency, and model/adapter size. Break results down by category and tags.
6. Save commands, scheduler/partition, GPU type, seed, package versions, model revision, adapter config, and output JSONL alongside the dataset content hash from `dataset_stats.json`.

Base and LoRA outputs must both pass `extract_json_object` and `validate_action` before semantic scoring. For update actions, compare the validator-produced merged runtime action, not merely the surface patch. Negative cases pass only when a separate safety/router layer chooses clarify/reject; JSON parse failure alone should not receive credit as safe abstention.

## Limitations

Most development data is synthetic and template-generated, so lexical breadth and real performer error patterns remain limited. The context reflects the prototype hardware and vocabulary. There are no audio/ASR artifacts, multi-turn state transitions, program-change/system-exclusive actions, or truly executable multi-action plans. The gold set is small and has one author. Before claiming generalization, collect consented commands from additional musicians, deduplicate them against this corpus, annotate independently, and report inter-annotator agreement.
