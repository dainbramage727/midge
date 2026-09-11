# Dataset QA review — pre-commit

Review date: 2026-09-11  
Reviewer stance: skeptical senior-ML-engineering review

## Scope

A deterministic stratified sample of 92 records was manually inspected after the final revisions: four records from every split/category bucket (23 buckets spanning train, validation, test, and all five gold categories). Earlier review passes inspected an additional 128-record stratified sample and all 48 original gold records. Automated corpus-wide checks covered all 688 records.

Special review targets included alias/group/source resolution, rule installation, `update_rule` wire semantics, noisy inputs, velocity and state exceptions, contextual rule references, ambiguity, unsupported multi-action requests, out-of-domain boundaries, and gold/train independence.

## Findings and changes

- Replaced arbitrary `generated-####` rule IDs and names with descriptions grounded in the requested trigger and behavior. Fixed-note IDs no longer contain an irrelevant chord group.
- Corrected unnatural generated language (`1th`, `2th`, `3th`, `every 1 hits`) and removed collision suffixes such as `for setup 2`.
- Diversified noisy language. The original robustness set repeatedly used the same `rout teh ... chanel` corruption; it now includes disfluencies, repairs, abbreviations, varied misspellings, and compressed speech-like constructions.
- Rebuilt the gold benchmark. Atomic cases no longer share one routing template; gold now includes routing, groups, passthrough, muting, clearing, capture, heterogeneous compositional rules, eight contextual/lifecycle cases (including five `update_rule` cases), varied noise, ambiguous references, ordered multi-action requests, and near-domain unsupported MIDI/audio requests.
- Corrected a semantic generation bug in which an utterance saying `velocity 64 or harder` could target `max_velocity: 72`. Corpus validation now checks numeric `or harder`/`or softer` alignment.
- Replaced redundant synthetic active-chord “exceptions” with meaningful `fill_mode == true` state guards. Active-context constraints remain only where the text explicitly asks for them.
- Reworked split allocation while keeping semantic paraphrase groups intact. Validation and test now both contain ambiguity, multiple-action inputs, out-of-domain inputs, `update_rule`, and enable/disable/delete lifecycle operations.
- Grouped repeated negative paraphrases by underlying intent so leakage checking is substantive rather than dependent on unique per-row group labels.
- Added corpus-wide lint failures for arbitrary generated IDs, cleanup suffixes, and malformed ordinals.

## Final checks

- Development split sizes: 480 train / 80 validation / 80 test
- Gold benchmark: 48 records
- Malformed records: 0
- Normalized duplicate utterances: 0
- Semantic-group split leaks: 0
- Gold/train nearest-string similarity: mean 0.582, maximum 0.736; no gold case at or above 0.80
- Reproducibility, validation, and dataset unit tests: all passing

## Remaining risks

The development corpus is still synthetic and produced by one generator, while the gold benchmark is handwritten by one author. Phrase diversity and label consistency therefore do not substitute for commands collected from multiple musicians or independent annotation. The 48-case gold set supports a portfolio experiment but is too small for narrow confidence intervals, especially by subcategory. Gold should remain sealed during LoRA and prompt selection.

Negative records are not directly SFT-eligible because Midge has no executable abstain/clarify action. Their evaluation requires a separate router or an explicit future protocol extension. Rule IDs remain model-authored metadata inferred from semantics rather than user-specified names, so primary semantic scoring should report both full exact match and an execution-field score that excludes cosmetic `id`/`name` differences.
