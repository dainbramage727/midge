#!/usr/bin/env python3
"""Validate Midge corpus structure, canonical actions, duplication, and leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
try:
    import requests  # noqa: F401
except ImportError:
    sys.modules["requests"] = types.ModuleType("requests")
try:
    import mido  # noqa: F401
except ImportError:
    stub = types.ModuleType("mido"); stub.Message = object; sys.modules["mido"] = stub
from llm_parser import validate_action  # noqa: E402

EXPECTED = {"train": 480, "validation": 80, "test": 80, "gold": 48}
FIELDS = {
    "id", "split", "category", "subcategory", "utterance", "context_ref",
    "target", "expected_behavior", "reason", "sft_eligible", "tags",
    "provenance", "template_id", "semantic_group",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def read_jsonl(path: Path):
    rows=[]
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        try: row=json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"{path}:{line_no}: {exc}") from exc
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data")
    p.add_argument("--stats-out",type=Path,default=ROOT/"data"/"dataset_stats.json")
    args=p.parse_args()
    context_doc=json.loads((args.data_dir/"contexts.json").read_text())
    sources=context_doc["sources"]; contexts=context_doc["contexts"]
    files={s: args.data_dir/"splits"/f"{s}.jsonl" for s in ("train","validation","test")}
    files["gold"]=args.data_dir/"gold_benchmark.jsonl"
    errors=[]; all_rows=[]; ids={}; utterances={}; group_splits=defaultdict(set)
    per_split={}
    for split,path in files.items():
        rows=read_jsonl(path); per_split[split]=rows
        if len(rows)!=EXPECTED[split]: errors.append(f"{split}: expected {EXPECTED[split]} rows, found {len(rows)}")
        for line_no,row in enumerate(rows,1):
            where=f"{path}:{line_no}"
            missing=FIELDS-set(row); extra=set(row)-FIELDS
            if missing: errors.append(f"{where}: missing fields {sorted(missing)}")
            if extra: errors.append(f"{where}: unknown fields {sorted(extra)}")
            if row.get("split")!=split: errors.append(f"{where}: split field mismatch")
            rid=row.get("id")
            if rid in ids: errors.append(f"{where}: duplicate id also in {ids[rid]}")
            ids[rid]=where
            utterance=row.get("utterance")
            if not isinstance(utterance,str) or not utterance.strip(): errors.append(f"{where}: utterance must be non-empty")
            else:
                norm=normalize_text(utterance)
                if norm in utterances: errors.append(f"{where}: normalized duplicate utterance also in {utterances[norm]}")
                utterances[norm]=where
            behavior=row.get("expected_behavior")
            if behavior not in {"execute","clarify","reject"}: errors.append(f"{where}: invalid expected_behavior")
            eligible=row.get("sft_eligible")
            if not isinstance(eligible,bool): errors.append(f"{where}: sft_eligible must be boolean")
            if behavior=="execute":
                if row.get("target") is None: errors.append(f"{where}: executable row has null target")
                if not eligible: errors.append(f"{where}: executable row must be SFT eligible")
                if row.get("reason") is not None: errors.append(f"{where}: executable row reason must be null")
                cref=row.get("context_ref")
                if cref not in contexts: errors.append(f"{where}: unknown context_ref {cref!r}")
                else:
                    try:
                        normalized=validate_action(row["target"],contexts[cref],sources)
                        if row["target"].get("action") != "update_rule" and normalized!=row["target"]:
                            errors.append(f"{where}: target is not canonicalized")
                    except Exception as exc: errors.append(f"{where}: invalid Midge action: {exc}")
                target=row.get("target") or {}; rule=target.get("rule") or {}; condition=rule.get("condition") or {}
                lower=utterance.lower() if isinstance(utterance,str) else ""
                hard=re.search(r"velocity\s+(\d+)\s+or harder",lower)
                soft=re.search(r"velocity(?: is)?\s+(\d+)\s+or softer",lower)
                if hard and condition.get("min_velocity")!=int(hard.group(1)):
                    errors.append(f"{where}: 'or harder' does not match condition.min_velocity")
                if soft and condition.get("max_velocity")!=int(soft.group(1)):
                    errors.append(f"{where}: 'or softer' does not match condition.max_velocity")
            else:
                if row.get("target") is not None: errors.append(f"{where}: non-executable row must have null target")
                if eligible: errors.append(f"{where}: negative row cannot be directly SFT eligible")
                if not row.get("reason"): errors.append(f"{where}: negative row requires a reason")
            if not isinstance(row.get("tags"),list) or not row.get("tags"): errors.append(f"{where}: tags must be a non-empty list")
            group_splits[row.get("semantic_group")].add(split)
            all_rows.append(row)
    serialized="\n".join(json.dumps(r,sort_keys=True).lower() for r in all_rows)
    for artifact in ("generated-", "generated rule", "for setup", "1th", "2th", "3th"):
        if artifact in serialized: errors.append(f"generation artifact found: {artifact!r}")
    for group,splits in group_splits.items():
        non_gold=splits-{"gold"}
        if len(non_gold)>1: errors.append(f"semantic leakage: group {group!r} occurs in {sorted(non_gold)}")
        if "gold" in splits and len(splits)>1: errors.append(f"gold leakage: group {group!r} also occurs in development data")

    stats={
        "dataset_version":"1.0.0", "generator_seed":5636,
        "content_sha256": hashlib.sha256("\n".join(json.dumps(r,sort_keys=True) for r in all_rows).encode()).hexdigest(),
        "counts": {s:len(rows) for s,rows in per_split.items()},
        "total_development":sum(len(per_split[s]) for s in ("train","validation","test")),
        "total_with_gold":len(all_rows),
        "sft_eligible":sum(r["sft_eligible"] for r in all_rows),
        "expected_behavior":dict(sorted(Counter(r["expected_behavior"] for r in all_rows).items())),
        "category":dict(sorted(Counter(r["category"] for r in all_rows).items())),
        "subcategory":dict(sorted(Counter(r["subcategory"] for r in all_rows).items())),
        "tag":dict(sorted(Counter(t for r in all_rows for t in r["tags"]).items())),
        "provenance":dict(sorted(Counter(r["provenance"] for r in all_rows).items())),
        "unique_utterances":len(utterances), "unique_semantic_groups":len(group_splits),
        "validation_errors":len(errors),
    }
    args.stats_out.parent.mkdir(parents=True,exist_ok=True)
    args.stats_out.write_text(json.dumps(stats,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if errors:
        print("Dataset validation FAILED:",file=sys.stderr)
        for error in errors: print(f"- {error}",file=sys.stderr)
        raise SystemExit(1)
    print(f"Dataset validation passed: {len(all_rows)} rows, 0 malformed, 0 duplicates, 0 split leaks")
    print(f"Statistics: {args.stats_out}")


if __name__=="__main__": main()
