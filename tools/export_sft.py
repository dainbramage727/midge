#!/usr/bin/env python3
"""Export executable Midge examples as OpenAI-style chat JSONL."""
from __future__ import annotations
import argparse, json, sys, types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
try: import requests  # noqa: F401
except ImportError: sys.modules["requests"]=types.ModuleType("requests")
try: import mido  # noqa: F401
except ImportError:
    stub=types.ModuleType("mido"); stub.Message=object; sys.modules["mido"]=stub
from llm_parser import build_system_prompt  # noqa: E402

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=["train","validation"],required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); data=ROOT/"data"
    context_doc=json.loads((data/"contexts.json").read_text()); sources=context_doc["sources"]
    rows=[]
    for line in (data/"splits"/f"{a.split}.jsonl").read_text().splitlines():
        row=json.loads(line)
        if not row["sft_eligible"]: continue
        state=context_doc["contexts"][row["context_ref"]]
        rows.append({"messages":[
            {"role":"system","content":build_system_prompt(state,sources)},
            {"role":"user","content":row["utterance"]},
            {"role":"assistant","content":json.dumps(row["target"],sort_keys=True,separators=(",",":"))},
        ],"metadata":{"id":row["id"],"category":row["category"],"context_ref":row["context_ref"]}})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    print(f"Wrote {len(rows)} SFT-eligible {a.split} examples to {a.output}")
if __name__=="__main__": main()
