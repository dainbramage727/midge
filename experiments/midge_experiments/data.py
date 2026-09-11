from __future__ import annotations
import json
from pathlib import Path
from .config import resolve_path

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def load_split(cfg, split):
    key={"train":"train","validation":"validation","test":"test","gold":"gold"}[split]
    rows=read_jsonl(resolve_path(cfg,cfg["data"][key]))
    if any(r["split"]!=split for r in rows): raise ValueError(f"Split contamination in {key}")
    return rows
def training_rows(cfg, split):
    if split not in {"train","validation"}: raise ValueError("Training may read only train or validation")
    return [r for r in load_split(cfg,split) if r["sft_eligible"]]
def contexts(cfg):
    return json.loads(resolve_path(cfg,cfg["data"]["contexts"]).read_text())

def prompt_messages(row, context_doc):
    # Import lazily: data validation/smoke mode does not need Midge runtime deps.
    import sys, types
    root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/"src"))
    try: import requests  # noqa:F401
    except ImportError: sys.modules["requests"]=types.ModuleType("requests")
    try: import mido  # noqa:F401
    except ImportError:
        stub=types.ModuleType("mido");stub.Message=object;sys.modules["mido"]=stub
    from llm_parser import build_system_prompt
    state=context_doc["contexts"][row["context_ref"]]
    return [{"role":"system","content":build_system_prompt(state,context_doc["sources"])},{"role":"user","content":row["utterance"]}]
def sft_records(cfg,split):
    ctx=contexts(cfg); return [{"messages":prompt_messages(r,ctx)+[{"role":"assistant","content":json.dumps(r["target"],sort_keys=True,separators=(",",":"))}],"id":r["id"]} for r in training_rows(cfg,split)]
