from __future__ import annotations
import json, re, sys, types
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
try: import requests  # noqa:F401
except ImportError: sys.modules["requests"]=types.ModuleType("requests")
try: import mido  # noqa:F401
except ImportError:
    stub=types.ModuleType("mido");stub.Message=object;sys.modules["mido"]=stub
from llm_parser import extract_json_object, validate_action

COSMETIC={"id","name","version"}
def strip_cosmetic(value):
    if isinstance(value,dict): return {k:strip_cosmetic(v) for k,v in value.items() if k not in COSMETIC}
    if isinstance(value,list): return [strip_cosmetic(v) for v in value]
    return value
def flatten(value,prefix=""):
    out={}
    if isinstance(value,dict):
        for k,v in value.items(): out.update(flatten(v,f"{prefix}.{k}" if prefix else k))
    elif isinstance(value,list):
        for i,v in enumerate(value): out.update(flatten(v,f"{prefix}[{i}]"))
    else: out[prefix]=value
    return out
def parse_prediction(pred):
    if pred.get("predicted_behavior") in {"clarify","reject"}: return pred["predicted_behavior"],None,None
    raw=pred.get("raw_output","")
    try:return "execute",extract_json_object(raw),None
    except Exception as exc:return "malformed",None,str(exc)
def score_rows(rows,predictions,context_doc):
    by_id={p["id"]:p for p in predictions};details=[]
    for row in rows:
        pred=by_id.get(row["id"],{"id":row["id"],"raw_output":""})
        behavior,action,error=parse_prediction(pred); strict=False; semantic=False; field_score=0.0; valid=False; normalized=None
        if row["expected_behavior"]=="execute" and action is not None:
            strict=action==row["target"]
            try:
                state=context_doc["contexts"][row["context_ref"]]; normalized=validate_action(deepcopy(action),deepcopy(state),context_doc["sources"])
                expected=validate_action(deepcopy(row["target"]),deepcopy(state),context_doc["sources"]);valid=True
                ef=flatten(strip_cosmetic(expected));af=flatten(strip_cosmetic(normalized));matched=sum(af.get(k)==v for k,v in ef.items());field_score=matched/len(ef) if ef else 1.0;semantic=field_score==1.0
            except Exception as exc:error=str(exc)
        elif row["expected_behavior"] in {"clarify","reject"}:
            semantic=behavior==row["expected_behavior"];strict=semantic;field_score=float(semantic)
        details.append({"id":row["id"],"split":row["split"],"category":row["category"],"subcategory":row["subcategory"],"expected_behavior":row["expected_behavior"],"predicted_behavior":behavior,"strict_exact":strict,"semantic_correct":semantic,"field_score":field_score,"valid_action":valid,"malformed":behavior=="malformed","error":error,"expected":row["target"],"parsed_prediction":action,"raw_output":pred.get("raw_output","")})
    def summary(items):
        n=len(items) or 1
        return {"count":len(items),"strict_exact_rate":sum(x["strict_exact"] for x in items)/n,"semantic_accuracy":sum(x["semantic_correct"] for x in items)/n,"mean_field_score":sum(x["field_score"] for x in items)/n,"malformed_output_rate":sum(x["malformed"] for x in items)/n,"validator_pass_rate":sum(x["valid_action"] for x in items)/n}
    capability={k:summary(v) for k,v in sorted(_group(details,"category").items())}
    behavior={k:summary(v) for k,v in sorted(_group(details,"expected_behavior").items())}
    regressions=[x for x in details if not x["semantic_correct"]][:25]
    return {"overall":summary(details),"per_capability":capability,"executable_vs_negative":behavior,"regression_examples":regressions},details
def _group(rows,key):
    out=defaultdict(list)
    for r in rows:out[r[key]].append(r)
    return out
