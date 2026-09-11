from __future__ import annotations
import argparse,json
from pathlib import Path
from .data import read_jsonl
from .metadata import write
def main(argv=None):
 p=argparse.ArgumentParser(description="Compare aligned base and tuned scored records")
 p.add_argument("--base",required=True);p.add_argument("--tuned",required=True);p.add_argument("--output",required=True);a=p.parse_args(argv)
 base={r["id"]:r for r in read_jsonl(a.base)};tuned={r["id"]:r for r in read_jsonl(a.tuned)}
 if set(base)!=set(tuned):raise SystemExit("Base/tuned record IDs do not align")
 improvements=[];regressions=[];unchanged_failures=[]
 for rid in sorted(base):
  b=base[rid];t=tuned[rid];pair={"id":rid,"category":b["category"],"expected":b["expected"],"base_prediction":b["parsed_prediction"],"tuned_prediction":t["parsed_prediction"],"base_semantic_correct":b["semantic_correct"],"tuned_semantic_correct":t["semantic_correct"]}
  if not b["semantic_correct"] and t["semantic_correct"]:improvements.append(pair)
  elif b["semantic_correct"] and not t["semantic_correct"]:regressions.append(pair)
  elif not t["semantic_correct"]:unchanged_failures.append(pair)
 payload={"count":len(base),"base_semantic_accuracy":sum(r["semantic_correct"] for r in base.values())/len(base),"tuned_semantic_accuracy":sum(r["semantic_correct"] for r in tuned.values())/len(tuned),"improvement_count":len(improvements),"regression_count":len(regressions),"improvements":improvements[:50],"regressions":regressions[:50],"unchanged_failures":unchanged_failures[:50]}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);write(out/"comparison.json",payload);print(json.dumps({k:v for k,v in payload.items() if not isinstance(v,list)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
