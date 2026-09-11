from __future__ import annotations
import argparse, hashlib, json
from copy import deepcopy
from pathlib import Path
from .config import load_config, resolve_path, run_id
from .data import contexts, load_split, prompt_messages
from .metadata import capture, write
from .scoring import score_rows

def write_jsonl(path,rows):Path(path).write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
def main(argv=None):
    p=argparse.ArgumentParser(description="Generate and score Midge predictions")
    p.add_argument("--config",required=True);p.add_argument("--split",choices=["validation","test","gold"],required=True);p.add_argument("--label",choices=["base","lora"],required=True);p.add_argument("--adapter",default=None);p.add_argument("--smoke",action="store_true")
    a=p.parse_args(argv);cfg=load_config(a.config)
    if a.label=="lora" and not a.adapter: raise ValueError("--adapter is required for --label lora")
    adapter_identity=""
    if a.adapter:
        adapter_config=Path(a.adapter)/"adapter_config.json"
        adapter_identity=(hashlib.sha256(adapter_config.read_bytes()).hexdigest()[:12] if adapter_config.is_file() else Path(a.adapter).name)
    stage=f"eval-{a.split}-{a.label}"+(f"-{adapter_identity}" if adapter_identity else "")
    rid=run_id(cfg,stage);out=resolve_path(cfg,cfg["experiment"]["output_root"])/rid;out.mkdir(parents=True,exist_ok=True)
    rows=load_split(cfg,a.split);ctx=contexts(cfg);write(out/"environment.json",capture(cfg,rid,f"eval-{a.split}-{a.label}"))
    if a.smoke or cfg["experiment"]["smoke_test"]:
        # Exercise parsing/scoring with explicit fixtures; never publish these as model results.
        subset=rows[:4];pred=[]
        for row in subset:
            pred.append({"id":row["id"],"raw_output":json.dumps(row["target"]) if row["target"] is not None else "not json","smoke_fixture":True})
        metrics,details=score_rows(subset,pred,ctx);payload={"run_id":rid,"status":"smoke_passed","not_an_experimental_result":True,"records_exercised":len(subset)}
        write_jsonl(out/"smoke_predictions.jsonl",pred);write(out/"smoke_status.json",payload);print(json.dumps(payload,indent=2));return 0
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:raise SystemExit("Install requirements-experiments.txt before model evaluation") from exc
    mcfg=cfg["model"];qcfg=cfg["quantization"];kwargs={"revision":mcfg["revision"],"trust_remote_code":mcfg["trust_remote_code"]}
    if qcfg["enabled"]:
        kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=qcfg["bits"]==4,load_in_8bit=qcfg["bits"]==8,bnb_4bit_compute_dtype=getattr(torch,qcfg["compute_dtype"]),bnb_4bit_quant_type=qcfg["quant_type"],bnb_4bit_use_double_quant=qcfg["double_quant"]);kwargs["device_map"]="auto"
    tokenizer=AutoTokenizer.from_pretrained(mcfg["name_or_path"],revision=mcfg["revision"],trust_remote_code=mcfg["trust_remote_code"]);model=AutoModelForCausalLM.from_pretrained(mcfg["name_or_path"],**kwargs)
    if a.adapter:model=PeftModel.from_pretrained(model,a.adapter)
    model.eval();g=cfg["generation"];predictions=[]
    for row in rows:
        rendered=tokenizer.apply_chat_template(prompt_messages(row,ctx),tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(rendered,return_tensors="pt").to(model.device)
        generation={"max_new_tokens":g["max_new_tokens"],"do_sample":g["do_sample"],"num_beams":g["num_beams"],"pad_token_id":tokenizer.eos_token_id}
        if g["do_sample"]:generation.update(temperature=g["temperature"],top_p=g["top_p"])
        with torch.inference_mode(): output=model.generate(**inputs,**generation)
        text=tokenizer.decode(output[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        predictions.append({"id":row["id"],"raw_output":text})
    metrics,details=score_rows(rows,predictions,ctx);metrics.update(run_id=rid,split=a.split,label=a.label,adapter=a.adapter,dataset_sha256=capture(cfg,rid,"eval")["dataset_sha256"])
    write_jsonl(out/"predictions.jsonl",predictions);write_jsonl(out/"scored_records.jsonl",details);write(out/"metrics.json",metrics);print(json.dumps(metrics["overall"],indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
