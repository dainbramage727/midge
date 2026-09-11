from __future__ import annotations
import argparse, json, random
from pathlib import Path
from .config import load_config, resolve_path, run_id
from .data import sft_records
from .metadata import capture, write

def main(argv=None):
    p=argparse.ArgumentParser(description="Train a Midge LoRA/QLoRA adapter")
    p.add_argument("--config",required=True);p.add_argument("--resume-from-checkpoint",default=None);p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv);cfg=load_config(a.config);rid=run_id(cfg,"train");out=resolve_path(cfg,cfg["experiment"]["output_root"])/rid
    train=sft_records(cfg,"train");validation=sft_records(cfg,"validation")
    manifest={"run_id":rid,"status":"dry_run" if a.dry_run else "starting","train_examples":len(train),"validation_examples":len(validation),"test_or_gold_loaded":False}
    out.mkdir(parents=True,exist_ok=True);write(out/"environment.json",capture(cfg,rid,"train"));write(out/"run_manifest.json",manifest)
    if a.dry_run or cfg["experiment"]["smoke_test"]:
        manifest["status"]="smoke_passed";manifest["note"]="No model was loaded and no experimental metric was produced."
        write(out/"run_manifest.json",manifest);print(json.dumps(manifest,indent=2));return 0
    if not cfg["lora"]["enabled"]: raise ValueError("Training requires lora.enabled=true; use evaluation for an untuned base model")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc: raise SystemExit("Install requirements-experiments.txt before real training") from exc
    set_seed(cfg["training"]["seed"]);mcfg=cfg["model"];qcfg=cfg["quantization"]
    tokenizer=AutoTokenizer.from_pretrained(mcfg["name_or_path"],revision=mcfg["revision"],trust_remote_code=mcfg["trust_remote_code"])
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model_kwargs={"revision":mcfg["revision"],"trust_remote_code":mcfg["trust_remote_code"]}
    if qcfg["enabled"]:
        dtype=getattr(torch,qcfg["compute_dtype"])
        model_kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=qcfg["bits"]==4,load_in_8bit=qcfg["bits"]==8,bnb_4bit_compute_dtype=dtype,bnb_4bit_quant_type=qcfg["quant_type"],bnb_4bit_use_double_quant=qcfg["double_quant"])
        model_kwargs["device_map"]="auto"
    model=AutoModelForCausalLM.from_pretrained(mcfg["name_or_path"],**model_kwargs)
    if qcfg["enabled"]: model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing",True))
    lcfg=cfg["lora"]
    peft=LoraConfig(r=lcfg["rank"],lora_alpha=lcfg["alpha"],lora_dropout=lcfg["dropout"],bias=lcfg.get("bias","none"),task_type="CAUSAL_LM",target_modules=lcfg["target_modules"])
    t=cfg["training"]
    args=SFTConfig(output_dir=str(out/"checkpoints"),learning_rate=t["learning_rate"],num_train_epochs=t["epochs"],per_device_train_batch_size=t["per_device_batch_size"],per_device_eval_batch_size=t.get("eval_batch_size",t["per_device_batch_size"]),gradient_accumulation_steps=t["gradient_accumulation_steps"],max_length=t["max_sequence_length"],seed=t["seed"],data_seed=t["seed"],save_steps=t["save_steps"],eval_steps=t["eval_steps"],logging_steps=t["logging_steps"],eval_strategy="steps",save_strategy="steps",load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,save_total_limit=t.get("save_total_limit",3),gradient_checkpointing=t.get("gradient_checkpointing",True),bf16=t.get("bf16",False),fp16=t.get("fp16",False),report_to=t.get("report_to",[]),packing=t.get("packing",False))
    trainer=SFTTrainer(model=model,args=args,train_dataset=Dataset.from_list(train),eval_dataset=Dataset.from_list(validation),processing_class=tokenizer,peft_config=peft)
    result=trainer.train(resume_from_checkpoint=a.resume_from_checkpoint)
    trainer.save_model(str(out/"adapter"));tokenizer.save_pretrained(str(out/"adapter"));trainer.save_state()
    manifest.update(status="completed",train_metrics=result.metrics,best_checkpoint=trainer.state.best_model_checkpoint)
    write(out/"run_manifest.json",manifest);return 0
if __name__=="__main__":raise SystemExit(main())
