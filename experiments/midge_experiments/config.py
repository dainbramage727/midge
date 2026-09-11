from __future__ import annotations
import hashlib, json, re, subprocess, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REQUIRED={
 "experiment":["name","output_root","smoke_test"], "model":["name_or_path","revision","trust_remote_code"],
 "data":["train","validation","test","gold","contexts","stats"],
 "training":["learning_rate","epochs","per_device_batch_size","gradient_accumulation_steps","max_sequence_length","seed","save_steps","eval_steps","logging_steps"],
 "lora":["enabled","rank","alpha","dropout","target_modules"],
 "quantization":["enabled","bits","compute_dtype","quant_type","double_quant"],
 "generation":["max_new_tokens","do_sample","temperature","top_p","num_beams"],
}
def load_config(path):
    path=Path(path).resolve(); cfg=tomllib.loads(path.read_text())
    missing=[f"{s}.{k}" for s,keys in REQUIRED.items() for k in keys if s not in cfg or k not in cfg[s]]
    if missing: raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    if cfg["quantization"]["enabled"] and cfg["quantization"]["bits"] not in (4,8): raise ValueError("quantization.bits must be 4 or 8")
    if cfg["lora"]["enabled"] and cfg["lora"]["rank"]<1: raise ValueError("lora.rank must be positive")
    cfg["_config_path"]=str(path); return cfg
def resolve_path(cfg,value):
    p=Path(value); return p if p.is_absolute() else ROOT/p
def dataset_hash(cfg):
    return json.loads(resolve_path(cfg,cfg["data"]["stats"]).read_text())["content_sha256"]
def run_id(cfg, stage):
    payload={k:v for k,v in cfg.items() if not k.startswith("_")}
    digest=hashlib.sha256(json.dumps({"config":payload,"dataset":dataset_hash(cfg),"code_commit":git_metadata()["commit"],"stage":stage},sort_keys=True,separators=(",",":")).encode()).hexdigest()[:12]
    model=re.sub(r"[^a-z0-9]+","-",cfg["model"]["name_or_path"].lower()).strip("-")[-36:]
    return f"{cfg['experiment']['name']}-{model}-{stage}-{digest}"
def git_metadata():
    def git(*args):
        try:return subprocess.check_output(["git",*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except Exception:return "unknown"
    return {"commit":git("rev-parse","HEAD"),"branch":git("branch","--show-current"),"dirty":bool(git("status","--porcelain"))}
