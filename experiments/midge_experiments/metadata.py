from __future__ import annotations
import json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from .config import dataset_hash, git_metadata
def capture(cfg,run_id,stage):
    packages=""
    try: packages=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    except Exception: pass
    slurm={k:v for k,v in os.environ.items() if k.startswith("SLURM_") and k in {"SLURM_JOB_ID","SLURM_JOB_NAME","SLURM_CLUSTER_NAME","SLURM_CPUS_PER_TASK","SLURM_NTASKS"}}
    return {"run_id":run_id,"stage":stage,"created_utc":datetime.now(timezone.utc).isoformat(),"dataset_sha256":dataset_hash(cfg),"git":git_metadata(),"python":sys.version,"platform":platform.platform(),"packages":packages.splitlines(),"slurm":slurm,"config":{k:v for k,v in cfg.items() if not k.startswith("_")}}
def write(path,payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
