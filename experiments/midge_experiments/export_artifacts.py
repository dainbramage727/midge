from __future__ import annotations
import argparse,hashlib,json,shutil,tarfile,tempfile
from datetime import datetime,timezone
from pathlib import Path
def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser(description="Create a portable, checksummed experiment archive")
 p.add_argument("--run-dir",required=True,type=Path);p.add_argument("--destination",required=True,type=Path);p.add_argument("--include",action="append",default=[],help="Additional log/file to include; repeatable")
 a=p.parse_args(argv);run=a.run_dir.resolve();dest=a.destination.resolve()
 if not run.is_dir():raise SystemExit(f"Run directory not found: {run}")
 dest.mkdir(parents=True,exist_ok=True);archive=dest/f"{run.name}.tar.gz"
 with tempfile.TemporaryDirectory(prefix="midge-export-") as tmp:
  stage=Path(tmp)/run.name;shutil.copytree(run,stage)
  extras=stage/"external_logs"
  for value in a.include:
   src=Path(value).resolve()
   if not src.is_file():raise SystemExit(f"Additional artifact not found: {src}")
   extras.mkdir(exist_ok=True);shutil.copy2(src,extras/src.name)
  files=[]
  for f in sorted(stage.rglob("*")):
   if f.is_file():files.append({"path":str(f.relative_to(stage)),"size":f.stat().st_size,"sha256":sha256(f)})
  (stage/"export_manifest.json").write_text(json.dumps({"run_id":run.name,"exported_utc":datetime.now(timezone.utc).isoformat(),"files":files},indent=2,sort_keys=True)+"\n")
  with tarfile.open(archive,"w:gz") as tar:tar.add(stage,arcname=run.name)
 digest=sha256(archive);(dest/f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")
 print(json.dumps({"archive":str(archive),"sha256":digest},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
