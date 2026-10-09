"""Run the published evaluator in an existing configured Isaac Lab environment."""
import argparse, hashlib, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--isaaclab',type=Path,required=True)
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--dx-mm',type=float,default=0)
p.add_argument('--dy-mm',type=float,default=0)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
checkpoint=root/'result/models/candidate.pt'
if not checkpoint.is_file():
    p.error('Model weights are not published. Supply the evaluated checkpoint at result/models/candidate.pt before running inference.')
with checkpoint.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='2cac0faa70e70ca69ad13553ef816aa8f2410213bc73f839cc9d0e3b441dace9'
assert a.dataset.is_file() and (a.isaaclab/'isaaclab.sh').is_file()
assert not a.output_dir.exists(), 'Use a new output directory'
env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['PYTHONDONTWRITEBYTECODE']='1'
cmd=[str((a.isaaclab/'isaaclab.sh').resolve()),'-p',str(root/'code/inference/evaluate.py'),
 '--checkpoint',str(checkpoint),'--instruction','pick up the cube','--dataset',str(a.dataset.resolve()),
 '--seed','2058','--horizon','1300','--device','cuda:0','--dx-mm',str(a.dx_mm),
 '--dy-mm',str(a.dy_mm),'--output-dir',str(a.output_dir.resolve())]
raise SystemExit(subprocess.call(cmd,cwd=a.isaaclab.resolve(),env=env))
