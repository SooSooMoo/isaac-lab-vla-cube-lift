PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -u - <<'PY'
from pathlib import Path
import sys,hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
root=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST')
sys.path.insert(0,str(root/'train_lift_stage_20260916_185701_630638_JST/code'))
from language_vla.model import LanguageConditionedVLA
from smoke_utils import tokens
base=root/'gripper_actual_fix_20260917_142308_983342/shared_full_candidate.pt'
ap=Path('/workspace/step3/current_hold_adapter_jtdqus8m/hold_adapter.pt')
assert hashlib.sha256(base.read_bytes()).hexdigest()=='3ad705b1224f552ccf6988364f5c9894fb55ab1003f390e5d04719033142177c'
assert hashlib.sha256(ap.read_bytes()).hexdigest()=='c49bcf3ab1c25556e7204337e413829293ab4418303ad38b00992792293f4ed5'
torch.backends.cudnn.benchmark=False
d='cuda:0';ck=torch.load(base,map_location=d,weights_only=False)
m=LanguageConditionedVLA(**{k:int(ck[k]) for k in ('state_dim','action_dim','vocab_size','language_dim')}).to(d)
m.load_state_dict(ck['model'])
class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))
    def forward(self,x):return x+self.net(x)
r=Residual().to(d);r.load_state_dict(torch.load(ap,map_location=d,weights_only=False)['adapter_state_dict'])
m.fusion=torch.nn.Sequential(m.fusion,r);m.eval();m._pick_mode=False

import time,copy,tempfile,os,atexit
ap2=Path('/workspace/step3/visual_consistency13_th1jfnmy/visual_adapter.pt')
assert hashlib.sha256(ap2.read_bytes()).hexdigest()=='aee282b49bc44f0d57255c16e994c402d97cb00621a2bc0190b6e43c9fb71f23'
pack=torch.load(ap2,map_location=d,weights_only=False)
assert all(torch.equal(v,pack['initial_adapter_state_dict'][k]) for k,v in r.state_dict().items())
class Projected(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))
        self.register_buffer('gripper_direction',m.gripper_head.weight.detach().reshape(-1).clone())
    def forward(self,x):
        v=self.net(x);w=self.gripper_direction
        return x+v-(v*w).sum(-1,keepdim=True)/w.square().sum().clamp_min(1e-12)*w
proj=Projected().to(d);proj.load_state_dict(pack['adapter_state_dict'])
m.fusion=torch.nn.Sequential(m.fusion,proj);m.eval()
for p in m.parameters():p.requires_grad=False
source=root/'collect_retention_inputs700_pick_20260918_163318_884185_JST/pick/retention_inputs.npz'
with np.load(source,allow_pickle=False) as z:
    h=torch.tensor(z['hidden'],device=d,dtype=torch.float32)
    y=torch.tensor(z['executed_actions'][:,6]<0,device=d,dtype=torch.float32)
    old_actions=torch.tensor(z['policy_actions'],device=d,dtype=torch.float32)
    assert np.array_equal(z['steps'],np.arange(700))
assert h.shape==(700,256)
above=root/'collect_settle_grasp_anchors_above_20260918_001143_047522_JST/above/preservation_anchors.npz'
with np.load(above,allow_pickle=False) as z,torch.no_grad():
    feat=torch.tensor(z['frozen_features'],device=d,dtype=torch.float32)
    above_y=torch.tensor(z['actions'][:,6]<0,device=d,dtype=torch.float32)
    lang=m.encode_language(torch.tensor(tokens('move above the cube'),device=d)[None]).expand(len(feat),-1)
    ah=m.fusion(torch.cat([feat,lang],-1)).detach()
with torch.no_grad():
    logits=m.gripper_head(h).squeeze(-1);above_logits=m.gripper_head(ah).squeeze(-1)
    arm_reference=torch.tanh(m.arm_head(h)).clone()
    assert torch.equal(logits>=0,old_actions[:,6]<0),'Cached grip reconstruction mismatch'
    assert float((arm_reference-old_actions[:,:6]).abs().max())<=.0001,'Cached arm reconstruction mismatch'
    assert torch.equal(above_logits>=0,above_y.bool()),'Above baseline mismatch'
torch.manual_seed(2060)
correction=torch.nn.Linear(256,1).to(d)
torch.nn.init.zeros_(correction.weight);torch.nn.init.zeros_(correction.bias)
opt=torch.optim.Adam(correction.parameters(),lr=1e-4)
def metrics():
    pred=logits+correction(h).squeeze(-1)
    apred=above_logits+correction(ah).squeeze(-1)
    return {'pick_before614_mismatches':int(((pred[:614]>=0)!=y[:614].bool()).sum()),'retention86_mismatches':int(((pred[614:]>=0)!=y[614:].bool()).sum()),'above_mismatches':int(((apred>=0)!=above_y.bool()).sum())}
with torch.no_grad():baseline=metrics()
best=None;selected=0;best_norm=float('inf');start=time.monotonic();updates=0
for update in range(1,10001):
    if time.monotonic()-start>=60:break
    opt.zero_grad()
    pred=logits+correction(h).squeeze(-1);apred=above_logits+correction(ah).squeeze(-1)
    loss=F.binary_cross_entropy_with_logits(pred[:614],y[:614])+F.binary_cross_entropy_with_logits(pred[614:],y[614:])+F.binary_cross_entropy_with_logits(apred,above_y)+.01*correction.weight.square().sum()
    loss.backward();opt.step();updates=update
    with torch.no_grad():
        current=metrics()
        if all(v==0 for v in current.values()):
            norm=float(correction.weight.square().sum()+correction.bias.square().sum())
            if norm<best_norm:best=copy.deepcopy(correction.state_dict());best_norm=norm;selected=update
            if update>=selected+200:break
if best is not None:correction.load_state_dict(best)
with torch.no_grad():
    final=metrics()
    arm_delta=float((torch.tanh(m.arm_head(h))-arm_reference).abs().max())
report={'baseline':baseline,'selected':final if best is not None else None,'candidate_found':best is not None,'updates':updates,'selected_update':selected,'optimization_seconds':round(time.monotonic()-start,2),'same_input_arm_max_change':arm_delta,'simulation_executed':False,'language_recheck_pending':True,'model_saved':False,'scope':'Shared learned linear gripper correction from final fusion features. No step input or runtime closure override. All earlier614 labels and above labels must match; all86 retention targets must close. Offline fit only; future trajectories may change.'}
if best is not None:
    out=Path(tempfile.mkdtemp(prefix='retention_gripper_',dir='/workspace/step3'))
    dest=out/'gripper_correction.pt'
    payload={'format':'shared_gripper_linear_correction_v1','state_dict':{k:v.cpu() for k,v in best.items()},'base_checkpoint':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'adapter_file':str(ap2),'adapter_sha256':hashlib.sha256(ap2.read_bytes()).hexdigest(),'data_file':str(source),'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    torch.save(payload,dest)
    check=torch.load(dest,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,check['state_dict'][k]) for k,v in payload['state_dict'].items())
    report.update(model_saved=True,correction_file=str(dest),correction_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(report,indent=2))
print('RETENTION_GRIPPER_TRAIN_RESULT',json.dumps(report),flush=True)
PY
