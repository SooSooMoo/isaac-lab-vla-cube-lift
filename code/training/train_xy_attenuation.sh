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
source=root/'collect_xy_correction879_pick_20260918_165517_256946_JST/pick/xy_correction_inputs.npz'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='374f0ba650066cccc582a64e288912d008e46e4529f98496153ad511d76c2178'
cp=Path('/workspace/step3/retention_gripper_da00uipp/gripper_correction.pt')
assert hashlib.sha256(cp.read_bytes()).hexdigest()=='48fc3c58dea3deaef9fbae6d1208d7df522967ff841162f5077f88eab0143863'
gp=torch.load(cp,map_location=d,weights_only=False)
gc=torch.nn.Linear(256,1).to(d);gc.load_state_dict(gp['state_dict']);gc.requires_grad_(False)
with np.load(source,allow_pickle=False) as z:
    h=torch.tensor(z['hidden'],device=d,dtype=torch.float32)
    old=torch.tensor(z['policy_actions'],device=d,dtype=torch.float32)
    labels=torch.tensor(z['executed_actions'],device=d,dtype=torch.float32)
    assert np.array_equal(z['steps'],np.arange(879))
assert h.shape==(879,256) and torch.isfinite(h).all()
above=root/'collect_settle_grasp_anchors_above_20260918_001143_047522_JST/above/preservation_anchors.npz'
assert hashlib.sha256(above.read_bytes()).hexdigest()=='884cc88e1830e3a790fc96f3764e2890dab932ad6bc53acdc6803791b83d3ac9'
with np.load(above,allow_pickle=False) as z,torch.no_grad():
    feat=torch.tensor(z['frozen_features'],device=d,dtype=torch.float32)
    lang=m.encode_language(torch.tensor(tokens('move above the cube'),device=d)[None]).expand(len(feat),-1)
    ah=m.fusion(torch.cat([feat,lang],-1)).detach()
with torch.no_grad():
    pre=m.arm_head(h).detach();apre=m.arm_head(ah).detach()
    original=torch.tanh(pre);aoriginal=torch.tanh(apre)
    grip=(m.gripper_head(h)+gc(h)).detach()
    assert float((original-old[:,:6]).abs().max())<=.0001,'Arm reconstruction failed'
    assert torch.equal(grip.squeeze(-1)>=0,old[:,6]<0),'Grip reconstruction failed'
    assert torch.equal(labels[:800],old[:800])
    assert torch.equal(labels[:,2:],old[:,2:])
    assert bool((labels[800:,:2]==0).all())
# Learned continuous attenuation: g=clamp(score,0,1); action_xy *= (1-g).
# No runtime step threshold or instruction branch. All samples now development data.
with torch.no_grad():
    x=torch.cat([h,ah],0).detach()
    positive=torch.zeros(len(x),device=d,dtype=torch.bool);positive[800:879]=True
    mean=x.mean(0);std=x.std(0).clamp_min(1e-4)
    normalized=((x-mean)/std).detach()
    design=torch.cat([normalized,torch.ones(len(x),1,device=d)],1).double()
    target=torch.where(positive,2.,-1.).double()
    weights=torch.where(positive,.5/int(positive.sum()),.5/int((~positive).sum())).double()
    coef=torch.linalg.solve(design.T@(weights[:,None]*design)+.0001*torch.eye(257,device=d,dtype=torch.float64),design.T@(weights*target))
linear=torch.nn.Linear(256,1).to(d)
with torch.no_grad():linear.weight.copy_(coef[:256].float()[None]);linear.bias.copy_(coef[256:].float())
opt=torch.optim.Adam(linear.parameters(),lr=.001)
best=None;bestloss=float('inf');selected=0;updates=0;start=time.monotonic();stop='update_limit'
for update in range(1,10001):
    if time.monotonic()-start>=60:stop='time_limit';break
    opt.zero_grad();score=linear(normalized).squeeze(-1)
    loss=torch.relu(score[~positive]+.1).square().mean()+torch.relu(1.1-score[positive]).square().mean()
    if not bool(torch.isfinite(loss)):stop='nonfinite_loss';break
    loss.backward()
    if not all(bool(torch.isfinite(p.grad).all()) for p in linear.parameters()):stop='nonfinite_gradient';break
    opt.step();updates=update
    with torch.no_grad():
        score=linear(normalized).squeeze(-1)
        valid=bool((score[~positive]<=0).all() and (score[positive]>=1).all())
        if valid:
            best=copy.deepcopy(linear.state_dict());selected=update;stop='all_cached_constraints_met';break
if best is not None:linear.load_state_dict(best)
with torch.no_grad():
    score=linear(normalized).squeeze(-1);gate=score.clamp(0,1)
    report={'candidate_found':best is not None,'updates':updates,'selected_update':selected,'stop_reason':stop,'seconds':round(time.monotonic()-start,2),'pick_first800_nonzero_correction_count':int((gate[:800]>0).sum()),'above_nonzero_correction_count':int((gate[879:]>0).sum()),'corrected79_not_fully_suppressed_count':int((gate[800:879]<1).sum()),'preserved_max_score':float(score[~positive].max()),'corrected_min_score':float(score[positive].min()),'model_saved':False,'simulation_executed':False,'language_recheck_pending':True,'scope':'Learned continuous XY attenuation from shared hidden features, no explicit runtime step switch. All cached observations reused for fitting; features include progress. Not physical-state understanding or autonomous success.'}
if best is not None:
    out=Path(tempfile.mkdtemp(prefix='learned_xy_attenuation_',dir='/workspace/step3'))
    payload={'format':'shared_xy_attenuation_v1','state_dict':{k:v.cpu() for k,v in best.items()},'feature_mean':mean.cpu(),'feature_std':std.cpu(),'base_checkpoint':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'adapter_file':str(ap2),'adapter_sha256':hashlib.sha256(ap2.read_bytes()).hexdigest(),'gripper_file':str(cp),'gripper_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'data_file':str(source),'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    dest=out/'xy_attenuation.pt';torch.save(payload,dest)
    check=torch.load(dest,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,check['state_dict'][k]) for k,v in payload['state_dict'].items())
    report.update(model_saved=True,correction_file=str(dest),correction_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('XY_ATTENUATION_TRAIN_RESULT',json.dumps(report,allow_nan=False),flush=True)
PY
