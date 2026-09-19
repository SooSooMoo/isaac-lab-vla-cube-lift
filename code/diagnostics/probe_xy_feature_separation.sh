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
# Offline separability probe only. No policy update, checkpoint or file writes.
# Labels describe where the recorded intervention was applied, not physical grasp truth.
with torch.no_grad():
    x=torch.cat([h,ah],0)
    y=torch.cat([torch.zeros(800,device=d),torch.ones(79,device=d),torch.zeros(len(ah),device=d)])
    pick_steps=torch.arange(879,device=d)
    above_steps=torch.arange(len(ah),device=d)
    # Hold out blocks; adjacent observations still belong to the same trajectories.
    test=torch.cat([(pick_steps//10)%5==2,(above_steps//10)%5==2])
    # Exclude nearest two steps around held-out samples from probe fitting.
    guard=torch.zeros_like(test)
    offset=0
    for length in (879,len(ah)):
        segment=test[offset:offset+length]
        for shift in (-2,-1,0,1,2):
            src=torch.arange(length,device=d)+shift
            valid=(src>=0)&(src<length)
            guard[offset:offset+length][valid]|=segment[src[valid]]
        offset+=length
    train=~guard
    assert bool((y[train]==1).any()) and bool((y[test]==1).any())
    mean=x[train].mean(0);std=x[train].std(0).clamp_min(1e-4)
    design=torch.cat([(x-mean)/std,torch.ones(len(x),1,device=d)],1).double()
    target=(2*y-1).double()
    weights=torch.zeros(len(x),device=d,dtype=torch.float64)
    for cls in (0,1):
        chosen=train&(y==cls)
        weights[chosen]=.5/int(chosen.sum())
    normal=design.T@(weights[:,None]*design)
    rhs=design.T@(weights*target)
    identity=torch.eye(design.shape[1],device=d,dtype=torch.float64)
    def measures(score,mask):
        positive=mask&(y==1);negative=mask&(y==0)
        return {'positive_samples':int(positive.sum()),'negative_samples':int(negative.sum()),'positive_missed':int((score[positive]<0).sum()),'negative_falsely_selected':int((score[negative]>=0).sum())}
    groups={'pick_0_552':(torch.arange(len(x),device=d)<553),'pick_553_799':(torch.arange(len(x),device=d)>=553)&(torch.arange(len(x),device=d)<800),'corrected_800_878':(torch.arange(len(x),device=d)>=800)&(torch.arange(len(x),device=d)<879),'above':torch.arange(len(x),device=d)>=879}
    for regularization in (.0001,.001,.01,.1):
        coef=torch.linalg.solve(normal+regularization*identity,rhs)
        score=design@coef
        assert bool(torch.isfinite(score).all())
        print('FEATURE_SEPARABILITY',json.dumps({'regularization':regularization,'train':measures(score,train),'heldout_blocks':measures(score,test),'heldout_by_region':{name:measures(score,test&mask) for name,mask in groups.items()}}),flush=True)
    # Nearest retained example for each corrected state, in training-standardized space.
    standardized=((x-mean)/std).float()
    distances=torch.cdist(standardized[800:879],torch.cat([standardized[:800],standardized[879:]],0))
    near,index=distances.min(1)
    examples=[]
    for local in torch.argsort(near)[:5].tolist():
        match=int(index[local]);label='pick' if match<800 else 'above'
        step=match if match<800 else match-800
        examples.append({'corrected_step':800+local,'nearest_retained_task':label,'nearest_retained_sample':step,'standardized_feature_distance':float(near[local])})
    print('FEATURE_NEAREST_OPPOSITE',json.dumps(examples),flush=True)
print('FEATURE_SEPARABILITY_COMPLETE',json.dumps({'policy_changed':False,'files_written':False,'scope':'Linear probe on cached features. Region labels are intervention membership, not proof of correction necessity or grasp understanding. Heldout blocks share trajectories and do not establish generalization. Probe is not installed in policy.'}),flush=True)
PY
