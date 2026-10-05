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
out=Path(tempfile.mkdtemp(prefix='visual_consistency13_',dir='/workspace/step3'))
reserve=out/'capacity.reserve'
atexit.register(lambda:reserve.unlink(missing_ok=True))
with reserve.open('xb') as f:
    f.write(os.urandom(1024*1024));f.flush();os.fsync(f.fileno())
torch.manual_seed(2059)
for p in m.parameters():p.requires_grad=False
paths=[root/'collect_visual_robustness13_base_20260918_154939_668595_JST/observations13.npz',root/'collect_visual_robustness13_tenth_20260918_155031_372274_JST/observations13.npz']
data=[]
for path in paths:
    with np.load(path,allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
mean=torch.tensor([.485,.456,.406],device=d)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=d)[None,:,None,None]
def hidden(feat,instruction):
    lang=m.encode_language(torch.tensor(tokens(instruction),device=d)[None]).expand(len(feat),-1)
    return m.fusion(torch.cat([feat,lang],-1))
paired=[]
with torch.no_grad():
    for step in range(13):
        key=str(step)+'_';q=data[0][key+'eef_quat'].copy()
        if q[3]<0:q=-q
        st=torch.tensor(np.concatenate([data[0][key+'eef_pos'],q,data[0][key+'gripper_pos'],[step/1200.]]),dtype=torch.float32,device=d)[None]
        states=m.state_encoder(st)
        images=np.stack([data[0][key+'table_cam'],data[0][key+'wrist_cam'],data[1][key+'wrist_cam']])
        images=torch.tensor(images,device=d).permute(0,3,1,2).float()/255.
        vision=m.vision((images-mean)/std)
        feats=torch.cat([vision[0:1].expand(2,-1),vision[1:3],states.expand(2,-1)],-1)
        paired.append(hidden(feats,'pick up the cube'))
    ph=torch.stack(paired).detach()
    target=torch.tanh(m.arm_head(ph[:,0])).detach()
anchor_specs=[('above',root/'collect_settle_grasp_anchors_above_20260918_001143_047522_JST/above/preservation_anchors.npz','move above the cube'),('pick',root/'collect_adapter_approach526_20260918_113414_371719_JST/pick/preservation_anchors.npz','pick up the cube')]
anchors=[];hashes={}
with torch.no_grad():
    for name,path,instruction in anchor_specs:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:
            feat=torch.tensor(z['frozen_features'],device=d,dtype=torch.float32)
            labels=torch.tensor(z['actions'],device=d,dtype=torch.float32)
        anchors.append((name,hidden(feat,instruction).detach(),labels))
class Projected(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('gripper_direction',m.gripper_head.weight.detach().reshape(-1).clone())
    def forward(self,x):
        v=self.net(x);w=self.gripper_direction
        return x+v-(v*w).sum(-1,keepdim=True)/w.square().sum().clamp_min(1e-12)*w
adapter=Projected().to(d)
def predict(h):return torch.tanh(m.arm_head(adapter(h)))
def preservation():
    values={};constraints=[]
    for name,h,y in anchors:
        h2=adapter(h);err=torch.tanh(m.arm_head(h2))-y[:,:6]
        rmse=err.square().mean().sqrt();maximum=err.abs().max()
        grip=m.gripper_head(h2).squeeze(-1);sign=torch.where(y[:,6]<0,1.,-1.)
        constraints.extend([rmse/.0005-1,maximum/.002-1,(-sign*grip+1e-5).max()])
        values[name]={'rmse':float(rmse.detach()),'max_abs':float(maximum.detach()),'gripper_mismatches':int(((grip>=0)!=(y[:,6]<0)).sum())}
    return torch.stack(constraints),values
def measure():
    pred=predict(ph);err=pred-target[:,None,:];gap=pred[:,1]-pred[:,0]
    return {'target_mse':float(err.square().mean()),'pair_max_action_difference':float(gap.abs().max()),'reference_max_action_change':float(err[:,0].abs().max())}
with torch.no_grad():
    baseline=measure();g,metrics=preservation();assert bool((g<=0).all()),metrics
normalizer=max(baseline['target_mse'],1e-12)
opt=torch.optim.Adam(adapter.parameters(),lr=1e-4)
mult=torch.zeros(6,device=d);best=None;best_score=baseline['target_mse'];best_update=0
start=time.monotonic()
for update in range(1,8001):
    if time.monotonic()-start>=60:break
    opt.zero_grad();err=predict(ph)-target[:,None,:];objective=err.square().mean()/normalizer
    g,_=preservation();loss=objective+((torch.relu(mult+g).square()-mult.square())/2).sum()
    loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),10);opt.step()
    with torch.no_grad():
        g,_=preservation()
        if update%20==0:mult=torch.relu(mult+g)
        if bool((g<=0).all()):
            score=measure()
            if score['target_mse']<best_score and score['pair_max_action_difference']<baseline['pair_max_action_difference'] and score['reference_max_action_change']<=.002:
                best_score=score['target_mse'];best=copy.deepcopy(adapter.state_dict());best_update=update
if best is not None:adapter.load_state_dict(best)
else:
    torch.nn.init.zeros_(adapter.net[-1].weight);torch.nn.init.zeros_(adapter.net[-1].bias)
with torch.no_grad():
    final=measure();g,metrics=preservation()
    grip_old=m.gripper_head(ph);grip_new=m.gripper_head(adapter(ph))
    grip_change=float((grip_new-grip_old).abs().max())
    grip_same=bool(torch.equal(grip_old>=0,grip_new>=0))
report={'baseline':baseline,'selected':final,'selected_update':best_update,'preservation':metrics,'thresholds':{'rmse':.0005,'max_abs':.002,'gripper_mismatches':0},'preservation_pass':bool((g<=0).all()),'gripper_logit_max_change':grip_change,'gripper_decisions_unchanged':grip_same,'language_recheck_pending':True,'simulation_executed':False,'model_saved':False,'scope':'13 reused observation pairs; nominal table/state fixed and alternate wrist image used as synthetic variation. Targets are c49 predictions, not expert corrective actions. No generalization or task-success claim.'}
reserve.unlink(missing_ok=True)
if best_update and report['preservation_pass'] and grip_same and grip_change<=1e-5:
    payload={'format':'shared_fusion_arm_residual_v2','base_checkpoint':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'initial_adapter_state_dict':{k:v.detach().cpu() for k,v in r.state_dict().items()},'adapter_state_dict':{k:v.detach().cpu() for k,v in adapter.state_dict().items()}}
    temp=out/'writing.pt';dest=out/'visual_adapter.pt';torch.save(payload,temp)
    check=torch.load(temp,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,check['adapter_state_dict'][k]) for k,v in payload['adapter_state_dict'].items())
    temp.replace(dest);report.update(model_saved=True,adapter_file=str(dest),adapter_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
report['anchor_hashes']=hashes
report['source_unchanged']=hashlib.sha256(base.read_bytes()).hexdigest()=='3ad705b1224f552ccf6988364f5c9894fb55ab1003f390e5d04719033142177c' and hashlib.sha256(ap.read_bytes()).hexdigest()=='c49bcf3ab1c25556e7204337e413829293ab4418303ad38b00992792293f4ed5'
(out/'results.json').write_text(json.dumps(report,indent=2))
print('VISUAL_CONSISTENCY_RESULT',json.dumps(report),flush=True)
print('RESULT_FILE',out/'results.json')

PY
