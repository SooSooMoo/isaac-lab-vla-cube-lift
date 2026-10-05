PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json,hashlib,time,copy
import numpy as np
import h5py,torch
import torch.nn.functional as F

root=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST')
system=root/'train_lift_stage_20260916_185701_630638_JST'
audit=root/'language_teacher_pairs_20260917_141110_240799_JST/results.json'
protocol=json.loads((system/'protocol.json').read_text())
source=root/'gripper_actual_fix_20260917_142308_983342/shared_full_candidate.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_hash=sha(source)
assert source_hash=='3ad705b1224f552ccf6988364f5c9894fb55ab1003f390e5d04719033142177c'
vertical_collection=root/'collect_settle_then_close623_verified_20260918_110336_081918_JST'
vertical_path=vertical_collection/'pick/vertical_recovery.npz'
vertical_report=json.loads((vertical_collection/'results.json').read_text())
assert vertical_report['arm_hold_pass'] and all(vertical_report['first623_exactly_equal'].values())
assert vertical_report['samples']==60 and vertical_report['teacher_actions_executed']
vertical_hash=sha(vertical_path)


import os,tempfile,atexit,signal
output_dir=Path(tempfile.mkdtemp(prefix='current_hold_adapter_',dir='/workspace/step3'))
reserve=output_dir/'save_capacity.reserve'
def release_reserve():reserve.unlink(missing_ok=True)
atexit.register(release_reserve)
def terminate(signum,frame):
    release_reserve()
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate)
try:
    with reserve.open('xb') as f:
        f.write(os.urandom(1024*1024));f.flush();os.fsync(f.fileno())
except BaseException:
    release_reserve();raise
print('SAVE_CAPACITY_RESERVED_1MIB',str(output_dir),flush=True)
print('SHARED_RESIDUAL_DIAGNOSTIC_STARTED NO_FILES_WRITTEN',flush=True)
sys.path.insert(0,str(system/'code'))
from language_vla.model import LanguageConditionedVLA
from smoke_utils import tokens
torch.manual_seed(2058);np.random.seed(2058)
torch.backends.cudnn.benchmark=False
device='cuda:0'
checkpoint=torch.load(source,map_location=device,weights_only=False)
model=LanguageConditionedVLA(**{k:int(checkpoint[k]) for k in ('state_dim','action_dim','vocab_size','language_dim')}).to(device)
model.load_state_dict(checkpoint['model']);model.eval()
# Original model is frozen; only the shared residual is trained.
model.language_encoder.eval()
assert not model.language_encoder.training and not model.fusion.training
# Always use the SAME fusion and heads. Language tokens never select a branch.
for p in model.parameters():p.requires_grad=False
class SharedResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        return x+self.net(x)
adapter=SharedResidual().to(device)
model.fusion=torch.nn.Sequential(model.fusion,adapter)
modules=(adapter,)
parameters=[p for module in modules for p in module.parameters()]
for p in parameters:p.requires_grad=True
records=json.loads(audit.read_text())['reports'][0]['details']
assert len(records)==24
steps=np.asarray([r['step'] for r in records])
train=np.flatnonzero(np.arange(24)%3!=0);heldout=np.flatnonzero(np.arange(24)%3==0)
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
features=[];targets=[]
with h5py.File(root/'two_instruction.hdf5','r') as f,torch.no_grad():
    demo=f['data/demo_0']
    for step,row in zip(steps,records):
        images=np.stack([demo['obs/table_cam'][step],demo['obs/wrist_cam'][step]])
        image=torch.as_tensor(images,device=device).permute(0,3,1,2).float()/255.
        vision=model.vision((image-mean)/std).reshape(1,-1)
        q=demo['obs/eef_quat'][step].copy()
        if q[3]<0:q=-q
        state=np.concatenate([demo['obs/eef_pos'][step],q,demo['obs/gripper_pos'][step],[step/1200.]])
        state_feature=model.state_encoder(torch.tensor(state,dtype=torch.float32,device=device)[None])
        features.append(torch.cat([vision,state_feature],dim=-1))
        target=np.repeat(demo['actions'][step][None],2,axis=0)
        target[0,:3]=row['teacher_above'];target[1,:3]=row['teacher_pick'];target[:,6]=1.
        targets.append(target)
features=torch.cat(features)
targets=torch.tensor(np.stack(targets),device=device,dtype=torch.float32)
language=torch.tensor(np.stack([tokens('move above the cube'),tokens('pick up the cube')]),device=device)
def forward(indices):
    index=torch.as_tensor(indices,device=device)
    encoded=model.encode_language(language)
    fused=torch.cat([features[index,None,:].expand(-1,2,-1),encoded[None].expand(len(indices),-1,-1)],dim=-1)
    hidden=model.fusion(fused)
    return torch.tanh(model.arm_head(hidden)),model.gripper_head(hidden).squeeze(-1)
def metrics(indices):
    with torch.no_grad():
        arm,grip=forward(indices);t=targets[indices]
        own=torch.linalg.vector_norm(arm[:,:,:3]-t[:,:,:3],dim=-1)
        other=torch.linalg.vector_norm(arm[:,:,:3]-t.flip(1)[:,:,:3],dim=-1)
        pair=((own+1e-6<other)&(grip<0)).all(dim=1)
        return {'paired_correct':int(pair.sum()),'pairs':len(indices),
                'arm_mse':float(F.mse_loss(arm,t[:,:,:6])),
                'max_relative_translation_error':float((own / torch.linalg.vector_norm(t[:,0,:3]-t[:,1,:3],dim=-1)[:,None].clamp_min(1e-8)).max()),
                'open_gripper_count':int((grip<0).sum())}

group_names=[]
correction_features=[];correction_y=[];correction_ids=[];correction_groups=[];offset=0
for task,path,n,instruction,first_step in ((1,vertical_path,60,'pick up the cube',623),):
    with np.load(path,allow_pickle=False) as data,torch.no_grad():
        assert np.array_equal(data['steps'],np.arange(first_step,first_step+n))
        assert str(data['instruction'].item())==instruction
        assert float(data['progress_denominator'])==1200.
        label=torch.tensor(data['teacher_actions'],dtype=torch.float32,device=device)
        assert label.shape==(n,7) and torch.isfinite(label).all()
        if path==vertical_path:
            expected=np.where(data['phase']=='settle_open',1.,-1.)
            assert np.array_equal(data['teacher_actions'][:,6],expected)
        else:
            assert torch.all(label[:,6]==1)
        if first_step==0:assert torch.all(label[:13,:6]==0)
        correction_y.append(label);correction_ids.extend([task]*n)
        if path==vertical_path:
            phases=data['phase']
            assert np.array_equal(phases,np.asarray(['settle_open']*20+['stationary_close']*40))
            correction_groups.extend([(instruction+'/settle_open',slice(offset,offset+20)),(instruction+'/stationary_close',slice(offset+20,offset+n))])
        elif first_step==0:
            correction_groups.extend([(instruction+'/settle',slice(offset,offset+13)),(instruction+'/approach',slice(offset+13,offset+n))])
        else:
            correction_groups.append((instruction+('/target_hold' if task==0 else ('/pregrasp_alignment' if path==pregrasp_path else '/current_descent_alignment' if path==current_path else ('/late_descent_alignment' if path==late_path else '/descent_alignment'))),slice(offset,offset+n)))
        for start in range(0,n,32):
            end=min(start+32,n);b=end-start
            images=np.concatenate([data['obs_table_cam'][start:end],data['obs_wrist_cam'][start:end]])
            image=torch.tensor(images,device=device).permute(0,3,1,2).float()/255.
            vision=model.vision((image-mean)/std)
            q=data['obs_eef_quat'][start:end].copy();q[q[:,3]<0]*=-1
            state=np.concatenate([data['obs_eef_pos'][start:end],q,data['obs_gripper_pos'][start:end],data['steps'][start:end,None]/1200.],axis=1)
            sf=model.state_encoder(torch.tensor(state,dtype=torch.float32,device=device))
            correction_features.append(torch.cat([vision[:b],vision[b:],sf],dim=-1))
            print('CORRECTION_CACHE',instruction,end,'/',n,flush=True)
        offset+=n
cx=torch.cat(correction_features);correction_targets=torch.cat(correction_y)
correction_ids=torch.tensor(correction_ids,device=device)

def correction_forward():
    language_feature=model.encode_language(language)[correction_ids]
    hidden=model.fusion(torch.cat([cx,language_feature],dim=-1))
    return torch.tanh(model.arm_head(hidden)),model.gripper_head(hidden).squeeze(-1)

def audit_corrections():
    with torch.no_grad():
        arm,grip=correction_forward();result={}
        for name,sl in correction_groups:
            error=arm[sl]-correction_targets[sl,:6]
            result[name]={'samples':len(error),'translation_rmse_action_units':float(error[:,:3].square().mean().sqrt()),'rotation_rmse_action_units':float(error[:,3:].square().mean().sqrt()),'gripper_mismatches':int(((grip[sl]>=0)!=(correction_targets[sl,6]<0)).sum())}
        return result


anchor_specs=[(0,root/'collect_settle_grasp_anchors_above_20260918_001143_047522_JST/above/preservation_anchors.npz',442),(1,root/'collect_current_approach623_20260918_110730_275711_JST/pick/preservation_anchors.npz',623)]
anchor_features=[];anchor_actions=[];anchor_ids=[];anchor_groups=[];anchor_hashes={};offset=0
for tid,path,count in anchor_specs:
    report=json.loads((path.parent.parent/'results.json').read_text())
    assert report['collection_pass'] and all(report['trajectory_exactly_equal'].values()) and report['checkpoint_sha256']==source_hash
    anchor_hashes[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as data:
        assert data['frozen_features'].shape==(count,1088) and np.array_equal(data['steps'],np.arange(count))
        assert str(data['checkpoint_sha256'].item())==source_hash
        assert str(data['instruction'].item())==('move above the cube' if tid==0 else 'pick up the cube')
        anchor_features.append(torch.tensor(data['frozen_features'],dtype=torch.float32,device=device))
        anchor_actions.append(torch.tensor(data['actions'],dtype=torch.float32,device=device))
        anchor_ids.extend([tid]*count);anchor_groups.append((str(tid),slice(offset,offset+count)));offset+=count
ax=torch.cat(anchor_features);ay=torch.cat(anchor_actions);aids=torch.tensor(anchor_ids,device=device)
def anchor_forward():
    hidden=model.fusion(torch.cat([ax,model.encode_language(language)[aids]],dim=-1))
    return torch.tanh(model.arm_head(hidden)),model.gripper_head(hidden).squeeze(-1)
with torch.no_grad():
    base_anchor_arm,base_anchor_grip=anchor_forward()
    base_anchor_prob=base_anchor_grip.sigmoid().detach().clone()
    initial_difference=float((base_anchor_arm-ay[:,:6]).abs().max())
    assert initial_difference<=0.0001,('Cached feature/action mismatch',initial_difference)
    assert torch.equal(base_anchor_grip>=0,ay[:,6]<0)
print('ANCHOR_BASELINE_PARITY',initial_difference,flush=True)
def anchor_metrics(groups_to_check=None):
    with torch.no_grad():
        arm,grip=anchor_forward();rows={};passed=True
        for name,sl in (anchor_groups if groups_to_check is None else groups_to_check):
            e=arm[sl]-ay[sl,:6];rmse=float(e.square().mean().sqrt());maximum=float(e.abs().max())
            mismatches=int(((grip[sl]>=0)!=(ay[sl,6]<0)).sum())
            ok=rmse<=0.0005 and maximum<=0.002 and mismatches==0
            rows[name]={'arm_rmse_action_units':rmse,'max_abs_arm_difference':maximum,'gripper_mismatches':mismatches,'pass':ok};passed=passed and ok
        return {'pass':passed,'tasks':rows}


# Optimize hold fit subject to explicit normalized inequalities.
# No instruction/time routing; all original anchor intervals remain included.
with torch.no_grad():
    reference_arm,_=correction_forward()
    arm_normalizers=[]
    for _,sl in correction_groups:
        error=reference_arm[sl]-correction_targets[sl,:6]
        arm_normalizers.append((error[:,:3].square().mean().clamp_min(1e-12),error[:,3:].square().mean().clamp_min(1e-12)))
def objective_and_constraints():
    ca,cg=correction_forward()
    # Equal relative importance for translation and rotation in both hold phases.
    terms=[]
    for (_,sl),(translation_scale,rotation_scale) in zip(correction_groups,arm_normalizers):
        error=ca[sl]-correction_targets[sl,:6]
        terms.extend([error[:,:3].square().mean()/translation_scale,
                      error[:,3:].square().mean()/rotation_scale])
    hold=torch.stack(terms).mean()
    aa,ag=anchor_forward();gs=[]
    for _,sl in anchor_groups:
        err=aa[sl]-ay[sl,:6]
        gs.extend([err.square().mean().clamp_min(1e-20).sqrt()/.0005-1,
                   err.abs().max()/.002-1])
        sign=torch.where(ay[sl,6]<0,1.,-1.)
        gs.append((1e-5-sign*ag[sl]).max())
    pa,pg=forward(train);tt=targets[train]
    own=torch.linalg.vector_norm(pa[:,:,:3]-tt[:,:,:3],dim=-1)
    other=torch.linalg.vector_norm(pa[:,:,:3]-tt.flip(1)[:,:,:3],dim=-1)
    separation=torch.linalg.vector_norm(tt[:,0,:3]-tt[:,1,:3],dim=-1).clamp_min(1e-8)
    gs.extend([((own-other+1e-6)/separation[:,None]).max(),pg.max()+1e-5])
    teacher_sign=torch.where(correction_targets[:,6]<0,1.,-1.)
    gs.append((1e-5-teacher_sign*cg).max())
    return hold,torch.stack(gs)
with torch.no_grad():
    initial_obj,_=objective_and_constraints()
    normalizer=max(float(initial_obj),1e-8)
baseline_fit=audit_corrections()
best_obj=float(initial_obj);best_update=0
best_adapter=copy.deepcopy(adapter.state_dict())
assert anchor_metrics()['pass'] and metrics(train)['paired_correct']==16
multipliers=torch.zeros(9,device=device)
rho=1.0
optimizer=torch.optim.Adam(parameters,lr=1e-4)
start=time.monotonic();stop='update_limit';updates=0;feasible_updates=0
for update in range(1,8001):
    if time.monotonic()-start>=60:stop='time_limit';break
    optimizer.zero_grad()
    hold,g=objective_and_constraints()
    # Standard inequality augmented penalty. Multipliers grow on violated bounds.
    positive=torch.relu(multipliers+rho*g)
    loss=hold/normalizer+(positive.square()-multipliers.square()).sum()/(2*rho)
    if not torch.isfinite(loss):stop='nonfinite_loss';break
    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10.)
    optimizer.step();updates=update
    with torch.no_grad():
        obj,post_g=objective_and_constraints()
        if update%20==0:multipliers=torch.relu(multipliers+rho*post_g)
        if bool((post_g<=0).all()):
            # Select by measured preservation and training language classification too.
            if anchor_metrics()['pass'] and metrics(train)['paired_correct']==16:
                feasible_updates+=1
                if float(obj)<best_obj and all(v['translation_rmse_action_units']<=baseline_fit[k]['translation_rmse_action_units'] and v['rotation_rmse_action_units']<=baseline_fit[k]['rotation_rmse_action_units'] for k,v in audit_corrections().items()):
                    best_obj=float(obj);best_update=update
                    best_adapter=copy.deepcopy(adapter.state_dict())
    if update%400==0:
        print('CONSTRAINT_PROGRESS',json.dumps({'update':update,'max_normalized_violation':max(0.,float(post_g.max())),
            'feasible_updates':feasible_updates,'best_feasible_update':best_update}),flush=True)
adapter.load_state_dict(best_adapter);model.eval()
assert sha(source)==source_hash and sha(vertical_path)==vertical_hash
assert all(sha(Path(p))==v for p,v in anchor_hashes.items())
assert all(torch.equal(v,checkpoint['model']['fusion.'+k]) for k,v in model.fusion[0].state_dict().items())
assert all(torch.equal(v,checkpoint['model'][k]) for k,v in model.state_dict().items() if not k.startswith('fusion.'))
print('CURRENT_HOLD_TRAINING_RESULT',json.dumps({'source_unchanged':True,
    'updates':updates,'stop_reason':stop,'optimization_seconds':round(time.monotonic()-start,2),
    'feasible_updates':feasible_updates,'selected_update':best_update,
    'improved_feasible_candidate':best_update>0,
    'teacher_label_gate':'All 60 supervised gripper labels must match for candidate selection; not a rule prohibiting physical grasp adjustments',
    'teacher_label_gate_pass':all(row['gripper_mismatches']==0 for row in audit_corrections().values()),
    'objective':'Mean of four arm MSE ratios to original baseline; no gripper BCE in objective. Gripper correctness remains a constraint.',
    'baseline_arm_objective':normalizer,'selected_arm_objective':best_obj,
    'baseline_hold_fit':baseline_fit,'selected_hold_fit':audit_corrections(),
    'full_preservation':anchor_metrics(),
    'thresholds':{'arm_rmse_max':.0005,'arm_max_abs_max':.002,'anchor_gripper_mismatches_max':0},
    'language_train':metrics(train),'language_development':metrics(heldout),
    'model_saved':False,'simulation_executed':False,
    'scope':'Offline constrained fitting with all original anchors. Selected update0 means original output retained, not an improved candidate. Hold fit is not physical grasp success.'}),flush=True)
fit=audit_corrections();preservation=anchor_metrics()
report={'source_unchanged':sha(source)==source_hash,'base_checkpoint':str(source),'base_sha256':source_hash,
    'updates':updates,'stop_reason':stop,'selected_update':best_update,
    'baseline_arm_objective':normalizer,'arm_objective':best_obj,
    'baseline_hold_fit':baseline_fit,'hold_fit':fit,'full_preservation':preservation,
    'thresholds':{'arm_rmse_max':.0005,'arm_max_abs_max':.002,'anchor_gripper_mismatches_max':0},
    'language_train':metrics(train),'language_development':metrics(heldout),
    'teacher_data':str(vertical_path),'teacher_sha256':vertical_hash,'anchor_hashes':anchor_hashes,
    'model_code_sha256':sha(system/'code/language_vla/model.py'),
    'torch_version':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),
    'model_saved':False,'simulation_executed':False,'runtime_integration_verified':False,
    'scope':'Current 623-step pick approach anchors and current teacher hold data; shared adapter, no inference stage switching. Offline improvement does not certify grasp.'}
release_reserve()
if best_update>0:
    assert preservation['pass'] and all(v['gripper_mismatches']==0 for v in fit.values())
    payload={'format':'shared_fusion_residual_v1','base_checkpoint':str(source),'base_sha256':source_hash,
        'architecture':{'input':256,'hidden':64,'output':256,'activation':'tanh','operation':'x + net(x)'},
        'adapter_state_dict':{k:v.detach().cpu() for k,v in adapter.state_dict().items()}}
    temp=output_dir/'adapter_writing.pt';dest=output_dir/'hold_adapter.pt'
    try:
        torch.save(payload,temp)
        loaded=torch.load(temp,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,loaded['adapter_state_dict'][k]) for k,v in payload['adapter_state_dict'].items())
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    report.update(adapter_file=str(dest),adapter_sha256=sha(dest),adapter_bytes=dest.stat().st_size,
        weights_roundtrip_pass=True,model_saved=True)
with (output_dir/'results.json').open('x',encoding='utf-8') as f:
    json.dump(report,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
print('CURRENT_HOLD_TRAIN_AND_SAVE_RESULT',json.dumps(report,ensure_ascii=False),flush=True)
print('RESULT_FILE',output_dir/'results.json',flush=True)
PY
