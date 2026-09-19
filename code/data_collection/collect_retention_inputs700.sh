PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -u - <<'PY'
from pathlib import Path
import os,json,hashlib,tempfile
base=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST/gripper_actual_fix_20260917_142308_983342/shared_full_candidate.pt')
adapter_path=Path('/workspace/step3/visual_consistency13_th1jfnmy/visual_adapter.pt')
assert hashlib.sha256(base.read_bytes()).hexdigest()=='3ad705b1224f552ccf6988364f5c9894fb55ab1003f390e5d04719033142177c'
assert hashlib.sha256(adapter_path.read_bytes()).hexdigest()=='aee282b49bc44f0d57255c16e994c402d97cb00621a2bc0190b6e43c9fb71f23'
fd,probe=tempfile.mkstemp(prefix='hold_eval_capacity_',dir='/workspace/step3')
try:
    with os.fdopen(fd,'wb') as f:
        for _ in range(8):f.write(os.urandom(1024*1024))
        f.flush();os.fsync(f.fileno())
finally:
    Path(probe).unlink(missing_ok=True)
print('ACTUAL_WRITE_8MIB_PASS NO_TRAINING',flush=True)

_candidate=str(base)
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, sys, json, hashlib, subprocess
import numpy as np

root=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST')
system=root/'train_lift_stage_20260916_185701_630638_JST'
checkpoint=Path(_candidate)
ngx=Path('/workspace/step3/ngx_runtime_580.65.06_cgbe5u2a/lib')
assert checkpoint.is_file() and (ngx/'libnvidia-ngx.so.580.65.06').is_file()
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()

out=root/('collect_retention_inputs700_pick_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f_JST'))
out.mkdir();dest=out/'pick'
print('OUTPUT_DIRECTORY',out,flush=True)
text=(root/'pick_branch_20260916_150628_449371_JST/evaluate_above.py').read_text()
anchor='import smoke_utils\n'
assert text.count(anchor)==1
replacement='''
def shared_load(args):
    import torch
    import numpy as np
    from language_vla.model import LanguageConditionedVLA
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model=LanguageConditionedVLA(**{k:int(ck[k]) for k in ('state_dim','action_dim','vocab_size','language_dim')}).to(args.device)
    model.load_state_dict(ck['model']);model.eval();model._pick_mode=False

    from pathlib import Path
    import hashlib
    ap=Path('/workspace/step3/visual_consistency13_th1jfnmy/visual_adapter.pt')
    assert hashlib.sha256(ap.read_bytes()).hexdigest()=='aee282b49bc44f0d57255c16e994c402d97cb00621a2bc0190b6e43c9fb71f23'
    pack=torch.load(ap,map_location=args.device,weights_only=False)
    assert pack['base_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    assert pack['format']=='shared_fusion_arm_residual_v2'
    class SharedResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))
        def forward(self,x):return x+self.net(x)
    initial_adapter=SharedResidual().to(args.device)
    initial_adapter.load_state_dict(pack['initial_adapter_state_dict'])
    class ArmOnlyResidual(torch.nn.Module):
        def __init__(self,w):
            super().__init__()
            self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))
            self.register_buffer('gripper_direction',w.detach().reshape(-1).clone())
        def forward(self,x):
            delta=self.net(x);w=self.gripper_direction
            return x+delta-(delta*w).sum(-1,keepdim=True)/w.square().sum().clamp_min(1e-12)*w
    assert isinstance(model.gripper_head,torch.nn.Linear) and model.gripper_head.weight.shape==(1,256)
    adapter=ArmOnlyResidual(model.gripper_head.weight).to(args.device)
    adapter.load_state_dict(pack['adapter_state_dict'])
    assert torch.equal(adapter.gripper_direction,model.gripper_head.weight.detach().reshape(-1))
    model.fusion=torch.nn.Sequential(model.fusion,initial_adapter,adapter)
    model.eval();model._pick_mode=False
    report=json.loads((ap.parent/'results.json').read_text())
    import language_vla.model as model_module
    assert hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest()=='57ba83d3a7eab86f46fd9b75315def0dc684fe3a4e7672b1f18423f57dafab58'
    runtime_checks=[]
    for path,expected_hash in report['anchor_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_hash
        with np.load(path,allow_pickle=False) as anchor_data,torch.inference_mode():
            feat=torch.tensor(anchor_data['frozen_features'],device=args.device,dtype=torch.float32)
            lab=torch.tensor(anchor_data['actions'],device=args.device,dtype=torch.float32)
            instruction=str(anchor_data['instruction'].item())
            tok=torch.tensor(smoke_utils.tokens(instruction),device=args.device)[None]
            lang=model.encode_language(tok).expand(len(feat),-1)
            hidden=model.fusion(torch.cat([feat,lang],-1))
            arm=torch.tanh(model.arm_head(hidden));grip=model.gripper_head(hidden).squeeze(-1)
            err=arm-lab[:,:6]
            rmse=float(err.square().mean().sqrt());maximum=float(err.abs().max())
            mismatches=int(((grip>=0)!=(lab[:,6]<0)).sum())
            assert rmse<=.0005 and maximum<=.002 and mismatches==0,(instruction,rmse,maximum,mismatches)
            runtime_checks.append({'instruction':instruction,'rmse':rmse,'max_difference':maximum,'gripper_mismatches':mismatches})
    print('ADAPTER_RUNTIME_CHECK_PASS',json.dumps(runtime_checks),flush=True)
    language=torch.tensor(smoke_utils.tokens(args.instruction),device=args.device)[None]
    mean=torch.tensor([.485,.456,.406],device=args.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=args.device)[None,:,None,None]
    print('SHARED_POLICY_LOADED',json.dumps({'checkpoint':str(args.checkpoint),'shared_head':True,'stage_switch':False,'instruction':args.instruction}),flush=True)
    retention_rows=[]
    retention_cache={}
    def capture_hidden(module,inputs,value):
        retention_cache['hidden']=value.detach().cpu().numpy()[0].copy()
    model.fusion.register_forward_hook(capture_hidden)
    def predict(obs,step):
        captured=smoke_utils.capture_obs(obs)
        def image(key):
            value=torch.tensor(captured[key],device=args.device).permute(2,0,1).float()[None]/255.
            return (value-mean)/std
        quat=np.asarray(captured['eef_quat']).copy()
        if quat[3]<0:quat=-quat
        state=np.concatenate([captured['eef_pos'],quat,captured['gripper_pos'],[step/1200.]])
        with torch.inference_mode():
            result=model(image('table_cam'),image('wrist_cam'),torch.tensor(state,dtype=torch.float32,device=args.device)[None],language)
            action=model.build_action(result).cpu().numpy()[0].copy()
        assert np.isfinite(action).all()
        original_action=action.copy()
        if step>=614:action[6]=-1.0
        retention_rows.append((step,retention_cache['hidden'].copy(),original_action,action.copy()))
        if step==699:
            np.savez_compressed(output/'retention_inputs.npz',
                steps=np.array([v[0] for v in retention_rows]),
                hidden=np.stack([v[1] for v in retention_rows]),
                policy_actions=np.stack([v[2] for v in retention_rows]),
                executed_actions=np.stack([v[3] for v in retention_rows]))
            print('RETENTION_INPUTS_SAVED',len(retention_rows),flush=True)
        return action
    return predict
smoke_utils.load_policy=shared_load
'''
text=text.replace(anchor,anchor+"\nfrom PIL import Image as _FrameImage\n_original_frame_save=_FrameImage.Image.save\ndef _save_without_frames(self,fp,*a,**kw):\n    if isinstance(fp,(str,Path)):\n        p=Path(fp)\n        if p.parent.resolve()==(output/'frames').resolve() and p.suffix.lower()=='.png':return None\n    return _original_frame_save(self,fp,*a,**kw)\n_FrameImage.Image.save=_save_without_frames\n"+replacement,1)
assert text.count("'--horizon', '550'")==1
text=text.replace("'--horizon', '550'","'--horizon', '700'")
compile(text,'shared_above.py','exec')
wrapper=out/'shared_pick.py';wrapper.write_text(text)
spec={'checkpoint':str(checkpoint),'sha256':digest,'horizon':700,'progress_denominator':1200.0,'shared_head':True,'stage_switch':False,'gripper_override_from_step':614,'teacher_at_evaluation':False,'ngx':str(ngx),'scope':'Fixed initial state; pick task evaluation using same unchanged shared model'}
(out/'protocol.json').write_text(json.dumps(spec,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
env['LD_LIBRARY_PATH']=str(ngx)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
log=out/'pick.log'
command=['timeout','300s','/workspace/IsaacLab-develop/isaaclab.sh','-p',str(wrapper),str(system),str(checkpoint),'pick up the cube',str(dest)]
with log.open('w') as f:
    proc=subprocess.Popen(command,cwd='/workspace/IsaacLab-develop',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        f.write(line);f.flush()
        if any(x in line for x in ('ADAPTER_RUNTIME_CHECK','[STEP]','INITIAL_INPUT_CHECK','SHARED_POLICY','Traceback','Error','TORCH_SETTINGS')):print(line.rstrip(),flush=True)
    rc=proc.wait()

assert rc==0, ('Evaluation failed',str(log),rc)
reference=root/'diagnose_keep_closed614_pick_20260918_162950_201300_JST/pick/teacher_trace.npz'
initial=json.loads((dest/'initial_input_check.json').read_text())
assert initial['pass']
with np.load(reference,allow_pickle=False) as a,np.load(dest/'teacher_trace.npz',allow_pickle=False) as b,np.load(dest/'retention_inputs.npz',allow_pickle=False) as z:
    keys=('actions','pre_eef','post_eef','pre_quat_xyzw','post_quat_xyzw','post_cube','gripper_pos')
    same={k:bool(np.array_equal(a[k],b[k])) for k in keys}
    assert all(same.values()),same
    assert z['hidden'].shape==(700,256) and np.isfinite(z['hidden']).all()
    assert np.array_equal(z['steps'],np.arange(700))
    assert np.array_equal(z['executed_actions'],b['actions'])
    result={'collection_pass':True,'samples':700,'retention_override_samples':86,'trajectory_exactly_equal':same,'data_file':str(dest/'retention_inputs.npz'),'adapter_sha256':hashlib.sha256(adapter_path.read_bytes()).hexdigest(),'training_performed':False,'vla_success':False,'scope':'Cached final fusion features from unchanged model; executed closure after614 is a diagnostic intervention target, not autonomous success. Earlier actions retained for preservation.'}
(out/'results.json').write_text(json.dumps(result,indent=2))
print('RETENTION_INPUT_COLLECTION_RESULT',json.dumps(result),flush=True)
print('RESULT_FILE',out/'results.json',flush=True)
PY
