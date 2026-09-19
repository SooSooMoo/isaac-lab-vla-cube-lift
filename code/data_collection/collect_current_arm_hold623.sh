PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -u - <<'PY'
from pathlib import Path
import json,hashlib
base=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST/gripper_actual_fix_20260917_142308_983342')
_candidate=str(base/'shared_full_candidate.pt')
assert hashlib.sha256(Path(_candidate).read_bytes()).hexdigest()=='3ad705b1224f552ccf6988364f5c9894fb55ab1003f390e5d04719033142177c'
from datetime import datetime, timezone, timedelta
import os, sys, json, hashlib, subprocess
import numpy as np

root=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST')
system=root/'train_lift_stage_20260916_185701_630638_JST'
checkpoint=Path(_candidate)
ngx=Path('/workspace/step3/ngx_runtime_580.65.06_cgbe5u2a/lib')
assert checkpoint.is_file() and (ngx/'libnvidia-ngx.so.580.65.06').is_file()
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()

out=root/('collect_settle_then_close623_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f_JST'))
out.mkdir();dest=out/'pick'
# Exercise the volume quota, rather than relying on shared-filesystem df output.
probe=out/'capacity_probe.tmp'
created=False
try:
    with probe.open('xb') as f:
        created=True
        for _ in range(32):f.write(os.urandom(1024*1024))
        f.flush();os.fsync(f.fileno())
    print('ACTUAL_WRITE_32MIB_PASS',flush=True)
finally:
    if created:probe.unlink()

print('OUTPUT_DIRECTORY',out,flush=True)

import shutil
original_system=system
system=out/'runtime'
shutil.copytree(original_system/'code',system/'code',ignore=shutil.ignore_patterns('__pycache__'))
(system/'two_instruction.hdf5').symlink_to(root/'two_instruction.hdf5')
epath=system/'code/evaluate.py';s=epath.read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,old
 s=s.replace(old,new,1)
replace('from smoke_utils import load_policy','from smoke_utils import load_policy, capture_obs')
replace('        predict = load_policy(args)',"        predict = load_policy(args)\n        recovery=[]\n        recovery_stage='align'\n        recovery_close=0\n        recovery_anchor=None")
override="\n                phase = 'policy'\n                if step >= 623:\n                    if recovery_anchor is None:\n                        recovery_anchor=(pos_w.clone(),quat_w.clone())\n                    tp,tq=subtract_frame_transforms(root_pos,root_quat,recovery_anchor[0],recovery_anchor[1])\n                    pe,re=compute_pose_error(ik_pos_b,ik_quat_b,tp,tq,rot_error_type='axis_angle')\n                    action=torch.cat([bounded(1.0*pe,.01)/scale,bounded(.2*re,.08)/scale,torch.full((1,1),1.0 if step < 643 else -1.0,device=env.device)],dim=-1)\n                    phase='teacher_stationary_close'\n                    recovery.append((step,capture_obs(obs),array(action[0]),'settle_open' if step < 643 else 'stationary_close'))\n"
replace("                phase = 'policy'",override)
save="""
        data={'steps':np.asarray([r[0] for r in recovery]),'teacher_actions':np.stack([r[2] for r in recovery]),'phase':np.asarray([r[3] for r in recovery]),'instruction':np.asarray(args.instruction),'progress_denominator':np.asarray(1200.)}
        for key in recovery[0][1]:data['obs_'+key]=np.stack([r[1][key] for r in recovery])
        np.savez_compressed(out/'vertical_recovery.npz',**data)
"""
replace("        np.savez_compressed(out / 'teacher_trace.npz',",save+"        np.savez_compressed(out / 'teacher_trace.npz',")
replace("        (out / 'summary.json').write_text", "        summary['teacher_recovery_success']=summary['success']\n        summary['success']=False\n        summary['vla_success']=False\n        summary['teacher_at_evaluation']=True\n        summary['teacher_type']='shared_policy_first550_then_teacher_recovery'\n        summary['learned_language_policy']=False\n        summary['limitations']=['Teacher-assisted recovery, not VLA task success; geometric lift proxy.']\n        (out / 'summary.json').write_text")
compile(s,str(epath),'exec');epath.write_text(s)

text=(root/'pick_branch_20260916_150628_449371_JST/evaluate_above.py').read_text()
anchor='import smoke_utils\n'
assert text.count(anchor)==1
replacement="\ndef shared_load(args):\n    import torch\n    import numpy as np\n    from language_vla.model import LanguageConditionedVLA\n    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)\n    model=LanguageConditionedVLA(**{k:int(ck[k]) for k in ('state_dim','action_dim','vocab_size','language_dim')}).to(args.device)\n    model.load_state_dict(ck['model']);model.eval();model._pick_mode=False\n\n    from pathlib import Path\n    import hashlib\n    ap=Path('/workspace/step3/hold_adapter_preserved_oypw0cwr/hold_adapter.pt')\n    assert hashlib.sha256(ap.read_bytes()).hexdigest()=='b9c4736114d1051b533329696c0ceb3f993ba3db46b99e97bd4fb1c34f6a7e9a'\n    pack=torch.load(ap,map_location=args.device,weights_only=False)\n    assert pack['base_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()\n    assert pack['format']=='shared_fusion_residual_v1'\n    class SharedResidual(torch.nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.net=torch.nn.Sequential(torch.nn.Linear(256,64),torch.nn.Tanh(),torch.nn.Linear(64,256))\n        def forward(self,x):return x+self.net(x)\n    adapter=SharedResidual().to(args.device)\n    adapter.load_state_dict(pack['adapter_state_dict'])\n    with torch.no_grad():\n        adapter.net[-1].weight.zero_();adapter.net[-1].bias.zero_()\n    print('CONTROL_ADAPTER_DISABLED',True,flush=True)\n    model.fusion=torch.nn.Sequential(model.fusion,adapter)\n    model.eval();model._pick_mode=False\n    report=json.loads((ap.parent/'results.json').read_text())\n    import language_vla.model as model_module\n    assert hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest()==report['model_code_sha256']\n    runtime_checks=[]\n    for path,expected_hash in report['anchor_hashes'].items():\n        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_hash\n        with np.load(path,allow_pickle=False) as anchor_data,torch.inference_mode():\n            feat=torch.tensor(anchor_data['frozen_features'],device=args.device,dtype=torch.float32)\n            lab=torch.tensor(anchor_data['actions'],device=args.device,dtype=torch.float32)\n            instruction=str(anchor_data['instruction'].item())\n            tok=torch.tensor(smoke_utils.tokens(instruction),device=args.device)[None]\n            lang=model.encode_language(tok).expand(len(feat),-1)\n            hidden=model.fusion(torch.cat([feat,lang],-1))\n            arm=torch.tanh(model.arm_head(hidden));grip=model.gripper_head(hidden).squeeze(-1)\n            err=arm-lab[:,:6]\n            rmse=float(err.square().mean().sqrt());maximum=float(err.abs().max())\n            mismatches=int(((grip>=0)!=(lab[:,6]<0)).sum())\n            assert rmse<=.0005 and maximum<=.002 and mismatches==0,(instruction,rmse,maximum,mismatches)\n            runtime_checks.append({'instruction':instruction,'rmse':rmse,'max_difference':maximum,'gripper_mismatches':mismatches})\n    print('ADAPTER_RUNTIME_CHECK_PASS',json.dumps(runtime_checks),flush=True)\n    language=torch.tensor(smoke_utils.tokens(args.instruction),device=args.device)[None]\n    mean=torch.tensor([.485,.456,.406],device=args.device)[None,:,None,None]\n    std=torch.tensor([.229,.224,.225],device=args.device)[None,:,None,None]\n    print('SHARED_POLICY_LOADED',json.dumps({'checkpoint':str(args.checkpoint),'shared_head':True,'stage_switch':False,'instruction':args.instruction}),flush=True)\n    def predict(obs,step):\n        captured=smoke_utils.capture_obs(obs)\n        def image(key):\n            value=torch.tensor(captured[key],device=args.device).permute(2,0,1).float()[None]/255.\n            return (value-mean)/std\n        quat=np.asarray(captured['eef_quat']).copy()\n        if quat[3]<0:quat=-quat\n        state=np.concatenate([captured['eef_pos'],quat,captured['gripper_pos'],[step/1200.]])\n        with torch.inference_mode():\n            result=model(image('table_cam'),image('wrist_cam'),torch.tensor(state,dtype=torch.float32,device=args.device)[None],language)\n            action=model.build_action(result).cpu().numpy()[0].copy()\n        assert np.isfinite(action).all()\n        return action\n    return predict\nsmoke_utils.load_policy=shared_load\n"
text=text.replace(anchor,anchor+"\nfrom PIL import Image as _FrameImage\n_original_frame_save=_FrameImage.Image.save\ndef _save_without_frames(self,fp,*a,**kw):\n    if isinstance(fp,(str,Path)):\n        p=Path(fp)\n        if p.parent.resolve()==(output/'frames').resolve() and p.suffix.lower()=='.png':return None\n    return _original_frame_save(self,fp,*a,**kw)\n_FrameImage.Image.save=_save_without_frames\n"+replacement,1)
assert text.count("'--horizon', '550'")==1
text=text.replace("'--horizon', '550'","'--horizon', '683'")
compile(text,'shared_above.py','exec')
wrapper=out/'shared_pick.py';wrapper.write_text(text)
spec={'checkpoint':str(checkpoint),'sha256':digest,'horizon':683,'progress_denominator':1200.0,'shared_head':True,'teacher_takeover_step':623,'teacher_at_evaluation':True,'ngx':str(ngx),'scope':'Fixed initial state; pick task evaluation using same unchanged shared model'}
(out/'protocol.json').write_text(json.dumps(spec,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
env['LD_LIBRARY_PATH']=str(ngx)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
log=out/'pick.log'
command=['timeout','300s','/workspace/IsaacLab-develop/isaaclab.sh','-p',str(wrapper),str(system),str(checkpoint),'pick up the cube',str(dest)]
with log.open('w') as f:
    proc=subprocess.Popen(command,cwd='/workspace/IsaacLab-develop',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        f.write(line);f.flush()
        if any(x in line for x in ('[STEP]','INITIAL_INPUT_CHECK','SHARED_POLICY','Traceback','Error','TORCH_SETTINGS')):print(line.rstrip(),flush=True)
    rc=proc.wait()

assert rc==0,'Recovery failed; inspect log'
initial=json.loads((dest/'initial_input_check.json').read_text())
reference=root/'verify_base_approach800_pick_20260918_104016_376435_JST/pick/teacher_trace.npz'
with np.load(dest/'teacher_trace.npz') as t,np.load(reference) as old,np.load(dest/'vertical_recovery.npz') as data:
    keys=('actions','pre_eef','post_eef','pre_quat_xyzw','post_quat_xyzw','post_cube','gripper_pos')
    parity={k:bool(np.array_equal(t[k][:623],old[k][:623])) for k in keys}
    assert len(t['actions'])==683 and np.array_equal(data['steps'],np.arange(623,683))
    assert np.array_equal(data['teacher_actions'],t['actions'][623:])
    eef=t['post_eef'][623:];cube=t['post_cube'][623:]
    eef_drift=float(np.linalg.norm(eef-t['pre_eef'][623],axis=1).max()*1000)
    cube_xy=float(np.linalg.norm(cube[:,:2]-t['post_cube'][622,:2],axis=1).max()*1000)
    relative_range=float(np.ptp((eef-cube)[40:],axis=0).max()*1000)
    closed=bool(np.all(t['actions'][643:,6]<0))
    settle_open=bool(np.all(t['actions'][623:643,6]>0))
    valid=bool(initial['pass'] and all(parity.values()) and closed and settle_open)
    result={'collection_pass':valid,'first623_exactly_equal':parity,'samples':60,'settle_open_steps':20,'hold_target_reanchored':False,'translation_gain':1.0,'translation_step_limit_m':0.01,'teacher_actions_executed':True,'training_performed':False,'vla_success':False,'eef_max_displacement_mm':eef_drift,'cube_xy_max_displacement_mm':cube_xy,'last20_relative_range_mm':relative_range,'continuous_closed40':closed,'arm_hold_pass':bool(valid and eef_drift<=5),'arm_limit_mm':5.0,'relative_stability_evaluated_for_pass':False,'secure_contact_verified':False,'data_file':str(dest/'vertical_recovery.npz'),'checkpoint_sha256':digest}
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
(out/'results.json').write_text(json.dumps(result,indent=2))
print('CURRENT_ARM_HOLD_TEACHER_RESULT',json.dumps(result),flush=True)
print('RESULT_FILE',out/'results.json',flush=True)

PY

