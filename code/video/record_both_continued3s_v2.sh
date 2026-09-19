set -e
PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -u - <<'PY'
from pathlib import Path
import os,json,hashlib,tempfile
import subprocess as _preflight_subprocess
try:
    import imageio_ffmpeg as _video_dependency
    _encoder=_video_dependency.get_ffmpeg_exe()
    _preflight_subprocess.run([_encoder,'-version'],check=True,stdout=_preflight_subprocess.DEVNULL,stderr=_preflight_subprocess.PIPE)
except Exception as exc:
    raise RuntimeError('VIDEO_PREFLIGHT_FAILED: '+repr(exc)+'; if imageio_ffmpeg is missing, run /isaac-sim/python.sh -m pip install --no-cache-dir imageio-ffmpeg') from exc
print('VIDEO_PREFLIGHT_PASS',_encoder,flush=True)
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

out=root/('record_continued3s_v2_pick_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f_JST'))
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
    cp=Path('/workspace/step3/retention_gripper_da00uipp/gripper_correction.pt')
    assert hashlib.sha256(cp.read_bytes()).hexdigest()=='48fc3c58dea3deaef9fbae6d1208d7df522967ff841162f5077f88eab0143863'
    correction_pack=torch.load(cp,map_location=args.device,weights_only=False)
    assert correction_pack['adapter_sha256']==hashlib.sha256(ap.read_bytes()).hexdigest()
    assert correction_pack['base_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    correction=torch.nn.Linear(256,1).to(args.device)
    correction.load_state_dict(correction_pack['state_dict'])
    with np.load(correction_pack['data_file'],allow_pickle=False) as data,torch.inference_mode():
        assert hashlib.sha256(Path(correction_pack['data_file']).read_bytes()).hexdigest()==correction_pack['data_sha256']
        h=torch.tensor(data['hidden'],device=args.device,dtype=torch.float32)
        target=torch.tensor(data['executed_actions'][:,6]<0,device=args.device)
        pred=(model.gripper_head(h)+correction(h)).squeeze(-1)>=0
        assert torch.equal(pred,target),'Retention reload mismatch'
    for path in report['anchor_hashes']:
        with np.load(path,allow_pickle=False) as data,torch.inference_mode():
            instruction=str(data['instruction'].item())
            if instruction!='move above the cube':continue
            feat=torch.tensor(data['frozen_features'],device=args.device,dtype=torch.float32)
            lang=model.encode_language(torch.tensor(smoke_utils.tokens(instruction),device=args.device)[None]).expand(len(feat),-1)
            h=model.fusion(torch.cat([feat,lang],-1))
            target=torch.tensor(data['actions'][:,6]<0,device=args.device)
            assert torch.equal((model.gripper_head(h)+correction(h)).squeeze(-1)>=0,target),'Above reload mismatch'
    class CorrectedGrip(torch.nn.Module):
        def __init__(self,original,extra):
            super().__init__();self.original=original;self.extra=extra
        def forward(self,x):return self.original(x)+self.extra(x)
    model.gripper_head=CorrectedGrip(model.gripper_head,correction)
    model.eval()
    print('LEARNED_GRIP_RELOAD_PASS NO_TIME_OVERRIDE',flush=True)
    language=torch.tensor(smoke_utils.tokens(args.instruction),device=args.device)[None]
    mean=torch.tensor([.485,.456,.406],device=args.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=args.device)[None,:,None,None]
    print('SHARED_POLICY_LOADED',json.dumps({'checkpoint':str(args.checkpoint),'shared_head':True,'stage_switch':False,'instruction':args.instruction}),flush=True)
    xp=Path('/workspace/step3/learned_xy_attenuation__zkv0syj/xy_attenuation.pt')
    assert hashlib.sha256(xp.read_bytes()).hexdigest()=='0f5b74efb4c9f34be3a9ab6b8985117869d8e6e1ff9565a1ab277f88ec2e3de5'
    xpack=torch.load(xp,map_location=args.device,weights_only=False)
    assert xpack['base_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    assert xpack['adapter_sha256']==hashlib.sha256(ap.read_bytes()).hexdigest()
    assert xpack['gripper_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
    selector=torch.nn.Linear(256,1).to(args.device)
    selector.load_state_dict(xpack['state_dict']);selector.eval()
    center=xpack['feature_mean'];spread=xpack['feature_std']
    assert torch.isfinite(center).all() and (spread>0).all()
    assert hashlib.sha256(Path(xpack['data_file']).read_bytes()).hexdigest()==xpack['data_sha256']
    with np.load(xpack['data_file'],allow_pickle=False) as data,torch.inference_mode():
        features=torch.tensor(data['hidden'],device=args.device,dtype=torch.float32)
        scores=selector((features-center)/spread).squeeze(-1)
        assert bool((scores[:800]<=0).all()) and bool((scores[800:]>=1).all()),'XY retention reload failed'
    for path in report['anchor_hashes']:
        with np.load(path,allow_pickle=False) as data,torch.inference_mode():
            instruction=str(data['instruction'].item())
            if instruction!='move above the cube':continue
            feat=torch.tensor(data['frozen_features'],device=args.device,dtype=torch.float32)
            lang=model.encode_language(torch.tensor(smoke_utils.tokens(instruction),device=args.device)[None]).expand(len(feat),-1)
            hidden=model.fusion(torch.cat([feat,lang],-1))
            assert bool((selector((hidden-center)/spread)<=0).all()),'Above attenuation reload failed'
    gate_cache={}
    def capture_attenuation(module,inputs,value):gate_cache['hidden']=value.detach()
    model.fusion.register_forward_hook(capture_attenuation)
    print('XY_ATTENUATION_RELOAD_PASS',flush=True)
    import subprocess as video_subprocess
    import imageio_ffmpeg
    video_process=None
    video_frames=0
    def record_frame(captured,step):
        nonlocal video_process,video_frames
        frame=np.ascontiguousarray(np.concatenate([captured['table_cam'],captured['wrist_cam']],axis=1),dtype=np.uint8)
        assert frame.ndim==3 and frame.shape[2]==3
        if video_process is None:
            height,width=frame.shape[:2]
            video_process=video_subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',str(width)+'x'+str(height),'-r','50','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'policy_view_continued3s.mp4')],stdin=video_subprocess.PIPE,stderr=video_subprocess.DEVNULL)
        video_process.stdin.write(frame.tobytes());video_frames+=1
        if step==1028:
            video_process.stdin.close()
            assert video_process.wait(timeout=30)==0,'Video encoder failed'
            print('VIDEO_SAVED',str(output/'policy_view_continued3s.mp4'),video_frames,flush=True)
    def predict(obs,step):
        captured=smoke_utils.capture_obs(obs)
        record_frame(captured,step)
        def image(key):
            value=torch.tensor(captured[key],device=args.device).permute(2,0,1).float()[None]/255.
            return (value-mean)/std
        quat=np.asarray(captured['eef_quat']).copy()
        if quat[3]<0:quat=-quat
        state=np.concatenate([captured['eef_pos'],quat,captured['gripper_pos'],[step/1200.]])
        with torch.inference_mode():
            result=model(image('table_cam'),image('wrist_cam'),torch.tensor(state,dtype=torch.float32,device=args.device)[None],language)
            action=model.build_action(result).cpu().numpy()[0].copy()
        with torch.inference_mode():
            gate=float(selector((gate_cache['hidden']-center)/spread).clamp(0,1).item())
        action[:2]*=(1.0-gate)
        assert np.isfinite(action).all()
        return action
    return predict
smoke_utils.load_policy=shared_load
'''
text=text.replace(anchor,anchor+"\nfrom PIL import Image as _FrameImage\n_original_frame_save=_FrameImage.Image.save\ndef _save_without_frames(self,fp,*a,**kw):\n    if isinstance(fp,(str,Path)):\n        p=Path(fp)\n        if p.parent.resolve()==(output/'frames').resolve() and p.suffix.lower()=='.png':return None\n    return _original_frame_save(self,fp,*a,**kw)\n_FrameImage.Image.save=_save_without_frames\n"+replacement,1)
assert text.count("'--horizon', '550'")==1
text=text.replace("'--horizon', '550'","'--horizon', '1300'")

import ast
runtime_source=(system/'code/evaluate.py').read_text()
class ExtendSuccess(ast.NodeTransformer):
    def __init__(self):self.loops=0;self.stops=0
    def visit_For(self,node):
        if ast.unparse(node.iter)!='range(args.horizon)':return self.generic_visit(node)
        self.loops+=1
        for statement in node.body:
            if isinstance(statement,ast.If) and 'args.instruction' in ast.unparse(statement.test) and any(isinstance(v,ast.Break) for v in statement.body) and any(isinstance(v,ast.Assign) and 'success' in ast.unparse(v.targets[0]) for v in statement.body):
                statement.body=[v for v in statement.body if not isinstance(v,ast.Break)]
                statement.body+=ast.parse("if _first_success_step is None:\n    _first_success_step=step\n").body
                self.stops+=1
        node.body+=ast.parse("if _first_success_step is not None and step >= _first_success_step+150:\n    reason='success_then_150_policy_steps'\n    (out/'continuation.json').write_text(json.dumps({'first_success_step':_first_success_step,'last_step':step,'extra_steps':step-_first_success_step,'dt':dt}))\n    break\n").body
        return ast.parse('_first_success_step=None').body+[node]
transformer=ExtendSuccess();tree=transformer.visit(ast.parse(runtime_source));ast.fix_missing_locations(tree)
assert transformer.loops==1 and transformer.stops==2,('Unexpected runtime stop structure',transformer.loops,transformer.stops)
class SaveRuntimeError(ast.NodeTransformer):
    def visit_Try(self,node):
        self.generic_visit(node)
        if any('env.close()' in ast.unparse(v) for v in node.finalbody) and not node.handlers:
            handler=ast.parse("try:\n    pass\nexcept BaseException:\n    import traceback\n    error=traceback.format_exc()\n    print('RUNTIME_ERROR_BEFORE_CLOSE',error,flush=True)\n    (out/'runtime_error.txt').write_text(error)\n    raise\n").body[0].handlers[0]
            node.handlers.append(handler)
        return node
tree=SaveRuntimeError().visit(tree);ast.fix_missing_locations(tree)
modified=ast.unparse(tree);compile(modified,'continued_evaluate.py','exec')
runtime_copy=out/'continued_evaluate.py';runtime_copy.write_text(modified)
wrapper_tree=ast.parse(text)
class RedirectRuntime(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='runpy' and node.func.attr=='run_path':
            self.count+=1;node.args[0]=ast.Constant(str(runtime_copy))
        return self.generic_visit(node)
redirect=RedirectRuntime();wrapper_tree=redirect.visit(wrapper_tree);ast.fix_missing_locations(wrapper_tree)
assert redirect.count==1,('Unexpected wrapper execution structure',redirect.count)
text=ast.unparse(wrapper_tree)

compile(text,'shared_above.py','exec')
wrapper=out/'shared_pick.py';wrapper.write_text(text)
spec={'checkpoint':str(checkpoint),'sha256':digest,'horizon':1300,'progress_denominator':1200.0,'shared_head':True,'stage_switch':False,'teacher_at_evaluation':False,'ngx':str(ngx),'scope':'Fixed initial state; pick task evaluation using same unchanged shared model'}
(out/'protocol.json').write_text(json.dumps(spec,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
env['LD_LIBRARY_PATH']=str(ngx)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
log=out/'pick.log'
command=['timeout','600s','/workspace/IsaacLab-develop/isaaclab.sh','-p',str(wrapper),str(system),str(checkpoint),'pick up the cube',str(dest)]
with log.open('w') as f:
    proc=subprocess.Popen(command,cwd='/workspace/IsaacLab-develop',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        f.write(line);f.flush()
        if any(x in line for x in ('XY_ATTENUATION_RELOAD','VIDEO_SAVED','LEARNED_GRIP_RELOAD','ADAPTER_RUNTIME_CHECK','[STEP]','INITIAL_INPUT_CHECK','SHARED_POLICY','Traceback','Error','TORCH_SETTINGS')):print(line.rstrip(),flush=True)
    rc=proc.wait()

assert rc==0, ('Evaluation failed',str(log),rc)
assert (dest/'continuation.json').is_file(), ('Continuation not completed; inspect',str(dest/'runtime_error.txt'),str(log))
continuation=json.loads((dest/'continuation.json').read_text())
assert continuation['extra_steps']==150 and abs(continuation['dt']-.02)<1e-8,continuation
reference=root/'verify_xy_attenuation1300_pick_20260918_173544_857706_JST/pick/teacher_trace.npz'
with np.load(reference,allow_pickle=False) as a,np.load(dest/'teacher_trace.npz',allow_pickle=False) as b:
    assert len(b['actions'])==1029
    same={k:bool(np.array_equal(a[k],b[k][:879])) for k in ('actions','pre_eef','post_eef','pre_quat_xyzw','post_quat_xyzw','post_cube','gripper_pos')}
    cube=b['post_cube'][879:];eef=b['post_eef'][879:]
    extra={'cube_height_min_mm':float(cube[:,2].min()*1000),'cube_height_final_mm':float(cube[-1,2]*1000),'cube_horizontal_motion_from_success_max_mm':float(np.linalg.norm(cube[:,:2]-b['post_cube'][878,:2],axis=1).max()*1000),'above_error_final_mm':float(np.linalg.norm(eef[-1]-cube[-1]-np.array([0,0,.1]))*1000),'closed_commands':int((b['actions'][879:,6]<0).sum())}
import imageio_ffmpeg
video=dest/'policy_view_continued3s.mp4'
reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');metadata=next(reader);frames=sum(1 for _ in reader)
assert frames==1029,frames
result={'task':'pick','original_trajectory_exactly_equal':same,'comparison_pass':all(same.values()),'extra_policy_steps':150,'extra_seconds':3,'video_file':str(video),'frames':frames,'fps':metadata['fps'],'extra_motion':extra,'visual_quality_verified':False,'scope':'Actual policy inference and environment steps continue after first success. No forced pose/gripper hold or duplicated freeze frames. Original videos unchanged. Last frame is pre-action.'}
(out/'video_result.json').write_text(json.dumps(result,indent=2))
print('CONTINUED_VIDEO_RESULT',json.dumps(result),flush=True)
PY

PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -u - <<'PY'
from pathlib import Path
import os,json,hashlib,tempfile
import subprocess as _preflight_subprocess
try:
    import imageio_ffmpeg as _video_dependency
    _encoder=_video_dependency.get_ffmpeg_exe()
    _preflight_subprocess.run([_encoder,'-version'],check=True,stdout=_preflight_subprocess.DEVNULL,stderr=_preflight_subprocess.PIPE)
except Exception as exc:
    raise RuntimeError('VIDEO_PREFLIGHT_FAILED: '+repr(exc)+'; if imageio_ffmpeg is missing, run /isaac-sim/python.sh -m pip install --no-cache-dir imageio-ffmpeg') from exc
print('VIDEO_PREFLIGHT_PASS',_encoder,flush=True)
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

out=root/('record_continued3s_v2_above_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f_JST'))
out.mkdir();dest=out/'above'
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
    cp=Path('/workspace/step3/retention_gripper_da00uipp/gripper_correction.pt')
    assert hashlib.sha256(cp.read_bytes()).hexdigest()=='48fc3c58dea3deaef9fbae6d1208d7df522967ff841162f5077f88eab0143863'
    correction_pack=torch.load(cp,map_location=args.device,weights_only=False)
    assert correction_pack['adapter_sha256']==hashlib.sha256(ap.read_bytes()).hexdigest()
    assert correction_pack['base_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    correction=torch.nn.Linear(256,1).to(args.device)
    correction.load_state_dict(correction_pack['state_dict'])
    with np.load(correction_pack['data_file'],allow_pickle=False) as data,torch.inference_mode():
        assert hashlib.sha256(Path(correction_pack['data_file']).read_bytes()).hexdigest()==correction_pack['data_sha256']
        h=torch.tensor(data['hidden'],device=args.device,dtype=torch.float32)
        target=torch.tensor(data['executed_actions'][:,6]<0,device=args.device)
        pred=(model.gripper_head(h)+correction(h)).squeeze(-1)>=0
        assert torch.equal(pred,target),'Retention reload mismatch'
    for path in report['anchor_hashes']:
        with np.load(path,allow_pickle=False) as data,torch.inference_mode():
            instruction=str(data['instruction'].item())
            if instruction!='move above the cube':continue
            feat=torch.tensor(data['frozen_features'],device=args.device,dtype=torch.float32)
            lang=model.encode_language(torch.tensor(smoke_utils.tokens(instruction),device=args.device)[None]).expand(len(feat),-1)
            h=model.fusion(torch.cat([feat,lang],-1))
            target=torch.tensor(data['actions'][:,6]<0,device=args.device)
            assert torch.equal((model.gripper_head(h)+correction(h)).squeeze(-1)>=0,target),'Above reload mismatch'
    class CorrectedGrip(torch.nn.Module):
        def __init__(self,original,extra):
            super().__init__();self.original=original;self.extra=extra
        def forward(self,x):return self.original(x)+self.extra(x)
    model.gripper_head=CorrectedGrip(model.gripper_head,correction)
    model.eval()
    print('LEARNED_GRIP_RELOAD_PASS NO_TIME_OVERRIDE',flush=True)
    language=torch.tensor(smoke_utils.tokens(args.instruction),device=args.device)[None]
    mean=torch.tensor([.485,.456,.406],device=args.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=args.device)[None,:,None,None]
    print('SHARED_POLICY_LOADED',json.dumps({'checkpoint':str(args.checkpoint),'shared_head':True,'stage_switch':False,'instruction':args.instruction}),flush=True)
    xp=Path('/workspace/step3/learned_xy_attenuation__zkv0syj/xy_attenuation.pt')
    assert hashlib.sha256(xp.read_bytes()).hexdigest()=='0f5b74efb4c9f34be3a9ab6b8985117869d8e6e1ff9565a1ab277f88ec2e3de5'
    xpack=torch.load(xp,map_location=args.device,weights_only=False)
    assert xpack['base_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    assert xpack['adapter_sha256']==hashlib.sha256(ap.read_bytes()).hexdigest()
    assert xpack['gripper_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
    selector=torch.nn.Linear(256,1).to(args.device)
    selector.load_state_dict(xpack['state_dict']);selector.eval()
    center=xpack['feature_mean'];spread=xpack['feature_std']
    assert torch.isfinite(center).all() and (spread>0).all()
    assert hashlib.sha256(Path(xpack['data_file']).read_bytes()).hexdigest()==xpack['data_sha256']
    with np.load(xpack['data_file'],allow_pickle=False) as data,torch.inference_mode():
        features=torch.tensor(data['hidden'],device=args.device,dtype=torch.float32)
        scores=selector((features-center)/spread).squeeze(-1)
        assert bool((scores[:800]<=0).all()) and bool((scores[800:]>=1).all()),'XY retention reload failed'
    for path in report['anchor_hashes']:
        with np.load(path,allow_pickle=False) as data,torch.inference_mode():
            instruction=str(data['instruction'].item())
            if instruction!='move above the cube':continue
            feat=torch.tensor(data['frozen_features'],device=args.device,dtype=torch.float32)
            lang=model.encode_language(torch.tensor(smoke_utils.tokens(instruction),device=args.device)[None]).expand(len(feat),-1)
            hidden=model.fusion(torch.cat([feat,lang],-1))
            assert bool((selector((hidden-center)/spread)<=0).all()),'Above attenuation reload failed'
    gate_cache={}
    def capture_attenuation(module,inputs,value):gate_cache['hidden']=value.detach()
    model.fusion.register_forward_hook(capture_attenuation)
    print('XY_ATTENUATION_RELOAD_PASS',flush=True)
    import subprocess as video_subprocess
    import imageio_ffmpeg
    video_process=None
    video_frames=0
    def record_frame(captured,step):
        nonlocal video_process,video_frames
        frame=np.ascontiguousarray(np.concatenate([captured['table_cam'],captured['wrist_cam']],axis=1),dtype=np.uint8)
        assert frame.ndim==3 and frame.shape[2]==3
        if video_process is None:
            height,width=frame.shape[:2]
            video_process=video_subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',str(width)+'x'+str(height),'-r','50','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'policy_view_continued3s.mp4')],stdin=video_subprocess.PIPE,stderr=video_subprocess.DEVNULL)
        video_process.stdin.write(frame.tobytes());video_frames+=1
        if step==573:
            video_process.stdin.close()
            assert video_process.wait(timeout=30)==0,'Video encoder failed'
            print('VIDEO_SAVED',str(output/'policy_view_continued3s.mp4'),video_frames,flush=True)
    def predict(obs,step):
        captured=smoke_utils.capture_obs(obs)
        record_frame(captured,step)
        def image(key):
            value=torch.tensor(captured[key],device=args.device).permute(2,0,1).float()[None]/255.
            return (value-mean)/std
        quat=np.asarray(captured['eef_quat']).copy()
        if quat[3]<0:quat=-quat
        state=np.concatenate([captured['eef_pos'],quat,captured['gripper_pos'],[step/1200.]])
        with torch.inference_mode():
            result=model(image('table_cam'),image('wrist_cam'),torch.tensor(state,dtype=torch.float32,device=args.device)[None],language)
            action=model.build_action(result).cpu().numpy()[0].copy()
        with torch.inference_mode():
            gate=float(selector((gate_cache['hidden']-center)/spread).clamp(0,1).item())
        action[:2]*=(1.0-gate)
        assert np.isfinite(action).all()
        return action
    return predict
smoke_utils.load_policy=shared_load
'''
text=text.replace(anchor,anchor+"\nfrom PIL import Image as _FrameImage\n_original_frame_save=_FrameImage.Image.save\ndef _save_without_frames(self,fp,*a,**kw):\n    if isinstance(fp,(str,Path)):\n        p=Path(fp)\n        if p.parent.resolve()==(output/'frames').resolve() and p.suffix.lower()=='.png':return None\n    return _original_frame_save(self,fp,*a,**kw)\n_FrameImage.Image.save=_save_without_frames\n"+replacement,1)
assert text.count("'--horizon', '550'")==1
text=text.replace("'--horizon', '550'","'--horizon', '1300'")

import ast
runtime_source=(system/'code/evaluate.py').read_text()
class ExtendSuccess(ast.NodeTransformer):
    def __init__(self):self.loops=0;self.stops=0
    def visit_For(self,node):
        if ast.unparse(node.iter)!='range(args.horizon)':return self.generic_visit(node)
        self.loops+=1
        for statement in node.body:
            if isinstance(statement,ast.If) and 'args.instruction' in ast.unparse(statement.test) and any(isinstance(v,ast.Break) for v in statement.body) and any(isinstance(v,ast.Assign) and 'success' in ast.unparse(v.targets[0]) for v in statement.body):
                statement.body=[v for v in statement.body if not isinstance(v,ast.Break)]
                statement.body+=ast.parse("if _first_success_step is None:\n    _first_success_step=step\n").body
                self.stops+=1
        node.body+=ast.parse("if _first_success_step is not None and step >= _first_success_step+150:\n    reason='success_then_150_policy_steps'\n    (out/'continuation.json').write_text(json.dumps({'first_success_step':_first_success_step,'last_step':step,'extra_steps':step-_first_success_step,'dt':dt}))\n    break\n").body
        return ast.parse('_first_success_step=None').body+[node]
transformer=ExtendSuccess();tree=transformer.visit(ast.parse(runtime_source));ast.fix_missing_locations(tree)
assert transformer.loops==1 and transformer.stops==2,('Unexpected runtime stop structure',transformer.loops,transformer.stops)
class SaveRuntimeError(ast.NodeTransformer):
    def visit_Try(self,node):
        self.generic_visit(node)
        if any('env.close()' in ast.unparse(v) for v in node.finalbody) and not node.handlers:
            handler=ast.parse("try:\n    pass\nexcept BaseException:\n    import traceback\n    error=traceback.format_exc()\n    print('RUNTIME_ERROR_BEFORE_CLOSE',error,flush=True)\n    (out/'runtime_error.txt').write_text(error)\n    raise\n").body[0].handlers[0]
            node.handlers.append(handler)
        return node
tree=SaveRuntimeError().visit(tree);ast.fix_missing_locations(tree)
modified=ast.unparse(tree);compile(modified,'continued_evaluate.py','exec')
runtime_copy=out/'continued_evaluate.py';runtime_copy.write_text(modified)
wrapper_tree=ast.parse(text)
class RedirectRuntime(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='runpy' and node.func.attr=='run_path':
            self.count+=1;node.args[0]=ast.Constant(str(runtime_copy))
        return self.generic_visit(node)
redirect=RedirectRuntime();wrapper_tree=redirect.visit(wrapper_tree);ast.fix_missing_locations(wrapper_tree)
assert redirect.count==1,('Unexpected wrapper execution structure',redirect.count)
text=ast.unparse(wrapper_tree)

compile(text,'shared_above.py','exec')
wrapper=out/'shared_pick.py';wrapper.write_text(text)
spec={'checkpoint':str(checkpoint),'sha256':digest,'horizon':1300,'progress_denominator':1200.0,'shared_head':True,'stage_switch':False,'teacher_at_evaluation':False,'ngx':str(ngx),'scope':'Fixed initial state; pick task evaluation using same unchanged shared model'}
(out/'protocol.json').write_text(json.dumps(spec,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
env['LD_LIBRARY_PATH']=str(ngx)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
log=out/'above.log'
command=['timeout','600s','/workspace/IsaacLab-develop/isaaclab.sh','-p',str(wrapper),str(system),str(checkpoint),'move above the cube',str(dest)]
with log.open('w') as f:
    proc=subprocess.Popen(command,cwd='/workspace/IsaacLab-develop',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        f.write(line);f.flush()
        if any(x in line for x in ('XY_ATTENUATION_RELOAD','VIDEO_SAVED','LEARNED_GRIP_RELOAD','ADAPTER_RUNTIME_CHECK','[STEP]','INITIAL_INPUT_CHECK','SHARED_POLICY','Traceback','Error','TORCH_SETTINGS')):print(line.rstrip(),flush=True)
    rc=proc.wait()

assert rc==0, ('Evaluation failed',str(log),rc)
assert (dest/'continuation.json').is_file(), ('Continuation not completed; inspect',str(dest/'runtime_error.txt'),str(log))
continuation=json.loads((dest/'continuation.json').read_text())
assert continuation['extra_steps']==150 and abs(continuation['dt']-.02)<1e-8,continuation
reference=root/'verify_xy_attenuation_above_20260918_174047_971979_JST/above/teacher_trace.npz'
with np.load(reference,allow_pickle=False) as a,np.load(dest/'teacher_trace.npz',allow_pickle=False) as b:
    assert len(b['actions'])==574
    same={k:bool(np.array_equal(a[k],b[k][:424])) for k in ('actions','pre_eef','post_eef','pre_quat_xyzw','post_quat_xyzw','post_cube','gripper_pos')}
    cube=b['post_cube'][424:];eef=b['post_eef'][424:]
    extra={'cube_height_min_mm':float(cube[:,2].min()*1000),'cube_height_final_mm':float(cube[-1,2]*1000),'cube_horizontal_motion_from_success_max_mm':float(np.linalg.norm(cube[:,:2]-b['post_cube'][423,:2],axis=1).max()*1000),'above_error_final_mm':float(np.linalg.norm(eef[-1]-cube[-1]-np.array([0,0,.1]))*1000),'closed_commands':int((b['actions'][424:,6]<0).sum())}
import imageio_ffmpeg
video=dest/'policy_view_continued3s.mp4'
reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');metadata=next(reader);frames=sum(1 for _ in reader)
assert frames==574,frames
result={'task':'above','original_trajectory_exactly_equal':same,'comparison_pass':all(same.values()),'extra_policy_steps':150,'extra_seconds':3,'video_file':str(video),'frames':frames,'fps':metadata['fps'],'extra_motion':extra,'visual_quality_verified':False,'scope':'Actual policy inference and environment steps continue after first success. No forced pose/gripper hold or duplicated freeze frames. Original videos unchanged. Last frame is pre-action.'}
(out/'video_result.json').write_text(json.dumps(result,indent=2))
print('CONTINUED_VIDEO_RESULT',json.dumps(result),flush=True)
PY
