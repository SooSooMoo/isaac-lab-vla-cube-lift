import json
import smoke_utils

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
        with torch.inference_mode():
            gate=float(selector((gate_cache['hidden']-center)/spread).clamp(0,1).item())
        action[:2]*=(1.0-gate)
        assert np.isfinite(action).all()
        return action
    return predict
smoke_utils.load_policy=shared_load
