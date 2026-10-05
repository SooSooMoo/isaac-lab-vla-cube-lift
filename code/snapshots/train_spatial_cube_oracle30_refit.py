"""Train a dedicated spatial-image position head; fixed position-held-out test."""
from pathlib import Path
import os,sys
ROOT=Path('/workspace/step4')
if '--prepared' not in sys.argv:
    paths=[ROOT/'.runtime_packages',ROOT/'python_packages',Path('/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle'),ROOT/'baseline_inputs/environment/runtime_packages']
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONPATH']=os.pathsep.join([str(p) for p in paths if p.is_dir()]+[env.get('PYTHONPATH','')])
    os.execve(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'--prepared'],env)
import json,hashlib,tempfile,traceback
import numpy as np
import h5py
import torch
from torch import nn
from torchvision.models import resnet18

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    out=Path(tempfile.mkdtemp(prefix='spatial_cube_oracle30_refit_',dir=ROOT))
    print('RUN_DIRECTORY',out,flush=True)
    def status(state,**kw):(out/'status.json').write_text(json.dumps(dict(state=state,**kw),indent=2))
    try:
        torch.manual_seed(2083);np.random.seed(2083)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        device='cuda:0';assert torch.cuda.is_available()
        parent=ROOT/'gripper_onset_fit_hq2qakz8/candidate.pt';parent_hash=digest(parent)
        ck=torch.load(parent,map_location='cpu',weights_only=False)
        backbone=resnet18(weights=None);backbone.fc=nn.Identity()
        vision={k[len('vision.'):]:v for k,v in ck['model'].items() if k.startswith('vision.')}
        backbone.load_state_dict(vision,strict=True)
        # Layer2 maps retain spatial structure; do not use the action model's fused features.
        encoder=nn.Sequential(*list(backbone.children())[:6],nn.AdaptiveAvgPool2d((5,5))).to(device).eval()
        for p in encoder.parameters():p.requires_grad_(False)
        del ck,backbone
        entries=json.loads((ROOT/'cube_vision_labels_57h41v26/manifest.json').read_text())['episodes']
        assert len(entries)==15
        new_manifest=ROOT/'cube_vision_full_data_5a28urn1/manifest.json'
        extra_entries=json.loads(new_manifest.read_text())['episodes']
        assert len(extra_entries)==6
        expected={(0,0),(10,0),(0,20),(0,-10),(20,0),(0,-20)}
        assert {tuple(e['offset_mm']) for e in extra_entries}==expected
        for e in extra_entries:
            assert e['pick_pass'] and e['hold_3s_pass']
            entries.append(dict(episode=len(entries),dataset=e['dataset_file'],dataset_sha256=e['sha256'],
                offset_mm=e['offset_mm'],samples=e['samples'],labels_embedded=True))

        capture=ROOT/'smoothed_vision_capture_ndgerrvl/trial_01/diagnostic_observations.hdf5'
        with h5py.File(capture,'r') as h:
            assert not bool(h.attrs['oracle_state_used_for_control'])
            assert not bool(h.attrs['pass'])
            assert (float(h.attrs['dx_mm']),float(h.attrs['dy_mm']))==(-20.,20.)
            capture_count=len(h['true_cube_pre'])
            assert capture_count==1300
            assert all(len(h['obs'][k])==capture_count for k in ('table_cam','wrist_cam','eef_pos','eef_quat','gripper_pos'))
        recovery_id=len(entries)
        entries.append(dict(episode=recovery_id,dataset=str(capture),dataset_sha256=digest(capture),
                            offset_mm=[-20,20],samples=capture_count,recovery_capture=True))
        oracle_manifest=ROOT/'oracle_vision_xp30ym30_d986vh_p/manifest.json'
        oracle_entries=json.loads(oracle_manifest.read_text())['episodes']
        assert len(oracle_entries)==1
        oe=oracle_entries[0]
        assert tuple(oe['offset_mm'])==(30, -30) and oe['pick_pass'] and oe['hold_3s_pass']
        oracle30_id=len(entries)
        entries.append(dict(episode=oracle30_id,dataset=oe['dataset_file'],dataset_sha256=oe['sha256'],
                            offset_mm=oe['offset_mm'],samples=oe['samples'],labels_embedded=True))
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        xs=[];ys=[];ids=[];positions=[]
        status('extracting_spatial_features')
        for index,e in enumerate(entries):
            assert digest(e['dataset'])==e['dataset_sha256']
            if e.get('recovery_capture'):
                with h5py.File(e['dataset'],'r') as h:
                    y=h['true_cube_pre'][:]-h['obs/eef_pos'][:]
                    assert y.shape==(e['samples'],3) and np.isfinite(y).all()
            elif e.get('labels_embedded'):
                with h5py.File(e['dataset'],'r') as h:
                    assert h.attrs['schema']=='cube_vision_full_episode_v1'
                    y=h['data/demo_0/labels/cube_minus_eef_env_m'][()]
            else:
                assert digest(e['labels'])==e['labels_sha256']
                with np.load(e['labels']) as lab:y=lab['cube_minus_eef_env_m'].copy()
            with h5py.File(e['dataset'],'r') as h:
                d=h['obs'] if e.get('recovery_capture') else h['data/demo_0/obs'];n=len(y)
                for start in range(0,n,32):
                    end=min(start+32,n);q=d['eef_quat'][start:end]
                    q=np.where(q[:,3:4]<0,-q,q)
                    state=np.concatenate([d['eef_pos'][start:end],q,d['gripper_pos'][start:end]],axis=1)
                    with torch.inference_mode():
                        maps=[]
                        for key in ('table_cam','wrist_cam'):
                            im=torch.from_numpy(d[key][start:end]).to(device).permute(0,3,1,2).float()/255
                            maps.append(encoder((im-mean)/std).flatten(1))
                        features=torch.cat(maps+[torch.tensor(state,device=device)],dim=1)
                    xs.append(features.cpu())
            ys.append(torch.from_numpy(y));ids.extend([index]*n);positions.extend([e['offset_mm']]*n)
            print('SPATIAL_FEATURES_READY',index,n,flush=True)
        x=torch.cat(xs).float().to(device);y=torch.cat(ys).float().to(device);del xs,ys
        ids=np.asarray(ids);positions=np.asarray(positions)
        heldout=np.all(positions==[20,-20],axis=1)
        recovery_train=ids==recovery_id
        oracle30_train=ids==oracle30_id
        recovery_check=np.all(positions==[-20,20],axis=1)&~recovery_train
        validation=np.isin(ids,[4,13])
        training=~heldout&~validation&~recovery_check
        assert np.all(training[recovery_train]) and np.all(training[oracle30_train])
        masks={k:torch.tensor(v,device=device) for k,v in [('train',training),('validation',validation),
               ('heldout',heldout),('recovery_train',recovery_train),('recovery_check',recovery_check),('oracle30_train',oracle30_train)]}
        xm=x[masks['train']].mean(0);xscl=x[masks['train']].std(0,unbiased=False).clamp_min(.01)
        ym=y[masks['train']].mean(0)
        z=(x-xm)/xscl;target=(y-ym)/.1
        head=nn.Sequential(nn.Linear(x.shape[1],256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,3)).to(device)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.0001)
        train_ids=torch.nonzero(masks['train'],as_tuple=True)[0]
        weights=np.zeros(len(y),dtype=np.float32)
        for pos in np.unique(positions[training],axis=0):
            m=training&np.all(positions==pos,axis=1);weights[m]=1/m.sum()
        weights=torch.tensor(weights,device=device)[train_ids]
        protocol=dict(parent=str(parent),parent_sha256=parent_hash,episodes=entries,
            change='Use the 25-position-success training data plus one full successful oracle episode at +30,-30; omit failed +30,-30 capture. Architecture, optimizer and selection unchanged',
            additional_manifest=str(new_manifest),additional_manifest_sha256=digest(new_manifest),
            encoder='frozen ResNet18 conv1 through layer2; adaptive 5x5 maps from each camera',
            head='6409 -> 256 GELU -> 64 GELU -> 3, when layer2 has 128 channels',
            target='cube minus EEF in environment axes, metres',heldout_positions=[[20,-20]], recovery_training_position=[-20,20], recovery_dataset=str(capture), oracle30_manifest=str(oracle_manifest), oracle30_manifest_sha256=digest(oracle_manifest),
            validation_episodes=[4,13],epochs=80,seed=2083,optimizer='AdamW lr=.0003 wd=.0001',
            loss='SmoothL1 beta=.05 in targets scaled by .1m',
            selection='lowest validation RMSE; heldout inspected only after training',
            limitations='Recovery train is fit error; recovery_check contains other trajectories at the now-trained position and is not position-heldout. Validation shares training positions. The remaining heldout position was inspected previously. No rollout claim.')
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        best=float('inf');best_state=None;best_epoch=None
        status('training')
        for epoch in range(1,81):
            head.train()
            order=train_ids[torch.multinomial(weights,len(train_ids),replacement=True)]
            for batch in order.split(128):
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.smooth_l1_loss(head(z[batch]),target[batch],beta=.05)
                assert bool(torch.isfinite(loss));loss.backward();optimizer.step()
            head.eval()
            with torch.no_grad():
                pred=torch.cat([head(v) for v in z[masks['validation']].split(256)])
                score=float(((pred-target[masks['validation']])*.1).square().mean().sqrt())
            if score<best:
                best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            if epoch%10==0:print('SPATIAL_TRAIN',epoch,'validation_rmse_mm=',round(score*1000,2),flush=True)
        head.load_state_dict(best_state);head.eval()
        with torch.no_grad():prediction=torch.cat([head(v) for v in z.split(256)])*.1+ym
        error=(prediction-y)*1000;rows=[]
        for split,mask in masks.items():
            err=error[mask];xy=err[:,:2].norm(dim=1);ez=err[:,2].abs()
            rows.append(dict(split=split,samples=int(mask.sum()),xyz_rmse_mm=float(err.square().mean().sqrt()),
                             xy_p95_mm=float(torch.quantile(xy,.95)),z_p95_mm=float(torch.quantile(ez,.95))))
        torch.save(dict(format='spatial_cube_estimator_v1',encoder_state={k:v.cpu() for k,v in encoder.state_dict().items()},
            head_state=best_state,input_dim=int(x.shape[1]),feature_mean=xm.cpu(),feature_scale=xscl.cpu(),
            target_mean=ym.cpu(),target_scale=.1,selected_epoch=best_epoch,protocol=protocol),out/'candidate.pt')
        np.savez_compressed(out/'predictions.npz',predicted_relative_m=prediction.cpu().numpy(),
                            true_relative_m=y.cpu().numpy(),episode_ids=ids,heldout=heldout,validation=validation)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        assert digest(parent)==parent_hash
        status('offline_complete',selected_epoch=best_epoch,rollout_verified=False)
        lines=['SPATIAL_ORACLE30_REFIT_SUMMARY','split | xyz_rmse_mm | xy_p95_mm | z_p95_mm']
        for r in rows:lines.append(f"{r['split']} | {r['xyz_rmse_mm']:.2f} | {r['xy_p95_mm']:.2f} | {r['z_p95_mm']:.2f}")
        lines+=['SELECTED_EPOCH '+str(best_epoch),'RESULT_DIRECTORY '+str(out)]
        text='\n'.join(lines)+'\n';(out/'summary.txt').write_text(text);print(text,flush=True)
    except BaseException as exc:
        status('error',error=str(exc));traceback.print_exc();return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
