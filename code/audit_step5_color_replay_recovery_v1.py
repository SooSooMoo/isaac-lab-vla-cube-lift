"""Audit one additional red regression teacher episode and combine with the previously audited manifest."""
from pathlib import Path
import argparse, ast, hashlib, json, math, tempfile, traceback
import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/workspace/step5/color_teacher4_e8slo5g1')
KEYS = ('table_cam', 'wrist_cam', 'eef_pos', 'eef_quat', 'gripper_pos')

def tokens(text):
    return np.asarray([1 + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), 'little') % 4095
                       for w in text.lower().replace('.', '').replace(',', '').split()[:16]], dtype=np.int64)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def audit_source(path):
    source = path.read_text()
    tree = ast.parse(source)
    loops = [x for x in ast.walk(tree) if isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 'step']
    require(len(loops) == 1, 'Unexpected collection step loop')
    loop = loops[0]
    calls = [x for x in ast.walk(loop) if isinstance(x, ast.Call)]
    capture = [x for x in calls if isinstance(x.func, ast.Name) and x.func.id == 'capture_obs']
    steps = [x for x in calls if isinstance(x.func, ast.Attribute) and isinstance(x.func.value, ast.Name)
             and x.func.value.id == 'env' and x.func.attr == 'step']
    actions = [x for x in calls if isinstance(x.func, ast.Name) and x.func.id == 'append'
               and len(x.args) > 1 and isinstance(x.args[1], ast.Constant) and x.args[1].value == 'actions']
    require(len(capture) == len(steps) == len(actions) == 1, 'Ambiguous observation/action recording')
    require(capture[0].lineno < actions[0].lineno < steps[0].lineno, 'Recording order is not pre-action')
    require(ast.unparse(capture[0]) == 'capture_obs(obs)' and ast.unparse(steps[0]) == 'env.step(action)', 'Unexpected data source')
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'pre_action_recording_order_verified': True}

def audit_episode(report, out):
    layout=report['layout']
    instruction=report['instruction']
    color=instruction.split()[3]
    folder=Path(report['dataset']).parent
    require(json.loads((folder/'result.json').read_text())==report,'External result mismatch')
    path = folder / 'episode.hdf5'
    with h5py.File(path, 'r') as f:
        d = f['data/demo_0']; obs = d['obs']; diag = f['diagnostics']
        n = len(d['actions']); dt = float(f.attrs['step_dt']); scale = float(f.attrs['action_scale'])
        require(n > 0 and n == int(d.attrs['num_samples']) == report['samples'], 'Sample count mismatch')
        require(str(d.attrs['layout']) == layout and str(d.attrs['target_color']) == color, 'Layout/color mismatch')
        require(str(d.attrs['language_instruction']) == instruction == report['instruction'], 'Instruction mismatch')
        require(np.array_equal(d['language_tokens'][()], tokens(instruction)), 'Language token mismatch')
        require(str(f.attrs['observation_timing']) == 'pre_action', 'Wrong observation timing')
        require(str(f.attrs['quaternion_order']) == 'xyzw', 'Wrong quaternion convention')
        require(set(obs.keys()) == set(KEYS), 'Unexpected learner observation keys')
        require(bool(d.attrs['success']) and bool(f.attrs['training_eligible']) and report['passed'], 'Episode not eligible')
        require(not bool(f.attrs['training_performed']) and bool(f.attrs['teacher_only']), 'Wrong collection provenance')
        require(json.loads(f.attrs['result_json']) == report, 'HDF5 and JSON result mismatch')
        require(int(d.attrs.get('dx_mm',0))==int(report.get('dx_mm',0)) and int(d.attrs.get('dy_mm',0))==int(report.get('dy_mm',0)), 'Offset metadata mismatch')
        expected = {'eef_pos': (3,), 'eef_quat': (4,), 'gripper_pos': (2,)}
        for k in KEYS:
            require(len(obs[k]) == n, 'Observation count mismatch: ' + k)
            if k.endswith('_cam'):
                require(obs[k].ndim == 4 and obs[k].shape[-1] == 3 and obs[k].dtype == np.uint8, 'Invalid RGB: ' + k)
                # Read every frame without keeping the entire episode in RAM.
                for start in range(0, n, 16):
                    a = obs[k][start:start+16]
                    require(np.all(a.reshape(len(a), -1).std(axis=1) > 1), 'Blank image: ' + k)
            else:
                require(obs[k].shape == (n, *expected[k]), 'Wrong state shape: ' + k)
                require(np.isfinite(obs[k][()]).all(), 'Nonfinite state: ' + k)
        q = obs['eef_quat'][()]
        require(np.max(np.abs(np.linalg.norm(q, axis=1)-1)) < 1e-3, 'Quaternion norm error')
        require(np.all(q[:, 3] >= -1e-6), 'Quaternion canonicalization error')
        action = d['actions'][()]
        require(action.shape == (n, 7) and np.isfinite(action).all(), 'Invalid actions')
        require(np.all(np.isin(action[:, 6], [-1., 1.])), 'Invalid gripper commands')
        require(np.linalg.norm(action[:, :3]*scale, axis=1).max() <= .010001, 'Translation action bound exceeded')
        require(np.linalg.norm(action[:, 3:6]*scale, axis=1).max() <= .080001, 'Rotation action bound exceeded')
        for k, shape in [('target_post_pos', (3,)), ('other_post_pos', (3,)), ('joint_pos', (9,)), ('phase', ())]:
            require(diag[k].shape == (n, *shape) and np.isfinite(diag[k][()]).all(), 'Invalid diagnostic: ' + k)
        phase = diag['phase'][()]
        require(set(phase.tolist()) == {0, 1, 2, 3} and np.all(np.diff(phase.astype(int)) >= 0), 'Unexpected phase sequence')
        require(np.array_equal(action[:, 6] < 0, phase >= 2), 'Phase/gripper mismatch')
        target = diag['target_post_pos'][()]; other = diag['other_post_pos'][()]
        h = math.ceil(3./dt)
        require(n > h, 'Hold window incomplete')
        require(np.all(target[-h:, 2] >= target[0, 2]+.1-1e-5), 'Saved trace fails 3-second lift check')
        require(np.all(action[-h:, 6] == -1.), 'Gripper opened during hold')
        require(np.max(other[:, 2]-other[0, 2]) < .02001, 'Other cube lifted')
        require(np.linalg.norm(other[:, :2]-other[0, :2], axis=1).max() <= .01001, 'Other cube moved')
        # Post-action diagnostics at t correspond to the next pre-action observation at t+1.
        joint = diag['joint_pos'][()]
        grip_error = float(np.max(np.abs(np.abs(obs['gripper_pos'][1:])-np.abs(joint[:-1, 7:9]))))
        require(grip_error <= 1e-4, 'Next observation/post-action finger state mismatch')
        proximity = np.linalg.norm(obs['eef_pos'][1:]-target[:-1], axis=1)
        require(np.all(proximity[-h:] <= .08001), 'Lifted target not near gripper in saved hold observations')
        png_matches = 0
        for step in range(0, n, 50):
            for cam in ('table_cam', 'wrist_cam'):
                saved = folder/'frames'/f'{step:04d}_{cam}.png'
                with Image.open(saved) as im:
                    require(np.array_equal(np.asarray(im.convert('RGB')), obs[cam][step]), 'PNG/HDF5 frame mismatch')
                png_matches += 1
        indices = [0] + [int(np.flatnonzero(phase == k)[0]) for k in (1, 2, 3)] + [n-1]
        captions = ['start', 'descend', 'close', 'lift', 'last recorded']
        tiles = []
        for cam in ('table_cam', 'wrist_cam'):
            row = Image.new('RGB', (1000, 238), 'white')
            for col, (idx, caption) in enumerate(zip(indices, captions)):
                im = Image.fromarray(obs[cam][idx]); im.thumbnail((200, 200))
                row.paste(im, (col*200, 38))
                draw = ImageDraw.Draw(row)
                draw.text((col*200+4, 3), cam + ' / ' + color, fill='black')
                draw.text((col*200+4, 19), caption + ' t=' + str(idx), fill='black')
            tiles.append(row)
        panel = Image.new('RGB', (1000, 500), 'white')
        ImageDraw.Draw(panel).text((4, 3), layout + ' / ' + instruction, fill='black')
        for i, row in enumerate(tiles):panel.paste(row, (0, 24+i*238))
        panel.save(out/(folder.name+'.png'))
    return {'dataset':str(path),'dx_mm':int(report.get('dx_mm',0)),'dy_mm':int(report.get('dy_mm',0)),'layout': layout, 'instruction': instruction, 'passed': True, 'samples': n,
            'matching_png_frames': png_matches, 'next_state_gripper_max_error': grip_error,
            'sha256': file_sha(path)}, panel

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

OLD_MANIFEST=Path('/workspace/step5/color_range20_red1_audit_rue_4izu/training_manifest.json')
NEW_ROOT=Path('/workspace/step5/color_teacher_replay_red1_r3rfvh01')
SOURCE=Path('/workspace/step5/color_policy_range20_recheck32_z6yrb9lx/x20_y-20__red_yminus_blue_yplus__pick_red/episode.hdf5')
SOURCE_SHA='9857ab1e930305bbb5c36c09d352fea508b597ced50a63ef5e004151043f16f5'

def validate_context(f, source):
    require(f.attrs['schema']=='color_recovery_teacher_v1','Recovery schema missing')
    require(int(f.attrs['replay_steps'])==500,'Unexpected replay length')
    require(not bool(f.attrs['context_is_teacher']),'Replay must not be a teacher label')
    require(str(f.attrs['replay_source_sha256'])==SOURCE_SHA,'Replay hash metadata mismatch')
    context=f['recovery_context/obs']
    require(set(context.keys())==set(KEYS),'Unexpected context keys')
    original=source['data/demo_0/obs']
    errors={}
    for key in KEYS:
        a=context[key]
        require(len(a)==500,'Context length mismatch: '+key)
        if key.endswith('_cam'):
            require(a.shape==(500,200,200,3) and a.dtype==np.uint8,'Invalid context RGB')
            for start in range(0,500,16):
                batch=a[start:start+16]
                require(np.all(batch.reshape(len(batch),-1).std(axis=1)>1),'Blank context RGB')
        else:
            require(a.shape==original[key][:500].shape,'Context shape mismatch')
            actual=a[()];expected=original[key][:500]
            require(np.isfinite(actual).all(),'Nonfinite context')
            if key=='eef_quat':
                require(np.max(np.abs(np.linalg.norm(actual,axis=1)-1))<1e-3,'Context quaternion norm')
                error=float(np.max(np.minimum(np.linalg.norm(actual-expected,axis=1),np.linalg.norm(actual+expected,axis=1))))
            else:error=float(np.max(np.abs(actual-expected)))
            errors[key]=error
            require(error<.001,'Context differs from recorded rollout: '+key)
    replay=f['recovery_context/replayed_actions'][()]
    require(np.array_equal(replay,source['data/demo_0/actions'][:500]),'Replay actions changed')
    handoff=f['data/demo_0/obs/eef_pos'][0]
    errors['handoff_eef_m']=float(np.linalg.norm(handoff-original['eef_pos'][500]))
    require(errors['handoff_eef_m']<.005,'Handoff position mismatch')
    return errors

def main():
    out=Path(tempfile.mkdtemp(prefix='color_replay_recovery_audit_',dir='/workspace/step5'))
    print('AUDIT_DIRECTORY',out,flush=True)
    error=None;manifest_path=None;comparison=None;context_errors=None;record=None
    try:
        old=json.loads(OLD_MANIFEST.read_text())
        prior=json.loads((OLD_MANIFEST.parent/'audit_results.json').read_text())
        require(prior.get('passed') is True and len(old['episodes'])==49,'Expected audited 49 episodes')
        for r in old['episodes']:require(file_sha(Path(r['dataset']))==r['sha256'],'Existing dataset changed')
        require(file_sha(SOURCE)==SOURCE_SHA,'Replay source changed')
        reports=json.loads((NEW_ROOT/'results.json').read_text())
        require(len(reports)==1,'Expected one recovery episode')
        r=reports[0]
        require((r['dx_mm'],r['dy_mm'],r['layout'],r['instruction'])==(20,-20,'red_yminus_blue_yplus','pick up the red cube'),'Wrong recovery condition')
        expected=NEW_ROOT/'x20_y-20__red_yminus_blue_yplus__pick_red/episode.hdf5'
        require(Path(r['dataset']).resolve()==expected.resolve(),'Unexpected recovery path')
        source_audit=audit_source(NEW_ROOT/'collect_color_teacher.py')
        record,panel=audit_episode(r,out)
        with h5py.File(expected,'r') as f,h5py.File(SOURCE,'r') as source:
            context_errors=validate_context(f,source)
            context_panel=Image.new('RGB',(1000,476),'white')
            for row,cam in enumerate(('table_cam','wrist_cam')):
                for col,step in enumerate((0,200,400,499)):
                    context_panel.paste(Image.fromarray(f['recovery_context/obs'][cam][step]),(col*200,row*238+38))
                    ImageDraw.Draw(context_panel).text((col*200+3,row*238+3),cam+' replay t='+str(step),fill='black')
                context_panel.paste(Image.fromarray(f['data/demo_0/obs'][cam][0]),(800,row*238+38))
                ImageDraw.Draw(context_panel).text((803,row*238+3),'teacher begins',fill='black')
        montage=Image.new('RGB',(1000,976),'white');montage.paste(context_panel,(0,0));montage.paste(panel,(0,476))
        comparison=out/'recovery_comparison.png';montage.save(comparison)
        record.update(episode_role='recovery',context_samples=500,context_group='recovery_context/obs',supervised_group='data/demo_0',context_actions_are_labels=False)
        for row in old['episodes']:row['episode_role']='standard'
        old['episodes'].append(record)
        old.update(schema='color_recovery_training_manifest_v1',visual_review='pending recovery image',context_training_rule='Warm recurrent state with context observations; supervised loss on teacher suffix only')
        manifest_path=out/'training_manifest.json';manifest_path.write_text(json.dumps(old,indent=2))
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    summary=dict(passed=error is None,total_episodes=50 if error is None else None,recovery_teacher_samples=record['samples'] if record else None,context_samples=500,context_errors=context_errors,manifest=str(manifest_path) if manifest_path else None,comparison=str(comparison) if comparison else None,training_performed=False,visual_review='pending recovery image',error=error)
    (out/'audit_results.json').write_text(json.dumps(summary,indent=2))
    text='STEP5_COLOR_REPLAY_DATA_AUDIT_SUMMARY '+json.dumps(summary)+'\n'
    (out/'share_results.txt').write_text(text);print(text,flush=True);print('SHARE_RESULTS_FILE',out/'share_results.txt',flush=True)
    return 0 if error is None else 1

if __name__=='__main__':raise SystemExit(main())
