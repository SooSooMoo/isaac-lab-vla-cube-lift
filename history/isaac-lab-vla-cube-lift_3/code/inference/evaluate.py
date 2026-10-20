"""Evaluate a pick-only learned VLA policy; cube truth is scoring-only."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', required=True)
p.add_argument('--reference-trace')
p.add_argument('--instruction', required=True, choices=['pick up the cube'])
p.add_argument('--dataset', default='/workspace/step3/data/lift_robomimic_language.hdf5')
p.add_argument('--demo', default='demo_0')
p.add_argument('--task', default='IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor')
p.add_argument('--output-dir', required=True)
p.add_argument('--seed', type=int, default=2058)
p.add_argument('--horizon', type=int, default=600)
p.add_argument('--dx-mm', type=float, default=0.0)
p.add_argument('--dy-mm', type=float, default=0.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args, headless=True, enable_cameras=True).app
from smoke_utils import capture_obs, tokens
import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error

def load_policy(args):
    import sys
    parent = Path(args.checkpoint).parent
    sys.path.insert(0, str(parent))
    from pick_model import LanguageConditionedVLA
    from pick_inputs import prepare
    from temporal_action import TemporalAction
    b = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert b['format'] == 'pick_temporal_action_v1'
    assert b['instruction'] == args.instruction == 'pick up the cube'
    assert b['parent_sha256'] == '4e9e5973f0c5addeea84a06bdd0cadb732f573f83be74dd2baf32b04291983ce'
    ck = b['parent_checkpoint']
    assert ck['instruction'] == args.instruction
    for name, key in [('pick_model.py', 'model_code_sha256'), ('pick_inputs.py', 'input_code_sha256')]:
        assert hashlib.sha256((parent / name).read_bytes()).hexdigest() == ck[key]
    assert hashlib.sha256((parent / 'temporal_action.py').read_bytes()).hexdigest() == b['temporal_code_sha256']
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = LanguageConditionedVLA(**{k: int(ck[k]) for k in ('state_dim', 'action_dim', 'vocab_size', 'language_dim')}).to(args.device).eval()
    model.load_state_dict(ck['model'], strict=True)
    decoder = TemporalAction().to(args.device).eval()
    decoder.load_state_dict(b['temporal_state'], strict=True)
    calls = {'language': 0, 'actions': 0}
    feature = {}
    hidden = None
    saved = {k: [] for k in ('features', 'arm', 'logit', 'target', 'source_logit')}

    def language_hook(m, i, o):
        calls['language'] += 1
    model.language_encoder.register_forward_hook(language_hook)
    model.fusion.register_forward_hook(lambda m, i, o: feature.update(value=o))

    def predict(obs, step):
        nonlocal hidden
        assert step == calls['actions'], 'Unexpected sequence reset/order'
        with torch.inference_mode():
            output = model(*prepare(capture_obs(obs), args.instruction, args.device))
            predict.last_feature = feature['value'][0].detach().cpu().clone()
            arm, g, hidden = decoder(feature['value'][:, None, :], hidden)
            action = model.build_action({'arm_action': arm[:, 0], 'gripper_logit': g[:, 0]}).cpu().numpy()[0].copy()
            saved['features'].append(torch.cat((feature['value'][0], torch.zeros(3, device=args.device))).cpu().clone())
            saved['arm'].append(output['arm_action'][0].cpu().clone())
            saved['logit'].append(output['gripper_logit'][0].cpu().clone())
            saved['source_logit'].append(g[0, 0].cpu().clone())
            saved['target'].append(torch.from_numpy(action.copy()))
        calls['actions'] += 1
        assert calls['language'] == calls['actions']
        return action
    predict.saved = saved
    predict.audit = calls
    predict.spatial_input = False
    return predict

def tensor(x):
    return x if torch.is_tensor(x) else x.torch

def array(x):
    return tensor(x).detach().cpu().numpy().copy()

def read_state(group):
    return {k: read_state(v) if isinstance(v, h5py.Group) else torch.as_tensor(v[()], device=args.device) for k, v in group.items()}

def bounded(v, limit):
    return v * torch.clamp(limit / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-08), max=1.0)

def remote_asset_fields(root):
    seen = set()

    def walk(obj, path):
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, type)) or callable(obj):
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            fields = list(obj.items())
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                yield from walk(value, path + '[' + str(index) + ']')
            return
        elif hasattr(obj, '__dict__'):
            fields = list(vars(obj).items())
        else:
            return
        for name, value in fields:
            field_path = path + '.' + str(name)
            if name in ('usd_path', 'texture_file') and isinstance(value, str) and value.startswith(('https://', 'http://')):
                yield (obj, name, value, field_path)
            else:
                yield from walk(value, field_path)
    return list(walk(root, 'cfg'))

def localize_scene_assets(cfg, report_dir):
    import hashlib, json, threading
    from pathlib import Path
    from urllib.parse import urljoin, urlsplit
    import omni.client
    from pxr import Sdf, UsdUtils
    cache = Path(os.environ.get('VLA_ASSET_CACHE', str(Path.home()/'.cache/vla_scene_assets')))
    cache.mkdir(parents=True, exist_ok=True)
    mapped, manifest = ({}, [])

    def localize(url):
        if url in mapped:
            return mapped[url]
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https'):
            raise RuntimeError('Unsupported asset URL: ' + url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix == '.mdl':
            raise RuntimeError('External MDL dependency needs packaging: ' + url)
        key = hashlib.sha256(url.encode()).hexdigest()
        raw = cache / (key + '.source' + suffix)
        dest = cache / (key + suffix)
        if not raw.exists():
            event, answer = (threading.Event(), [])

            def callback(result, version, content):
                try:
                    answer.append((result, bytes(content) if result == omni.client.Result.OK else b''))
                except Exception as exc:
                    answer.append((str(exc), b''))
                finally:
                    event.set()
            request = omni.client.read_file_with_callback(url, callback)
            if not event.wait(60):
                request.stop()
                raise RuntimeError('Asset download timeout: ' + url)
            result, data = answer[0]
            if result != omni.client.Result.OK or not data:
                raise RuntimeError('Asset download failed: ' + url + ' ' + str(result))
            temp = raw.with_name(raw.name + '.part')
            temp.write_bytes(data)
            temp.replace(raw)
        mapped[url] = str(dest)
        dependencies = []
        if suffix in ('.usd', '.usda', '.usdc'):
            layer = Sdf.Layer.FindOrOpen(str(raw))
            if not layer:
                raise RuntimeError('Cannot open local USD: ' + str(raw))

            def rewrite(asset):
                if not asset:
                    return asset
                if asset.endswith('.mdl') and '/' not in asset and ('\\' not in asset):
                    return asset
                if '<' in asset or '[' in asset:
                    raise RuntimeError('Asset template/package path needs packaging: ' + asset)
                dependency = urljoin(url, asset)
                dependencies.append(dependency)
                return localize(dependency)
            copy = Sdf.Layer.CreateAnonymous('.usda')
            copy.TransferContent(layer)
            UsdUtils.ModifyAssetPaths(copy, rewrite)
            if not copy.Export(str(dest)):
                raise RuntimeError('Local USD export failed: ' + str(dest))
        else:
            dest.write_bytes(raw.read_bytes())
        manifest.append({'url': url, 'local': str(dest), 'source_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'dependencies': dependencies})
        return str(dest)
    replacements = []
    for owner, field, value, field_path in remote_asset_fields(cfg):
        target = localize(value)
        if isinstance(owner, dict):
            owner[field] = target
        else:
            setattr(owner, field, target)
        replacements.append(field_path)
    if not replacements:
        raise RuntimeError('No scene assets found; refusing empty localization')
    missing = [p for p in mapped.values() if not Path(p).is_file()]
    if missing:
        raise RuntimeError('Missing localized files: ' + str(missing))
    remaining = remote_asset_fields(cfg)
    if remaining:
        raise RuntimeError('Unlocalized configuration assets: ' + str([row[3] for row in remaining]))
    (report_dir / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('LOCAL_ASSETS_READY', json.dumps({'files': len(mapped), 'scene_fields': replacements}), flush=True)

def main():
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    with h5py.File(args.dataset, 'r') as f:
        demo = f['data'][args.demo]
        initial = read_state(demo['initial_state'])
        initial['rigid_object']['object']['root_pose'][0, 0] += args.dx_mm / 1000.0
        initial['rigid_object']['object']['root_pose'][0, 1] += args.dy_mm / 1000.0
        expected_eef = demo['obs/eef_pos'][0]
    zero = torch.zeros((1, 3), device=args.device)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=args.device)
    x90 = torch.tensor([[2 ** (-0.5), 0.0, 0.0, 2 ** (-0.5)]], device=args.device)
    _, probe = compute_pose_error(zero, identity, zero, x90, rot_error_type='axis_angle')
    torch.testing.assert_close(probe, torch.tensor([[np.pi / 2, 0.0, 0.0]], device=args.device), atol=1e-05, rtol=1e-05)
    print('[CONVENTION] installed compute_pose_error uses xyzw: verified', flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.seed = args.seed
    cfg.observations.policy.concatenate_terms = False
    cfg.recorders = None
    for name in ('time_out', 'object_dropping', 'success'):
        setattr(cfg.terminations, name, None)
    sensor_offset = cfg.scene.ee_frame.target_frames[0].offset
    cfg.actions.arm_action.body_offset.pos = tuple(sensor_offset.pos)
    cfg.actions.arm_action.body_offset.rot = tuple(sensor_offset.rot)
    scale = float(cfg.actions.arm_action.scale)
    if scale <= 0:
        raise ValueError('Invalid action scale')
    localize_scene_assets(cfg, out)
    env = gym.make(args.task, cfg=cfg).unwrapped
    records = {k: [] for k in ('actions', 'pre_eef', 'post_eef', 'pre_quat_xyzw', 'post_quat_xyzw', 'post_cube', 'gripper_pos', 'phase', 'above_error_m', 'orientation_error_rad', 'target_position', 'ik_sensor_position_error_m', 'ik_sensor_angle_error_rad')}
    try:
        obs, _ = env.reset_to(initial, env_ids=None, seed=args.seed, is_relative=True)

        def poses():
            sensor = env.scene['ee_frame'].data
            return (tensor(sensor.target_pos_w)[:, 0].clone(), tensor(sensor.target_quat_w)[:, 0].clone())

        def cube():
            return array(tensor(env.scene['object'].data.root_pos_w)[0] - env.scene.env_origins[0])
        pos_w, quat_w = poses()
        initial_eef = array(pos_w[0] - env.scene.env_origins[0])
        initial_cube = cube()
        reset = {'eef_first_obs_error_m': float(np.linalg.norm(array(obs['policy']['eef_pos'])[0] - expected_eef)), 'cube_position_error_m': float(np.linalg.norm(initial_cube - array(initial['rigid_object']['object']['root_pose'])[0, :3])), 'robot_joint_max_abs_error_rad': float(np.max(np.abs(array(env.scene['robot'].data.joint_pos)[0] - array(initial['articulation']['robot']['joint_position'])[0])))}
        if max(reset.values()) > 0.0001:
            raise RuntimeError(f'Reset mismatch: {reset}')
        predict = load_policy(args)
        action_term = env.action_manager.get_term('arm_action')
        target_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        dt = float(cfg.sim.dt) * int(cfg.decimation)
        required_hold = int(np.ceil(1.0 / dt))
        stage, stable, close_steps = ('approach', 0, 0)
        hold_streak = lift_streak = max_hold = max_lift = 0
        settled_height = None
        pick_anchor = None
        success = False
        first_success_step = None
        extra_hold_steps = int(np.ceil(3.0 / dt))
        joint_samples = []
        reason = 'horizon'
        with torch.inference_mode():
            for step in range(args.horizon + extra_hold_steps):
                pos_w, quat_w = poses()
                pre = array(pos_w[0] - env.scene.env_origins[0])
                obj = cube()
                robot = env.scene['robot'].data
                root_pos, root_quat = (tensor(robot.root_pos_w), tensor(robot.root_quat_w))
                current_pos_b, current_quat_b = subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)
                ik_pos_b, ik_quat_b = action_term._compute_frame_pose()
                offset_error, angle_error = compute_pose_error(current_pos_b, current_quat_b, ik_pos_b, ik_quat_b, rot_error_type='axis_angle')
                offset_norm, angle_norm = (float(offset_error.norm()), float(angle_error.norm()))
                if offset_norm > 0.0001 or angle_norm > 0.001:
                    raise RuntimeError(f'IK/sensor frame mismatch: {offset_norm} m, {angle_norm} rad')
                target = obj + [0.0, 0.0, 0.1]
                grip = 1.0
                if args.instruction == 'pick up the cube':
                    if stage in ('descend', 'close'):
                        target = obj.copy()
                    elif stage == 'lift':
                        target = pick_anchor + [0.0, 0.0, 0.18]
                    if stage in ('close', 'lift'):
                        grip = -1.0
                target_w = torch.as_tensor(target, dtype=pos_w.dtype, device=env.device)[None] + env.scene.env_origins
                goal_pos_b, goal_quat_b = subtract_frame_transforms(root_pos, root_quat, target_w, target_quat_w)
                pos_err, rot_err = compute_pose_error(ik_pos_b, ik_quat_b, goal_pos_b, goal_quat_b, rot_error_type='axis_angle')
                orientation_error = float(rot_err.norm())
                action = torch.as_tensor(predict(obs, step), device=env.device).unsqueeze(0)
                phase = 'policy'
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite teacher action')
                obs, _, terminated, truncated, _ = env.step(action)
                joint_samples.append(array(env.scene['robot'].data.joint_pos)[0])
                post_pos_w, post_quat_w = poses()
                post = array(post_pos_w[0] - env.scene.env_origins[0])
                post_cube = cube()
                fingers = array(obs['policy']['gripper_pos'])[0]
                error = float(np.linalg.norm(post - (post_cube + [0.0, 0.0, 0.1])))
                if step == 10:
                    settled_height = float(post_cube[2])
                valid_above = error <= 0.03 and fingers[0] >= 0.03 and (fingers[1] <= -0.03)
                hold_streak = hold_streak + 1 if valid_above else 0
                max_hold = max(max_hold, hold_streak)
                lifted = settled_height is not None and post_cube[2] >= settled_height + 0.1 and (float(action[0, 6]) < 0) and (np.linalg.norm(post - post_cube) <= 0.08)
                lift_streak = lift_streak + 1 if lifted else 0
                max_lift = max(max_lift, lift_streak)
                values = (array(action)[0], pre, post, array(quat_w)[0], array(post_quat_w)[0], post_cube, fingers, phase, error, orientation_error, target, offset_norm, angle_norm)
                for key, value in zip(records, values):
                    records[key].append(value)
                if step % 25 == 0:
                    print('[STEP]', step + 1, phase, 'above_error_m=', error, 'rotation_error_rad=', orientation_error, flush=True)
                if bool(terminated[0]) or bool(truncated[0]):
                    reason = 'unexpected_environment_termination'
                    break
                if args.instruction == 'pick up the cube' and lift_streak >= 10 and (first_success_step is None):
                    first_success_step = step
                    success = True
                if first_success_step is not None and step >= first_success_step + extra_hold_steps:
                    reason = 'success_then_3s_policy_control'
                    break
                if first_success_step is None and step >= args.horizon - 1:
                    reason = 'acquisition_horizon'
                    break
        np.savez_compressed(out / 'teacher_trace.npz', joint_pos=np.asarray(joint_samples), **{k: np.asarray(v) for k, v in records.items()}, initial_eef=initial_eef, initial_cube=initial_cube, instruction=args.instruction)
        summary = {'instruction': args.instruction, 'success': success, 'stop_reason': reason, 'teacher_type': 'raw_corrected_data_vla', 'learned_language_policy': True, 'executed_steps': len(records['actions']), 'step_dt': dt, 'reset_check': reset, 'max_consecutive_above_hold_steps': max_hold, 'required_above_hold_steps': required_hold, 'max_consecutive_lift_steps': max_lift, 'final_above_error_m': records['above_error_m'][-1], 'final_cube_height_m': float(records['post_cube'][-1][2]), 'settled_cube_height_m': settled_height, 'last_phase': records['phase'][-1], 'settings': {'quaternion': 'xyzw; state canonicalized to w>=0', 'progress_denominator': 1200.0, 'forced_warmup': False, 'action_filtering': False}, 'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), 'progress_denominator': 1200.0, 'limitations': ['Fixed-state evaluation, not unseen-position generalization.', 'Model acts from step 0 without teacher warmup or output correction.', 'Pick is a lift proxy, not contact-force verification.', 'Cube coordinates are used for diagnostics and scoring only.'], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'frames': str(out / 'frames')}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
        print('SUMMARY=' + str(out / 'summary.json'), flush=True)
        trace_cube = np.asarray(records['post_cube'])
        trace_actions = np.asarray(records['actions'])
        horizontal = np.linalg.norm(trace_cube[:, :2] - trace_cube[10, :2], axis=1) * 1000
        threshold = (settled_height + 0.1) * 1000
        closed = np.flatnonzero(trace_actions[:, 6] < 0)
        first_close = int(closed[0]) if len(closed) else None
        first = first_success_step
        pick_xy = float(horizontal[first_close:first + 1].max()) if first is not None and first_close is not None else None
        pick_pass = bool(first is not None and pick_xy is not None and (pick_xy <= 50))
        complete = first is not None and len(trace_cube) - 1 - first >= extra_hold_steps
        hold = False
        min_height = None
        hold_xy = None
        if first is not None and len(trace_cube) > first + 1:
            heights = trace_cube[first + 1:, 2] * 1000
            min_height = float(heights.min())
            hold_xy = float(horizontal[first + 1:].max())
            hold = bool(complete and np.all(heights >= threshold) and (hold_xy <= 50))
        robot = env.scene['robot']
        ids = [robot.joint_names.index('panda_joint' + str(i)) for i in range(1, 8)]
        bounds = array(robot.data.joint_pos_limits)[0, ids]
        joints = np.asarray(joint_samples)[:, ids]
        margins = np.minimum(joints - bounds[:, 0], bounds[:, 1] - joints)
        finite = all((np.isfinite(np.asarray(records[k])).all() for k in ('actions', 'post_eef', 'post_cube'))) and np.isfinite(joints).all()
        margin = float(margins.min())
        phases = np.asarray(records['phase'])
        target_error = np.linalg.norm(np.asarray(records['post_eef']) - np.asarray(records['target_position']), axis=1) * 1000
        phase_best = {phase: float(target_error[phases == phase].min()) for phase in sorted(set(records['phase']))}
        result = {'offset_mm': [args.dx_mm, args.dy_mm], 'teacher_only': False, 'learned_policy': True, 'uses_corrections': True, 'uses_progress_input': False, 'pick_pass': pick_pass, 'hold_3s_pass': hold, 'first_success_step': first, 'first_close_step': first_close, 'samples': len(trace_cube), 'last_phase': records['phase'][-1], 'stop_reason': reason, 'required_height_mm': threshold, 'maximum_cube_height_after_settle_mm': float(trace_cube[10:, 2].max() * 1000), 'hold_minimum_height_mm': min_height, 'pick_maximum_xy_mm': pick_xy, 'hold_maximum_xy_mm': hold_xy, 'minimum_arm_joint_margin_rad': margin, 'phase_best_position_error_mm': phase_best, 'pass': bool(pick_pass and hold and finite and (margin >= -0.001)), 'dataset_collected': False, 'training_performed': False, 'result_directory': str(out)}
        result.update(language_encoder_calls=predict.audit['language'], model_action_calls=predict.audit['actions'], runtime_action_override=False)
        assert predict.audit['actions'] == len(records['actions'])
        if result['pass']:
            data = {k: torch.stack(v) for k, v in predict.saved.items()}
            assert all((len(v) == len(records['actions']) for v in data.values()))
            assert np.array_equal(data['target'].numpy(), np.asarray(records['actions']))
            data.update(instruction=args.instruction, offset_mm=[args.dx_mm, args.dy_mm], warmup=0, source='successful_learned_history_policy', teacher_phase_is_input=False, parent_sha256='4e9e5973f0c5addeea84a06bdd0cadb732f573f83be74dd2baf32b04291983ce', source_checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
            torch.save(data, out / 'preserve_features.pt')
            result['preserve_features_file'] = str(out / 'preserve_features.pt')
            result['reference_exact'] = None
            result['preserve_features_sha256'] = hashlib.sha256((out / 'preserve_features.pt').read_bytes()).hexdigest()
        (out / 'direct_result.json').write_text(json.dumps(result, indent=2))
        print('DIRECT_RESULT', json.dumps(result), flush=True)
    finally:
        env.close()
if __name__ == '__main__':
    import traceback, os, sys
    failed = False
    try:
        main()
    except BaseException:
        failed = True
        message = traceback.format_exc()
        Path(str(args.output_dir) + '_worker_exception.txt').write_text(message)
        print(message, flush=True)
    finally:
        try:
            app.close()
        except BaseException:
            failed = True
    if failed:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)