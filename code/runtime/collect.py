"""Test a new teacher using the installed Isaac Lab pose convention. No training or dataset overwrite."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--instruction', required=True, choices=['pick up the cube', 'move above the cube'])
p.add_argument('--dataset', default='/workspace/step3/data/lift_robomimic_language.hdf5')
p.add_argument('--demo', default='demo_0')
p.add_argument('--task', default='IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor')
p.add_argument('--output-dir', required=True)
p.add_argument('--seed', type=int, default=2058)
p.add_argument('--horizon', type=int, default=600)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args, headless=True, enable_cameras=True).app

from smoke_utils import capture_obs, export_episode
import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error


def tensor(x):
    return x if torch.is_tensor(x) else x.torch


def array(x):
    return tensor(x).detach().cpu().numpy().copy()


def read_state(group):
    return {k: read_state(v) if isinstance(v, h5py.Group) else
            torch.as_tensor(v[()], device=args.device) for k, v in group.items()}


def bounded(v, limit):
    return v * torch.clamp(limit / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8), max=1.)


def main():
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    with h5py.File(args.dataset, 'r') as f:
        demo = f['data'][args.demo]
        initial = read_state(demo['initial_state'])
        expected_eef = demo['obs/eef_pos'][0]
    # Check the actual math implementation, not only its documentation, before moving.
    zero = torch.zeros((1, 3), device=args.device)
    identity = torch.tensor([[0., 0., 0., 1.]], device=args.device)
    x90 = torch.tensor([[2**-.5, 0., 0., 2**-.5]], device=args.device)
    _, probe = compute_pose_error(zero, identity, zero, x90, rot_error_type='axis_angle')
    torch.testing.assert_close(probe, torch.tensor([[np.pi/2, 0., 0.]], device=args.device), atol=1e-5, rtol=1e-5)
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
    env = gym.make(args.task, cfg=cfg).unwrapped
    training_observations = []
    records = {k: [] for k in ('actions', 'pre_eef', 'post_eef', 'pre_quat_xyzw', 'post_quat_xyzw',
                              'post_cube', 'gripper_pos', 'phase', 'above_error_m', 'orientation_error_rad',
                              'target_position', 'ik_sensor_position_error_m', 'ik_sensor_angle_error_rad')}
    try:
        obs, _ = env.reset_to(initial, env_ids=None, seed=args.seed, is_relative=True)
        def poses():
            sensor = env.scene['ee_frame'].data
            return tensor(sensor.target_pos_w)[:, 0].clone(), tensor(sensor.target_quat_w)[:, 0].clone()
        def cube():
            return array(tensor(env.scene['object'].data.root_pos_w)[0] - env.scene.env_origins[0])
        pos_w, quat_w = poses()
        initial_eef = array(pos_w[0] - env.scene.env_origins[0])
        initial_cube = cube()
        reset = {
            'eef_first_obs_error_m': float(np.linalg.norm(array(obs['policy']['eef_pos'])[0] - expected_eef)),
            'cube_position_error_m': float(np.linalg.norm(initial_cube - array(initial['rigid_object']['object']['root_pose'])[0, :3])),
            'robot_joint_max_abs_error_rad': float(np.max(np.abs(array(env.scene['robot'].data.joint_pos)[0] - array(initial['articulation']['robot']['joint_position'])[0])))
        }
        if max(reset.values()) > 1e-4:
            raise RuntimeError(f'Reset mismatch: {reset}')
        action_term = env.action_manager.get_term('arm_action')
        # x-axis half-turn in WORLD frame, xyzw. Keep the gripper pointing down.
        target_quat_w = torch.tensor([[1., 0., 0., 0.]], device=env.device)
        dt = float(cfg.sim.dt) * int(cfg.decimation)
        required_hold = int(np.ceil(1. / dt))
        stage, stable, close_steps = 'approach', 0, 0
        hold_streak = lift_streak = max_hold = max_lift = 0
        settled_height = None
        pick_anchor = None
        success = False
        reason = 'horizon'
        with torch.inference_mode():
            for step in range(args.horizon):
                pos_w, quat_w = poses()
                pre = array(pos_w[0] - env.scene.env_origins[0])
                obj = cube()
                robot = env.scene['robot'].data
                root_pos, root_quat = tensor(robot.root_pos_w), tensor(robot.root_quat_w)
                current_pos_b, current_quat_b = subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)
                ik_pos_b, ik_quat_b = action_term._compute_frame_pose()
                offset_error, angle_error = compute_pose_error(current_pos_b, current_quat_b, ik_pos_b, ik_quat_b,
                                                               rot_error_type='axis_angle')
                offset_norm, angle_norm = float(offset_error.norm()), float(angle_error.norm())
                if offset_norm > 1e-4 or angle_norm > 1e-3:
                    raise RuntimeError(f'IK/sensor frame mismatch: {offset_norm} m, {angle_norm} rad')
                target = obj + [0., 0., .1]
                grip = 1.
                if args.instruction == 'pick up the cube':
                    if stage in ('descend', 'close'):
                        target = obj.copy()
                    elif stage == 'lift':
                        target = pick_anchor + [0., 0., .18]
                    if stage in ('close', 'lift'):
                        grip = -1.
                target_w = torch.as_tensor(target, dtype=pos_w.dtype, device=env.device)[None] + env.scene.env_origins
                goal_pos_b, goal_quat_b = subtract_frame_transforms(root_pos, root_quat, target_w, target_quat_w)
                pos_err, rot_err = compute_pose_error(ik_pos_b, ik_quat_b, goal_pos_b, goal_quat_b,
                                                       rot_error_type='axis_angle')
                orientation_error = float(rot_err.norm())
                delta = torch.cat([bounded(.2 * pos_err, .01), bounded(.2 * rot_err, .08)], dim=-1)
                action = torch.cat([delta / scale, torch.tensor([[grip]], device=env.device)], dim=-1)
                phase = stage
                if step < 13:
                    action.zero_()
                    action[:, 6] = 1.
                    phase = 'settle'
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite teacher action')
                training_observations.append(capture_obs(obs))
                obs, _, terminated, truncated, _ = env.step(action)
                post_pos_w, post_quat_w = poses()
                post = array(post_pos_w[0] - env.scene.env_origins[0])
                post_cube = cube()
                fingers = array(obs['policy']['gripper_pos'])[0]
                error = float(np.linalg.norm(post - (post_cube + [0., 0., .1])))
                if step == 10:
                    settled_height = float(post_cube[2])
                valid_above = error <= .03 and fingers[0] >= .03 and fingers[1] <= -.03
                hold_streak = hold_streak + 1 if valid_above and step >= 13 else 0
                max_hold = max(max_hold, hold_streak)
                lifted = (settled_height is not None and post_cube[2] >= settled_height + .1
                          and float(action[0, 6]) < 0 and np.linalg.norm(post - post_cube) <= .08)
                lift_streak = lift_streak + 1 if lifted else 0
                max_lift = max(max_lift, lift_streak)
                values = (array(action)[0], pre, post, array(quat_w)[0], array(post_quat_w)[0], post_cube,
                          fingers, phase, error, orientation_error, target, offset_norm, angle_norm)
                for key, value in zip(records, values):
                    records[key].append(value)
                if step % 10 == 0:
                    Image.fromarray(array(obs['policy']['table_cam'])[0, ..., :3].astype(np.uint8)).save(out/'frames'/f'frame_{step:04d}.png')
                if step % 25 == 0:
                    print('[STEP]', step+1, phase, 'above_error_m=', error, 'rotation_error_rad=', orientation_error, flush=True)
                if bool(terminated[0]) or bool(truncated[0]):
                    reason = 'unexpected_environment_termination'
                    break
                if args.instruction == 'move above the cube' and hold_streak >= required_hold:
                    success, reason = True, 'above_position_held'
                    break
                if args.instruction == 'pick up the cube' and lift_streak >= 10:
                    success, reason = True, 'cube_lifted'
                    break
                if step >= 13 and args.instruction == 'pick up the cube':
                    reached = np.linalg.norm(post - target) <= .01 and orientation_error <= .1
                    stable = stable + 1 if reached else 0
                    if stage == 'approach' and stable >= 5:
                        stage, stable = 'descend', 0
                    elif stage == 'descend' and stable >= 5:
                        stage, stable, close_steps = 'close', 0, 0
                        pick_anchor = post_cube.copy()
                    elif stage == 'close':
                        close_steps += 1
                        if close_steps >= 20:
                            stage = 'lift'
        if success:
            export_episode(out, args, records, training_observations)
        np.savez_compressed(out/'teacher_trace.npz', **{k: np.asarray(v) for k, v in records.items()},
                            initial_eef=initial_eef, initial_cube=initial_cube, instruction=args.instruction)
        summary = {
            'instruction': args.instruction, 'success': success, 'stop_reason': reason,
            'teacher_type': 'official_pose_error_in_robot_frame_bounded_steps', 'learned_language_policy': False,
            'executed_steps': len(records['actions']), 'step_dt': dt, 'reset_check': reset,
            'max_consecutive_above_hold_steps': max_hold, 'required_above_hold_steps': required_hold,
            'max_consecutive_lift_steps': max_lift, 'final_above_error_m': records['above_error_m'][-1],
            'final_cube_height_m': float(records['post_cube'][-1][2]),
            'settled_cube_height_m': settled_height, 'last_phase': records['phase'][-1],
            'settings': {'quaternion': 'xyzw, runtime checked', 'target_quaternion_world': [1,0,0,0],
                         'hand_offset_m': list(sensor_offset.pos), 'scale': scale, 'gain': .2,
                         'max_translation_increment_m': .01, 'max_rotation_increment_rad': .08,
                         'settle_steps': 13},
            'limitations': ['Teacher diagnostic only; not VLA success.',
                            'Pick uses lift proxy, not contact-force validation.',
                            'Changes convention, frame alignment, desired orientation and gains together; not a single-variable causal test.'],
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'frames': str(out/'frames')}
        (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
        print('SUMMARY=' + str(out/'summary.json'), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
