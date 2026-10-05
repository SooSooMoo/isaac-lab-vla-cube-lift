from pathlib import Path
import hashlib
import inspect
import json
import runpy
import sys
import traceback
root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
instruction = sys.argv[3]
output = Path(sys.argv[4])
sys.path.insert(0, str(root / 'code'))
import smoke_utils
original_load = smoke_utils.load_policy

def guarded_load(args):
    import h5py
    import numpy as np
    caller = inspect.currentframe().f_back
    obs = caller.f_locals['obs']
    captured = smoke_utils.capture_obs(obs)
    demo_name = {'pick up the cube': 'demo_0', 'move above the cube': 'demo_1'}[instruction]
    comparisons = {}
    with h5py.File(root / 'two_instruction.hdf5', 'r', swmr=True) as h5:
        saved = h5['data'][demo_name]['obs']
        for key in ('table_cam', 'wrist_cam', 'eef_pos', 'eef_quat', 'gripper_pos'):
            live = np.asarray(captured[key])
            reference = np.asarray(saved[key][0])
            same_shape = live.shape == reference.shape
            difference = float(np.max(np.abs(live.astype(np.float64) - reference.astype(np.float64)))) if same_shape else None
            comparisons[key] = {'same_shape': same_shape, 'max_abs_difference': difference, 'pass': bool(same_shape and np.array_equal(live, reference))}
    passed = all((item['pass'] for item in comparisons.values()))
    report = {'pass': passed, 'comparisons': comparisons}
    (output / 'initial_input_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('INITIAL_INPUT_CHECK', json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError('Initial input mismatch: evaluation stopped')
    import os
    import torch

    def settings():
        return {'algorithms': torch.are_deterministic_algorithms_enabled(), 'benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    original = settings()
    expected = {'algorithms': False, 'benchmark': True, 'cudnn_deterministic': False, 'matmul_tf32': False, 'cudnn_tf32': True, 'workspace': None}
    if original != expected:
        raise RuntimeError(f'Runtime defaults changed: {original}')
    torch.backends.cudnn.benchmark = False
    record = {'before': original, 'after': settings()}
    (output / 'torch_settings.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('TORCH_SETTINGS', json.dumps(record), flush=True)
    predict = original_load(args)
    trace = output / 'policy_steps.jsonl'

    def recorded_predict(obs, step):
        action = predict(obs, step)
        captured = smoke_utils.capture_obs(obs)
        row = {'step': int(step), 'pre_eef': np.asarray(captured['eef_pos']).tolist(), 'action': np.asarray(action).tolist()}
        with trace.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        return action
    return recorded_predict
smoke_utils.load_policy = guarded_load
sys.argv = [str(root / 'code/evaluate.py'), '--checkpoint', str(checkpoint), '--instruction', instruction, '--dataset', '/workspace/step3/data/lift_robomimic_language.hdf5', '--demo', 'demo_0', '--seed', '2058', '--device', 'cuda:0', '--horizon', '550', '--output-dir', str(output)]
try:
    runpy.run_path(str(root / 'code/evaluate.py'), run_name='__main__')
except BaseException:
    error = traceback.format_exc()
    (output.parent / (output.name + '_error.txt')).write_text(error, encoding='utf-8')
    print(error, flush=True)
    raise