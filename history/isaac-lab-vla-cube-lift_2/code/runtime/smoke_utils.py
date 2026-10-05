
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

INSTRUCTIONS = ('pick up the cube', 'move above the cube')
OBS_KEYS = ('table_cam', 'wrist_cam', 'eef_pos', 'eef_quat', 'gripper_pos')

def tokens(text):
    words = text.lower().replace('.', '').replace(',', '').split()[:16]
    return np.asarray([1 + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(),
                       'little') % 4095 for w in words] or [0], dtype=np.int64)

def to_array(value):
    if hasattr(value, 'torch'):
        value = value.torch
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()

def capture_obs(obs):
    result = {}
    for key in OBS_KEYS:
        value = to_array(obs['policy'][key])[0]
        if key.endswith('_cam'):
            if value.ndim != 3 or value.shape[-1] < 3 or value.dtype != np.uint8:
                raise ValueError(f'{key}: expected HWC uint8 RGB(A), got {value.shape}/{value.dtype}')
            value = value[..., :3].copy()
        elif not np.isfinite(value).all():
            raise ValueError(f'Nonfinite observation: {key}')
        if key == 'eef_quat' and value[3] < 0:
            value = -value
        result[key] = value
    return result

def export_episode(out, args, records, observations):
    path = Path(out) / 'training_episode.hdf5'
    length = len(records['actions'])
    if length != len(observations) or length == 0:
        raise ValueError('Observation/action alignment failure')
    with h5py.File(args.dataset, 'r') as source, h5py.File(path, 'x') as dest:
        data = dest.create_group('data')
        for key, value in source['data'].attrs.items():
            data.attrs[key] = value
        data.attrs['total'] = length
        data.attrs['language_conditioning'] = 'per_demo_token_ids'
        data.attrs['observation_timing'] = 'pre_action'
        data.attrs['progress_denominator'] = 1200.0
        data.attrs['experiment_protocol'] = 'fixed_state_fit_check_not_generalization'
        data.attrs['language_instruction'] = args.instruction
        demo = data.create_group('demo_0')
        demo.attrs['num_samples'] = length
        demo.attrs['success'] = True
        demo.attrs['language_instruction'] = args.instruction
        demo.attrs['language_tokenizer'] = 'blake2b_word_hash_v1'
        demo.attrs['language_vocab_size'] = 4096
        demo.attrs['source_demo'] = args.demo
        demo.attrs['source_dataset'] = str(Path(args.dataset).resolve())
        demo.attrs['seed'] = args.seed
        source.copy(source['data'][args.demo]['initial_state'], demo, name='initial_state')
        demo.create_dataset('actions', data=np.asarray(records['actions'], dtype=np.float32))
        demo.create_dataset('language_tokens', data=tokens(args.instruction))
        obs = demo.create_group('obs')
        for key in OBS_KEYS:
            obs.create_dataset(key, data=np.stack([row[key] for row in observations]),
                               compression='gzip', compression_opts=1)
    print('[DATASET]', path, flush=True)

def check_episode(demo):
    instruction = demo.attrs['language_instruction']
    if instruction not in INSTRUCTIONS or not bool(demo.attrs.get('success', False)):
        raise ValueError('Expected a successful episode of one of the two instructions')
    actions = demo['actions'][:]
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError('Invalid actions')
    if not np.array_equal(demo['language_tokens'][:], tokens(instruction)):
        raise ValueError('Instruction/token mismatch')
    for key in OBS_KEYS:
        values = demo['obs'][key]
        if len(values) != len(actions):
            raise ValueError(f'Observation/action length mismatch: {key}')
        if key.endswith('_cam'):
            if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
                raise ValueError(f'Invalid RGB images: {key}')
        elif values.shape[1:] != {'eef_pos': (3,), 'eef_quat': (4,), 'gripper_pos': (2,)}[key]:
            raise ValueError(f'Invalid state shape: {key}')
    return instruction, len(actions)

def state_equal(left, right):
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, h5py.Group) != isinstance(b, h5py.Group):
            return False
        if isinstance(a, h5py.Group):
            if not state_equal(a, b):
                return False
        elif not np.array_equal(a[()], b[()]):
            return False
    return True

def load_policy(args):
    import torch
    from language_vla.model import LanguageConditionedVLA
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get('experiment_protocol') != 'fixed_state_fit_check_not_generalization':
        raise ValueError('Use a checkpoint produced by this experiment')
    if checkpoint.get('progress_denominator') != 1200.0:
        raise ValueError('Use the corrected-data checkpoint with denominator 1200')
    model = LanguageConditionedVLA(**{k: int(checkpoint[k]) for k in
                                    ('state_dim', 'action_dim', 'vocab_size', 'language_dim')})
    model.to(args.device).load_state_dict(checkpoint['model'])
    model.eval()
    model.set_instruction(args.instruction)
    language = torch.tensor(tokens(args.instruction), device=args.device).unsqueeze(0)
    def predict(obs, step):
        captured = capture_obs(obs)
        def image(key):
            value = torch.tensor(captured[key], device=args.device).permute(2, 0, 1).float()[None] / 255.
            mean = torch.tensor([.485, .456, .406], device=args.device).view(1, 3, 1, 1)
            std = torch.tensor([.229, .224, .225], device=args.device).view(1, 3, 1, 1)
            return (value - mean) / std
        state = np.concatenate([captured['eef_pos'], captured['eef_quat'], captured['gripper_pos'],
                                np.asarray([step / 1200.], dtype=np.float32)])
        output = model(table_image=image('table_cam'), wrist_image=image('wrist_cam'),
                       state=torch.tensor(state, dtype=torch.float32, device=args.device)[None],
                       language=language)
        return model.build_action(output).detach().cpu().numpy()[0].copy()
    return predict


_stage350_original_load_policy = load_policy
def load_policy(args):
    import copy
    import hashlib
    from pathlib import Path
    base_path = Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST/pick_branch_20260916_150628_449371_JST/training/best_pick_branch.pt')
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == '97405cf86e1b500013eedf4b94a827cd527c9db6ef9810940302e4b9994b64a9'
    prefix_args = copy.copy(args)
    prefix_args.checkpoint = str(base_path)
    prefix = _stage350_original_load_policy(prefix_args)
    if args.instruction == "move above the cube":
        return prefix
    assert args.instruction == "pick up the cube"
    suffix = _stage350_original_load_policy(args)
    def predict(obs, step):
        if step == 350:
            print("LEARNED_SUFFIX_TAKEOVER STEP 350 TEACHER False", flush=True)
        return prefix(obs, step) if step < 350 else suffix(obs, step)
    return predict


_before_lift_load_policy=load_policy
def load_policy(args):
    import copy,hashlib
    from pathlib import Path
    source_path=Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST/train_descent_alignment_20260916_181508_562003_JST/training/last_updates.pt')
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='20684032dd42bac61a8d9730c2c8377b5239584f5de831135232cdce99f11d5a'
    before_args=copy.copy(args)
    before_args.checkpoint=str(source_path)
    before=_before_lift_load_policy(before_args)
    if args.instruction=='move above the cube':return before
    assert args.instruction=='pick up the cube'
    lift=_before_lift_load_policy(args)
    def predict(obs,step):
        if step==551:print('LEARNED_LIFT_TAKEOVER STEP 551 TEACHER False',flush=True)
        return before(obs,step) if step<551 else lift(obs,step)
    return predict
