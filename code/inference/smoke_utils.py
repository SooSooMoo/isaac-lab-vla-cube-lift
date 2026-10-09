import hashlib
import numpy as np
OBS_KEYS = ('table_cam','wrist_cam','eef_pos','eef_quat','gripper_pos')

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
