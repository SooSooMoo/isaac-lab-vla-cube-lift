import hashlib
import numpy as np
import torch

def tokens(text):
    words = text.lower().replace('.', '').replace(',', '').split()[:16]
    return np.asarray([1 + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(),
                       'little') % 4095 for w in words] or [0], dtype=np.int64)

def read_instruction(h5):
    def decode(x):
        return x.decode('utf-8') if isinstance(x,bytes) else str(x)
    root=decode(h5.attrs['instruction'])
    episode=decode(h5['data/demo_0'].attrs['instruction'])
    if root != episode or root != 'pick up the cube':
        raise ValueError('Instruction missing, inconsistent or unsupported')
    return episode

def prepare(obs, instruction, device='cpu'):
    if instruction != 'pick up the cube':raise ValueError('Unsupported instruction')
    images=[]
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    for name in ('table_cam','wrist_cam'):
        a=np.asarray(obs[name])
        if a.shape!=(200,200,3) or a.dtype!=np.uint8:raise ValueError('Invalid image: '+name)
        t=torch.tensor(a.copy(),device=device).permute(2,0,1)[None].float()/255.
        images.append((t-mean)/std)
    q=np.asarray(obs['eef_quat']).copy()
    if q[3]<0:q=-q
    state=np.concatenate([obs['eef_pos'],q,obs['gripper_pos']]).astype(np.float32)
    if state.shape!=(9,) or not np.isfinite(state).all():raise ValueError('Invalid state')
    language=torch.tensor(tokens(instruction),device=device)[None]
    return (*images,torch.tensor(state,device=device)[None],language)
