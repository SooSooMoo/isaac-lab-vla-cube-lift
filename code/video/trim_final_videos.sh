#!/usr/bin/env bash
set -euo pipefail
/isaac-sim/python.sh -u - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg

root = Path('/workspace/step3/fixed_vla_4090_20260916_095545_374496_JST')
jobs = [
    ('above', 'record_continued3s_v2_above_20260919_105322_814504_JST', 574, 495),
    ('pick', 'record_continued3s_v2_pick_20260919_105151_114828_JST', 1029, 930),
]
encoder = imageio_ffmpeg.get_ffmpeg_exe()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def inspect(path):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        metadata = next(reader)
        count = sum(1 for _ in reader)
    finally:
        reader.close()
    return count, float(metadata['fps'])

prepared = []
for task, run, expected, keep in jobs:
    source = root / run / task / 'policy_view_continued3s.mp4'
    destination = source.with_name('policy_view_trimmed.mp4')
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f'Existing edited video preserved: {destination}')
    count, fps = inspect(source)
    if count != expected or abs(fps - 50) > 0.01:
        raise RuntimeError(f'Unexpected source: {task}, frames={count}, fps={fps}')
    prepared.append((task, source, destination, keep, sha(source)))

for task, source, destination, keep, original_sha in prepared:
    subprocess.run([
        encoder, '-hide_banner', '-loglevel', 'error', '-n',
        '-i', str(source), '-map', '0:v:0', '-an',
        '-frames:v', str(keep), '-c:v', 'libx264', '-crf', '18',
        '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(destination),
    ], check=True)
    count, fps = inspect(destination)
    if count != keep or abs(fps - 50) > 0.01:
        raise RuntimeError(f'Edited video verification failed: {destination}')
    if sha(source) != original_sha:
        raise RuntimeError(f'Source hash changed: {source}')
    print('TRIMMED_VIDEO_RESULT', json.dumps({
        'task': task, 'video_file': str(destination),
        'decoded_frames': count, 'fps': fps, 'duration_seconds': count / fps,
        'source_unchanged': True, 'verification_pass': True,
        'editing': 'Tail removed only; no freeze frames or simulation changes',
    }), flush=True)
print('BOTH_VIDEO_TRIMS_COMPLETE')
PY
