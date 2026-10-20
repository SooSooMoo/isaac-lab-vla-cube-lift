from pathlib import Path
import hashlib, json, os
root=Path(__file__).resolve().parent
manifest=json.loads((root/'checkpoint_parts.json').read_text())
target=root/'best.pt'
if target.exists():raise SystemExit('best.pt already exists; refusing overwrite')
temp=root/'best.pt.restoring'
full=hashlib.sha256()
with temp.open('xb') as out:
    for item in manifest['parts']:
        data=(root/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise RuntimeError('Part checksum mismatch: '+item['file'])
        out.write(data);full.update(data)
    out.flush();os.fsync(out.fileno())
if full.hexdigest()!=manifest['sha256']:raise RuntimeError('Model checksum mismatch')
temp.replace(target)
print('MODEL_RESTORED',target,'SHA256',full.hexdigest())
