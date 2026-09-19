"""Read-only package integrity check; no training or simulator execution."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def check(repo, require_imported=False):
    errors = []
    counts = {'python': 0, 'json': 0, 'markdown': 0, 'shell_python_blocks': 0}
    for p in repo.rglob('*'):
        parts = p.relative_to(repo).parts
        if not p.is_file() or '.git' in parts or any(x.startswith('pod_before_update_') for x in parts):
            continue
        try:
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8'))
                counts['python'] += 1
            elif p.suffix == '.json':
                json.loads(p.read_text(encoding='utf-8'))
                counts['json'] += 1
            elif p.suffix == '.sh':
                text = p.read_text(encoding='utf-8')
                for block in re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY(?:\r?\n|$)", text, re.S):
                    tree = ast.parse(block)
                    counts['shell_python_blocks'] += 1
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            if any(isinstance(t, ast.Name) and t.id == 'replacement' for t in node.targets):
                                ast.parse(node.value.value)
            elif p.suffix == '.md':
                counts['markdown'] += 1
                for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
                    if '://' not in target and not target.startswith('#'):
                        if not (p.parent / target.split('#')[0]).exists():
                            errors.append(f'Missing link: {p}: {target}')
        except Exception as e:
            errors.append(f'{p}: {e}')
    inventory = json.loads((repo / 'source_inventory.json').read_text(encoding='utf-8'))
    for row in inventory['files']:
        p = repo / row['path']
        if not p.is_file() or sha(p) != row['sha256']:
            errors.append('Source hash mismatch: ' + row['path'])
    manifest = json.loads((repo / 'result/model/manifest.json').read_text(encoding='utf-8'))
    cfg = json.loads((repo / 'code/configs/success.json').read_text(encoding='utf-8'))
    if cfg['base_sha256'] != manifest['components'][0]['sha256']:
        errors.append('Base identity mismatch')
    if require_imported:
        imported = repo / 'result/evidence/import_manifest.json'
        if not imported.exists():
            errors.append('Pod import manifest missing')
        else:
            report = json.loads(imported.read_text(encoding='utf-8'))
            for row in report['files']:
                p = repo / row['destination']
                if not p.is_file() or sha(p) != row['sha256']:
                    errors.append('Imported file mismatch: ' + row['destination'])
            actual = {r['destination'] for r in report['files']}
            expected = json.loads((repo / 'code/configs/artifact_import.json').read_text(encoding='utf-8'))
            for row in expected['artifacts']:
                if row['required'] and row['destination'] not in actual:
                    errors.append('Required artifact missing: ' + row['destination'])
    return {'checks': counts, 'errors': errors, 'pass': not errors, 'simulator_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-imported', action='store_true')
    args = parser.parse_args()
    result = check(Path(__file__).resolve().parents[2], args.require_imported)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
