#!/usr/bin/env python3
"""Check that every shipped file is intact and that each task has 400 matching records and prompts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Package file missing or changed: {name}')
    counts = {}
    for task in ('AD', 'IMP', 'FC'):
        def rows(name):
            return [json.loads(s) for s in (ROOT / task / name).read_text().splitlines() if s.strip()]
        records, prompts = rows('records.jsonl'), rows('prompts.jsonl')
        ids = [r['id'] for r in records]
        if not (len(ids) == len(set(ids)) == 400 and ids == [p['id'] for p in prompts]):
            raise ValueError(f'{task}: unexpected counts or mismatched IDs')
        if task == 'AD':
            counts[task] = {'queries': 400, 'scenes': len({r['scene_id'] for r in records}), 'pairs': len(rows('pairs.jsonl'))}
        else:                                                   # IMP/FC: one L1 and one L2 query per scene, no A/B pairs
            scenes = rows('scenes.jsonl')
            if len(scenes) != 200 or sorted(i for s in scenes for i in (s['L1'], s['L2'])) != sorted(ids):
                raise ValueError(f'{task}: scenes.jsonl does not list every query once')
            counts[task] = {'queries': 400, 'scenes': len(scenes)}
    return {'status': 'PASS', 'package': manifest['package'], 'protocol_version': manifest['protocol_version'],
            'tasks': counts, 'verified_files': len(manifest['files_sha256'])}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
