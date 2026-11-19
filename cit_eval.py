"""CiT-Bench 2027 Final protocol (answer-only, untagged output) for AD, IMP and FC.

Parsing, sampling seeds, the retry schedule and scoring, with NumPy as the only dependency.
The rules are those used for the reported results; `protocol.json` holds the fixed settings.
IMP/FC (t3 sets): a sample that writes a value as an arithmetic expression, e.g. "(108, 24.1+21.7)", is a format
failure (a value followed on the same line by +, * or / and another number); the minus sign is not treated as an operator.
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
TASKS = ('ad', 'imp', 'fc')
CELLS = ('L1/magnitude', 'L1/temporal', 'L2/magnitude', 'L2/temporal')
SPAN = re.compile(r'"start"\s*:\s*(-?\d+)\s*,\s*"end"\s*:\s*(-?\d+)')
PAIR = re.compile(r"\(?\s*(\d+)\s*[,:]\s*(-?\d+(?:\.\d+)?)\s*\)?")
ARITHMETIC = re.compile(r"\(?[ \t]*\d+[ \t]*[,:][ \t]*-?\d+(?:\.\d+)?[ \t]*[+*/][ \t]*-?\d")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(task):
    return PROTOCOL['tasks'][task]


def load(task):
    """-> (records by id, prompts by id), in dataset order."""
    folder = ROOT / task.upper()
    records = {r['id']: r for r in read_jsonl(folder / 'records.jsonl')}
    prompts = {p['id']: p for p in read_jsonl(folder / 'prompts.jsonl')}
    if list(records) != list(prompts):
        raise ValueError(f'{task}: records and prompts differ')
    return records, prompts


def seed(task, query_id, round_index):
    namespace = settings(task)['seed_namespace']
    return int(hashlib.sha256(f'{namespace}|{query_id}|{round_index}'.encode()).hexdigest()[:8], 16) % (2 ** 31)


def required_steps(task, record):
    inp = record['input']
    if task == 'imp':
        return range(*inp['gap'])
    return range(inp['forecast_start'], inp['forecast_start'] + inp['horizon'])


def parse(task, record, text):
    """AD -> (start, end) or None. IMP/FC -> array of values for the required steps, or None."""
    if not isinstance(text, str):
        return None
    if task == 'ad':
        m = SPAN.search(text)                      # the first "start": s, "end": e anywhere in the output
        if not m:
            return None
        s, e = int(m[1]), int(m[2])
        return (s, e) if 0 <= s <= e < record['L'] else None
    if ARITHMETIC.search(text):                            # "(i, 24.1+21.7)": format failure
        return None
    got = {int(k): float(v) for k, v in PAIR.findall(text)}   # "(i, v)", "i, v" or "i:v"; a repeated index keeps its last value
    steps = required_steps(task, record)
    if not all(t in got for t in steps):
        return None
    values = np.array([got[t] for t in steps], dtype=float)
    return values if np.isfinite(values).all() else None


def is_valid(task, record, text):
    return parse(task, record, text) is not None


def collect_batch(task, records, ids, generate):
    """Final retry schedule for many queries. generate(live_ids, n, seeds) -> {id: [n strings in draw order]}.

    AD: rounds of one draw, up to five, until one valid answer. IMP/FC: 25, then up to three rounds of five,
    until 25 valid answers. Additional draws depend only on parsing, never on correctness."""
    cfg = settings(task)
    out = {i: {'id': i, 'samples': [], 'retry_rounds': [], 'n_valid_total': 0} for i in ids}
    live = list(ids)
    for index, n in enumerate(cfg['rounds']):
        if not live:
            break
        seeds = {i: seed(task, i, index) for i in live}
        drawn = generate(live, n, seeds)
        for i in live:
            texts = drawn[i]
            if not isinstance(texts, list) or len(texts) != n or not all(isinstance(t, str) for t in texts):
                raise ValueError('generate must return exactly n strings per query')
            valid = sum(is_valid(task, records[i], t) for t in texts)
            out[i]['samples'] += texts
            out[i]['n_valid_total'] += valid
            out[i]['retry_rounds'].append({'round': index, 'requested': n, 'valid': valid, 'seed': seeds[i]})
        live = [i for i in live if out[i]['n_valid_total'] < cfg['need_valid']]
    for i in ids:
        out[i].update(dataset_version=settings(task)['dataset_version'], protocol_version=PROTOCOL['version'],
                      max_tokens=PROTOCOL['common']['max_tokens'])
    return out


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    return max(inter, 0) / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - max(inter, 0))


def fair_crps(samples, truth):
    n = len(samples)
    w = 2 * np.arange(1, n + 1) - n - 1
    spread = (np.sort(samples, axis=0) * w[:, None]).sum(axis=0) / (n * (n - 1))
    return np.abs(samples - truth).mean(axis=0) - spread


def rcrps(crps, roi, rho):
    if roi.all() or not roi.any():
        return float(rho * crps.mean())
    return float(rho * (crps[roi].mean() + crps[~roi].mean()) / 2)


def normalization(task, record):
    """1 / range of the values the model sees (IMP: observed values of the series; FC: the history)."""
    values = [v for v in record['input']['values'] if v is not None]
    return 1.0 / float(np.ptp(values))


def score_item(task, record, samples):
    cell = f"L{record['level']}/{record['axis']}"
    item = {'id': record['id'], 'cell': cell, 'pair_id': record['pair_id'], 'condition': record['condition'],
            'n_samples': len(samples)}
    parsed = [parse(task, record, t) for t in samples]
    item['n_valid'] = sum(p is not None for p in parsed)
    if task == 'ad':
        idx = next((k for k, p in enumerate(parsed) if p is not None), None)
        item.update(first_valid=idx, correct=False, chosen=None, iou=None)
        if idx is not None:
            span = parsed[idx]
            overlaps = [iou(span, c['span']) for c in record['candidates']]
            best = int(np.argmax(overlaps))
            chosen = 'elsewhere' if overlaps[best] == 0 else ('target' if best == record['target_idx'] else 'other_candidate')
            item.update(span=list(span), chosen=chosen, iou=overlaps[record['target_idx']],
                        correct=chosen == 'target' and overlaps[record['target_idx']] >= 0.5)
        return item
    need = settings(task)['need_valid']
    good = [p for p in parsed if p is not None][:need]
    item.update(covered=len(good) == need, rcrps=settings(task)['failure_score'])
    if item['covered']:
        c = fair_crps(np.asarray(good), np.asarray(record['private']['truth'], dtype=float))
        item['rcrps'] = rcrps(c, np.asarray(record['private']['roi_mask'], dtype=bool), normalization(task, record))
    return item


def evaluate(task, records, responses):
    """records: list in dataset order; responses: rows with 'id' and ordered 'samples'. Missing rows score as failures."""
    known = {r['id']: r for r in records}
    supplied = {}
    for row in responses:
        if row['id'] not in known or row['id'] in supplied:
            raise ValueError(f"Unknown or duplicate response id: {row['id']}")
        if row.get('protocol_version', PROTOCOL['version']) != PROTOCOL['version']:
            raise ValueError(f"Response made with another protocol: {row['id']}")
        if len(row['samples']) > sum(settings(task)['rounds']):
            raise ValueError(f"More samples than the retry schedule allows: {row['id']}")
        supplied[row['id']] = row
    items = [score_item(task, r, supplied.get(r['id'], {}).get('samples', [])) for r in records]

    def summary(part):
        draws = sum(i['n_samples'] for i in part)
        out = {'queries': len(part), 'sample_valid_rate': sum(i['n_valid'] for i in part) / draws if draws else None}
        if task == 'ad':
            pairs = defaultdict(list)
            for i in part:
                pairs[i['pair_id']].append(i['correct'])
            out.update(accuracy=sum(i['correct'] for i in part) / len(part), pairs=len(pairs),
                       consistency=sum(all(v) for v in pairs.values()) / len(pairs))
        else:
            ok = [i['rcrps'] for i in part if i['covered']]
            out.update(coverage=len(ok) / len(part), rcrps=float(np.mean([i['rcrps'] for i in part])),
                       rcrps_valid_only=float(np.mean(ok)) if ok else None)
        return out

    return {'task': task, 'dataset_version': settings(task)['dataset_version'], 'protocol_version': PROTOCOL['version'],
            'cells': {c: summary([i for i in items if i['cell'] == c]) for c in CELLS},
            'overall': summary(items), 'items': items}
