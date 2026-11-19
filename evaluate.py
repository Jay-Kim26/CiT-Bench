#!/usr/bin/env python3
"""Score the responses of one task under the Final protocol and print the Level x axis table."""
import argparse
import json
from pathlib import Path

from cit_eval import CELLS, evaluate, load, read_jsonl, sha, write_json
from verify import verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', type=str.lower, choices=('ad', 'imp', 'fc'), required=True)
    ap.add_argument('--responses', type=Path, required=True, help='JSONL rows with "id" and ordered "samples"')
    ap.add_argument('--out', type=Path, required=True, help='JSON file for the cell summaries and per-item scores')
    args = ap.parse_args()
    verify()
    records, _ = load(args.task)
    result = evaluate(args.task, list(records.values()), read_jsonl(args.responses))
    result['responses_sha256'] = sha(args.responses)
    write_json(args.out, result)
    rows = [*CELLS, 'overall']
    table = {c: result['cells'][c] if c != 'overall' else result['overall'] for c in rows}
    if args.task == 'ad':
        line = {c: f"ACC {100 * s['accuracy']:.1f} / Cons {100 * s['consistency']:.1f}" for c, s in table.items()}
    else:
        line = {c: f"Cov {100 * s['coverage']:.0f} / RCRPS {s['rcrps']:.3f}"
                   + (f" ({s['rcrps_valid_only']:.3f})" if s['rcrps_valid_only'] is not None else '') for c, s in table.items()}
    print(json.dumps({'task': args.task, **line}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
