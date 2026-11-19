#!/usr/bin/env python3
"""CPU-only checks of the shipped data, parsers, retry schedule, seeds and scoring. No model is run."""
import json

from cit_eval import CELLS, PROTOCOL, collect_batch, evaluate, load, parse, seed
from verify import verify


def exact(task, record):
    """A perfect answer in the requested output format."""
    if task == 'ad':
        s, e = record['candidates'][record['target_idx']]['span']
        return json.dumps({'start': s, 'end': e})
    inp = record['input']
    steps = range(*inp['gap']) if task == 'imp' else range(inp['forecast_start'], inp['forecast_start'] + inp['horizon'])
    return '\n'.join(f'({t}, {v:.10f})' for t, v in zip(steps, record['private']['truth']))


def main():
    checks = {'files': verify()}
    for task in ('ad', 'imp', 'fc'):
        records, _ = load(task)
        rows = list(records.values())
        need = PROTOCOL['tasks'][task]['need_valid']
        perfect = evaluate(task, rows, [{'id': r['id'], 'samples': [exact(task, r)] * need} for r in rows])
        empty = evaluate(task, rows, [])
        for cell in CELLS:
            p, e = perfect['cells'][cell], empty['cells'][cell]
            assert p['queries'] == e['queries'] == 100
            if task == 'ad':
                assert p['accuracy'] == p['consistency'] == 1 and e['accuracy'] == e['consistency'] == 0
            else:
                assert p['coverage'] == 1 and p['rcrps'] < 1e-8 and e['coverage'] == 0 and e['rcrps'] == 1
        if task == 'ad':
            first = {r['pair_id']: exact(task, r) for r in rows if r['condition'] == 'A'}
            fixed = evaluate(task, rows, [{'id': r['id'], 'samples': [first[r['pair_id']]]} for r in rows])
            assert all(c['accuracy'] == .5 and c['consistency'] == 0 for c in fixed['cells'].values())
        # Parsing rules on the first record.
        r = rows[0]
        if task == 'ad':
            s, e = r['candidates'][r['target_idx']]['span']
            assert parse(task, r, f'Answer: {{"start": {s}, "end": {e}}}') == (s, e)
            assert parse(task, r, f'{{"start": {e}, "end": {s}}}') is None or s == e
            assert parse(task, r, f'{{"start": 0, "end": {r["L"]}}}') is None
            assert parse(task, r, 'no answer') is None
        else:
            text = exact(task, r)
            lines = text.split('\n')
            assert parse(task, r, 'Here are the values:\n' + text) is not None
            assert parse(task, r, text.replace('(', '').replace(')', '')) is not None
            assert parse(task, r, '\n'.join(l.strip('()').replace(', ', ':') for l in lines)) is not None
            assert parse(task, r, '\n'.join(lines[:-1])) is None          # the last required step is missing
            first = lines[0].rstrip(')')
            assert parse(task, r, '\n'.join([first + '+1.0)'] + lines[1:])) is None      # arithmetic expression: format failure
            assert parse(task, r, '\n'.join([first + ' * 2)'] + lines[1:])) is None
            assert parse(task, r, '* ' + '\n* '.join(l.strip('()') for l in lines)) is not None   # "* i, v" bullets are not operators
        # Retry schedule with a scripted generator.
        valid, invalid = exact(task, r), 'no answer'
        calls = []

        def scripted(live, n, seeds, plan):
            calls.append((n, dict(seeds)))
            k = len(calls) - 1
            return {i: [valid if j < plan[k] else invalid for j in range(n)] for i in live}

        if task == 'ad':
            out = collect_batch(task, records, [r['id']], lambda live, n, s: scripted(live, n, s, [0, 0, 1, 1, 1]))
            assert len(out[r['id']]['samples']) == 3 and out[r['id']]['n_valid_total'] == 1
        else:
            out = collect_batch(task, records, [r['id']], lambda live, n, s: scripted(live, n, s, [20, 5, 5, 5]))
            assert len(out[r['id']]['samples']) == 30 and out[r['id']]['n_valid_total'] == 25
            calls.clear()
            out = collect_batch(task, records, [r['id']], lambda live, n, s: scripted(live, n, s, [0, 0, 0, 0]))
            assert len(out[r['id']]['samples']) == 40 and out[r['id']]['n_valid_total'] == 0
            assert [n for n, _ in calls] == [25, 5, 5, 5]
        # Seeds reproduce the ones recorded in the reported runs.
        for example in PROTOCOL['tasks'][task]['seed_examples']:
            assert seed(task, example['id'], example['round']) == example['seed'], example
        checks[task] = {'perfect': 'PASS', 'failures': 'PASS', 'parsing': 'PASS', 'retry': 'PASS',
                        'seeds': len(PROTOCOL['tasks'][task]['seed_examples'])}
    print(json.dumps({'status': 'PASS', **checks}, indent=2))


if __name__ == '__main__':
    main()
