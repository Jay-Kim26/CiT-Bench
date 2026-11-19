#!/usr/bin/env python3
"""Generate Final-protocol responses with vLLM for a text-input model (resumable, batched).

Weights and vLLM are installed separately. Each chunk of queries runs the retry schedule of the task and is
appended to --out; rerunning the same command resumes after the last completed chunk.
"""
import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

from cit_eval import PROTOCOL, ROOT, collect_batch, load, read_jsonl, sha, write_json
from verify import verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='Local model directory or Hugging Face model ID')
    ap.add_argument('--task', type=str.lower, choices=('ad', 'imp', 'fc'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--chunk', type=int, default=50, help='Queries per batch')
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--gpu-memory-utilization', type=float, default=.85)
    ap.add_argument('--no-mm-limit', action='store_true', help='Do not pass limit_mm_per_prompt (for engines that reject it)')
    args = ap.parse_args()
    verify()
    records, prompts = load(args.task)
    metadata = {'model': args.model, 'task': args.task, 'protocol': PROTOCOL,
                'package_manifest_sha256': sha(ROOT / 'MANIFEST.json'), 'runner_sha256': sha(__file__),
                'max_model_len': args.max_model_len, 'tensor_parallel_size': args.tensor_parallel_size,
                'gpu_memory_utilization': args.gpu_memory_utilization, 'chunk': args.chunk,
                'packages': {p: version(p) for p in ('vllm', 'torch', 'transformers', 'numpy')}}
    stamp = args.out.with_suffix(args.out.suffix + '.meta.json')
    if stamp.exists():
        if json.loads(stamp.read_text()) != metadata:
            raise ValueError('Resume metadata differs; use a new output file')
    elif args.out.exists():
        raise ValueError('Existing output has no metadata file')
    else:
        write_json(stamp, metadata)
    done = {r['id'] for r in read_jsonl(args.out)} if args.out.exists() else set()
    pending = [i for i in records if i not in done]
    if not pending:
        print('All queries are already completed.')
        return
    from vllm import LLM, SamplingParams
    extra = {} if args.no_mm_limit else {'limit_mm_per_prompt': {'image': 0, 'video': 0}}
    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=args.max_model_len, max_num_seqs=256,
              tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True, **extra)
    tk = llm.get_tokenizer()
    encoded = {}
    for i in pending:
        p = prompts[i]
        text = tk.apply_chat_template(p['messages'], tokenize=False, add_generation_prompt=True, **p['chat_template_kwargs'])
        ids = tk.encode(text, add_special_tokens=False)
        if len(ids) + p['max_tokens'] > args.max_model_len:
            raise ValueError(f'Prompt plus output budget exceeds --max-model-len: {i}')
        encoded[i] = ids
    usage = {i: {'finish_reasons': [], 'output_tokens': []} for i in pending}

    def generate(live, n, seeds):
        params = [SamplingParams(n=n, seed=seeds[i], max_tokens=prompts[i]['max_tokens'], temperature=prompts[i]['temperature'],
                                 top_p=prompts[i]['top_p'], top_k=prompts[i]['top_k']) for i in live]
        outputs = llm.generate([{'prompt_token_ids': encoded[i]} for i in live], params, use_tqdm=False)
        drawn = {}
        for i, output in zip(live, outputs):
            choices = sorted(output.outputs, key=lambda c: c.index)
            drawn[i] = [c.text for c in choices]
            usage[i]['finish_reasons'] += [c.finish_reason for c in choices]
            usage[i]['output_tokens'] += [len(c.token_ids) for c in choices]
        return drawn

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), args.chunk):
        chunk = pending[start:start + args.chunk]
        rows = collect_batch(args.task, records, chunk, generate)
        with args.out.open('a') as stream:
            for i in chunk:
                stream.write(json.dumps({**rows[i], 'model': args.model, 'usage': usage[i]}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(f'{args.task}: {len(done) + start + len(chunk)}/{len(records)} queries', flush=True)


if __name__ == '__main__':
    main()
