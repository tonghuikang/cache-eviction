#!/usr/bin/env python3
"""Write gpt-oss-20b/data/<task>-s<seed>.json: the lost attention mass of the 20 Table 1 reference traces.

The traces are the 20 full-cache gpt-oss-20b runs on AIME 2026 problems 04, 11,
13, 26 and 29 with seeds 0 to 3. The report's Table 1 replays them with
code/cache_eviction/sglang_static.py. Nothing here is recomputed. The script reads what
that replay wrote.

Inputs, relative to the repository root:
  gpt-oss-20b/runs/perhead/decode/trials-005.jsonl.gz    the full-cache trials (policy "full")
  gpt-oss-20b/runs/perhead/decode/prompts.json.gz        prompt ids, key <task>|<effort>|<date>
  gpt-oss-20b/runs/perhead/static-full20/static.jsonl.gz one summary row per trace, policy, budget and metric
  gpt-oss-20b/runs/perhead/static-full20/static/<task>-s<seed>/
    reference.npz              ids, logprob_full, logprob_full_repeat (one value per token)
    lost-<policy>-<budget>.npz lost, one float32 per token, budgets 2048 to 32768
    replay-<policy>-<budget>.npz logprob and layer_error per token, budgets 8192 and 16384 only

What lost holds, from _diagnose in code/cache_eviction/sglang_evict_runtime.py:
  The unevicted model is teacher-forced on the trace while the server applies
  each policy's per-head eviction at every 2048-token boundary in shadow.
  lost[q] is the attention probability that query position q puts on keys the
  policy has already evicted, summed over keys and averaged over the 64 query
  heads and the 12 full-attention layers. It is 0 until the first eviction,
  which happens at the first boundary past the budget.
  Its mean over the trace is the static.jsonl "regret" row, the Table 1
  attention weight difference.

Shading: each token is shaded by lost at its own query position, the share of
that token's attention which went to keys the policy had evicted.

Output, one JSON object per trace:
  schema          "gpt-oss-20b-reference-lost-v1"
  task, seed, prompt_tokens, generated, length (prompt + generated)
  finish_reason, quality, prediction, answer   from the full-cache trial
  pieces          decoded text per token, prompt then output, streamed so a
                  multi-byte character split across tokens lands on the token
                  that completes it and earlier ones are ""
  kind_names      harmony labels, kinds = base64 uint8 index per token
  scale           {log10_min, log10_max, steps}: code 0 means lost < 10**log10_min,
                  code c >= 1 means lost = 10 ** (log10_min + (c - 1) * (log10_max - log10_min) / steps)
  lost            {policy: {budget: {start, codes, mean}}}. codes is base64
                  uint8, one code per position from start to the end. Every
                  position before start has lost exactly 0. mean is the exact
                  float mean of lost over the whole trace.

Run with an interpreter that has openai_harmony and numpy, from anywhere:
  python gpt-oss-20b/build_data.py
The output is deterministic.
"""
import base64
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DECODE = ROOT / 'gpt-oss-20b' / 'runs' / 'perhead' / 'decode'
STATIC = ROOT / 'gpt-oss-20b' / 'runs' / 'perhead' / 'static-full20'
OUT = Path(__file__).resolve().parent / 'data'
TASKS = ['aime2026-04', 'aime2026-11', 'aime2026-13', 'aime2026-26', 'aime2026-29']
SEEDS = [0, 1, 2, 3]
POLICIES = ['random_2048', 'random_reservoir_2048', 'prefix_sliding_2048', 'head_sum_2048', 'head_max_sum_2048']
BUDGETS = [2048, 4096, 8192, 16384, 32768]
LOG_MIN, LOG_MAX, STEPS = -4.0, 0.0, 254
CONTROL = {'<|start|>', '<|end|>', '<|return|>', '<|message|>', '<|channel|>', '<|constrain|>', '<|call|>'}
MAX_CARRY = 4  # the longest UTF-8 sequence, so a split can never span more tokens than this


def stream_decode(ids, encoding):
    """Decode token by token, carrying bytes that do not yet form a character.

    Copied from individual-project/report/make_trace_view.py in gstar.
    """
    pieces, carry = [], []
    for token in ids:
        carry.append(token)
        text = encoding.decode(carry)
        if '�' not in text or len(carry) >= MAX_CARRY:
            pieces.extend([''] * (len(carry) - 1))
            pieces.append(text)
            carry = []
    if carry:
        pieces.extend([''] * (len(carry) - 1))
        pieces.append(encoding.decode(carry))
    if len(pieces) != len(ids):
        raise ValueError(f'decoded {len(pieces)} pieces for {len(ids)} tokens')
    return pieces


def channels(pieces):
    """Label each token with its harmony channel, or its role in the prompt.

    The channel logic is make_trace_view.py's, as in gpt-oss-20b/build_data.py.
    The role labels the prompt's system, developer and user messages, and the
    channel replaces it once it is named. Control tokens are their own label.
    """
    names, kinds = [], []
    current, pending = 'control', None
    for piece in pieces:
        if piece in ('<|start|>', '<|channel|>'):
            pending = piece
            kinds.append('control')
            continue
        if pending and piece not in CONTROL:
            current = piece.strip() or 'control'
            pending = None
            kinds.append('control')
            continue
        pending = None
        kinds.append('control' if piece in CONTROL else current)
    for kind in kinds:
        if kind not in names:
            names.append(kind)
    return names, [names.index(k) for k in kinds]


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode('ascii')


def quantize(values):
    """Log-scale uint8 codes, 0 for anything below 10**LOG_MIN."""
    codes = np.zeros(values.shape, dtype=np.uint8)
    positive = values >= 10 ** LOG_MIN
    scaled = (np.log10(values[positive].astype(np.float64)) - LOG_MIN) / (LOG_MAX - LOG_MIN) * STEPS
    codes[positive] = 1 + np.clip(np.rint(scaled), 0, STEPS - 1).astype(np.uint8)
    return codes


def main():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    with gzip.open(DECODE / 'prompts.json.gz', 'rt') as stream:
        prompts = json.load(stream)
    trials = {}
    with gzip.open(DECODE / 'trials-005.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['policy']['name'] == 'full' and r['task'] in TASKS and r['seed'] in SEEDS:
                key = (r['task'], r['seed'])
                if key in trials:
                    raise ValueError(f'two full-cache trials for {key}')
                trials[key] = r
    regret = {}
    with gzip.open(STATIC / 'static.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['metric'] == 'regret':
                regret[(r['task'], r['seed'], r['policy'], r['budget'])] = r['regret']
    OUT.mkdir(exist_ok=True)
    total = 0
    means = {}
    for task in TASKS:
        for seed in SEEDS:
            trial = trials[(task, seed)]
            prompt = prompts[f"{task}|{trial['effort']}|{trial['date']}"]
            ids = list(prompt) + list(trial['output_ids'])
            directory = STATIC / 'static' / f'{task}-s{seed}'
            with np.load(directory / 'reference.npz') as reference:
                if not np.array_equal(reference['ids'], ids):
                    raise ValueError(f'{task} s{seed}: reference ids differ from the trial')
            pieces = stream_decode(ids, encoding)
            kind_names, kinds = channels(pieces)
            lost = {}
            for policy in POLICIES:
                lost[policy] = {}
                for budget in BUDGETS:
                    path = directory / f'lost-{policy}-{budget}.npz'
                    if not path.exists():
                        continue
                    with np.load(path) as z:
                        values = z['lost']
                    if values.shape != (len(ids),) or not np.all(np.isfinite(values)) or values.min() < 0:
                        raise ValueError(f'{path}: bad lost array')
                    mean = float(np.mean(values))
                    if not math.isclose(mean, regret[(task, seed, policy, budget)], rel_tol=1e-5, abs_tol=1e-7):
                        raise ValueError(f'{path}: mean differs from static.jsonl')
                    nonzero = np.flatnonzero(values)
                    start = int(nonzero[0]) if nonzero.size else len(ids)
                    lost[policy][str(budget)] = {'start': start, 'codes': b64(quantize(values[start:])),
                                                 'mean': round(mean, 6)}
                    means.setdefault((policy, budget), []).append(mean)
            data = {'schema': 'gpt-oss-20b-reference-lost-v1', 'task': task, 'seed': seed,
                    'prompt_tokens': len(prompt), 'generated': len(trial['output_ids']), 'length': len(ids),
                    'finish_reason': trial['finish_reason'].get('type'), 'quality': trial['quality'],
                    'prediction': trial['prediction'], 'answer': trial['answer'],
                    'scale': {'log10_min': LOG_MIN, 'log10_max': LOG_MAX, 'steps': STEPS},
                    'pieces': pieces, 'kind_names': kind_names, 'kinds': b64(np.array(kinds, dtype=np.uint8)),
                    'lost': lost}
            path = OUT / f'{task}-s{seed}.json'
            path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
            size = path.stat().st_size
            total += size
            print(f'{path.name}: {len(ids)} tokens, {size / 1e6:.2f} MB')
    print(f'total {total / 1e6:.1f} MB')
    for policy in POLICIES:
        print(policy, ' '.join(f'{b // 1024}K {np.mean(means[(policy, b)]):.4f}' for b in BUDGETS if (policy, b) in means))


if __name__ == '__main__':
    main()
