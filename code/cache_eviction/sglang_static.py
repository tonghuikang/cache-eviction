"""Static metrics on frozen full-cache reference traces, with every per-token payload.

Each trace (prompt plus a full-cache completion) is sent as a prompt with
max_new_tokens = 1, so the model is teacher-forced on the reference tokens:

  full, full_repeat    chosen-token logprobs without eviction (the repeat is the noise floor)
  policy at budget     gstar_replay: per-head eviction at every 2048-token prefill
                       boundary, mask-only, so chosen-token logprobs and the first
                       full-attention layer's per-token output error can be compared
  diagnostic           one gstar_diagnostic request: the per-token attention mass
                       each policy and budget would lose, scored by the unevicted model

The server returns the chosen-token logprobs in the response and writes the
per-token layer errors and lost mass as payloads, fetched over HTTP. Results
only ever grow: static/{task}-s{seed}/reference.npz holds the token ids and
both unevicted logprob arrays, replay-{policy}-{budget}.npz holds one
replay's logprobs and layer errors, lost-{policy}-{budget}.npz one
diagnostic item's lost mass (one value per token), and static.jsonl gets one
summary row per new item. A rerun computes only the items that are
missing, so adding a trace, policy or budget adds to the results. Requests run one at a
time, so prefill chunks end on eviction boundaries. The server needs
--chunked-prefill-size 2048 and, for the diagnostic, a sliding-window pool of
at least 2048 + 128 tokens per request (--max-running-requests 16 or more).
"""
import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

from . import sglang_client as client
from .files import load
from .sglang_tasks import render

POLICIES = ["random_2048", "random_reservoir_2048", "prefix_sliding_2048", "head_sum_2048", "head_max_sum_2048"]


def protection(name, budget):
    """k1 = k2 = k3 / 2: prefix and recent are each a quarter of the budget, and Prefix Sliding's window takes k2 + k3."""
    quarter = budget // 4
    return {"prefix": quarter, "recent": budget - quarter if name == "prefix_sliding_2048" else quarter}


def logprobs(url, ids, custom=None):
    params = {"max_new_tokens": 1, "temperature": 0.0}
    if custom:
        params["custom_params"] = custom
    out = client.generate(url, ids, params, return_logprob=True)
    values = [row[0] for row in out["meta_info"]["input_token_logprobs"]]
    if len(values) != len(ids):
        raise ValueError("input logprobs do not cover the trace")
    return out["meta_info"]["id"], np.array([np.nan if v is None else v for v in values], dtype=np.float32)


def trace_ids(task, trials, seed, model_dir):
    """Prompt plus output ids of one full-cache trial, from a decode dataset or an older sweep."""
    opener = gzip.open if trials.endswith(".gz") else open
    with opener(trials, "rt") as stream:
        row = next(r for r in map(json.loads, stream)
                   if r["task"] == task and r["policy"]["name"] == "full" and r["seed"] == int(seed))
    suffix = ".gz" if trials.endswith(".gz") else ""
    prompts = Path(trials).with_name(f"prompts.json{suffix}")
    run = Path(trials).with_name(f"run.json{suffix}")
    read = lambda path: json.loads(gzip.decompress(path.read_bytes()) if suffix else path.read_text())
    if prompts.exists():
        # sglang_decode records each prompt once per (task, effort, date).
        prompt = read(prompts)[f"{task}|{row['effort']}|{row['date']}"]
    elif "prompt_ids" in (meta := read(run)):
        prompt = meta["prompt_ids"][task]
    else:
        # Older runs did not record prompt ids. Re-render with the run's own date stamp.
        date = time.strftime("%Y-%m-%d", time.localtime(meta["started"]))
        prompt = render(load(f"tasks/aime2026/{task}.json"), model_dir, meta["effort"], date)
    if len(prompt) != row["prompt_tokens"]:
        raise ValueError(f"{task}: prompt length differs from the reference trial")
    return prompt + row["output_ids"], len(prompt), row.get("rid") or row["session"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:30123")
    parser.add_argument("--trace", nargs=3, action="append", required=True, metavar=("TASK_ID", "TRIALS", "SEED"),
                        help="full-cache trial to replay: task id, trials.jsonl(.gz) with run.json beside it, seed")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384], help="replay budgets")
    parser.add_argument("--diagnostic-budgets", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    for task, trials, seed in args.trace:
        directory = output / "static" / f"{task}-s{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        ids, prompt_tokens, trial_rid = trace_ids(task, trials, seed, args.model_dir)
        reference = directory / "reference.npz"
        if reference.exists():
            with np.load(reference) as stored:
                if not np.array_equal(stored["ids"], ids):
                    raise ValueError(f"{task}: the trace differs from its stored reference")
                full = stored["logprob_full"]
        else:
            _, full = logprobs(args.url, ids)
            _, repeat = logprobs(args.url, ids)
            write(reference, ids=np.array(ids, dtype=np.int32), logprob_full=full, logprob_full_repeat=repeat)
            append(output, {"task": task, "seed": int(seed), "trial_rid": trial_rid, "metric": "replay",
                            "policy": "full_repeat", "budget": None, "tokens": len(ids), "prompt_tokens": prompt_tokens,
                            "logprob_abs_difference": float(np.mean(np.abs(full[1:] - repeat[1:])))})
        lost = [(name, budget) for name in args.policies for budget in args.diagnostic_budgets
                if not (directory / f"lost-{name}-{budget}.npz").exists()]
        replays = [(name, budget) for name in args.policies for budget in args.budgets
                   if not (directory / f"replay-{name}-{budget}.npz").exists()]
        if not lost and not replays:
            continue
        started = time.time()
        common = {"task": task, "seed": int(seed), "trial_rid": trial_rid, "tokens": len(ids),
                  "prompt_tokens": prompt_tokens, "hardware": os.environ.get("GSTAR_HARDWARE"), "url": args.url}
        if lost:
            specs = [(name, budget, *protection(name, budget).values()) for name, budget in lost]
            rid, _ = logprobs(args.url, ids, {"gstar_diagnostic": {
                "seed": int(seed), "policies": ",".join(f"{n}:{b}:{p}:{r}" for n, b, p, r in specs)}})
            diagnostic = client.payload(args.url, rid)
            if diagnostic is None or diagnostic["error"]:
                raise RuntimeError(f"{task}: diagnostic payload {diagnostic and diagnostic['error']}")
            for j, (name, budget, prefix, recent) in enumerate(specs):
                values = diagnostic["lost"][j]
                check(task, f"lost {name} {budget}", values, ids)
                write(directory / f"lost-{name}-{budget}.npz", lost=values)
                append(output, {**common, "metric": "regret", "policy": name, "budget": budget, "prefix": prefix,
                                "recent": recent, "regret": float(np.mean(values)), "time": time.time()})
            print(task, "diagnostic", len(specs), "items", flush=True)
        for name, budget in replays:
            cfg = {"name": name, "budget": budget, "seed": int(seed), **protection(name, budget)}
            rid, evicted = logprobs(args.url, ids, {"gstar_replay": cfg})
            replay = client.payload(args.url, rid)
            if replay is None or replay["error"]:
                raise RuntimeError(f"{task} {name} {budget}: replay payload {replay and replay['error']}")
            check(task, f"logprob {name} {budget}", evicted, ids)
            check(task, f"layer error {name} {budget}", replay["layer_error"], ids)
            write(directory / f"replay-{name}-{budget}.npz", logprob=evicted, layer_error=replay["layer_error"])
            append(output, {**common, "metric": "replay", "policy": name, "budget": budget, "prefix": cfg["prefix"],
                            "recent": cfg["recent"], "logprob_abs_difference": float(np.mean(np.abs(full[1:] - evicted[1:]))),
                            "logprob_nll_increase": float(np.mean(full[1:] - evicted[1:])),
                            "layer_relative_l2": float(np.mean(replay["layer_error"])), "events": replay["events"],
                            "time": time.time()})
            print(task, name, budget, "replayed", flush=True)
        print(task, len(lost), "lost and", len(replays), "replay items in", round(time.time() - started), "s", flush=True)


def check(task, label, values, ids):
    if len(values) != len(ids):
        raise ValueError(f"{task}: {label} has {len(values)} values for {len(ids)} tokens")


def write(path, **arrays):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.rename(path)


def append(output, row):
    with (output / "static.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
