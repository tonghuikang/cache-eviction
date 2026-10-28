"""Live decode under physical per-head eviction, one streaming session per run.

Each run opens a streaming session and decodes in calls that end on multiples
of 2048 tokens, so eviction state stays on the server between calls. A run
cannot outlive its client: SGLang discards a session's KV when its in-flight
call is aborted, so a restarted client closes the sessions left behind and
runs those again. A run that fails or is retracted is requeued from scratch,
up to three attempts. The same sampling_seed is sent on every call, and
SGLang keys its sampling noise on (seed, position), so a run split into calls
samples the same way as one long call.

A memory-aware scheduler keeps as many runs in flight as the KV pool can hold
without retraction. A run under budget B never holds more than
prompt + B + 2048 full-attention entries (plus one call of slack), so runs are
admitted, budgets interleaved, while the sum of those worst cases stays
under --pool-fraction of the server's pool and under its request limit.

--record TASK_FILE:BUDGET adds one run per policy at that budget, at the
lowest seed no other recorded run of that cell uses, that keeps the
retention record of every boundary, fetched over HTTP into
retention/. Everything comes back through the server's API, so the server can
be local or on Modal.

Results only ever grow. Each invocation appends the runs it asks for to
plan.jsonl, then runs every planned run that has no trial yet. --add-seeds N
plans N more seeds for every (task, policy) cell, taking the lowest seeds
that no planned or archived run of that cell uses, so rerunning the same
command adds another N seeds rather than replacing anything. --seeds plans
explicit seeds. With neither flag, an invocation only finishes the plan,
which is how to resume after a crash. Trials, errors and invocations are
appended, and --reuse names archived trials.jsonl.gz files that count as done.
"""
import argparse
import gzip
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from . import sglang_client as client
from .files import digest, load, save
from .sglang_tasks import answer_text, render, score

CONTEXT = 131072


def job_id(task, policy, seed):
    budget = policy.get("budget", 0)
    return f"{task}-{policy['name']}-{budget}-s{seed}"


def footprint(prompt, policy, max_new_tokens):
    if policy["name"] == "full":
        return prompt + max_new_tokens
    return prompt + min(policy["budget"], max_new_tokens) + 2 * client.INTERVAL


class Runner:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.output = Path(args.output)
        self.lock = threading.Lock()
        self.generated = 0

    def params(self, policy, seed, count, record, temperature):
        params = {"max_new_tokens": count, "temperature": temperature, "top_p": 1.0,
                  "sampling_seed": seed, "skip_special_tokens": False, "stop_token_ids": client.STOP_TOKEN_IDS}
        if policy["name"] != "full":
            params["custom_params"] = {"gstar_eviction": {**policy, "seed": seed, "record": record}}
        return params

    def run(self, job):
        task, prompt, policy, seed, record = job["task"], job["prompt"], job["policy"], job["seed"], job["record"]
        url, path = self.args.url, self.output / "state" / f"{job['id']}.json"
        state = self.fresh(job, path)
        finish = None
        while True:
            total = len(prompt) + len(state["output_ids"])
            count = min(client.INTERVAL - total % client.INTERVAL, job["max_new_tokens"] - len(state["output_ids"]),
                        CONTEXT - 1 - total)
            if count <= 0:
                finish = {"type": "length"}
                break
            began = time.time()
            try:
                out = client.generate(url, prompt if state["calls"] == 0 else [],
                                      self.params(policy, seed, count, record, job["temperature"]),
                                      session=state["session"])
                if out["meta_info"]["prompt_tokens"] != total:
                    raise RuntimeError("the session history differs from the client's tokens")
            except Exception:
                # The call may have run on the server without the client seeing its tokens,
                # so the session cannot be continued. The run is requeued from scratch.
                self.forget(path, state)
                raise
            ids = out["output_ids"]
            state["output_ids"] += ids
            state["calls"] += 1
            state["seconds"] += time.time() - began
            state["retractions"] = state.get("retractions", 0) + (out["meta_info"].get("num_retractions") or 0)
            with self.lock:
                self.generated += len(ids)
            finish = out["meta_info"]["finish_reason"]
            if finish["type"] != "length" or len(ids) < count:
                break
        key = f"session-{state['session']}"
        events = client.payload(url, f"{key}-events", wait=60) if policy["name"] != "full" else None
        retracted = client.payload(url, f"{key}-retracted", wait=0) is not None
        if record:
            total = len(prompt) + len(state["output_ids"])
            records = [r for n in range(client.INTERVAL, total + 1, client.INTERVAL)
                       if (r := client.payload(url, f"{key}-b{n}", wait=0, raw=True)) is not None]
            (self.output / "retention").mkdir(exist_ok=True)
            with gzip.open(self.output / "retention" / f"{job['id']}.json.gz", "wt") as stream:
                json.dump({"task": task["id"], "policy": policy, "seed": seed, "prompt_tokens": len(prompt),
                           "records": records}, stream)
        self.forget(path, state)
        if retracted or state.get("retractions") or (events and events.get("retracted")):
            # A retracted run recomputed its KV and lost its eviction state, so it does not count.
            raise RuntimeError("the run was retracted")
        text = self.tokenizer.decode(state["output_ids"])
        return {"task": task["id"], "policy": policy, "seed": seed, "record": record, "effort": job["effort"],
                "date": job["date"], "temperature": job["temperature"], "max_new_tokens": job["max_new_tokens"],
                **score(answer_text(text, finish["type"]), task), "answer": task["answer"],
                "prompt_tokens": len(prompt), "completion_tokens": len(state["output_ids"]),
                "finish_reason": finish, "seconds": state["seconds"], "calls": state["calls"],
                "session": state["session"], "retractions": state.get("retractions", 0),
                "retracted": retracted or bool(events and events.get("retracted")),
                "eviction_error": events.get("error") if events else None,
                "events": events.get("events") if events else None,
                "peak_full_entries": events.get("peak_full_entries") if events else None,
                "freed": events.get("freed") if events else None,
                "output_ids": state["output_ids"], "text": text}

    def forget(self, path, state):
        client.close_session(self.args.url, state["session"])
        path.unlink(missing_ok=True)

    def fresh(self, job, path):
        state = {"session": f"{job['id']}-{uuid.uuid4().hex[:8]}", "output_ids": [], "calls": 0, "seconds": 0.0}
        client.open_session(self.args.url, state["session"])
        # Only the session id is kept, so a later client can close a session this one leaves open.
        path.write_text(json.dumps({"session": state["session"]}))
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:30123")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--task", nargs="+", default=[], help="task JSON files")
    parser.add_argument("--policies", help="JSON list of {name, budget, prefix, recent}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[], help="explicit seeds to plan")
    parser.add_argument("--add-seeds", type=int, default=0, help="plan this many new seeds per (task, policy) cell")
    parser.add_argument("--record", action="append", default=[], metavar="TASK_FILE:BUDGET",
                        help="plan one more run of every policy at BUDGET on this task, with retention records")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--date", help="chat template date stamp, to match prompts rendered on another day")
    parser.add_argument("--max-new-tokens", type=int, default=130000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pool-fraction", type=float, default=0.95)
    parser.add_argument("--max-running", type=int, help="defaults to the server's max_running_requests")
    parser.add_argument("--reuse", nargs="*", default=[], help="trials.jsonl(.gz) files whose runs are skipped")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    output = Path(args.output)
    (output / "state").mkdir(parents=True, exist_ok=True)
    read = lambda f: (gzip.decompress(Path(f).read_bytes()) if str(f).endswith(".gz") else Path(f).read_bytes()).decode()
    rows = lambda f: [json.loads(line) for line in read(f).splitlines()] if Path(f).exists() else []
    key = lambda task, policy, seed, record: (task, json.dumps(policy, sort_keys=True), seed, record)
    done = {key(r["task"], r["policy"], r["seed"], r.get("record", False))
            for f in [*args.reuse, output / "trials.jsonl"] for r in rows(f)}
    planned = rows(output / "plan.jsonl")
    seen = {key(e["task"], e["policy"], e["seed"], e["record"]) for e in planned}
    additions = []

    def plan(file, policy, seed, record):
        task = load(file)
        k = key(task["id"], policy, seed, record)
        if k not in seen and k not in done:
            seen.add(k)
            additions.append({"task": task["id"], "task_file": file, "task_sha256": digest(task), "policy": policy,
                              "seed": seed, "record": record, "effort": args.effort, "date": args.date,
                              "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                              "planned": time.time()})

    def lowest(task_id, policy, record):
        used = {k[2] for k in seen | done if k[:2] == (task_id, json.dumps(policy, sort_keys=True)) and k[3] == record}
        return next(seed for seed in range(1 << 30) if seed not in used)

    policies = json.loads(Path(args.policies).read_text()) if args.policies else []
    if (args.seeds or args.add_seeds) and not (args.task and policies) or args.record and not policies:
        parser.error("--seeds and --add-seeds need --task and --policies, and --record needs --policies")
    for file in args.task:
        task_id = load(file)["id"]
        for policy in policies:
            for seed in args.seeds:
                plan(file, policy, seed, False)
            for _ in range(args.add_seeds):
                plan(file, policy, lowest(task_id, policy, False), False)
    for spec in args.record:
        file, budget = spec.rsplit(":", 1)
        task_id = load(file)["id"]
        for policy in policies:
            if policy.get("budget") == int(budget):
                plan(file, policy, lowest(task_id, policy, True), True)
    with (output / "plan.jsonl").open("a") as stream:
        for entry in additions:
            stream.write(json.dumps(entry) + "\n")
    planned += additions
    tasks = {e["task"]: load(e["task_file"]) for e in planned}
    for e in planned:
        if digest(tasks[e["task"]]) != e["task_sha256"]:
            raise ValueError(f"{e['task_file']} changed since it was planned")
    # One rendering per (task, effort, date). prompts.json only grows, and a prompt never changes once recorded.
    prompts_path = output / "prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
    for name in sorted({f"{e['task']}|{e['effort']}|{e['date']}" for e in planned}):
        task_id, effort, date = name.split("|")
        ids = render(tasks[task_id], args.model_dir, effort, None if date == "None" else date)
        if prompts.setdefault(name, ids) != ids:
            raise ValueError(f"{name} renders differently from its recorded prompt")
    save(prompts_path, prompts)
    jobs = [{**e, "id": job_id(e["task"], e["policy"], e["seed"]) + ("-record" if e["record"] else ""),
             "task": tasks[e["task"]], "prompt": prompts[f"{e['task']}|{e['effort']}|{e['date']}"],
             "footprint": footprint(len(prompts[f"{e['task']}|{e['effort']}|{e['date']}"]), e["policy"],
                                    e["max_new_tokens"])}
            for e in planned if key(e["task"], e["policy"], e["seed"], e["record"]) not in done]
    # Interleave the budgets, so the runs in flight are an average mix and many
    # fit at once. Largest first would fill the pool with a few 32K runs.
    classes = {}
    for job in sorted(jobs, key=lambda j: (-j["footprint"], j["id"])):
        classes.setdefault(job["policy"].get("budget", 0), []).append(job)
    jobs = [job for rank in range(max(map(len, classes.values()), default=0))
            for _, members in sorted(classes.items(), key=lambda c: -c[0]) if rank < len(members)
            for job in [members[rank]]]
    info = client.server_info(args.url)
    pool = int(info["max_total_num_tokens"])
    limit = args.max_running or int(info.get("max_running_requests") or 1 << 30)
    capacity = int(args.pool_fraction * pool)
    with (output / "invocations.jsonl").open("a") as stream:
        stream.write(json.dumps({"started": time.time(), "argv": sys.argv[1:], "added": len(additions),
                                 "to_run": len(jobs), "reused": args.reuse, "pool_tokens": pool, "capacity": capacity,
                                 "max_running": limit, "hardware": os.environ.get("GSTAR_HARDWARE"),
                                 "url": args.url}) + "\n")
    print(f"planned {len(additions)} new runs, {len(jobs)} to run, pool {pool} tokens, capacity {capacity}", flush=True)
    # SGLang discards a session's KV when its in-flight call is aborted, so runs a
    # stopped client left behind cannot be resumed. Close their sessions and run them again.
    for path in (output / "state").glob("*.json"):
        client.close_session(args.url, json.loads(path.read_text())["session"])
        path.unlink()
    runner = Runner(args, tokenizer)
    condition = threading.Condition()
    pending = list(jobs)
    flight = {}
    finished = []

    def work(job):
        try:
            row = runner.run(job)
            with condition:
                with (output / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                (output / "state" / f"{job['id']}.json").unlink(missing_ok=True)
            print(job["id"], row["prediction"], row["quality"], row["completion_tokens"], round(row["seconds"]),
                  "retracted" if row["retracted"] else "", flush=True)
        except Exception as error:
            with condition, (output / "errors.jsonl").open("a") as stream:
                stream.write(json.dumps({"job": job["id"], "error": repr(error), "time": time.time()}) + "\n")
            print("ERROR", job["id"], repr(error), flush=True)
            with condition:
                job["attempts"] = job.get("attempts", 1) + 1
                if job["attempts"] <= 3:
                    del flight[job["id"]]
                    pending.append(job)  # starts again from scratch, its state is gone
                    condition.notify_all()
                    return
        with condition:
            del flight[job["id"]]
            finished.append(job["id"])
            condition.notify_all()

    started, last = time.time(), (time.time(), 0)
    with condition:
        while pending or flight:
            reserved = sum(j["footprint"] for j in flight.values())
            for job in list(pending):
                if len(flight) >= limit:
                    break
                # A run larger than the whole capacity still starts once the pool is empty.
                if reserved + job["footprint"] <= capacity or not flight:
                    pending.remove(job)
                    flight[job["id"]] = job
                    reserved += job["footprint"]
                    threading.Thread(target=work, args=(job,), daemon=True).start()
            condition.wait(timeout=60)
            now, generated = time.time(), runner.generated
            if now - last[0] >= 60:
                save(output / "progress.json", {
                    "time": now, "elapsed": now - started, "done": len(finished), "in_flight": len(flight),
                    "pending": len(pending), "reserved_tokens": reserved, "capacity": capacity,
                    "tokens_per_second": (generated - last[1]) / (now - last[0]), "generated": generated})
                last = (now, generated)
    print("done", len(finished), "runs in", round(time.time() - started), "s", flush=True)


if __name__ == "__main__":
    main()
