"""Export the per-head gpt-oss-20b study as the report's schema-3 dataset.

Reads the decode trials written by sglang_decode (the eviction runs and their
full-cache baseline), the static.jsonl summaries written by sglang_static on
the full-cache reference traces, and sglang_timing output. Every source file
is recorded by SHA-256. Counts are kept as integers, never as rounded rates.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from .sglang_tasks import answer_text, score

TASKS = {"aime": ("GPT-OSS-20B: AIME 2026", "Accuracy (%)", "AIME 2026")}
IDS = ["full", "random_2048", "random_reservoir_2048", "prefix_sliding_2048", "head_sum_2048", "head_max_sum_2048"]
LABELS = {"random_2048": "Random", "random_reservoir_2048": "Reservoir", "prefix_sliding_2048": "Prefix Sliding",
          "head_sum_2048": "Head-sum", "head_max_sum_2048": "Head-maxsum"}
# The reference-trace table columns, in the order the report prints them.
METRIC_COLUMNS = [("logprob_difference", "Logprob difference"),
                  ("attention_weight_difference", "Attention weight difference"),
                  ("first_layer_output_difference", "First layer output difference")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(path):
    """Read a record file, transparently decompressing the runs/ copies."""
    data = Path(path).read_bytes()
    return (gzip.decompress(data) if str(path).endswith(".gz") else data).decode()


def quarter(policy):
    """k1 = k2 = k3 / 2: each protected end is a quarter of the budget."""
    q = policy["budget"] // 4
    recent = policy["budget"] - q if policy["name"] == "prefix_sliding_2048" else q
    return policy.get("prefix") == q and policy.get("recent") == recent


def trials(paths, tasks):
    """Rescore every trial of the given tasks with the single answer rule.

    Eviction trials count only under k1 = k2 = k3 / 2, and the retention
    representatives (record runs) are left out of the accuracy counts.
    """
    rows = [json.loads(line) for p in paths for line in text(p).splitlines()]
    rows = [r for r in rows if r["task"] in tasks and not r.get("record")
            and (r["policy"]["name"] == "full" or quarter(r["policy"]))]
    keys = [(r["task"], json.dumps(r["policy"], sort_keys=True), r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/policy/seed trials")
    if any(r.get("retractions") or r.get("retracted") or r.get("eviction_error") for r in rows):
        raise ValueError("a retracted or failed trial lost its eviction state; rerun it")
    for r in rows:
        r.update(score(answer_text(r["text"], r["finish_reason"]["type"]), tasks[r["task"]]))
    return rows


def counts(rows):
    return [sum(r["quality"] for r in rows), len(rows)]


def task_series(rows, budgets, label, ylabel, tasks):
    """One series per eviction policy against the full-cache reference. There is no selected policy."""
    reference = [r for r in rows if r["policy"]["name"] == "full"]
    if not reference:
        raise ValueError(f"{label}: no full-cache reference trials")
    mean = lambda group: sum(r["completion_tokens"] for r in group) / len(group)
    policies = {}
    for name in IDS[1:]:
        groups = [[r for r in rows if r["policy"]["name"] == name and r["policy"]["budget"] == b] for b in budgets]
        if not all(groups):
            raise ValueError(f"{label}: {LABELS[name]} has no trials at some budgets")
        policies[LABELS[name]] = {"measured": True, "counts": [counts(g) for g in groups],
                                  "tokens": [mean(g) for g in groups],
                                  "finish_length": [sum(r["finish_reason"]["type"] == "length" for r in g)
                                                    for g in groups]}
    return {"label": label, "ylabel": ylabel, "problems": sorted(tasks), "budget_tokens": budgets,
            "reference": counts(reference), "tokens_reference": mean(reference),
            "finish_length_reference": sum(r["finish_reason"]["type"] == "length" for r in reference),
            "policies": policies}


# Two-sided 95% t quantiles by degrees of freedom, so the exporter needs no scipy.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def static_metrics(path):
    """Means over the reference traces, keyed by metric, policy label and budget, plus the trace count.

    Each mean comes with the half-width of its 95% confidence interval, using
    the t distribution over the traces, since every trace is one sample.

    The attention weight column is the mean lost reference attention mass, the
    logprob column the mean absolute chosen-token logprob difference against an
    unevicted replay, and the first layer column the mean relative L2 error of
    the first full-attention layer's output.
    """
    rows = [json.loads(line) for line in text(path).splitlines()]
    traces = {(r["task"], r["seed"], r["trial_rid"]): r["tokens"] for r in rows}
    if any(r["logprob_abs_difference"] != 0 for r in rows if r["policy"] == "full_repeat"):
        raise ValueError("an unevicted replay was not reproducible, so the logprob column has a noise floor")
    cells = {}
    for r in rows:
        if r["policy"] not in LABELS:
            continue
        key = (LABELS[r["policy"]], r["budget"])
        if r["metric"] == "regret":
            cells.setdefault("attention_weight_difference", {}).setdefault(key, []).append(r["regret"])
        else:
            cells.setdefault("logprob_difference", {}).setdefault(key, []).append(r["logprob_abs_difference"])
            cells.setdefault("first_layer_output_difference", {}).setdefault(key, []).append(r["layer_relative_l2"])
    measured, ci95 = {}, {}
    for metric, values in cells.items():
        for (label, budget), v in values.items():
            if len(v) != len(traces):
                raise ValueError(f"{metric}: {label} at {budget} covers {len(v)} of {len(traces)} traces")
            mean_v = sum(v) / len(v)
            sd = (sum((x - mean_v) ** 2 for x in v) / (len(v) - 1)) ** 0.5
            measured.setdefault(metric, {}).setdefault(label, {})[budget] = mean_v
            ci95.setdefault(metric, {}).setdefault(label, {})[budget] = T975[len(v) - 1] * sd / len(v) ** 0.5
    return measured, ci95, len(traces), sum(traces.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aime", nargs="+", required=True, help="decode trials.jsonl(.gz) files")
    parser.add_argument("--aime-tasks", nargs="+", required=True, help="frozen AIME task files to include")
    parser.add_argument("--aime-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--static", required=True, help="static.jsonl(.gz) from sglang_static on the reference traces")
    parser.add_argument("--trace-label", required=True, help="how the report names the reference traces")
    parser.add_argument("--timing", required=True)
    parser.add_argument("--regret-budget", type=int, required=True)
    parser.add_argument("--reference-metric-budgets", type=int, nargs="+", default=[8192, 16384],
                        help="retained budgets to tabulate in the reference-trace metric tables")
    parser.add_argument("--settings", required=True, help="JSON with run settings")
    parser.add_argument("--findings", required=True, help="JSON with sweep and conclusion sentences")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [*args.aime, args.static, args.timing, args.settings, args.findings, *args.aime_tasks]
    measured, ci95, trace_count, trace_tokens = static_metrics(args.static)
    regret = measured["attention_weight_difference"]
    budgets = sorted({b for v in regret.values() for b in v})
    rows = [[0.0]] + [[regret[LABELS[name]][args.regret_budget]] for name in IDS[1:]]
    timing = json.loads(text(args.timing))
    resident = timing[0]["resident_tokens"]
    by_policy = {t["policy"]: t for t in timing if t["resident_tokens"] == resident}
    off = sum(t["off_ms"] for t in by_policy.values()) / len(by_policy)
    timing_rows = []
    for name in IDS:
        t = by_policy.get(name)
        timing_rows.append({"policy": name, "on_us": 1000 * (t["on_ms"] if t else off),
                            "delta_us": 1000 * t["paired_delta_mean_ms"] if t else 0.0,
                            "stdev_us": 1000 * t["paired_delta_stdev_ms"] if t else None})
    settings = json.loads(Path(args.settings).read_text())
    files = {t["id"]: t for t in (json.loads(Path(p).read_text()) for p in args.aime_tasks)}
    label, ylabel, scope = TASKS["aime"]
    reference = {"trace": args.trace_label, "budgets": args.reference_metric_budgets,
                 "columns": [{"key": key, "label": name} for key, name in METRIC_COLUMNS],
                 "cells": {LABELS[name]: {key: [{"value": measured[key][LABELS[name]][b], "ci95": ci95[key][LABELS[name]][b], "measured": True}
                                                for b in args.reference_metric_budgets]
                                          for key, _ in METRIC_COLUMNS} for name in IDS[1:]}}
    data = {"schema_version": 3, "cadence_tokens": 2048,
            "provenance": {"kind": "measured", "model": "openai/gpt-oss-20b",
                           "model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                           "hardware": settings["hardware"],
                           "runtime": "SGLang 0.5.21 with the gstar_evict overlay",
                           "overlay_runtime_sha256": sha(root / "cache_eviction/sglang_evict_runtime.py"),
                           "scope": "gpt-oss-20b on " + scope,
                           "sources": {str(Path(p).resolve().relative_to(root.parent)): sha(p) for p in sources}},
            "settings": settings,
            "regret": {"budget": args.regret_budget,
                       "traces": [{"label": args.trace_label, "rid": "mean", "traces": trace_count,
                                   "queries": trace_tokens}],
                       "rows": rows, "budgets": budgets,
                       "by_budget": {args.trace_label: {LABELS[n]: [regret[LABELS[n]][b] for b in budgets]
                                                        for n in IDS[1:]}}},
            "reference_metrics": reference,
            "timing": {"resident_tokens": resident, "rows": timing_rows},
            "findings": json.loads(Path(args.findings).read_text()),
            "tasks": {"aime": task_series(trials(args.aime, files), args.aime_budgets, label, ylabel, files)}}
    Path(args.output).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
