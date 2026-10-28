"""Paired CUDA timing of gpt-oss full-layer decode attention with gstar_evict counters.

One full-attention layer of gpt-oss-20b (64 query heads, 8 KV heads, head
dimension 64, bf16, sinks) over an identical resident cache. "off" is
SGLang's Triton decode attention alone. "on" adds the head-sum or head-max
score kernel, which keeps one score per KV head. Pairs alternate order, every sample is synchronized with CUDA
events, and counters are reset outside the timed region. Run with the GPU idle.
"""
import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch


def case(resident, batch, splits, seed=0):
    from sglang.srt import gstar_evict as ge
    torch.manual_seed(seed)
    heads, kv_heads, dim, slots = 64, 8, 64, resident * batch + 1
    keys = torch.randn((slots, kv_heads, dim), device="cuda").to(torch.bfloat16)
    values = torch.randn_like(keys)
    q = torch.randn((batch, heads, dim), device="cuda").to(torch.bfloat16)
    sinks = torch.randn(heads, device="cuda").to(torch.bfloat16)
    indices = torch.randperm(slots - 1, device="cuda") + 1
    indptr = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * resident
    num_splits = torch.full((batch,), splits, dtype=torch.int32, device="cuda")
    lse = torch.empty((batch, heads, splits), device="cuda")
    logits = torch.empty((batch, heads, splits, dim), device="cuda")
    out = torch.empty_like(q)
    pool = SimpleNamespace(layers_mapping={0: (0, False)}, get_key_buffer=lambda layer: keys)
    backend = SimpleNamespace(token_to_kv_pool=pool, max_kv_splits=splits,
                              req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(
                                  (batch + 1, resident + 1), dtype=torch.int32, device="cuda")),
                              kv_indptr=torch.zeros(batch + 2, dtype=torch.int32, device="cuda"),
                              forward_metadata=SimpleNamespace(attn_lse=lse, kv_indptr=indptr, kv_indices=indices,
                                                               num_kv_splits=num_splits))
    ge.STATE = None
    st = ge.state(backend)  # sizes the per-KV-head scores from the key buffer
    layer = SimpleNamespace(sliding_window_size=-1, tp_q_head_num=heads, tp_k_head_num=kv_heads,
                            qk_head_dim=dim, layer_id=0, scaling=dim ** -0.5)
    batch_info = SimpleNamespace(out_cache_loc=torch.zeros(batch, dtype=torch.int64, device="cuda"),
                                 req_pool_indices=torch.arange(1, batch + 1, device="cuda"))
    return SimpleNamespace(ge=ge, st=st, backend=backend, layer=layer, batch=batch_info, q=q, keys=keys,
                           values=values, out=out, indptr=indptr, indices=indices, logits=logits, lse=lse,
                           splits=num_splits, sinks=sinks, max_splits=splits)


def call(c, counters):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    decode_attention_fwd(c.q, c.keys, c.values, c.out, c.indptr, c.indices, c.logits, c.lse, c.splits,
                         c.max_splits, c.layer.scaling, 1.0, 1.0, sinks=c.sinks)
    if counters:
        c.ge.after_decode(c.backend, c.q.reshape(c.q.shape[0], -1), c.layer, c.batch, c.sinks)


def timed(c, counters):
    c.st.scores.zero_()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    call(c, counters)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def measure(policy, resident, batch, splits, repeats, warmup):
    c = case(resident, batch, splits)
    c.st.mode[:] = {"head_sum_2048": 1, "head_max_sum_2048": 2}[policy]
    for _ in range(warmup):
        timed(c, False), timed(c, True)
    pairs = []
    for i in range(repeats):
        order = ("off", "on") if i % 2 == 0 else ("on", "off")
        sample = {k: timed(c, k == "on") for k in order}
        pairs.append({"order": "-".join(order), "state_restored": True, "off_ms": sample["off"], "on_ms": sample["on"]})
    deltas = [p["on_ms"] - p["off_ms"] for p in pairs]
    off, on = statistics.mean(p["off_ms"] for p in pairs), statistics.mean(p["on_ms"] for p in pairs)
    return {"kind": "measurement", "policy": policy, "resident_tokens": resident, "batch": batch,
            "kv_splits": splits, "pairs": len(pairs), "warmup": warmup, "synchronization": "cuda_event_per_sample",
            "off_ms": off, "on_ms": on, "extra_ms": on - off, "paired_delta_mean_ms": statistics.mean(deltas),
            "paired_delta_stdev_ms": statistics.stdev(deltas), "overhead_percent": 100 * (on - off) / off,
            "hardware": torch.cuda.get_device_name(), "torch": torch.__version__,
            "scope": "one full-attention layer decode step: Triton decode attention, plus score kernel when on",
            "samples": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [measure(p, r, args.batch, args.splits, args.repeats, args.warmup)
            for r in args.resident for p in ("head_sum_2048", "head_max_sum_2048")]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(r["policy"], r["resident_tokens"], f"off {1000*r['off_ms']:.1f} us", f"on {1000*r['on_ms']:.1f} us",
              f"+{r['overhead_percent']:.1f}%")


if __name__ == "__main__":
    main()
