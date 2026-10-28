"""Per-layer, per-KV-head physical KV eviction for gpt-oss on SGLang 0.5.21.

Installed as sglang.srt.gstar_evict by cache_eviction.sglang_overlay and active
only when GSTAR_EVICT=1. Requests opt in through sampling_params.custom_params:

  gstar_eviction   {name, budget, prefix, recent, seed[, record]}  live decode
  gstar_replay     {name, budget, prefix, recent, seed}            static replay
  gstar_diagnostic {policies: "name:B:P:R,...", seed}              static regret

Every full-attention layer and KV head chooses its own survivors, as in the
Random Attention and H2O papers. Protected entries (prefix, recent window and
the sliding-window guard) keep their own slots, shared by every head. Each
head's chosen middle entries are gathered into one shared set of destination
slots, so all heads read one index list
and the leftover slots return to the allocator. The logical req_to_token row
keeps real slots from the protected recent region on, packs the other owned
slots at its start and zeroes the rest, so every owned slot appears once and
the stock release path frees it. RoPE positions and scheduler lengths stay
logical. Boundaries fall before queries 2048, 4096, ... in prefill and decode.

Live state is keyed by the KV row and the streaming session (or request), so
a session keeps its eviction state across 2048-token calls. Static requests
and recorded boundaries write per-token payloads, served over HTTP at
/gstar/payload/{key}, so a remote client never reads server files.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

INTERVAL = 2048
POLICIES = {"random_2048": 3, "random_reservoir_2048": 3, "prefix_sliding_2048": 3,
            "head_sum_2048": 1, "head_max_sum_2048": 2}
BLOCK_N = 128
MIN_BLOCK_KV = 32  # decode_attention._MIN_BLOCK_KV, which fixes the split layout
STATE = None
RETRACTED = set()


def enabled():
    return os.environ.get("GSTAR_EVICT") == "1"


# Payloads -----------------------------------------------------------------

def payload_dir():
    return Path(os.environ.get("GSTAR_PAYLOAD_DIR", "/tmp/gstar-payload"))


def array(values, dtype):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return {"dtype": dtype, "shape": list(data.shape), "data": base64.b64encode(data.tobytes()).decode()}


def write_payload(key, obj):
    directory = payload_dir()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{key}.tmp"
    temporary.write_bytes(gzip.compress(json.dumps({"format": "gstar-payload-1", "key": key, **obj}).encode()))
    temporary.rename(directory / f"{key}.json.gz")


def payload_response(key):
    from fastapi.responses import JSONResponse, Response

    if not key.replace("-", "").replace("_", "").isalnum():
        return JSONResponse({"error": "bad key"}, status_code=400)
    path = payload_dir() / f"{key}.json.gz"
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return Response(content=path.read_bytes(), media_type="application/gzip")


# State ----------------------------------------------------------------------

class State:
    def __init__(self, backend):
        self.pool = backend.token_to_kv_pool
        self.req_to_token = backend.req_to_token_pool.req_to_token
        rows, width = self.req_to_token.shape
        device = self.req_to_token.device
        self.full_layers = sorted(i for i, (_, swa) in self.pool.layers_mapping.items() if not swa)
        self.index = {layer: i for i, layer in enumerate(self.full_layers)}
        self.first_full = self.full_layers[0]
        keys = self.pool.get_key_buffer(self.first_full)
        self.kv_heads, self.dim = keys.shape[1], keys.shape[2]
        self.frontier = torch.zeros(rows, dtype=torch.int32, device=device)
        self.count = torch.zeros(rows, dtype=torch.int32, device=device)
        self.mode = torch.zeros(rows, dtype=torch.int32, device=device)
        self.table = torch.zeros((rows, width), dtype=torch.int32, device=device)
        # Head-sum and head-max scores per full layer, KV head and slot. A
        # relocation moves a score with its key and value.
        self.scores = torch.zeros((len(self.full_layers), self.kv_heads, keys.shape[0]),
                                  dtype=torch.float32, device=device)
        self.blocks = triton.cdiv(width, BLOCK_N)
        self.entries = {}
        self.verify = os.environ.get("GSTAR_EVICT_VERIFY") == "1"


def state(backend=None):
    global STATE
    if STATE is None and backend is not None:
        STATE = State(backend)
        logger.info("gstar_evict active: full layers %s, %d KV heads", STATE.full_layers, STATE.kv_heads)
    return STATE


def _reset(st, row):
    st.entries.pop(row, None)
    st.mode[row] = 0
    st.frontier[row] = 0
    st.count[row] = 0


def freed(row):
    """Slots this row's evictions returned to the allocator, for SGLang's session accounting."""
    entry = STATE.entries.get(row) if STATE is not None else None
    return entry.get("freed", 0) if entry is not None else 0


# Attention metadata -------------------------------------------------------

def fill(backend, bs, seq_lens, req_pool_indices, kv_indices):
    """Full-layer CSR indices: shared resident slots below the frontier, then the tail."""
    from sglang.kernels.ops.kvcache.kv_indices import create_flashinfer_kv_indices_triton

    st = state(backend)
    rows = req_pool_indices[:bs]
    lengths = seq_lens[:bs]
    frontier = st.frontier[rows]
    count = st.count[rows]
    kv_indptr = backend.kv_indptr[: bs + 1]
    kv_indptr[1:] = torch.cumsum(lengths - frontier + count, dim=0)
    create_flashinfer_kv_indices_triton[(bs,)](
        st.table, rows, count, kv_indptr, None, kv_indices, st.table.stride(0))
    create_flashinfer_kv_indices_triton[(bs,)](
        st.req_to_token, rows, lengths - frontier, kv_indptr[:bs] + count, frontier,
        kv_indices, st.req_to_token.stride(0))
    return kv_indptr


@triton.jit
def _decode_scores(Q, K, kv_indptr, kv_indices, Lse, num_kv_splits, Sinks, Mode, Rows, Scores,
                   stride_qb, stride_qh, stride_kb, stride_kh, stride_lb, stride_lh, stride_sg, sm_scale,
                   GROUP: tl.constexpr, ROWS: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr,
                   MAX_SPLITS: tl.constexpr, BLOCK: tl.constexpr, MIN_KV: tl.constexpr):
    b = tl.program_id(0)
    block = tl.program_id(1)
    mode = tl.load(Mode + tl.load(Rows + b))
    if mode != 1 and mode != 2:
        return
    start = tl.load(kv_indptr + b)
    length = tl.load(kv_indptr + b + 1) - start
    if block * BLOCK >= length:
        return
    splits = tl.load(num_kv_splits + b)
    per = tl.cdiv(tl.cdiv(length, splits), MIN_KV) * MIN_KV
    offs = block * BLOCK + tl.arange(0, BLOCK)
    valid = offs < length
    loc = tl.load(kv_indices + start + offs, mask=valid, other=0).to(tl.int64)
    dims = tl.arange(0, D)
    rows = tl.arange(0, ROWS)  # GROUP query heads padded for tl.dot
    live = rows < GROUP
    for g in tl.static_range(KVH):
        heads = g * GROUP + rows
        # Same normalizer as decode stage 2: valid split LSEs, then the sink.
        m = tl.load(Sinks + heads, mask=live, other=0.0).to(tl.float32)
        s = tl.zeros([ROWS], dtype=tl.float32) + 1.0
        for i in tl.static_range(MAX_SPLITS):
            if i * per < length:
                lse = tl.load(Lse + b * stride_lb + heads * stride_lh + i, mask=live, other=0.0)
                n = tl.maximum(m, lse)
                s = s * tl.exp(m - n) + tl.exp(lse - n)
                m = n
        norm = m + tl.log(s)
        q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + dims[None, :], mask=live[:, None], other=0.0)
        k = tl.load(K + loc[None, :] * stride_kb + g * stride_kh + dims[:, None],
                    mask=valid[None, :], other=0.0)
        qk = tl.dot(q.to(K.dtype.element_ty), k) * sm_scale
        p = tl.where(live[:, None], tl.exp(qk - norm[:, None]), 0.0)
        # Each KV head keeps its own score: the sum or the max over its query heads.
        if mode == 1:
            total = tl.sum(p, axis=0)
        else:
            total = tl.max(p, axis=0)
        old = tl.load(Scores + g * stride_sg + loc, mask=valid, other=0.0)
        tl.store(Scores + g * stride_sg + loc, old + total, mask=valid)


def after_decode(backend, q, layer, forward_batch, sinks, out=None):
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        return
    st = state(backend)
    meta = backend.forward_metadata
    heads, kv_heads, dim = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim
    queries = q.view(-1, heads, dim)
    bs = queries.shape[0]
    if layer.layer_id == st.first_full:
        st.scores.index_fill_(2, forward_batch.out_cache_loc.long(), 0.0)
        if st.verify:
            _verify_decode(st, queries, out, layer, forward_batch, sinks)
    keys = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
    scores = st.scores[st.index[layer.layer_id]]
    lse = meta.attn_lse
    _decode_scores[(bs, st.blocks)](
        queries, keys, meta.kv_indptr, meta.kv_indices, lse, meta.num_kv_splits, sinks,
        st.mode, forward_batch.req_pool_indices, scores,
        queries.stride(0), queries.stride(1), keys.stride(0), keys.stride(1),
        lse.stride(0), lse.stride(1), scores.stride(0), layer.scaling,
        GROUP=heads // kv_heads, ROWS=max(16, triton.next_power_of_2(heads // kv_heads)), KVH=kv_heads, D=dim,
        MAX_SPLITS=backend.max_kv_splits, BLOCK=BLOCK_N, MIN_KV=MIN_BLOCK_KV)


def _shadow(st, entry, length):
    """First full layer keys and values by logical position, for checks that need the unevicted cache."""
    current = entry["shadow_k"].shape[0] if "shadow_k" in entry else 0
    if current < length:
        grown = max(length, 2 * current)
        for name in ("shadow_k", "shadow_v"):
            new = torch.zeros((grown, st.kv_heads, st.dim), dtype=st.pool.get_key_buffer(st.first_full).dtype,
                              device=st.req_to_token.device)
            if name in entry:
                new[:current] = entry[name]
            entry[name] = new
    return entry["shadow_k"], entry["shadow_v"]


def _visible(entry, end, kv_heads, device):
    """Per-KV-head mask over logical positions [0, end): each head's survivors, then the tail."""
    visible = torch.ones((kv_heads, end), dtype=torch.bool, device=device)
    if entry["count"]:
        visible[:, :entry["frontier"]] = False
        visible.scatter_(1, entry["pos"][0], True)
    return visible


def _verify_decode(st, queries, out, layer, forward_batch, sinks):
    """Eager-only check: the first full layer's output equals attention over each head's claimed positions."""
    keys = st.pool.get_key_buffer(st.first_full)
    values = st.pool.get_value_buffer(st.first_full)
    loc = forward_batch.out_cache_loc.long()
    positions = forward_batch.positions.tolist()
    heads, kv_heads = layer.tp_q_head_num, layer.tp_k_head_num
    group = heads // kv_heads
    for i, row in enumerate(forward_batch.req_pool_indices.tolist()):
        entry = st.entries.get(row)
        if entry is None or entry["kind"] != "live" or "shadow_k" not in entry:
            continue
        end = positions[i] + 1
        shadow_k, shadow_v = _shadow(st, entry, end)
        shadow_k[end - 1], shadow_v[end - 1] = keys[loc[i]], values[loc[i]]
        visible = _visible(entry, end, kv_heads, keys.device)
        q = queries[i].float().view(kv_heads, group, -1)
        logits = torch.einsum("gjd,kgd->gjk", q, shadow_k[:end].float()) * layer.scaling
        logits.masked_fill_(~visible[:, None, :], -torch.inf)
        sink = sinks.float().view(kv_heads, group, 1)
        p = torch.softmax(torch.cat((logits, sink), -1), -1)[..., :-1]
        expected = torch.einsum("gjk,kgd->gjd", p, shadow_v[:end].float()).reshape(heads, -1)
        error = float((out.view(-1, heads, expected.shape[-1])[i].float() - expected).abs().max())
        entry["verify_max_error"] = max(entry.get("verify_max_error", 0.0), error)
        entry["verify_steps"] = entry.get("verify_steps", 0) + 1


def after_extend(backend, q, layer, forward_batch, sinks):
    """Exact prefill probabilities for head scores, static payloads and the regret diagnostic."""
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        return
    st = state(backend)
    entries = [st.entries.get(r) for r in forward_batch.req_pool_indices.tolist()]
    if not any(entries):
        return
    meta = backend.forward_metadata
    heads = layer.tp_q_head_num
    loc = forward_batch.out_cache_loc.long()
    keys = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
    values = backend.token_to_kv_pool.get_value_buffer(layer.layer_id)
    queries = q.view(-1, heads, layer.qk_head_dim)
    qo, kv = meta.qo_indptr.tolist(), meta.kv_indptr.tolist()
    index = st.index[layer.layer_id]
    if layer.layer_id == st.first_full:
        st.scores.index_fill_(2, loc, 0.0)
    for i, entry in enumerate(entries):
        if entry is None or entry["error"]:
            continue
        start = forward_batch.extend_prefix_lens_cpu[i]
        new = loc[qo[i]:qo[i + 1]]
        chunk = queries[qo[i]:qo[i + 1]]
        if layer.layer_id == st.first_full and (entry["kind"] == "replay" or st.verify and entry["kind"] == "live"):
            shadow_k, shadow_v = _shadow(st, entry, start + new.numel())
            shadow_k[start:start + new.numel()] = keys[new]
            shadow_v[start:start + new.numel()] = values[new]
            if entry["kind"] == "replay":
                _replay_layer_difference(st, entry, chunk, layer, sinks, start)
        prefix = meta.kv_indices[kv[i]:kv[i + 1]].long()
        if entry["kind"] == "diagnostic":
            _diagnose(st, entry, index, layer, chunk, keys, sinks, torch.cat((prefix, new)), start)
        elif entry["mode"] in (1, 2):
            _prefill_scores(st, index, entry["mode"], layer, chunk, keys, sinks, prefix, new)


def _probabilities(queries, keys, sinks, scaling, kv_heads, query_pos, key_pos, visible=None):
    """Exact fp32 probabilities [KV head, query head in group, query, key], sink in the normalizer."""
    count, heads, dim = queries.shape
    group = heads // kv_heads
    logits = torch.einsum("qgjd,kgd->gjqk", queries.float().view(count, kv_heads, group, dim), keys.float()) * scaling
    logits.masked_fill_(key_pos[None, None, None, :] > query_pos[None, None, :, None], -torch.inf)
    if visible is not None:
        logits.masked_fill_(~visible[:, None, None, :], -torch.inf)
    sink = sinks.float().view(kv_heads, group, 1, 1)
    return logits.sub_(torch.logaddexp(torch.logsumexp(logits, -1, keepdim=True), sink)).exp_()


def _prefill_scores(st, index, mode, layer, queries, keys, sinks, prefix, new):
    """Add one chunk's per-KV-head sum or max probabilities to the slot scores."""
    heads, kv_heads = layer.tp_q_head_num, layer.tp_k_head_num
    slots = torch.cat((prefix, new))
    k = keys[slots]
    total = slots.numel()
    key_pos = torch.arange(total, device=k.device)
    step = max(8, min(2048, (1 << 28) // (heads * total)))
    for lo in range(0, queries.shape[0], step):
        hi = min(lo + step, queries.shape[0])
        query_pos = prefix.numel() + torch.arange(lo, hi, device=k.device)
        p = _probabilities(queries[lo:hi], k, sinks, layer.scaling, kv_heads, query_pos, key_pos)
        reduced = (p.sum(1) if mode == 1 else p.amax(1)).sum(1)  # [KV head, key]
        st.scores[index].index_add_(1, slots, reduced)
        del p


# Per-head boundary selection ------------------------------------------------

_MASK = (1 << 64) - 1


def _signed(value):
    value &= _MASK
    return value - (1 << 64) if value >= 1 << 63 else value


def _shift(x, bits):
    """Logical right shift of wrapping int64 tensors."""
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def reservoir_scores(seed, pos, layers, heads):
    """A fixed uniform score per (seed, layer, KV head, position): splitmix64 of the four.

    Keeping the k highest fixed scores among every candidate seen so far keeps
    a uniform random subset of the history, so a token's chance of staying in
    the cache does not decay with its age. cache_eviction.core.reservoir_score is
    the same function in plain Python.
    """
    x = pos.long() * _signed(0x9E3779B97F4A7C15)
    x = x + layers.long()[:, None, None] * _signed(0xBF58476D1CE4E5B9)
    x = x + heads.long()[None, :, None] * _signed(0x94D049BB133111EB)
    x = x + _signed(seed * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019)
    x = (x ^ _shift(x, 30)) * _signed(0xBF58476D1CE4E5B9)
    x = (x ^ _shift(x, 27)) * _signed(0x94D049BB133111EB)
    x = x ^ _shift(x, 31)
    return _shift(x, 11).double() / float(1 << 53)


def choose(name, pos, n, budget, prefix, recent, score, cursor=None):
    """Survivors of one boundary for every (layer, KV head) row of pos [L, G, N].

    Protected positions (prefix, recent window, sliding-window guard) are kept
    by every head. The rest of the budget goes to the highest scores, with ties
    retaining newer positions, as in cache_eviction.core.Policy. Returns kept
    column indices [L, G, K] with the protected ones first, and their count.
    """
    protected = (pos < prefix) | (pos >= n - recent)
    if cursor is not None:
        protected |= pos >= cursor
    count = int(protected[0, 0].sum())
    if not bool((protected.sum(-1) == count).all()):
        raise RuntimeError("protected positions differ between heads")
    if count > budget:
        return None, count
    if name == "prefix_sliding_2048":
        score = torch.zeros(pos.shape, device=pos.device, dtype=torch.float64)
    key = torch.where(protected, torch.inf, score.double())
    newer = torch.argsort(pos, dim=-1, descending=True, stable=True)
    order = torch.gather(newer, -1, torch.argsort(torch.gather(key, -1, newer), dim=-1, descending=True, stable=True))
    return order[..., :count if name == "prefix_sliding_2048" else budget], count


def _scores(st, entry, name, pos, slots, n):
    device = pos.device
    seed = int(entry["cfg"].get("seed", 0))
    if name == "random_2048":
        # A fresh draw at every boundary, as in the Random Attention paper, so survival decays with age.
        generator = torch.Generator(device=device).manual_seed((seed * 1000003 + n) & 0x7FFFFFFF)
        return torch.rand(pos.shape, generator=generator, device=device, dtype=torch.float64)
    if name == "random_reservoir_2048":
        return reservoir_scores(seed, pos, torch.arange(pos.shape[0], device=device),
                                torch.arange(pos.shape[1], device=device))
    if name in ("head_sum_2048", "head_max_sum_2048"):
        return st.scores[:, :, slots].double()
    return torch.zeros(pos.shape, device=device, dtype=torch.float64)


def _boundary(st, row, entry, n, cursor=None, allocator=None):
    """Evict at logical length n: choose per head, relocate survivors, free the rest.

    With allocator=None (static replay) survivors are relocated but nothing is
    freed and req_to_token is untouched, so the request's release frees every
    original slot exactly once.
    """
    cfg = entry["cfg"]
    budget, prefix, recent = int(cfg["budget"]), int(cfg["prefix"]), int(cfg["recent"])
    layers, heads = len(st.full_layers), st.kv_heads
    device = st.req_to_token.device
    frontier, count = entry["frontier"], entry["count"]
    started = time.perf_counter()
    slots = torch.cat((st.table[row, :count].long(), st.req_to_token[row, frontier:n].long()))
    pos = torch.arange(frontier, n, device=device).expand(layers, heads, -1)
    if count:
        pos = torch.cat((entry["pos"], pos), -1)
    entry["last_boundary"] = n
    if slots.numel() <= budget:
        entry["events"].append({"boundary": n, "before": int(slots.numel()), "after": int(slots.numel())})
        return
    keep, protected = choose(cfg["name"], pos, n, budget, prefix, recent,
                             _scores(st, entry, cfg["name"], pos, slots, n), cursor)
    if keep is None:
        # The sliding-window guard still covers too much. Retry at the next boundary.
        entry["events"].append({"boundary": n, "deferred": True, "protected": protected})
        return
    protected_cols = keep[0, 0, :protected]
    protected_cols = protected_cols[torch.argsort(pos[0, 0, protected_cols])]
    middle = keep[..., protected:]
    middle = torch.gather(middle, -1, torch.argsort(torch.gather(pos, -1, middle), dim=-1))
    middle_pos = torch.gather(pos, -1, middle)
    is_protected = torch.zeros(slots.numel(), dtype=torch.bool, device=device)
    is_protected[protected_cols] = True
    candidates = slots[~is_protected]
    width = middle.shape[-1]
    targets, leftover = candidates[:width], candidates[width:]
    # Every head's j-th middle survivor moves to targets[j]. All of a layer's
    # sources are gathered before any target is written, and the leftover
    # candidate slots are disjoint from the targets, so no copy needs scratch KV.
    group = torch.arange(heads, device=device)[:, None]
    destination = targets.expand(heads, -1)
    for index, layer in enumerate(st.full_layers):
        keys, values = st.pool.get_key_buffer(layer), st.pool.get_value_buffer(layer)
        source = slots[middle[index]]  # [KV head, width]
        moved_k, moved_v, moved_s = keys[source, group], values[source, group], st.scores[index][group, source]
        keys[destination, group] = moved_k
        values[destination, group] = moved_v
        st.scores[index][group, destination] = moved_s
    victims = None
    if cfg.get("record"):
        kept = torch.zeros(pos.shape, dtype=torch.bool, device=device).scatter_(-1, keep, True)
        victims = [pos[l, g][~kept[l, g]].sort().values.cpu().numpy() for l in range(layers) for g in range(heads)]
    table = torch.cat((slots[protected_cols], targets))
    entry["pos"] = torch.cat((pos[..., protected_cols], middle_pos), -1)
    st.table[row, :table.numel()] = table.int()
    st.count[row] = table.numel()
    st.frontier[row] = n
    entry.update(frontier=n, count=int(table.numel()))
    torch.cuda.synchronize()
    ranked_at = time.perf_counter()
    if allocator is not None:
        mapping = getattr(allocator, "full_to_swa_index_mapping", None)
        if mapping is not None and bool((mapping[leftover] != 0).any()):
            raise RuntimeError("gstar_evict would free a slot that still holds a sliding-window entry")
        allocator.free_full(leftover)
        st.scores[:, :, leftover] = 0.0
        entry["freed"] = entry.get("freed", 0) + int(leftover.numel())
        # Positions from `true` on keep their real slots for the sliding-window
        # layers. The other owned slots are packed at the start of the row.
        true = min(n - recent, cursor if cursor is not None else n)
        real = st.req_to_token[row, true:n].long()
        packed = table[~torch.isin(table, real)]
        if packed.numel() > true:
            raise RuntimeError("packed ownership overlaps the sliding-window region")
        st.req_to_token[row, :true] = 0
        st.req_to_token[row, :packed.numel()] = packed.int()
    torch.cuda.synchronize()
    finished = time.perf_counter()
    event = {"boundary": n, "before": int(slots.numel()), "after": int(table.numel()), "freed": int(leftover.numel()),
             "swa_cursor": cursor, "ranking_ms": 1000 * (ranked_at - started),
             "compaction_ms": 1000 * (finished - ranked_at)}
    if st.verify and "shadow_k" in entry:
        event["relocation_ok"] = _relocation_ok(st, entry, table)
    entry["events"].append(event)
    if victims is not None:
        # Victims per (full layer, KV head), ascending and delta-coded within each head.
        lengths = [len(v) for v in victims]
        deltas = np.concatenate([np.diff(v, prepend=0) for v in victims]) if victims else np.zeros(0)
        write_payload(f"{entry['key']}-b{n}", {"kind": "retention", "boundary": n, "cfg": cfg,
                                               "layers": st.full_layers, "kv_heads": heads,
                                               "evicted_delta": array(deltas, "uint32"),
                                               "evicted_counts": array(lengths, "int32")})


def _relocation_ok(st, entry, table):
    """Each head's keys at the table slots equal the unevicted keys of the positions it claims."""
    keys = st.pool.get_key_buffer(st.first_full)
    pos = entry["pos"][0]
    group = torch.arange(st.kv_heads, device=pos.device)[:, None]
    return bool(torch.equal(keys[table.expand(st.kv_heads, -1), group], entry["shadow_k"][pos, group]))


# Static payloads ----------------------------------------------------------

def _replay_layer_difference(st, entry, queries, layer, sinks, start):
    """Per-token relative L2 error of the first full layer's attention output, evicted against full.

    The layer before it is a sliding-window layer, so its input does not depend
    on eviction. Both outputs use the unevicted keys, the evicted one masked to
    each head's survivors.
    """
    shadow_k, shadow_v = entry["shadow_k"], entry["shadow_v"]
    count = queries.shape[0]
    end = start + count
    heads, kv_heads = layer.tp_q_head_num, layer.tp_k_head_num
    visible = _visible(entry, end, kv_heads, queries.device)
    key_pos = torch.arange(end, device=queries.device)
    k, v = shadow_k[:end], shadow_v[:end].float()
    step = max(8, min(256, (1 << 27) // (heads * end)))
    errors = []
    for lo in range(0, count, step):
        hi = min(lo + step, count)
        query_pos = start + torch.arange(lo, hi, device=queries.device)
        full = _probabilities(queries[lo:hi], k, sinks, layer.scaling, kv_heads, query_pos, key_pos)
        a = torch.einsum("gjqk,kgd->qgjd", full, v).reshape(hi - lo, -1)
        del full
        evicted = _probabilities(queries[lo:hi], k, sinks, layer.scaling, kv_heads, query_pos, key_pos, visible)
        b = torch.einsum("gjqk,kgd->qgjd", evicted, v).reshape(hi - lo, -1)
        del evicted
        errors.append(((b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)).cpu())
    entry["layer_error"].append(torch.cat(errors))


def _diagnostic_entry(req, cfg, device, layers, heads):
    specs = []
    for item in str(cfg["policies"]).split(","):
        name, budget, prefix, recent = item.split(":")
        specs.append((name, int(budget), int(prefix), int(recent)))
    length = len(req.origin_input_ids)
    return {"kind": "diagnostic", "key": _identity(req), "mode": 0, "specs": specs, "error": None, "frontier": 0,
            "seed": int(cfg.get("seed", 0)),
            "resident": torch.ones((len(specs), layers, heads, length), dtype=torch.bool, device=device),
            "score_sum": torch.zeros((layers, heads, length), device=device),
            "score_max": torch.zeros((layers, heads, length), device=device),
            "lost": torch.zeros((len(specs), length), device=device), "layers_seen": set()}


def _diagnostic_boundary(entry, n):
    """Every spec's per-head boundary at n, scored by the unevicted reference."""
    layers, heads = entry["resident"].shape[1:3]
    device = entry["resident"].device
    for j, (name, budget, prefix, recent) in enumerate(entry["specs"]):
        resident = entry["resident"][j, :, :, :n]
        count = int(resident[0, 0].sum())
        if count <= budget:
            continue
        pos = resident.nonzero()[:, -1].view(layers, heads, count)
        if name == "random_2048":
            generator = torch.Generator(device=device).manual_seed((entry["seed"] * 1000003 + n * 31 + j) & 0x7FFFFFFF)
            score = torch.rand(pos.shape, generator=generator, device=device, dtype=torch.float64)
        elif name == "random_reservoir_2048":
            score = reservoir_scores(entry["seed"], pos, torch.arange(layers, device=device),
                                     torch.arange(heads, device=device))
        else:
            table = entry["score_max" if name == "head_max_sum_2048" else "score_sum"][..., :n]
            score = torch.gather(table, -1, pos).double()
        keep, _ = choose(name, pos, n, budget, prefix, recent, score)
        entry["resident"][j, :, :, :n] = torch.zeros_like(resident).scatter_(-1, torch.gather(pos, -1, keep), True)
    entry["frontier"] = n


def _diagnose(st, entry, index, layer, queries, keys, sinks, slots, start):
    """Exact reference probabilities for one full layer of one prefill chunk, and per-token lost mass."""
    count = queries.shape[0]
    end = start + count
    if start // INTERVAL != (end - 1) // INTERVAL:
        entry["error"] = "prefill chunk crosses a boundary; use --chunked-prefill-size 2048 and run alone"
        return
    if layer.layer_id == st.first_full and start % INTERVAL == 0 and start > entry["frontier"]:
        _diagnostic_boundary(entry, start)
    heads, kv_heads = layer.tp_q_head_num, layer.tp_k_head_num
    k = keys[slots]
    key_pos = torch.arange(end, device=k.device)
    evicted = (~entry["resident"][:, index, :, :end]).float()  # [spec, KV head, key]
    step = max(8, min(256, (1 << 28) // (heads * end)))
    layers = len(st.full_layers)
    for lo in range(0, count, step):
        hi = min(lo + step, count)
        query_pos = start + torch.arange(lo, hi, device=k.device)
        p = _probabilities(queries[lo:hi], k, sinks, layer.scaling, kv_heads, query_pos, key_pos)
        entry["score_sum"][index, :, :end] += p.sum(1).sum(1)
        entry["score_max"][index, :, :end] += p.amax(1).sum(1)
        grouped = p.sum(1)  # [KV head, query, key], summed over the KV head's query heads
        del p
        # Lost mass per query, averaged over query heads and full layers.
        entry["lost"][:, start + lo:start + hi] += torch.einsum("gqk,sgk->sq", grouped, evicted) / (heads * layers)
        del grouped
    entry["layers_seen"].add(layer.layer_id)


# Scheduler hooks ----------------------------------------------------------

def _identity(req):
    session = getattr(req, "session", None)
    if session is not None and getattr(session, "streaming", False):
        return f"session-{session.session_id}"
    return req.rid


def _live_entry(req, cfg, kind):
    entry = {"kind": kind, "key": _identity(req), "cfg": dict(cfg), "mode": POLICIES.get(cfg.get("name"), 0),
             "events": [], "error": None, "frontier": 0, "count": 0, "last_boundary": 0,
             "peak_full_entries": 0, "layer_error": []}
    try:
        budget, prefix, recent = int(cfg["budget"]), int(cfg["prefix"]), int(cfg["recent"])
        if entry["mode"] == 0 or min(prefix, recent) < 0 or prefix + recent > budget:
            raise ValueError("invalid gstar eviction settings")
        if cfg["name"] == "prefix_sliding_2048" and prefix + recent != budget:
            raise ValueError("Prefix Sliding requires B = P + R")
    except (KeyError, TypeError, ValueError) as error:
        entry["error"], entry["mode"] = str(error), 0
    return entry


def on_extend(batch):
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    st = STATE
    if st is None:
        return
    for req in batch.reqs:
        row = req.kv.req_pool_idx
        params = getattr(req.sampling_params, "custom_params", None) or {}
        entry = st.entries.get(row)
        if entry is not None and entry["key"] != _identity(req):
            _reset(st, row)  # the row now belongs to another request or session
            entry = None
        if entry is None:
            if "gstar_diagnostic" in params:
                entry = _diagnostic_entry(req, params["gstar_diagnostic"], st.req_to_token.device,
                                          len(st.full_layers), st.kv_heads)
            elif "gstar_replay" in params:
                entry = _live_entry(req, params["gstar_replay"], "replay")
            elif "gstar_eviction" in params:
                entry = _live_entry(req, params["gstar_eviction"], "live")
            else:
                continue
            st.entries[row] = entry
            st.mode[row] = entry["mode"]
            st.frontier[row], st.count[row] = 0, 0
        if entry["kind"] == "diagnostic" or entry["error"]:
            continue
        if getattr(req, "retraction_count", 0) or entry["key"] in RETRACTED:
            # A retracted request recomputes its KV from scratch, which evicted state cannot survive.
            RETRACTED.add(entry["key"])
            entry["error"] = "retracted"
            write_payload(f"{entry['key']}-retracted", {"kind": "retracted", "cfg": entry["cfg"]})
            st.mode[row] = 0
            continue
        start, length = req.extend_range.start, req.extend_range.length
        if start < entry["frontier"]:
            entry["error"] = "the session rewound below the eviction frontier"
        elif length and start // INTERVAL != (start + length - 1) // INTERVAL:
            entry["error"] = "prefill chunk crosses a boundary; use --chunked-prefill-size 2048"
        if entry["error"]:
            st.mode[row] = 0
            continue
        if start and start % INTERVAL == 0 and start > entry["last_boundary"]:
            torch.cuda.synchronize()  # the previous chunk's keys and scores are complete
            if entry["kind"] == "replay":
                _boundary(st, row, entry, start)
            else:
                _boundary(st, row, entry, start, req.kv.get_evicted_seqlen(ComponentType.SWA),
                          batch.tree_cache.token_to_kv_pool_allocator)


def on_decode(batch):
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    st = STATE
    if st is None or not st.entries:
        return
    lengths = batch.seq_lens_cpu.tolist()
    for i, req in enumerate(batch.reqs):
        row = req.kv.req_pool_idx
        entry = st.entries.get(row)
        if entry is None or entry["kind"] != "live" or entry["error"] or entry["key"] != _identity(req):
            continue
        n = lengths[i]
        entry["peak_full_entries"] = max(entry["peak_full_entries"], entry["count"] + n - entry["frontier"])
        if n % INTERVAL or n <= entry["last_boundary"]:
            continue
        torch.cuda.synchronize()  # the previous decode step wrote query n-1's key and scores
        _boundary(st, row, entry, n, req.kv.get_evicted_seqlen(ComponentType.SWA),
                  batch.tree_cache.token_to_kv_pool_allocator)


def _events(entry, req):
    keys = ("cfg", "error", "events", "peak_full_entries", "freed", "verify_max_error", "verify_steps")
    return {"kind": "events", **{k: entry.get(k) for k in keys},
            "length": len(req.origin_input_ids) + len(req.output_ids), "retracted": entry["key"] in RETRACTED}


def release(req):
    """At a request's KV release: write its payload, and drop its state unless its session keeps the row."""
    st = STATE
    if st is None or not req.kv.holds_kv:
        return
    row = req.kv.req_pool_idx
    entry = st.entries.get(row)
    if entry is None or entry["key"] != _identity(req):
        return
    if entry["kind"] == "diagnostic":
        lost = entry["lost"][:, :len(req.origin_input_ids)].cpu().numpy()
        write_payload(req.rid, {"kind": "diagnostic", "error": entry["error"], "specs": entry["specs"],
                                "layers": sorted(entry["layers_seen"]), "lost": array(lost, "float32")})
    elif entry["kind"] == "replay":
        error = torch.cat(entry["layer_error"]).numpy() if entry["layer_error"] else np.zeros(0)
        write_payload(req.rid, {"kind": "replay", "cfg": entry["cfg"], "error": entry["error"],
                                "events": entry["events"], "layer": st.first_full,
                                "layer_error": array(error, "float32")})
    else:
        session = entry["key"] != req.rid
        write_payload(f"{entry['key']}-events" if session else req.rid, _events(entry, req))
        if session:
            return  # the session keeps its row and eviction state for the next call
    _reset(st, row)


def filter_holes(segments):
    """Holes are zero and below the sliding-window cursor, so only full-side frees see them."""
    return [(kv[kv != 0], start) for kv, start in segments]
