"""Small causal reference implementations. No dependency on a GPU runtime."""
import hashlib
import math
import random
import struct
from collections import defaultdict

INTERVAL = 2048
POLICIES = {"full", "random_2048", "random_reservoir_2048", "h2o_2048", "prefix_sliding_2048",
            "streamingllm_2048", "snapkv_prompt", "head_sum_2048", "head_max_sum_2048"}


def chunks(length, start=0, maximum=2048):
    """Sequential spans. Each span can compute causal queries in parallel."""
    if length < start or start < 0 or maximum < 1:
        raise ValueError("invalid chunk bounds")
    while start < length:
        end = min(length, start + maximum, (start // INTERVAL + 1) * INTERVAL)
        yield start, end
        start = end


def reservoir_score(seed, layer, head, position):
    """Fixed uniform score per (seed, layer, KV head, position), splitmix64 as in sglang_evict_runtime."""
    mask = (1 << 64) - 1
    x = (position * 0x9E3779B97F4A7C15 + layer * 0xBF58476D1CE4E5B9 + head * 0x94D049BB133111EB
         + seed * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return ((x ^ (x >> 31)) >> 11) / float(1 << 53)


class Policy:
    """Per-layer/KV-head token policy, original logical positions never renumbered.

    Observe scores from the unmodified reference, restricted to hypothetical
    residents. This is explicitly a reference-driven approximation.
    All ties retain newer logical positions. No evicted token can reenter.
    random_2048 draws fresh scores at every boundary, so survival decays with
    age. random_reservoir_2048 gives each position one fixed score for the
    (seed, stream) pair, so the kept middle is a uniform sample of history.
    """
    def __init__(self, name, budget, prefix=0, recent=0, seed=0,
                 prompt_length=0, window=32, pool=5, stream=(0, 0)):
        if name not in POLICIES or min(budget, prefix, recent, prompt_length) < 0:
            raise ValueError("invalid policy")
        if budget < 1 or prefix + recent > budget:
            raise ValueError("protected allocation exceeds retained budget")
        if name == "snapkv_prompt" and (window < 1 or pool < 1 or pool % 2 == 0
                                        or budget <= window or prompt_length < window):
            raise ValueError("SnapKV needs prompt >= window, odd pool, budget > window")
        self.name, self.budget, self.prefix, self.recent = name, budget, prefix, recent
        self.prompt_length, self.window, self.pool = prompt_length, window, pool
        self.rng = random.Random(seed)
        self.seed, self.stream = seed, stream
        self.resident = set()
        self.scores = {}
        self.next_position = 0
        self.peak = 0
        self.events = []

    def append(self, position):
        if position != self.next_position:
            raise ValueError("queries must be committed sequentially, no skipped positions")
        self.resident.add(position)
        self.scores[position] = 0.0
        self.next_position += 1
        self.peak = max(self.peak, len(self.resident))

    def observe(self, query, weights):
        """weights are normalized reference mass reduced by mean over GQA heads."""
        if query != self.next_position - 1:
            raise ValueError("observe before append or wrong query")
        active = self.name in {"h2o_2048", "head_sum_2048", "head_max_sum_2048"} or (
            self.name == "snapkv_prompt" and
            self.prompt_length - self.window <= query < self.prompt_length)
        if active:
            for key, value in weights.items():
                if key in self.resident:
                    self.scores[key] += value

    def finish(self):
        n = self.next_position
        if self.name == "full":
            return
        if self.name == "snapkv_prompt":
            if n != self.prompt_length:
                return
        elif n % INTERVAL:
            return
        if len(self.resident) <= self.budget:
            return
        if self.name == "snapkv_prompt":
            protected = {i for i in self.resident if i >= n - self.window}
            radius = self.pool // 2
            scores = {i: sum(self.scores.get(j, 0.) for j in range(i-radius, i+radius+1)
                              if j < n-self.window) / self.pool
                      for i in self.resident - protected}
        else:
            protected = {i for i in self.resident if i < self.prefix or i >= n-self.recent}
            scores = self.scores
        available = self.budget - len(protected)
        if available < 0:
            raise ValueError("actual protection exceeds retained budget")
        candidates = sorted(self.resident - protected)
        if self.name == "random_2048":
            scores = {i: self.rng.random() for i in candidates}
        elif self.name == "random_reservoir_2048":
            scores = {i: reservoir_score(self.seed, *self.stream, i) for i in candidates}
        elif self.name in {"prefix_sliding_2048", "streamingllm_2048"}:
            scores = {i: i for i in candidates}
        if self.name == "prefix_sliding_2048":
            available = 0  # Prefix Sliding retains exactly P union R.
        keep = protected | set(sorted(candidates, key=lambda i: (scores[i], i),
                                      reverse=True)[:available])
        self.events.append({"boundary": n, "before": len(self.resident), "after": len(keep)})
        self.resident = keep
        self.scores = {i: self.scores[i] for i in keep}


class ReferenceObserver:
    """Consume normalized probability tiles already computed by attention.

    API order: begin_query(t), tile(layer, head, positions, probabilities), end_query().
    All layers/heads must be observed exactly once over their complete native
    support (possibly many disjoint tiles). Stores O(resident KV) counters and
    one query's key scores, never a T x T attention matrix. The Python adapter
    is a correctness oracle, not a fused GPU implementation or timing proxy.
    """
    def __init__(self, layers, policies):
        # layers: {layer_id: [kv_head_for_query_head, ...]}.
        if not layers or any(not mapping for mapping in layers.values()):
            raise ValueError("instrumented attention layers and head mapping required")
        self.layers = {str(k): v for k, v in layers.items()}
        self.states = {}
        for label, specification in policies.items():
            for index, (layer, mapping) in enumerate(self.layers.items()):
                for kv in set(mapping):
                    args = dict(specification)
                    seed = args.pop("seed", 0)
                    if args["name"] == "random_reservoir_2048":
                        # The runtime's fixed scores: one stream per layer index and KV head.
                        self.states[label, layer, kv] = Policy(**args, seed=seed, stream=(index, kv))
                        continue
                    independent = int.from_bytes(hashlib.sha256(
                        f"{seed}:{layer}:{kv}".encode()).digest()[:8], "little")
                    self.states[label, layer, kv] = Policy(**args, seed=independent)
        self.labels = list(policies)
        self.t = -1
        self.active = False
        self.totals = defaultdict(float)
        self.count = 0

    def begin_query(self, t):
        if self.active or t != self.t + 1:
            raise ValueError("missing query completion or nonsequential reference")
        self.t, self.active = t, True
        self.norm = defaultdict(float)
        self.loss = defaultdict(float)
        self.updates = defaultdict(lambda: defaultdict(float))
        self.seen = defaultdict(set)
        for state in self.states.values():
            state.append(t)

    def tile(self, layer, head, positions, probabilities):
        layer = str(layer)
        if not self.active or layer not in self.layers or not 0 <= head < len(self.layers[layer]):
            raise ValueError("unknown query/layer/head")
        if len(positions) != len(probabilities):
            raise ValueError("tile shape mismatch")
        kv = self.layers[layer][head]
        group_size = self.layers[layer].count(kv)
        for pos, probability in zip(positions, probabilities):
            if not isinstance(pos, int) or not 0 <= pos <= self.t:
                raise ValueError("invalid position or future leakage")
            if pos in self.seen[layer, head]:
                raise ValueError("duplicate probability tile key")
            if not math.isfinite(probability) or probability < 0:
                raise ValueError("invalid probability")
            self.seen[layer, head].add(pos)
            self.norm[layer, head] += probability
            for label in self.labels:
                state = self.states[label, layer, kv]
                if pos not in state.resident:
                    self.loss[label, layer, head] += probability
                elif state.name in {"h2o_2048", "snapkv_prompt", "head_sum_2048", "head_max_sum_2048"}:
                    if state.name == "head_max_sum_2048":
                        self.updates[label, layer, kv][pos] = max(self.updates[label, layer, kv][pos], probability)
                    elif state.name == "head_sum_2048":
                        self.updates[label, layer, kv][pos] += probability
                    else:
                        self.updates[label, layer, kv][pos] += probability / group_size

    def sink(self, layer, head, probability):
        """Architectural non-token softmax sink, never a KV eviction candidate."""
        layer=str(layer)
        if not self.active or layer not in self.layers or not 0 <= head < len(self.layers[layer]):
            raise ValueError("unknown sink query/layer/head")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid sink probability")
        self.norm[layer,head] += probability

    def end_query(self):
        if not self.active:
            raise ValueError("no active query")
        for layer, mapping in self.layers.items():
            for head in range(len(mapping)):
                if not math.isclose(self.norm[layer, head], 1., abs_tol=1e-5):
                    raise ValueError("incomplete/non-normalized attention, maxima are not accepted")
        row = {"query": self.t, "loss": {}, "per_layer": {}}
        for label in self.labels:
            means = {layer: sum(self.loss[label, layer, h] for h in range(len(mapping))) / len(mapping)
                     for layer, mapping in self.layers.items()}
            row["per_layer"][label] = means
            row["loss"][label] = sum(means.values()) / len(means)
            self.totals[label] += row["loss"][label]
        # Evaluation precedes the decision, so query t never scores a new victim as lost.
        for key, state in self.states.items():
            state.observe(self.t, self.updates[key])
            state.finish()
        self.active = False
        self.count += 1
        return row

    def summary(self):
        if self.active or not self.count:
            raise ValueError("no complete query coverage")
        return {"interpretation": "post-hoc reference access loss, not causal quality prediction",
                "queries": self.count, "query_range": [0, self.t], "layers": self.layers,
                "averaging": "sum evicted mass, mean heads within layer, mean layers, mean queries",
                "regret": {k: v/self.count for k, v in self.totals.items()},
                "peak_entries": {f"{a}/{b}/{c}": s.peak for (a,b,c),s in self.states.items()}}

