"""GPU checks for the gstar_evict SGLang overlay. Skipped without SGLang and CUDA."""
import random
import unittest
from types import SimpleNamespace

try:
    import torch
    from sglang.srt import gstar_evict as ge
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    READY = torch.cuda.is_available()
except ImportError:
    READY = False

from cache_eviction.core import Policy, reservoir_score

FULL = (1, 3)


def backend(rows=4, width=8192, slots=20000, splits=8, kv_heads=8, dim=64):
    """Two full layers (1, 3) and two sliding-window layers, each with distinct random keys and values."""
    keys = {l: torch.randn((slots, kv_heads, dim), device="cuda").to(torch.bfloat16) for l in range(4)}
    values = {l: torch.randn((slots, kv_heads, dim), device="cuda").to(torch.bfloat16) for l in range(4)}
    pool = SimpleNamespace(layers_mapping={l: (l, l not in FULL) for l in range(4)},
                           get_key_buffer=keys.__getitem__, get_value_buffer=values.__getitem__)
    return SimpleNamespace(token_to_kv_pool=pool, max_kv_splits=splits,
                           req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(
                               (rows, width), dtype=torch.int32, device="cuda")),
                           kv_indptr=torch.zeros(rows + 1, dtype=torch.int32, device="cuda"))


def allocator(slots=20000):
    freed = []
    return freed, SimpleNamespace(free_full=lambda s: freed.append(s.clone()),
                                  full_to_swa_index_mapping=torch.zeros(slots, dtype=torch.int64, device="cuda"))


def entry(name, budget, prefix, recent, seed, kind="live", record=False):
    cfg = {"name": name, "budget": budget, "prefix": prefix, "recent": recent, "seed": seed, "record": record}
    return ge._live_entry(SimpleNamespace(rid="r", session=None), cfg, kind)


@unittest.skipUnless(READY, "needs SGLang and CUDA")
class Overlay(unittest.TestCase):
    def setUp(self):
        ge.STATE = None
        torch.manual_seed(0)

    def test_decode_scores_are_per_kv_head(self):
        b = backend()
        st = ge.state(b)
        heads, kv_heads, dim, lengths = 64, 8, 64, [37, 1500, 3001]
        keys, values = b.token_to_kv_pool.get_key_buffer(1), b.token_to_kv_pool.get_value_buffer(1)
        q = torch.randn((3, heads, dim), device="cuda").to(torch.bfloat16)
        sinks = torch.randn(heads, device="cuda").to(torch.bfloat16)
        perm = torch.randperm(19999, device="cuda")[:sum(lengths)] + 1
        indptr = torch.tensor([0, 37, 1537, 4538], dtype=torch.int32, device="cuda")
        splits = torch.tensor([1, 5, 8], dtype=torch.int32, device="cuda")
        lse = torch.empty((3, heads, 8), device="cuda")
        decode_attention_fwd(q, keys, values, torch.empty_like(q), indptr, perm,
                             torch.empty((3, heads, 8, dim), device="cuda"), lse, splits, 8,
                             0.125, 1.0, 1.0, sinks=sinks)
        rows = torch.tensor([1, 2, 3], device="cuda")
        b.forward_metadata = SimpleNamespace(attn_lse=lse, kv_indptr=indptr, kv_indices=perm, num_kv_splits=splits)
        layer = SimpleNamespace(sliding_window_size=-1, tp_q_head_num=heads, tp_k_head_num=kv_heads,
                                qk_head_dim=dim, layer_id=1, scaling=0.125)
        for mode in (1, 2):
            st.mode[:] = mode
            st.scores.zero_()
            ge.after_decode(b, q.reshape(3, -1), layer, SimpleNamespace(
                out_cache_loc=torch.zeros(3, dtype=torch.int64, device="cuda"), req_pool_indices=rows), sinks)
            expected = torch.zeros_like(st.scores)
            for i in range(3):
                slots = perm[indptr[i]:indptr[i + 1]].long()
                k = keys[slots].float().repeat_interleave(heads // kv_heads, 1)
                logits = torch.einsum("hd,khd->hk", q[i].float(), k) * 0.125
                p = torch.softmax(torch.cat((logits, sinks.float()[:, None]), -1), -1)[:, :-1].view(kv_heads, 8, -1)
                expected[0][:, slots] += p.sum(1) if mode == 1 else p.amax(1)
            torch.testing.assert_close(st.scores, expected, rtol=2e-3, atol=1e-6)

    def test_fill_reads_compact_residents_then_tail(self):
        b = backend()
        st = ge.state(b)
        st.req_to_token[1, :10] = torch.arange(101, 111, dtype=torch.int32)
        st.req_to_token[2, :6] = torch.arange(201, 207, dtype=torch.int32)
        st.table[1, :4] = torch.tensor([101, 102, 106, 107], dtype=torch.int32)
        st.count[1], st.frontier[1] = 4, 7
        out = torch.full((32,), -1, dtype=torch.int64, device="cuda")
        indptr = ge.fill(b, 2, torch.tensor([10, 6], device="cuda"), torch.tensor([1, 2], device="cuda"), out)
        self.assertEqual(indptr.tolist(), [0, 7, 13])
        self.assertEqual(out[:13].tolist(), [101, 102, 106, 107, 108, 109, 110, 201, 202, 203, 204, 205, 206])

    def check_boundaries(self, name, kind):
        """Two boundaries per head against core.Policy, with relocation, ownership and freeing checks."""
        b = backend()
        st = ge.state(b)
        n, budget = 4096 + 300, 1024
        prefix = budget // 4
        recent = budget - prefix if name == "prefix_sliding_2048" else budget // 4
        slot_of = torch.randperm(19000, device="cuda")[:n] + 1  # scattered original slots
        st.req_to_token[1, :n] = slot_of.int()
        original = {l: b.token_to_kv_pool.get_key_buffer(l)[slot_of].clone() for l in FULL}  # by position
        original_v = {l: b.token_to_kv_pool.get_value_buffer(l)[slot_of].clone() for l in FULL}
        score = torch.rand((2, 8, n), device="cuda")
        st.scores[:, :, slot_of] = score
        e = entry(name, budget, prefix, recent, seed=4, kind=kind)
        freed, alloc = allocator()
        policies = {(l, g): Policy(name, budget, prefix, recent, seed=4, stream=(l, g)) for l in range(2) for g in range(8)}
        for boundary in (2048, 4096):
            cursor = boundary - 200
            for p in policies.values():
                for t in range(p.next_position, boundary):
                    p.append(t)
            for (l, g), p in policies.items():
                p.scores = {t: float(score[l, g, t]) for t in p.resident}
                p.finish()
            if kind == "replay":
                ge._boundary(st, 1, e, boundary)
            else:
                ge._boundary(st, 1, e, boundary, cursor, alloc)
            table = st.table[1, :e["count"]].long()
            self.assertEqual(e["count"], budget if name != "prefix_sliding_2048" else prefix + recent)
            for index, layer in enumerate(FULL):
                keys = b.token_to_kv_pool.get_key_buffer(layer)
                values = b.token_to_kv_pool.get_value_buffer(layer)
                for g in range(8):
                    held = e["pos"][index, g]
                    if name != "random_2048":
                        self.assertEqual(set(held.tolist()), policies[index, g].resident, (name, boundary, index, g))
                    # Relocated keys, values and scores belong to the positions the head claims.
                    self.assertTrue(torch.equal(keys[table, g], original[layer][held, g]))
                    self.assertTrue(torch.equal(values[table, g], original_v[layer][held, g]))
                    torch.testing.assert_close(st.scores[index, g, table], score[index, g, held])
            self.assertEqual(int(st.frontier[1]), boundary)
        if kind == "replay":
            self.assertTrue(torch.equal(st.req_to_token[1, :n], slot_of.int()))
            self.assertEqual(freed, [])
            return
        # Every owned slot appears once in the row, and the freed slots are exactly the rest.
        row = st.req_to_token[1, :n].long()
        owned = row[row != 0]
        true = min(4096 - recent, 4096 - 200)
        self.assertTrue(torch.equal(row[true:], slot_of[true:]))
        self.assertEqual(owned.numel(), torch.unique(owned).numel())
        self.assertEqual(set(owned.tolist()), set(st.table[1, :e["count"]].tolist()) | set(slot_of[4096:].tolist()))
        released = torch.cat(freed).tolist()
        self.assertEqual(len(released), len(set(released)))
        self.assertEqual(set(released) | set(owned.tolist()), set(slot_of.tolist()))
        self.assertEqual(e["freed"], len(released))

    def test_live_boundaries_relocate_and_free(self):
        for name in ("head_sum_2048", "random_reservoir_2048", "prefix_sliding_2048", "random_2048"):
            ge.STATE = None
            self.check_boundaries(name, "live")

    def test_replay_boundaries_relocate_without_freeing(self):
        for name in ("head_max_sum_2048", "random_reservoir_2048"):
            ge.STATE = None
            self.check_boundaries(name, "replay")

    def test_random_redraws_but_reservoir_is_fixed(self):
        pos = torch.arange(1000, device="cuda").expand(2, 8, -1)
        layers, heads = torch.arange(2, device="cuda"), torch.arange(8, device="cuda")
        fixed = ge.reservoir_scores(7, pos, layers, heads)
        self.assertAlmostEqual(float(fixed[1, 5, 321]), reservoir_score(7, 1, 5, 321), places=15)
        self.assertGreater(float((fixed[0, 0] - fixed[1, 3]).abs().mean()), 0.2)  # independent heads
        st = SimpleNamespace(scores=None)
        a = ge._scores(st, {"cfg": {"seed": 7}}, "random_2048", pos, None, 2048)
        c = ge._scores(st, {"cfg": {"seed": 7}}, "random_2048", pos, None, 4096)
        self.assertGreater(float((a - c).abs().mean()), 0.2)

    def test_reservoir_survival_does_not_decay_with_age(self):
        # With no protection, after many boundaries a reservoir keeps old and new history equally.
        budget, total = 2048, 2048 * 16
        layers, heads = torch.arange(1, device="cuda"), torch.arange(1, device="cuda")
        fractions = {}
        for name in ("random_reservoir_2048", "random_2048"):
            resident = torch.arange(0, device="cuda", dtype=torch.long).view(1, 1, 0)
            for n in range(2048, total + 1, 2048):
                pos = torch.cat((resident, torch.arange(n - 2048, n, device="cuda").view(1, 1, -1)), -1)
                if name == "random_2048":
                    g = torch.Generator(device="cuda").manual_seed(n)
                    score = torch.rand(pos.shape, generator=g, device="cuda", dtype=torch.float64)
                else:
                    score = ge.reservoir_scores(3, pos, layers, heads)
                keep, _ = ge.choose(name, pos, n, budget, 0, 0, score)
                resident = torch.gather(pos, -1, keep)
            fractions[name] = float((resident < total // 2).float().mean())
        self.assertAlmostEqual(fractions["random_reservoir_2048"], 0.5, delta=0.05)
        self.assertLess(fractions["random_2048"], 0.1)

    def test_diagnostic_lost_mass_matches_reference_observer_with_gqa(self):
        from cache_eviction.core import ReferenceObserver
        heads, kv_heads, length, dim = 8, 2, 2200, 16
        keys = (torch.randn((length + 1, kv_heads, dim), device="cuda") * 2).to(torch.bfloat16)
        queries = (torch.randn((length, heads, dim), device="cuda") * 2).to(torch.bfloat16)
        sinks = torch.randn(heads, device="cuda")
        st = SimpleNamespace(first_full=1, full_layers=[1])
        layer = SimpleNamespace(layer_id=1, tp_q_head_num=heads, tp_k_head_num=kv_heads, scaling=0.25)
        specs = {"head_sum_2048": (900, 16, 64), "head_max_sum_2048": (900, 16, 64),
                 "prefix_sliding_2048": (900, 16, 884), "random_reservoir_2048": (900, 16, 64)}
        req = SimpleNamespace(rid="r", session=None, origin_input_ids=[0] * length)
        e = ge._diagnostic_entry(req, {"seed": 5, "policies": ",".join(
            f"{k}:{b}:{p}:{r}" for k, (b, p, r) in specs.items())}, "cuda", 1, kv_heads)
        for lo, hi in ((0, 2048), (2048, length)):
            ge._diagnose(st, e, 0, layer, queries[lo:hi], keys, sinks, torch.arange(1, hi + 1, device="cuda"), lo)
        observer = ReferenceObserver({"1": [h // 4 for h in range(heads)]}, {
            k: {"name": k, "budget": b, "prefix": p, "recent": r, "seed": 5} for k, (b, p, r) in specs.items()})
        k = keys[1:].float().repeat_interleave(4, 1)
        expected = []
        for t in range(length):
            logits = torch.einsum("hd,khd->hk", queries[t].float(), k[:t + 1]) * 0.25
            p = torch.softmax(torch.cat((logits, sinks[:, None]), -1), -1).cpu()
            observer.begin_query(t)
            for h in range(heads):
                observer.tile("1", h, list(range(t + 1)), p[h, :-1].tolist())
                observer.sink("1", h, float(p[h, -1]))
            expected.append(observer.end_query()["loss"])
        for j, name in enumerate(specs):
            want = torch.tensor([row[name] for row in expected])
            self.assertGreater(float(want.sum()), 0)
            torch.testing.assert_close(e["lost"][j].cpu(), want.float(), rtol=1e-3, atol=1e-5)

    def test_probabilities_match_softmax_with_sinks_and_masks(self):
        heads, kv_heads, dim, n = 8, 2, 16, 50
        q = torch.randn(5, heads, dim, device="cuda")
        k = torch.randn(n, kv_heads, dim, device="cuda")
        sinks = torch.randn(heads, device="cuda")
        visible = torch.rand(kv_heads, n, device="cuda") > 0.3
        got = ge._probabilities(q, k, sinks, 0.25, kv_heads, torch.arange(45, 50, device="cuda"),
                                torch.arange(n, device="cuda"), visible)
        kk = k.repeat_interleave(heads // kv_heads, 1)
        mask = visible.repeat_interleave(heads // kv_heads, 0)
        for i in range(5):
            logits = torch.einsum("hd,khd->hk", q[i], kk[:46 + i]) * 0.25
            logits = logits.masked_fill(~mask[:, :46 + i], -torch.inf)
            p = torch.softmax(torch.cat((logits, sinks[:, None]), -1), -1)[:, :-1]
            torch.testing.assert_close(got[:, :, i, :46 + i].reshape(heads, -1), p)
            self.assertEqual(float(got[:, :, i, 46 + i:].abs().sum()), 0.0)

    def test_diagnostic_specs_take_their_own_protection(self):
        req = SimpleNamespace(rid="r", session=None, origin_input_ids=[0] * 10)
        e = ge._diagnostic_entry(req, {"policies": "random_2048:8192:2048:2048,prefix_sliding_2048:8192:2048:6144"},
                                 "cuda", 12, 8)
        self.assertEqual(e["specs"], [("random_2048", 8192, 2048, 2048), ("prefix_sliding_2048", 8192, 2048, 6144)])


if __name__ == "__main__":
    unittest.main()
