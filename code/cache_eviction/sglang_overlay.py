"""Install the gstar_evict overlay into an unmodified SGLang 0.5.21 site directory.

Every edit is a literal replacement that must match exactly once, against
source files with pinned SHA-256 hashes. Originals are kept beside each file
as *.gstar-orig, and reinstalling starts from them. The overlay is inert
unless the server runs with GSTAR_EVICT=1.
"""
import argparse
import hashlib
from pathlib import Path

HASHES = {
    "srt/layers/attention/triton_backend.py": "9924224f0dd4150fe1fc9ef7de6a556c49857608792479379179a973436762c1",
    "srt/mem_cache/allocation.py": "b721edde1b0da5176e6787e40173ac23e62f943d13afd2e0ca90ec84f0852212",
    "srt/mem_cache/common.py": "c7eb492978794bbd4015043199f36927faf591ab4009f90d03455f36ed5f764f",
    "srt/session/streaming_session.py": "bf73c5866c1d592673a58f9e5a5d9627d0586cfefbaab48d9796e5e6992a20dc",
    "srt/entrypoints/http_server.py": "1fbb46671e86d0ebf23046efa09e8e03d251754e681ba03b818dbb7a75c79c55",
}

EDITS = {
    "srt/layers/attention/triton_backend.py": [
        ("        kv_indices: torch.Tensor,\n    ) -> torch.Tensor:\n"
         "        kv_indptr = self.kv_indptr[: bs + 1]\n",
         "        kv_indices: torch.Tensor,\n    ) -> torch.Tensor:\n"
         "        from sglang.srt import gstar_evict\n\n"
         "        if gstar_evict.enabled():\n"
         "            return gstar_evict.fill(self, bs, seq_lens, req_pool_indices, kv_indices)\n"
         "        kv_indptr = self.kv_indptr[: bs + 1]\n"),
        ("            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,\n        )\n        return o\n",
         "            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,\n        )\n"
         "        from sglang.srt import gstar_evict\n\n"
         "        if gstar_evict.enabled():\n"
         "            gstar_evict.after_extend(self, q, layer, forward_batch, sinks)\n"
         "        return o\n"),
        ("            lean_locks=self.forward_metadata.lean_locks,\n        )\n        return o\n\n\n"
         "class TritonMultiStepDraftBackend",
         "            lean_locks=self.forward_metadata.lean_locks,\n        )\n"
         "        from sglang.srt import gstar_evict\n\n"
         "        if gstar_evict.enabled():\n"
         "            gstar_evict.after_decode(self, q, layer, forward_batch, sinks, o)\n"
         "        return o\n\n\nclass TritonMultiStepDraftBackend"),
    ],
    "srt/mem_cache/allocation.py": [
        ("    batch.maybe_evict_swa()\n\n    seq_lens_gpu = batch.seq_lens\n",
         "    batch.maybe_evict_swa()\n"
         "    from sglang.srt import gstar_evict\n\n"
         "    if gstar_evict.enabled():\n"
         "        gstar_evict.on_decode(batch)\n\n"
         "    seq_lens_gpu = batch.seq_lens\n"),
        ("    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu\n",
         "    from sglang.srt import gstar_evict\n\n"
         "    if gstar_evict.enabled():\n"
         "        gstar_evict.on_extend(batch)\n"
         "    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu\n"),
    ],
    "srt/mem_cache/common.py": [
        ('    what it can key."""\n    assert (not req.kv.holds_kv) == req.kv.is_kv_released\n',
         '    what it can key."""\n'
         "    from sglang.srt import gstar_evict\n\n"
         "    if gstar_evict.enabled():\n"
         "        gstar_evict.release(req)\n"
         "    assert (not req.kv.holds_kv) == req.kv.is_kv_released\n"),
        ("    if swa_dead:\n        allocator.free_full_segments(swa_dead)\n",
         "    if swa_dead:\n"
         "        from sglang.srt import gstar_evict\n\n"
         "        if gstar_evict.enabled():\n"
         "            swa_dead = gstar_evict.filter_holes(swa_dead)\n"
         "        allocator.free_full_segments(swa_dead)\n"),
    ],
    # A session's evicted slots are back in the allocator, so they are not held.
    "srt/session/streaming_session.py": [
        ("                total += allocated - slot.kv.cache_protected_len\n        return total\n\n"
         "    def session_held_full_tokens",
         "                total += allocated - slot.kv.cache_protected_len\n"
         "                from sglang.srt import gstar_evict\n\n"
         "                if gstar_evict.enabled():\n"
         "                    total -= gstar_evict.freed(slot.kv.req_pool_idx)\n"
         "        return total\n\n    def session_held_full_tokens"),
    ],
    # Payloads written by the scheduler, served so a remote client never reads server files.
    "srt/entrypoints/http_server.py": [
        ('@app.api_route("/open_session", methods=["GET", "POST"])\n',
         '@app.get("/gstar/payload/{key}")\n'
         "async def gstar_payload(key: str):\n"
         "    from sglang.srt import gstar_evict\n\n"
         "    return gstar_evict.payload_response(key)\n\n\n"
         '@app.api_route("/open_session", methods=["GET", "POST"])\n'),
    ],
}


def patched(relative, source):
    for before, after in EDITS[relative]:
        if source.count(before) != 1:
            raise ValueError(f"{relative}: expected one occurrence of {before[:60]!r}")
        source = source.replace(before, after, 1)
    return source


def install(site):
    root = Path(site) / "sglang"
    pending = {}
    for relative, expected in HASHES.items():
        path = root / relative
        original = path.with_name(path.name + ".gstar-orig")
        source = (original if original.exists() else path).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise ValueError(f"unexpected SGLang source: {relative}")
        changed = patched(relative, source)
        compile(changed, relative, "exec")
        pending[path] = (source, changed)
    runtime = Path(__file__).with_name("sglang_evict_runtime.py").read_text()
    compile(runtime, "gstar_evict.py", "exec")
    for path, (source, changed) in pending.items():
        original = path.with_name(path.name + ".gstar-orig")
        if not original.exists():
            original.write_text(source)
        path.write_text(changed)
    (root / "srt/gstar_evict.py").write_text(runtime)
    files = {**{str(p.relative_to(root)): c for p, (_, c) in pending.items()}, "srt/gstar_evict.py": runtime}
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="site-packages directory containing sglang/")
    for name, digest in install(parser.parse_args().site).items():
        print(digest, name)


if __name__ == "__main__":
    main()
