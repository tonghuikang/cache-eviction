# gpt-oss-20b on SGLang with physical KV eviction

The study runs gpt-oss-20b on SGLang 0.5.21 with the `gstar_evict` overlay, on AIME 2026 with no tool use.
The [top-level README](../README.md) gives the commands in the order the report used them.

## Environment

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease allow --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com \
  "sglang==0.5.21" "torch==2.13.0" ninja modal
.venv/bin/python -m cache_eviction.sglang_overlay --site .venv/lib/python3.12/site-packages
```

The installer checks the SHA-256 hashes of five upstream SGLang files, keeps each original as `*.gstar-orig`, applies literal edits that must each match once, and installs `sglang/srt/gstar_evict.py`.
The checkpoint is `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`.

## Serving

`modal_server.py` serves the model on one RTX PRO 6000 on Modal, with the overlay installed and `GSTAR_EVICT=1`.
It exposes the server through a Modal tunnel, which has no HTTP request timeout, and the clients run locally against its URL.

```sh
.venv/bin/modal run --detach modal_server.py
URL=$(.venv/bin/modal run modal_server.py --action url)
.venv/bin/modal app stop gstar-eviction-server
```

Without `GSTAR_EVICT=1` the overlay is inert and the server is stock SGLang.

## Overlay

A request opts in through `sampling_params.custom_params`, in one of three ways.

| Key | Fields | Use |
| --- | --- | --- |
| `gstar_eviction` | `name`, `budget`, `prefix`, `recent`, `seed`, optional `record` | Live decode with physical eviction |
| `gstar_replay` | `name`, `budget`, `prefix`, `recent`, `seed` | Mask-only eviction on a frozen trace, for the parity metrics |
| `gstar_diagnostic` | `policies` as `name:B:P:R,...`, `seed` | The attention mass each policy would lose on a frozen trace |

The policy names are `random_2048`, `random_reservoir_2048`, `prefix_sliding_2048`, `head_sum_2048` and `head_max_sum_2048`, described in the [study README](README.md#policies).
Boundaries fall before queries 2048, 4096 and so on.

Every full-attention layer and KV head chooses its own survivors.
Protected entries, the prefix, the recent window and the sliding-window guard, keep their own slots, shared by every head.
Each head's chosen middle entries are copied into one shared set of slots, so all heads read one index list and the leftover slots return to the allocator.
RoPE positions and scheduler lengths stay logical.
The 12 sliding-window layers keep their native 128-token window and are never evicted.

Head-sum and Head-maxsum keep one score per layer, KV head and slot.
During decode a CUDA-graph-safe Triton kernel rebuilds each head's softmax normalizer from the decode kernel's split log-sum-exps and the learned sink, then recomputes QK once for the resident keys.
Prefill scores are computed exactly in fp32.
Sinks enter the normalizer but are never eviction candidates.

Live state is kept per streaming session, so a run keeps its eviction state across 2048-token calls.
Per-token payloads, such as recorded retention and the static metrics, are served over HTTP at `/gstar/payload/{key}`, so a remote client never reads server files.

## Tasks

```sh
.venv/bin/python -m cache_eviction.sglang_tasks aime --parquet PARQUET \
  --revision d61f486805306e6e6b5913504c07763a99e1b112 --output tasks/aime2026
```

AIME 2026 comes from `MathArena/aime_2026` at revision `d61f486`, 30 problems with integer answers from 0 to 999.
`PARQUET` is that revision's `data/train-00000-of-00001.parquet`.
The final channel is scored, a run that stopped without one is scored on its last boxed integer, and a run truncated inside its analysis has no answer.

## Clients

`sglang_decode` runs each trial as a streaming session that decodes in calls ending on multiples of 2048 tokens.
It admits runs while the sum of their worst-case cache reservations fits the pool, and requeues a failed or retracted run from scratch.
`--add-seeds` plans more seeds for each problem, policy and budget, and `--reuse` skips runs already in earlier trial files.

`sglang_static` replays frozen full-cache traces as prompts with one output token.
For each policy and budget it records the chosen-token logprobs, the first full-attention layer's output error and the attention mass lost to evicted keys.
It needs a server started with `--chunked-prefill-size 2048`, so prefill chunks end on eviction boundaries, and it sends one request at a time.

`sglang_timing` times one full-attention decode layer with paired, alternating CUDA-event samples.
Off is SGLang's Triton decode attention alone, and on adds the score kernel.
Run it with the GPU otherwise idle.

## Checks

```sh
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m unittest tests.test_sglang_evict -v
```

The tests compare the decode score kernel with an exact softmax including sinks, check the slot relocation, and compare boundary selection with `core.Policy`.
`GSTAR_EVICT_VERIFY=1` with `--disable-cuda-graph` also recomputes every full-layer decode output over the resident entries and records the largest error per request.
