# Code

This directory holds the code behind the report [Comparing KV Cache Eviction Policies](../Hui-Kang%20Tong%20-%20Comparing%20KV%20Cache%20Eviction%20Policies.pdf).
The [top-level README](../README.md) gives the exact steps that reproduce every gpt-oss-20b number in the report.
[SGLANG.md](SGLANG.md) documents the eviction overlay and the clients that drive it.

## Layout

| Path | Contents |
| --- | --- |
| `cache_eviction/sglang_overlay.py` | Installs the eviction overlay into SGLang 0.5.21 |
| `cache_eviction/sglang_evict_runtime.py` | The overlay's runtime module, installed as `sglang.srt.gstar_evict` |
| `cache_eviction/sglang_client.py` | The HTTP client the decode and static clients share |
| `cache_eviction/sglang_decode.py` | Live decode under eviction, one streaming session per run |
| `cache_eviction/sglang_static.py` | Parity metrics on frozen full-cache reference traces |
| `cache_eviction/sglang_report.py` | Exports the report dataset from the run archives |
| `cache_eviction/sglang_tasks.py` | Builds the AIME 2026 task files |
| `cache_eviction/sglang_timing.py` | Paired CUDA timing of the score kernel |
| `cache_eviction/core.py` | Reference policies in plain Python, which the tests compare the overlay against |
| `cache_eviction/files.py` | JSON and hashing helpers |
| `modal_server.py` | The server on one RTX PRO 6000 on Modal |
| `sync_runs.sh` | Archives the ignored `artifacts/` into `../gpt-oss-20b/runs/` as gzip files |
| `configs/sweep/` | Policy and budget lists, `policies-perhead.json` for the sweep and `full.json` for full cache |
| `configs/report/` | Settings and findings text the exporter writes into the dataset |
| `tasks/aime2026/` | The 30 AIME 2026 problems as task files |

## Policies

Every policy keeps a protected prefix `P` and a protected recent window `R`, and decides once every 2048 tokens.
The sweep sets `P = R = B/4` for a budget `B`, except Prefix Sliding, which keeps `P = B/4` and `R = 3B/4`.

| Policy ID | Rule for the rest of the budget |
| --- | --- |
| `full` | No eviction |
| `random_2048` | A fresh uniform score for every resident entry at every boundary, so an old entry faces the draw many times |
| `random_reservoir_2048` | One fixed random score per entry, drawn when it arrives, so the kept middle is a uniform sample of the history |
| `prefix_sliding_2048` | Nothing beyond `P` and `R`, which together fill the budget |
| `head_sum_2048` | Attention received, summed over the query heads that share a KV head and over queries |
| `head_max_sum_2048` | Attention received, the maximum over the query heads that share a KV head, summed over queries |

Each of the 12 full-attention layers and each of its 8 KV heads chooses its own survivors.
Ties keep newer positions.
Evicted entries never return, and positions are never renumbered.

## Tests

```sh
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m unittest tests.test_sglang_evict -v
```

The tests need a CUDA GPU and the SGLang environment from the top-level README.
