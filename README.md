# KV cache eviction

The code, run archives and viewers behind the report [Comparing KV Cache Eviction Policies](Hui-Kang%20Tong%20-%20Comparing%20KV%20Cache%20Eviction%20Policies.pdf).
The PDF and its LaTeX source, `source.tex`, sit at the root of this repository.
`slides.pptx` holds the presentation slides.

| Folder | Contents |
| --- | --- |
| [`code/`](code/) | The SGLang eviction overlay, the clients that run the experiments, and the exporter for the report's dataset |
| [`gpt-oss-20b/`](gpt-oss-20b/) | The gpt-oss-20b run archives in `runs/`, and the reference-trace viewer |
| [`arc-agi-3/`](arc-agi-3/) | One ARC-AGI-3 capture and the eviction viewer for it |

## Reproducing the report

The gpt-oss-20b results in the report can be reproduced with these steps, which write the archives in `gpt-oss-20b/runs/`.
Run them from `code/`.
`MODEL` is a local copy of `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`, which the clients read for the tokenizer.

1. Set up the environment and install the eviction overlay.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease allow --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com \
  "sglang==0.5.21" "torch==2.13.0" ninja modal
.venv/bin/python -m cache_eviction.sglang_overlay --site .venv/lib/python3.12/site-packages
```

2. Start the server on one RTX PRO 6000 on Modal, and note the URL it prints.

```sh
.venv/bin/modal run --detach modal_server.py
URL=$(.venv/bin/modal run modal_server.py --action url)
```

3. Run the five policies at 2K to 32K on the five AIME problems with four seeds, 20 runs per point.

```sh
T=tasks/aime2026
.venv/bin/python -m cache_eviction.sglang_decode --url $URL --model-dir MODEL --policies configs/sweep/policies-perhead.json \
  --date 2026-10-22 --task $T/aime2026-04.json $T/aime2026-11.json $T/aime2026-13.json $T/aime2026-26.json $T/aime2026-29.json \
  --add-seeds 4 --output artifacts/perhead/decode
```

4. Run full cache on the same problems and seeds, the 20-run baseline, into the same trials file as step 3.

```sh
.venv/bin/python -m cache_eviction.sglang_decode --url $URL --model-dir MODEL --policies configs/sweep/full.json \
  --date 2026-10-22 --task $T/aime2026-04.json $T/aime2026-11.json $T/aime2026-13.json $T/aime2026-26.json $T/aime2026-29.json \
  --seeds 0 1 2 3 --output artifacts/perhead/decode
```

5. Replay those 20 full-cache runs for the Table 1 metrics.
In the archive they sit in `gpt-oss-20b/runs/perhead/decode/trials-005.jsonl.gz`, which can stand in for the trials file here.

```sh
traces=""
for t in 04 11 13 26 29; do for s in 0 1 2 3; do
  traces="$traces --trace aime2026-$t artifacts/perhead/decode/trials.jsonl $s"
done; done
.venv/bin/python -m cache_eviction.sglang_static --url $URL --model-dir MODEL $traces --output artifacts/perhead/static-full20
```

6. Stop the server, then archive the outputs into `gpt-oss-20b/runs/`.

```sh
.venv/bin/modal app stop gstar-eviction-server
sh sync_runs.sh
```

7. Export the report dataset, the JSON file that the report's tables and figures are built from.

```sh
R=../gpt-oss-20b/runs
.venv/bin/python -m cache_eviction.sglang_report \
  --aime $R/perhead/decode/trials-*.jsonl.gz \
  --aime-tasks $T/aime2026-04.json $T/aime2026-11.json $T/aime2026-13.json $T/aime2026-26.json $T/aime2026-29.json \
  --aime-budgets 2048 4096 8192 16384 32768 \
  --static $R/perhead/static-full20/static.jsonl.gz --trace-label "20 full-cache AIME traces" \
  --timing $R/timing/gpt-oss-20b-decode-perhead.json.gz --regret-budget 8192 \
  --reference-metric-budgets 8192 16384 \
  --settings configs/report/settings.json --findings configs/report/findings.json \
  --output artifacts/report-data.json
```

The report's statement that full cache needed 55K to 67K tokens on these five problems comes from an earlier full-cache run over AIME 2026 in `gpt-oss-20b/runs/sweep/aime2026/`.
The ARC-AGI-3 figures come from recorded competition runs, not from these steps.

## The viewers

Both pages fetch their data, so they need to be served rather than opened from the filesystem.
GitHub Pages serves them at [`/arc-agi-3`](https://tonghuikang.github.io/cache-eviction/arc-agi-3) and [`/gpt-oss-20b`](https://tonghuikang.github.io/cache-eviction/gpt-oss-20b).
To serve them locally, run `python3 -m http.server 8000` from the repository root and open `http://127.0.0.1:8000/`.

[`arc-agi-3/`](arc-agi-3/) shows an ARC-AGI-3 AR25 capture at level 2.
It covers 79,720 prompt tokens over 12 sparse-attention layers, 64,885 text tokens and 38 images, under a cache of 512 prefix, 4,096 recent and 24,064 scored history tokens.
A token is shaded by how many of the 12 layers still hold its block.
`rollout-ar25-v85.jsonl` is the rollout, `eviction-ar25-v85.json` is the per-layer eviction state, and `tokenizer.json` decodes the token IDs.

[`gpt-oss-20b/`](gpt-oss-20b/) shows the 20 full-cache runs that Table 1 replays, AIME 2026 problems 4, 11, 13, 26 and 29 with seeds 0 to 3.
A token is shaded by the attention its query put on keys the policy had already evicted, averaged over heads and full-attention layers and scored by the unevicted model.
The picker switches run, policy and budget from 2K to 32K, and the trace means, averaged over the 20 traces, give the Table 1 attention weight difference.
`gpt-oss-20b/data/` is written by `gpt-oss-20b/build_data.py` from `gpt-oss-20b/runs/perhead/`, and the script's docstring documents the format.
