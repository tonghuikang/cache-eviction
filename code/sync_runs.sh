#!/bin/sh
# Mirror run records from the git-ignored artifacts/ into ../gpt-oss-20b/runs/ as gzip files.
# Trials, run settings, eviction logs and timing are kept.
# The per-head static arrays (.npz) and retention records (.json.gz)
# are already compressed and copied as they are. In-progress run state is not kept.
# With --watch it repeats every five minutes while the clients are running.
here=$(cd "$(dirname "$0")" && pwd)
sync() {
  cd "$here/artifacts" || exit 1
  find sweep timing perhead -type f -not -path '*/state/*' \
    -not -path 'perhead/*/trials.jsonl' \( -name '*.jsonl' -o -name '*.json' \) 2>/dev/null | while read -r f; do
    out="$here/../gpt-oss-20b/runs/$f.gz"
    if [ ! -e "$out" ] || [ "$f" -nt "$out" ]; then
      mkdir -p "$(dirname "$out")" && gzip -9nc "$f" > "$out.tmp" && mv "$out.tmp" "$out"
    fi
  done
  # The per-head trials only grow, so they are archived in shards of 100 lines.
  # A full shard never changes, and every shard stays far below GitHub's file limit.
  find perhead -type f -name trials.jsonl 2>/dev/null | while read -r f; do
    out="$here/../gpt-oss-20b/runs/$(dirname "$f")"
    mkdir -p "$out" && rm -f "$out/trials.jsonl.gz"
    tmp=$(mktemp -d) && split -l 100 -d -a 3 --additional-suffix=.jsonl "$f" "$tmp/trials-"
    for shard in "$tmp"/trials-*.jsonl; do
      gzip -9nc "$shard" > "$out/$(basename "$shard").gz.tmp" && mv "$out/$(basename "$shard").gz.tmp" "$out/$(basename "$shard").gz"
    done
    rm -r "$tmp"
  done
  find perhead -type f \( -name '*.npz' -o -name '*.json.gz' \) 2>/dev/null | while read -r f; do
    out="$here/../gpt-oss-20b/runs/$f"
    if [ ! -e "$out" ] || [ "$f" -nt "$out" ]; then
      mkdir -p "$(dirname "$out")" && cp "$f" "$out.tmp" && mv "$out.tmp" "$out"
    fi
  done
}
sync
if [ "$1" = "--watch" ]; then
  while pgrep -f "cache_eviction[.]sglang_(tasks|decode|static)" >/dev/null; do sleep 300; sync; done
  sync
fi
