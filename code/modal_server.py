"""The gstar SGLang inference server on Modal, one RTX PRO 6000, nothing else.

The container installs the hash-checked gstar_evict overlay on the official
SGLang 0.5.21 wheel and serves openai/gpt-oss-20b at its pinned revision with
GSTAR_EVICT=1. It exposes the server through a public Modal tunnel, which has
no HTTP request timeout (Modal web endpoints stop at 150 seconds, shorter
than one 2048-token call under full load), and publishes the tunnel URL in
the modal.Dict "gstar-eviction-server". The clients run locally and use only
the server's HTTP API, so nothing is read from inside Modal.

  .venv/bin/modal run --detach modal_server.py          # start, print the URL once it is healthy
  .venv/bin/modal run modal_server.py --action url      # print the URL of the running server
  .venv/bin/modal app stop gstar-eviction-server        # stop it
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

MODEL_REPO = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
HF_MOUNT = "/hf-cache"
MODEL_PATH = f"{HF_MOUNT}/hub/models--openai--gpt-oss-20b/snapshots/{MODEL_REVISION}"
REMOTE = "/opt/gstar"
PORT = 30123
HERE = Path(__file__).resolve().parent

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
registry = modal.Dict.from_name("gstar-eviction-server", create_if_missing=True)
app = modal.App("gstar-eviction-server")

LOCK = "/opt/wheelhouse/requirements.lock.txt"
SGLANG_WHEEL = "sglang-0.5.21-cp312-cp312-manylinux_2_34_x86_64.whl"
SGLANG_WHEEL_SHA256 = "ac300998eb6b1afcb2971b31317b66ef12891003bdf9d1a5c7e34f9d4e3dcf0d"
FLASHINFER_JIT_CACHE_URL = (
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/"
    "flashinfer_jit_cache-0.6.18%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl"
)
# The pinned stack: the registry image, its dependency lock, the official release wheel, and FlashInfer's AOT cache.
image = modal.Image.from_registry("lmsysorg/sglang:v0.5.21-cu130")
if modal.is_local():
    image = image.add_local_file(HERE / "configs/modal/sglang-v0521.lock.txt", LOCK, copy=True)
image = image.run_commands(
    "python -m pip install --only-binary=:all: "
    "--extra-index-url https://download.pytorch.org/whl/cu130 "
    "--extra-index-url https://pypi.nvidia.com "
    f"--find-links {FLASHINFER_JIT_CACHE_URL} -r {LOCK}",
    "python -m pip download --no-deps --only-binary=:all: sglang==0.5.21 -d /opt/wheelhouse/stock",
    f"echo '{SGLANG_WHEEL_SHA256}  /opt/wheelhouse/stock/{SGLANG_WHEEL}' | sha256sum -c -",
    f"python -m pip install --no-deps --force-reinstall /opt/wheelhouse/stock/{SGLANG_WHEEL}",
    "python -c 'import sglang, pydantic; assert sglang.__version__ == \"0.5.21\"; assert pydantic.VERSION.startswith(\"2.\")'",
)
if modal.is_local():
    image = image.add_local_file(HERE / "cache_eviction/sglang_overlay.py", f"{REMOTE}/cache_eviction/sglang_overlay.py",
                                 copy=True)
    image = image.add_local_file(HERE / "cache_eviction/sglang_evict_runtime.py",
                                 f"{REMOTE}/cache_eviction/sglang_evict_runtime.py", copy=True)
image = image.run_commands(
    # Fails the build if any patched SGLang source differs from the pinned hashes.
    f"touch {REMOTE}/cache_eviction/__init__.py && cd {REMOTE} && python -m cache_eviction.sglang_overlay "
    "--site $(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')",
).env({"HF_HUB_CACHE": f"{HF_MOUNT}/hub", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
       "GSTAR_EVICT": "1", "GSTAR_PAYLOAD_DIR": "/tmp/gstar-payload"})


def _healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


@app.function(image=image, gpu="RTX-PRO-6000:1", cpu=16, memory=131072, timeout=24 * 3600,
              volumes={HF_MOUNT: hf_volume})
def serve(memory_fraction: float = 0.9, max_running: int = 320, graph_batch: int = 256):
    """Serve until stopped or for 24 hours. The pool and limits are what the clients read from /get_server_info."""
    server = subprocess.Popen(
        [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL_PATH, "--host", "0.0.0.0",
         "--port", str(PORT), "--attention-backend", "triton", "--mem-fraction-static", str(memory_fraction),
         "--context-length", "131072", "--disable-radix-cache", "--chunked-prefill-size", "2048",
         "--enable-streaming-session", "--max-running-requests", str(max_running),
         "--cuda-graph-max-bs-decode", str(graph_batch), "--disable-prefill-cuda-graph"])
    while not _healthy():
        if server.poll() is not None:
            raise RuntimeError("the SGLang server exited during startup")
        time.sleep(5)
    with modal.forward(PORT) as tunnel:
        registry["url"] = tunnel.url
        registry["started"] = time.time()
        print("SERVING", tunnel.url, flush=True)
        try:
            server.wait()
        finally:
            registry.pop("url", None)
    raise RuntimeError(f"the SGLang server exited with {server.returncode}")


@app.local_entrypoint()
def main(action: str = "serve", memory_fraction: float = 0.9, max_running: int = 320, graph_batch: int = 256):
    if action == "url":
        print(registry.get("url"))
        return
    registry.pop("url", None)
    call = serve.spawn(memory_fraction, max_running, graph_batch)
    while registry.get("url") is None:
        time.sleep(10)
    print("SERVING", registry["url"], call.object_id, flush=True)
