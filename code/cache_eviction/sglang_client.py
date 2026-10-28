"""HTTP client for a gstar SGLang server, local or on Modal.

Everything goes through the server's public API: /generate, streaming
sessions (/open_session, /close_session) and /gstar/payload/{key}, which
serves the per-token payloads and retention records the overlay writes. The
client never reads server files.
"""
import base64
import gzip
import json
import time
import urllib.error
import urllib.request

import numpy as np

INTERVAL = 2048
STOP_TOKEN_IDS = [200002, 200012]  # gpt-oss <|return|> and <|call|>


def request(url, path, body=None, timeout=None, attempts=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data, {"Content-Type": "application/json"})
    for attempt in range(attempts):
        try:
            # No timeout by default: a long call under heavy load can take many minutes.
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError:
            # urllib raises URLError only while connecting or sending, before the
            # server has the whole request, so even a session call is safe to retry.
            # The Modal tunnel drops a connection now and then.
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 2 ** attempt))
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw) if raw else None


def healthy(url):
    try:
        request(url, "/health", timeout=10)
        return True
    except (OSError, ValueError):
        return False


def wait_healthy(url, seconds=1800):
    deadline = time.time() + seconds
    while not healthy(url):
        if time.time() > deadline:
            raise TimeoutError(f"{url} is not healthy")
        time.sleep(10)


def server_info(url):
    return request(url, "/get_server_info", timeout=60)


def generate(url, input_ids, params, session=None, return_logprob=False):
    body = {"input_ids": input_ids, "sampling_params": params}
    if session is not None:
        body["session_params"] = {"id": session}
    if return_logprob:
        body.update(return_logprob=True, logprob_start_len=0)
    out = request(url, "/generate", body)
    if out["meta_info"]["finish_reason"] and out["meta_info"]["finish_reason"].get("type") == "abort":
        raise RuntimeError(f"request aborted: {out['meta_info']['finish_reason']}")
    return out


def open_session(url, session, capacity=131072):
    out = request(url, "/open_session", {"capacity_of_str_len": capacity, "session_id": session, "streaming": True})
    if out != session:
        raise RuntimeError(f"could not open session {session}: {out}")


def close_session(url, session):
    try:
        request(url, "/close_session", {"session_id": session}, timeout=60)
    except urllib.error.HTTPError:
        pass


def decode_arrays(obj):
    """Turn the overlay's {dtype, shape, data} records back into numpy arrays."""
    if isinstance(obj, dict):
        if set(obj) == {"dtype", "shape", "data"}:
            return np.frombuffer(base64.b64decode(obj["data"]), dtype=obj["dtype"]).reshape(obj["shape"])
        return {k: decode_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_arrays(v) for v in obj]
    return obj


def payload(url, key, wait=120, raw=False):
    """Fetch one payload, retrying while the server has not written it yet. None if it never appears."""
    deadline = time.time() + wait
    while True:
        try:
            obj = request(url, f"/gstar/payload/{key}", timeout=120)
            return obj if raw else decode_arrays(obj)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        if time.time() >= deadline:
            return None
        time.sleep(1)


def retained_sets(records):
    """Rebuild each (full layer, KV head)'s retained positions after every recorded boundary.

    records: retention payloads sorted by boundary. A boundary that evicted
    nothing writes no record. Before a boundary a head holds its previous
    survivors plus every newer position. Yields
    (boundary, list of sorted position arrays, one per layer-major head).
    """
    held = None
    last = 0
    for record in records:
        n = record["boundary"]
        counts = record["evicted_counts"]
        evicted = np.split(np.asarray(record["evicted_delta"], dtype=np.int64), np.cumsum(counts)[:-1])
        new = np.arange(last, n)
        held = [new.copy() for _ in counts] if held is None else [np.concatenate((h, new)) for h in held]
        held = [np.setdiff1d(h, np.cumsum(e), assume_unique=True) for h, e in zip(held, evicted)]
        last = n
        yield n, held
