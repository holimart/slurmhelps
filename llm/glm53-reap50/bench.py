#!/usr/bin/env python3
"""Batched-throughput benchmark for an OpenAI-compatible LLM server.

The client is deliberately stdlib-only so it can run inside a Slurm job without
a venv, and it can use several processes when one CPython process would become
the bottleneck.

It measures, per concurrency point and repetition:

  * aggregate output tokens/s (the headline decode number)
  * aggregate total tokens/s (input + output)
  * request throughput
  * TTFT p50/p90/p99
  * TPOT p50/p99 (decode time per output token)
  * goodput at a TTFT/TPOT SLO
  * prefix-cache hit counters scraped from /metrics, when available

Protocol (see llm/glm53-reap50/README.md): closed-loop `request-rate=inf` with
`--max-concurrency C`, `num_prompts >= factor * C`, warmup, and a fresh
`cache_salt` (or a cache flush) per measured run so later points do not inherit
an earlier point's prefix cache.

usage:
  python3 bench.py --base-url http://127.0.0.1:30000 \
      --concurrency "1 2 4 8 16 32 64" --isl 2048 --osl 256 --runs 3
  python3 bench.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty
from typing import Any, Sequence

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega one two three four "
    "five six seven eight nine ten red green blue amber violet silver golden "
    "run jump walk read write build test measure serve cache batch token layer "
    "model server client request response latency throughput memory compute "
    "prefix suffix stream decode prefill sample route expert attention sparse "
    "dense weight matrix vector tensor kernel graph node edge path queue slot "
    "cold warm local remote fast slow stable random fixed known hidden open "
    "closed first second third last next previous small large long short"
).split()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Config:
    base_url: str = "http://127.0.0.1:30000"
    model: str = "glm53-reap50"
    endpoint: str = "/v1/completions"
    api_key: str = ""
    isl: int = 2048
    osl: int = 256
    timeout: float = 600.0
    stream: bool = True
    stream_options: bool = True
    chat: bool = False
    ignore_eos: bool = True
    cache_salt: str = ""
    shared_prefix: int = 0
    slo_ttft_ms: float = 2000.0
    slo_tpot_ms: float = 80.0
    metrics_url: str = ""
    flush_cache: bool = True
    seed: int = 1


@dataclass
class Sample:
    ok: bool
    error: str = ""
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0

    @property
    def tpot_s(self) -> float:
        if self.out_tokens <= 1:
            return 0.0
        return max(0.0, (self.e2e_s - self.ttft_s) / (self.out_tokens - 1))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def make_prompt(isl: int, rng: random.Random, shared_prefix: Sequence[str] = ()) -> str:
    """A deterministic, roughly token-length prompt (word ≈ token for English)."""
    words = list(shared_prefix)
    need = max(1, int(isl) - len(words))
    words.extend(rng.choice(WORDS) for _ in range(need))
    return " ".join(words)


# ---------------------------------------------------------------------------
# HTTP + SSE
# ---------------------------------------------------------------------------
def _headers(cfg: Config) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if cfg.api_key:
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


def _payload(cfg: Config, prompt: str) -> dict[str, Any]:
    if cfg.chat:
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.osl,
            "temperature": 0,
            "stream": cfg.stream,
        }
    else:
        body = {
            "model": cfg.model,
            "prompt": prompt,
            "max_tokens": cfg.osl,
            "temperature": 0,
            "stream": cfg.stream,
            "ignore_eos": cfg.ignore_eos,
        }
    if cfg.stream and cfg.stream_options:
        body["stream_options"] = {"include_usage": True}
    if cfg.cache_salt:
        body["cache_salt"] = cfg.cache_salt
    return body


def _extract(obj: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    text = ""
    choices = obj.get("choices") or []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if "text" in choice:
            text += choice.get("text") or ""
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            text += delta.get("content") or ""
            text += delta.get("reasoning_content") or ""
        message = choice.get("message") or {}
        if isinstance(message, dict):
            text += message.get("content") or ""
    usage = obj.get("usage")
    return text, usage if isinstance(usage, dict) else None


def request_once(cfg: Config, prompt: str) -> Sample:
    url = cfg.base_url.rstrip("/") + cfg.endpoint
    body = json.dumps(_payload(cfg, prompt)).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_headers(cfg), method="POST")
    start = time.perf_counter()
    first: float | None = None
    chunks = 0
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            if not cfg.stream:
                obj = json.loads(resp.read().decode("utf-8", "replace"))
                text, usage = _extract(obj)
                end = time.perf_counter()
                out = int((usage or {}).get("completion_tokens") or (len(text.split()) if text else 0))
                inn = int((usage or {}).get("prompt_tokens") or 0)
                if out <= 0:
                    return Sample(False, "no output tokens", in_tokens=inn)
                return Sample(True, "", end - start, end - start, inn, out)
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text, got_usage = _extract(obj)
                if got_usage:
                    usage = got_usage
                if text:
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
        end = time.perf_counter()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        return Sample(False, f"HTTP {exc.code}: {detail}")
    except Exception as exc:  # timeout, connection reset, ...
        return Sample(False, f"{type(exc).__name__}: {exc}")

    out = int((usage or {}).get("completion_tokens") or chunks)
    inn = int((usage or {}).get("prompt_tokens") or 0)
    if out <= 0:
        return Sample(False, "no output tokens", in_tokens=inn)
    ttft = first if first is not None else end
    return Sample(True, "", ttft - start, end - start, inn, out)


def flush_cache(base_url: str, timeout: float = 10.0) -> str:
    """Best-effort prefix-cache flush. Returns the endpoint that answered."""
    for path in ("/flush_cache", "/reset_prefix_cache"):
        req = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return path
        except Exception:
            continue
    return ""


def scrape_metrics(url: str, timeout: float = 10.0) -> dict[str, float]:
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, value = line.rsplit(" ", 1)
            value_f = float(value)
        except (ValueError, IndexError):
            continue
        out[name] = value_f
        base = name.split("{", 1)[0]
        out[base] = out.get(base, 0.0) + value_f
    return out


def _delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in after.items():
        if key in before:
            out[key] = value - before[key]
    return out


def cache_stats(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Normalise vLLM/SGLang/llama.cpp prefix-cache counters."""
    stats: dict[str, float] = {}
    for key in ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total", "sglang:cache_hit_rate"):
        if key in after:
            stats[key] = after[key]
    d = _delta(before, after)
    hits = 0.0
    queries = 0.0
    for key in ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total",
                "sglang:prefix_cache_hits", "sglang:prefix_cache_hits_total"):
        if key in d:
            hits = max(hits, d[key])
    for key in ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total",
                "sglang:prefix_cache_queries", "sglang:prefix_cache_queries_total"):
        if key in d:
            queries = max(queries, d[key])
    if queries > 0:
        stats["hits"] = hits
        stats["queries"] = queries
        stats["hit_rate"] = hits / queries
    elif "sglang:cache_hit_rate" in after:
        stats["hit_rate"] = after["sglang:cache_hit_rate"]
    return stats


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
def run_batch(cfg: Config, prompts: Sequence[str], concurrency: int) -> tuple[list[Sample], float]:
    samples: list[Sample] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(request_once, cfg, p) for p in prompts]
        for fut in as_completed(futures):
            samples.append(fut.result())
    return samples, time.perf_counter() - start


def _mp_worker(cfg_dict: dict[str, Any], prompts: list[str], concurrency: int, queue: Any) -> None:
    try:
        cfg = Config(**cfg_dict)
        samples, wall = run_batch(cfg, prompts, concurrency)
        queue.put(([asdict(s) for s in samples], wall))
    except Exception as exc:  # keep the parent from blocking forever
        queue.put(([asdict(Sample(False, f"worker: {type(exc).__name__}: {exc}"))], 0.0))


def run_batch_mp(cfg: Config, prompts: Sequence[str], concurrency: int, workers: int) -> tuple[list[Sample], float]:
    workers = max(1, min(workers, concurrency, len(prompts)))
    if workers <= 1:
        return run_batch(cfg, prompts, concurrency)
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    chunks: list[list[str]] = [[] for _ in range(workers)]
    for i, prompt in enumerate(prompts):
        chunks[i % workers].append(prompt)
    per_worker = max(1, int(math.ceil(concurrency / workers)))
    start = time.perf_counter()
    procs = [
        ctx.Process(target=_mp_worker, args=(asdict(cfg), chunk, per_worker, queue))
        for chunk in chunks if chunk
    ]
    for proc in procs:
        proc.start()
    samples: list[Sample] = []
    try:
        for _ in procs:
            raw, _ = queue.get(timeout=cfg.timeout + 300)
            samples.extend(Sample(**item) for item in raw)
    except Empty:
        for proc in procs:
            proc.terminate()
        raise RuntimeError("benchmark worker timed out; raise --timeout or lower --workers")
    finally:
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
    return samples, time.perf_counter() - start


def summarize(samples: Sequence[Sample], wall_s: float, cfg: Config) -> dict[str, Any]:
    ok = [s for s in samples if s.ok]
    failed = len(samples) - len(ok)
    out_tokens = sum(s.out_tokens for s in ok)
    in_tokens = sum(s.in_tokens for s in ok)
    ttfts = [s.ttft_s * 1000.0 for s in ok]
    tpots = [s.tpot_s * 1000.0 for s in ok]
    e2es = [s.e2e_s * 1000.0 for s in ok]
    good = sum(
        1 for s in ok
        if s.ttft_s * 1000.0 <= cfg.slo_ttft_ms and s.tpot_s * 1000.0 <= cfg.slo_tpot_ms
    )
    wall = max(wall_s, 1e-9)
    # For a single request these are exact prefill/decode rates. For a batch
    # they are the aggregate prompt/output rates over the longest prefill or
    # decode window, which is the useful capacity number for a long-context
    # point (end-to-end output tok/s is dominated by prefill).
    prefill_s = max((s.ttft_s for s in ok), default=0.0)
    decode_s = max((s.e2e_s - s.ttft_s for s in ok), default=0.0)
    return {
        "completed": len(ok),
        "failed": failed,
        "duration_s": round(wall, 4),
        "output_tokens": out_tokens,
        "input_tokens": in_tokens,
        "output_tokens_per_s": round(out_tokens / wall, 3),
        "total_tokens_per_s": round((out_tokens + in_tokens) / wall, 3),
        "prefill_tokens_per_s": round(in_tokens / prefill_s, 3) if prefill_s else 0.0,
        "decode_tokens_per_s": round(out_tokens / decode_s, 3) if decode_s else 0.0,
        "request_throughput": round(len(ok) / wall, 3),
        "ttft_ms": {
            "mean": round(statistics.fmean(ttfts), 3) if ttfts else 0.0,
            "p50": round(percentile(ttfts, 50), 3),
            "p90": round(percentile(ttfts, 90), 3),
            "p99": round(percentile(ttfts, 99), 3),
        },
        "tpot_ms": {
            "mean": round(statistics.fmean(tpots), 3) if tpots else 0.0,
            "p50": round(percentile(tpots, 50), 3),
            "p90": round(percentile(tpots, 90), 3),
            "p99": round(percentile(tpots, 99), 3),
        },
        "e2e_ms": {
            "p50": round(percentile(e2es, 50), 3),
            "p99": round(percentile(e2es, 99), 3),
        },
        "goodput_req_per_s": round(good / wall, 3),
        "slo_attainment": round(good / len(ok), 4) if ok else 0.0,
        "slo": {"ttft_ms": cfg.slo_ttft_ms, "tpot_ms": cfg.slo_tpot_ms},
    }


def median_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    keys = ("output_tokens_per_s", "total_tokens_per_s", "prefill_tokens_per_s",
            "decode_tokens_per_s", "request_throughput",
            "goodput_req_per_s", "slo_attainment")
    out: dict[str, Any] = {k: round(statistics.median([r[k] for r in runs]), 3) for k in keys}
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        out[metric] = {
            p: round(statistics.median([r[metric][p] for r in runs]), 3)
            for p in runs[0][metric]
        }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def parse_concurrency(text: str) -> list[int]:
    values = [int(x) for x in text.replace("+", " ").replace(",", " ").split() if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("concurrency must be positive integers")
    return sorted(set(values))


def benchmark(cfg: Config, concs: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    shared = make_prompt(cfg.shared_prefix, random.Random(cfg.seed + 7)).split() if cfg.shared_prefix else []
    for conc in concs:
        n_prompts = max(args.min_prompts, conc * args.prompt_factor)
        n_prompts = min(n_prompts, args.max_prompts)
        if args.warmup < 0:
            warmup = 0  # explicitly disabled (important at 256K/1M)
        elif args.warmup > 0:
            warmup = args.warmup
        else:
            warmup = min(args.warmup_max, max(2 * conc, 32))
        point_runs: list[dict[str, Any]] = []
        for run_idx in range(1, args.runs + 1):
            rng = random.Random(cfg.seed + conc * 1000 + run_idx)
            prompts = [make_prompt(cfg.isl, rng, shared) for _ in range(n_prompts)]
            run_cfg = Config(**asdict(cfg))
            run_cfg.cache_salt = f"{args.cache_salt_prefix}-c{conc}-r{run_idx}" if args.cache_salt_prefix else ""
            # Cold regime: clear the prefix cache before warmup and again before
            # the measured run, so the measurement cannot inherit a previous
            # point. Warmup still pays for CUDA-graph/allocator warm state.
            if args.cache_regime == "cold" and args.flush_cache:
                flush_cache(cfg.base_url)
            if warmup:
                warm_prompts = prompts[: min(warmup, len(prompts))]
                run_batch_mp(run_cfg, warm_prompts, min(conc, max(1, len(warm_prompts))), args.workers)
            if args.cache_regime == "cold" and args.flush_cache:
                flush_cache(cfg.base_url)
            before = scrape_metrics(cfg.metrics_url) if cfg.metrics_url else {}
            samples, wall = run_batch_mp(run_cfg, prompts, conc, args.workers)
            after = scrape_metrics(cfg.metrics_url) if cfg.metrics_url else {}
            summary = summarize(samples, wall, cfg)
            summary["concurrency"] = conc
            summary["run"] = run_idx
            summary["num_prompts"] = n_prompts
            summary["warmup"] = warmup
            summary["cache_regime"] = args.cache_regime
            summary["prefix_cache"] = cache_stats(before, after)
            errors = [s.error for s in samples if not s.ok][:3]
            if errors:
                summary["error_samples"] = errors
            point_runs.append(summary)
            print(
                f"C={conc:<4} r={run_idx} out={summary['output_tokens_per_s']:>9.1f} tok/s "
                f"total={summary['total_tokens_per_s']:>10.1f} "
                f"ttft p50/p99={summary['ttft_ms']['p50']:>7.1f}/{summary['ttft_ms']['p99']:>8.1f} ms "
                f"tpot p50={summary['tpot_ms']['p50']:>6.2f} ms "
                f"goodput={summary['goodput_req_per_s']:>7.2f} req/s",
                flush=True,
            )
        points.append({
            "concurrency": conc,
            "runs": point_runs,
            "median": median_summary(point_runs),
        })
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "server": {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "endpoint": cfg.endpoint,
            "metrics_url": cfg.metrics_url,
        },
        "workload": {
            "isl": cfg.isl,
            "osl": cfg.osl,
            "shared_prefix": cfg.shared_prefix,
            "cache_regime": args.cache_regime,
            "request_rate": "inf",
            "stream": cfg.stream,
            "ignore_eos": cfg.ignore_eos,
            "runs": args.runs,
            "prompt_factor": args.prompt_factor,
            "min_prompts": args.min_prompts,
            "max_prompts": args.max_prompts,
            "workers": args.workers,
            "slo": {"ttft_ms": cfg.slo_ttft_ms, "tpot_ms": cfg.slo_tpot_ms},
        },
        "points": points,
    }


def write_outputs(result: dict[str, Any], outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "results.json"
    csv_path = outdir / "results.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = [
        "concurrency", "run", "num_prompts", "completed", "failed", "duration_s",
        "output_tokens_per_s", "total_tokens_per_s", "prefill_tokens_per_s",
        "decode_tokens_per_s", "request_throughput",
        "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
        "goodput_req_per_s", "slo_attainment", "cache_hit_rate", "cache_regime",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in result["points"]:
            for run in point["runs"]:
                writer.writerow({
                    "concurrency": run["concurrency"],
                    "run": run["run"],
                    "num_prompts": run["num_prompts"],
                    "completed": run["completed"],
                    "failed": run["failed"],
                    "duration_s": run["duration_s"],
                    "output_tokens_per_s": run["output_tokens_per_s"],
                    "total_tokens_per_s": run["total_tokens_per_s"],
                    "prefill_tokens_per_s": run["prefill_tokens_per_s"],
                    "decode_tokens_per_s": run["decode_tokens_per_s"],
                    "request_throughput": run["request_throughput"],
                    "ttft_p50_ms": run["ttft_ms"]["p50"],
                    "ttft_p99_ms": run["ttft_ms"]["p99"],
                    "tpot_p50_ms": run["tpot_ms"]["p50"],
                    "tpot_p99_ms": run["tpot_ms"]["p99"],
                    "goodput_req_per_s": run["goodput_req_per_s"],
                    "slo_attainment": run["slo_attainment"],
                    "cache_hit_rate": run.get("prefix_cache", {}).get("hit_rate", ""),
                    "cache_regime": run.get("cache_regime", ""),
                })
    return json_path, csv_path


# ---------------------------------------------------------------------------
# self-test: a tiny OpenAI-compatible mock
# ---------------------------------------------------------------------------
class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:  # silence test output
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/metrics":
            body = (
                "# HELP mock\n"
                "vllm:prefix_cache_hits 0\n"
                "vllm:prefix_cache_queries 0\n"
                "vllm:num_requests_running 0\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path in ("/flush_cache", "/reset_prefix_cache"):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        n = int(body.get("max_tokens", 8))
        if not body.get("stream"):
            payload = {
                "choices": [{"text": "tok " * n, "message": {"content": "tok " * n}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": n, "total_tokens": 4 + n},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for _ in range(n):
            chunk = {"choices": [{"index": 0, "text": "tok ", "delta": {"content": "tok "}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.001)
        usage = {"prompt_tokens": 4, "completion_tokens": n, "total_tokens": 4 + n}
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def self_test() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = None
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = Config(
            base_url=f"http://127.0.0.1:{port}",
            model="mock",
            isl=8,
            osl=8,
            timeout=10,
            metrics_url=f"http://127.0.0.1:{port}/metrics",
        )
        samples, wall = run_batch(cfg, [make_prompt(cfg.isl, random.Random(i)) for i in range(4)], 2)
        summary = summarize(samples, wall, cfg)
        assert summary["completed"] == 4, summary
        assert summary["failed"] == 0, summary
        assert summary["output_tokens"] == 32, summary
        assert summary["ttft_ms"]["p50"] > 0, summary
        assert summary["tpot_ms"]["p50"] > 0, summary
        assert percentile([1, 2, 3, 4], 50) == 2.5
        assert percentile([], 99) == 0.0
        print("self-test OK:", json.dumps({k: summary[k] for k in ("completed", "output_tokens_per_s", "ttft_ms", "tpot_ms")}))
        return 0
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="glm53-reap50")
    p.add_argument("--endpoint", default="")
    p.add_argument("--chat", action="store_true", help="use /v1/chat/completions")
    p.add_argument("--api-key", default="")
    p.add_argument("--concurrency", default="1 2 4 8 16 32 64",
                   help="space/comma separated concurrency points")
    p.add_argument("--isl", type=int, default=2048, help="approx. input tokens")
    p.add_argument("--osl", type=int, default=256, help="output tokens per request")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--prompt-factor", type=int, default=5)
    p.add_argument("--min-prompts", type=int, default=200)
    p.add_argument("--max-prompts", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=0,
                   help="0 = auto max(2*C, 32) capped by --warmup-max; -1 = no warmup")
    p.add_argument("--warmup-max", type=int, default=256)
    p.add_argument("--shared-prefix", type=int, default=0,
                   help="tokens shared by every prompt (prefix-cache measurement)")
    p.add_argument("--cache-regime", choices=("cold", "warm-shared"), default="cold")
    p.add_argument("--cache-salt-prefix", default="glm53bench")
    p.add_argument("--flush-cache", dest="flush_cache", action="store_true", default=True)
    p.add_argument("--no-flush-cache", dest="flush_cache", action="store_false")
    p.add_argument("--metrics-url", default="")
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--no-stream-options", dest="stream_options", action="store_false", default=True,
                   help="do not send stream_options.include_usage (for a server that rejects it)")
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=80.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--workers", type=int, default=1,
                   help="multiprocess client workers; use >1 when aggregate >3k tok/s")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    endpoint = args.endpoint or ("/v1/chat/completions" if args.chat else "/v1/completions")
    metrics_url = "" if args.no_metrics else (args.metrics_url or args.base_url.rstrip("/") + "/metrics")
    cfg = Config(
        base_url=args.base_url,
        model=args.model,
        endpoint=endpoint,
        api_key=args.api_key,
        isl=args.isl,
        osl=args.osl,
        timeout=args.timeout,
        stream_options=args.stream_options,
        chat=args.chat,
        cache_salt="",
        shared_prefix=args.shared_prefix,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_tpot_ms=args.slo_tpot_ms,
        metrics_url=metrics_url,
        flush_cache=args.flush_cache,
        seed=args.seed,
    )
    concs = parse_concurrency(args.concurrency)
    if args.dry_run:
        for conc in concs:
            n = min(args.max_prompts, max(args.min_prompts, conc * args.prompt_factor))
            if args.warmup < 0:
                warm = 0
            elif args.warmup > 0:
                warm = args.warmup
            else:
                warm = min(args.warmup_max, max(2 * conc, 32))
            print(f"C={conc:<4} prompts={n:<5} warmup={warm:<4} runs={args.runs}")
        print(f"endpoint={cfg.base_url}{cfg.endpoint} metrics={cfg.metrics_url or '-'} workers={args.workers}")
        return 0
    result = benchmark(cfg, concs, args)
    outdir = Path(args.output_dir) if args.output_dir else Path(f"glm53-bench-{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    json_path, csv_path = write_outputs(result, outdir)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    best = max(
        ((p["median"]["output_tokens_per_s"], p["concurrency"]) for p in result["points"]),
        default=(0.0, 0),
    )
    print(f"peak median output: {best[0]:.1f} tok/s at C={best[1]}")
    completed = sum(run["completed"] for p in result["points"] for run in p["runs"])
    if completed == 0:
        print("no requests completed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
