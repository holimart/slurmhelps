#!/usr/bin/env python3
"""HTTP smoke matrix for the serving gateway.

This is deliberately stdlib-only so it can run from a login node, a Slurm job,
or a CI test. It is *not* a load generator; it verifies the API, streaming,
cache-hit accounting, batching response, limits and auth. Hard GPU/controller
tests (preemption, spill, load ETA) live in the pytest suite.

Usage:
  python tests/serve_smoke.py --base-url http://127.0.0.1:8000 \
      --model glm53-reap50 --contexts "500 800" --concurrency "1 2 4 8"

  # gateway with auth and a queue cap
  python tests/serve_smoke.py ... --api-key "$KEY" --auth-required \
      --max-active 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega one two three four "
    "five six seven eight nine ten red green blue amber violet silver golden "
    "run jump walk read write build test measure serve cache batch token layer"
).split()


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "data": self.data}


def prompt(tokens: int, seed: int, prefix: str = "") -> str:
    rng = random.Random(seed)
    words = prefix.split() if prefix else []
    words.extend(rng.choice(WORDS) for _ in range(max(1, tokens - len(words))))
    return " ".join(words)


def headers(args) -> dict[str, str]:
    out = {"Content-Type": "application/json"}
    if args.api_key:
        out["Authorization"] = "Bearer " + args.api_key
    return out


def post_json(url: str, body: dict, args, timeout: float = 300.0) -> tuple[int, dict, float]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers(args), method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            elapsed = time.perf_counter() - started
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                obj = {"_raw": data.decode("utf-8", "replace")}
            return response.status, obj, elapsed
    except urllib.error.HTTPError as exc:
        data = exc.read()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            obj = {"_raw": data.decode("utf-8", "replace")}
        return exc.code, obj, time.perf_counter() - started


def stream_chat(url: str, body: dict, args, timeout: float = 300.0) -> tuple[int, int, float, bool]:
    body = dict(body, stream=True)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers(args), method="POST"
    )
    started = time.perf_counter()
    first = None
    chunks = 0
    done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return response.status, 0, 0.0, False
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                chunks += 1
                if first is None:
                    first = time.perf_counter() - started
            return response.status, chunks, (first or 0.0), done
    except urllib.error.HTTPError as exc:
        return exc.code, 0, 0.0, False


def check_health(args) -> Result:
    try:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=10) as r:
            return Result("health", "pass", f"HTTP {r.status}")
    except Exception as exc:  # noqa: BLE001 - smoke reports the exception
        return Result("health", "fail", str(exc))


def check_models(args) -> Result:
    try:
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/models", headers=headers(args))
        with urllib.request.urlopen(req, timeout=20) as r:
            obj = json.loads(r.read())
        ids = [m.get("id") for m in obj.get("data", [])]
        return Result("models", "pass", ",".join(x for x in ids if x))
    except Exception as exc:  # noqa: BLE001
        return Result("models", "fail", str(exc))


def chat_body(args, n: int, out: int, prefix: str = "", salt: str = "") -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt(n, 7, prefix)}],
        "max_tokens": out,
        "temperature": 0,
        "stream": False,
    }
    if salt:
        body["cache_salt"] = salt
    return body


def check_chat(args) -> Result:
    code, obj, elapsed = post_json(
        args.base_url.rstrip("/") + "/v1/chat/completions", chat_body(args, args.contexts[0], 32), args
    )
    if code != 200:
        return Result("chat_nonstream", "fail", f"HTTP {code}: {obj}")
    usage = obj.get("usage") or {}
    return Result("chat_nonstream", "pass", f"{elapsed:.3f}s, usage={usage}",
                  {"elapsed_s": elapsed, "usage": usage})


def check_stream(args) -> Result:
    code, chunks, first, done = stream_chat(
        args.base_url.rstrip("/") + "/v1/chat/completions", chat_body(args, args.contexts[0], 32), args
    )
    if code != 200 or chunks == 0 or not done:
        return Result("chat_stream", "fail", f"HTTP {code}, chunks={chunks}, done={done}")
    return Result("chat_stream", "pass", f"{chunks} chunks, first={first:.3f}s",
                  {"chunks": chunks, "first_s": first})


def check_cache(args) -> Result:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    body = chat_body(args, args.contexts[0], 8, salt="smoke-cache")
    code1, obj1, t1 = post_json(url, body, args)
    code2, obj2, t2 = post_json(url, body, args)
    if code1 != 200 or code2 != 200:
        return Result("cache_hit", "fail", f"HTTP {code1}/{code2}")
    u1 = obj1.get("usage") or {}
    u2 = obj2.get("usage") or {}
    c1 = ((u1.get("prompt_tokens_details") or {}).get("cached_tokens"))
    c2 = ((u2.get("prompt_tokens_details") or {}).get("cached_tokens"))
    hit = (c2 or 0) > 0 or (t2 < t1 * 0.8)
    return Result("cache_hit", "pass" if hit else "warn",
                  f"cached={c1}->{c2}, ttft={t1:.3f}->{t2:.3f}s",
                  {"cached_tokens": [c1, c2], "elapsed_s": [t1, t2]})


def check_shared_prefix(args) -> Result:
    prefix = prompt(args.contexts[0], 11)
    outputs = [16, 64, 128, 256, 512]
    results: list[dict] = []
    lock = threading.Lock()

    def one(i: int) -> None:
        body = chat_body(args, args.contexts[0] + 20, outputs[i], prefix=prefix, salt="smoke-shared")
        code, obj, elapsed = post_json(args.base_url.rstrip("/") + "/v1/chat/completions", body, args)
        with lock:
            results.append({"i": i, "code": code, "elapsed": elapsed,
                            "usage": (obj.get("usage") or {})})

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(outputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.timeout)
    ok = len(results) == len(outputs) and all(r["code"] == 200 for r in results)
    if not ok:
        return Result("shared_prefix_concurrent", "fail", json.dumps(results))
    # The shortest answer should normally finish before the longest. Do not
    # fail on timer noise; report it as a warning when it does not.
    by_i = {r["i"]: r for r in results}
    ordered = by_i[0]["elapsed"] <= by_i[4]["elapsed"]
    return Result("shared_prefix_concurrent", "pass" if ordered else "warn",
                  f"short={by_i[0]['elapsed']:.3f}s long={by_i[4]['elapsed']:.3f}s",
                  {"results": results})


def check_concurrency(args) -> Result:
    data = []
    for ctx in args.contexts:
        for conc in args.concurrency:
            bodies = [chat_body(args, ctx, 32, salt=f"smoke-c{ctx}-{conc}-{i}") for i in range(conc)]
            codes: list[int] = []
            lock = threading.Lock()

            def one(body: dict) -> None:
                code, _obj, _elapsed = post_json(
                    args.base_url.rstrip("/") + "/v1/chat/completions", body, args
                )
                with lock:
                    codes.append(code)

            started = time.perf_counter()
            threads = [threading.Thread(target=one, args=(b,)) for b in bodies]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=args.timeout)
            elapsed = time.perf_counter() - started
            ok = len(codes) == conc and all(c == 200 for c in codes)
            data.append({"context": ctx, "concurrency": conc, "ok": ok,
                         "elapsed_s": elapsed, "codes": sorted(codes),
                         "output_tokens_per_s": (conc * 32 / elapsed) if elapsed else 0})
    failed = [d for d in data if not d["ok"]]
    return Result("concurrency_sweep", "fail" if failed else "pass",
                  f"{len(data)} points", {"points": data})


def check_max_len(args) -> Result:
    body = chat_body(args, 8, args.model_max_len + 1)
    code, obj, _ = post_json(args.base_url.rstrip("/") + "/v1/chat/completions", body, args)
    return Result("max_model_len", "pass" if code in (400, 413, 422) else "warn",
                  f"HTTP {code}", {"body": obj})


def check_auth(args) -> Result:
    if not args.auth_required:
        return Result("auth", "skip", "server has no application key")
    bad = argparse.Namespace(**vars(args))
    bad.api_key = "definitely-wrong"
    code, _obj, _ = post_json(
        args.base_url.rstrip("/") + "/v1/chat/completions", chat_body(args, 8, 8), bad
    )
    good_code, _obj, _ = post_json(
        args.base_url.rstrip("/") + "/v1/chat/completions", chat_body(args, 8, 8), args
    )
    return Result("auth", "pass" if code == 401 and good_code == 200 else "fail",
                  f"bad={code} good={good_code}")


def check_queue(args) -> Result:
    if not args.max_active:
        return Result("queue_limit", "skip", "no --max-active configured")
    bodies = [chat_body(args, 64, args.max_active + 4) for _ in range(args.max_active + 1)]
    codes: list[int] = []
    lock = threading.Lock()

    def one(body: dict) -> None:
        code, _obj, _ = post_json(args.base_url.rstrip("/") + "/v1/chat/completions", body, args)
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=one, args=(b,)) for b in bodies]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.timeout)
    return Result("queue_limit", "pass" if codes.count(429) >= 1 else "warn",
                  f"codes={sorted(codes)}")


CHECKS = (
    ("health", check_health),
    ("models", check_models),
    ("chat", check_chat),
    ("stream", check_stream),
    ("cache", check_cache),
    ("shared_prefix", check_shared_prefix),
    ("concurrency", check_concurrency),
    ("max_len", check_max_len),
    ("auth", check_auth),
    ("queue", check_queue),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm53-reap50")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--auth-required", action="store_true")
    parser.add_argument("--contexts", default="500 800",
                        help="space/comma separated prompt-token targets")
    parser.add_argument("--concurrency", default="1 2 4 8")
    parser.add_argument("--model-max-len", "--max-model-len", dest="model_max_len",
                        type=int, default=786432)
    parser.add_argument("--max-active", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--only", default="", help="comma-separated check names")
    args = parser.parse_args(argv)
    args.contexts = [int(x) for x in args.contexts.replace(",", " ").split()]
    args.concurrency = [int(x) for x in args.concurrency.replace(",", " ").split()]
    only = {x for x in args.only.replace(",", " ").split() if x}

    results = []
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        try:
            result = fn(args)
        except Exception as exc:  # noqa: BLE001 - a smoke matrix reports, not crashes
            result = Result(name, "fail", f"{type(exc).__name__}: {exc}")
        results.append(result)
        print(f"{result.status:4s} {result.name:24s} {result.detail}", flush=True)

    report = {"base_url": args.base_url, "model": args.model,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "results": [r.as_dict() for r in results]}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    failed = [r for r in results if r.status == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
