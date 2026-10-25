#!/usr/bin/env python3
"""Stub-server test for strata_probe warm mode (no GPU, no Slurm, no Strata).

Proves three things:
  1. the default (cold) JSONL contract is unchanged: exactly the same key set
     as before warm mode existed, and no warm-only field;
  2. --mode warm emits the extra fields (mode, ttft_cold_s, warm p50/p95,
     reused_tokens_p50, per_request, conversation_cache, ...);
  3. repeats produce a deterministic TTFT drop and engine-reported reuse:
     tests/stub_server.py --cache-reuse reports reused=prompt_tokens-1 and
     prefill x 0.2 for a byte-identical prompt.

Run: python3 llm/strata/tests/test_warm_probe.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "strata_probe.py")
STUB = os.path.join(HERE, "stub_server.py")

# The pre-warm-mode cold contract.  If this set changes, cold mode changed.
COLD_KEYS = {
    "ts", "model", "context", "concurrency", "prefill_tok_s",
    "decode_tok_s_p50", "decode_tok_s_p95", "aggregate_tok_s",
    "ttft_s_p50", "ttft_s_p95", "completion_tokens", "prompt_tokens",
    "engine_metrics", "error", "requests_ok", "requests_failed", "samples",
    "wall_s", "max_tokens", "requests_per_slot", "prompt_target_tokens",
    "prompt_clamped", "max_context", "pad_safety", "calibration",
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_ready(port, timeout=20.0):
    url = "http://127.0.0.1:%d/v1/models" % port
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if b'"id"' in r.read(1 << 20):
                    return
        except Exception:  # noqa: BLE001 - retry
            pass
        time.sleep(0.1)
    raise SystemExit("stub server never became ready on port %d" % port)


def run_probe(port, out, extra):
    cmd = [sys.executable, PROBE, "--base-url", "http://127.0.0.1:%d" % port,
           "--model", "strata-selftest", "--context", "1024",
           "--concurrency", "1", "--max-context", "8192",
           "--max-tokens", "8", "--json-out", out] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lines = [l for l in open(out) if l.strip()] if os.path.exists(out) else []
    if not lines:
        raise SystemExit("probe wrote no JSONL\nstdout:\n%s\nstderr:\n%s"
                         % (proc.stdout, proc.stderr))
    return json.loads(lines[-1]), proc


def main():
    tmp = tempfile.mkdtemp(prefix="strata-warm-test-")
    port = free_port()
    stub = subprocess.Popen(
        [sys.executable, STUB, "--port", str(port),
         "--token-delay-ms", "5", "--prefill-ms-per-token", "0.2",
         "--cache-reuse"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_ready(port)
        rc = 0

        cold, p1 = run_probe(port, os.path.join(tmp, "cold.jsonl"), [])
        explicit, p2 = run_probe(port, os.path.join(tmp, "cold-explicit.jsonl"),
                                 ["--mode", "cold"])
        warm, p3 = run_probe(port, os.path.join(tmp, "warm.jsonl"),
                             ["--mode", "warm", "--warm-repeats", "3"])

        def check(name, cond, detail=""):
            nonlocal rc
            print("  %-52s %s%s" % (name, "ok" if cond else "FAIL",
                                    (" (" + detail + ")") if detail and not cond else ""))
            if not cond:
                rc = 1

        print("[warm-test] default cold JSONL contract")
        check("default mode writes the pre-warm key set exactly",
              set(cold) == COLD_KEYS,
              "missing=%s extra=%s" % (sorted(COLD_KEYS - set(cold)),
                                       sorted(set(cold) - COLD_KEYS)))
        check("explicit --mode cold is identical to the default",
              set(explicit) == set(cold))
        check("cold mode has no warm-only fields",
              not any(k in cold for k in
                      ("mode", "ttft_cold_s", "ttft_warm_s_p50",
                       "reused_tokens_p50", "per_request")))
        check("cold requests are salted (no reuse in stub cache)",
              all(r.get("reused") == 0
                  for r in (cold.get("engine_metrics", {}).get("after", {})
                            .get("requests") or [])))
        check("cold probe exit 0", p1.returncode == 0 and p2.returncode == 0)

        print("[warm-test] warm mode extra fields")
        for key in ("mode", "warm_repeats", "ttft_cold_s", "ttft_warm_s_p50",
                    "ttft_warm_s_p95", "reused_tokens_p50",
                    "reused_tokens_total", "cache_reuse_engaged",
                    "conversation_cache_before", "conversation_cache_after",
                    "conversation_cache_budget_mib", "conversation_cache_slots",
                    "repeat_breakdown", "per_request", "warm_ttft_speedup"):
            check("field %s present" % key, key in warm)
        check("mode == warm", warm.get("mode") == "warm")
        check("warm_repeats == 3", warm.get("warm_repeats") == 3)
        check("1 cold + 3 warm requests recorded",
              len(warm.get("per_request") or []) == 4,
              "got %s" % len(warm.get("per_request") or []))
        check("request_index is 1..4",
              [r.get("request_index") for r in warm.get("per_request") or []]
              == [1, 2, 3, 4])
        check("repeat breakdown roles",
              [r.get("role") for r in warm.get("repeat_breakdown") or []]
              == ["cold", "warm", "warm", "warm"])
        check("cache budget reported (8192 MiB / 4 slots)",
              warm.get("conversation_cache_budget_mib") == 8192
              and warm.get("conversation_cache_slots") == 4)

        print("[warm-test] deterministic stub TTFT drop on repeats")
        rows = warm.get("per_request") or []
        check("first repeat (cold) reports no reuse",
              rows[0].get("reused_tokens") == 0,
              "got %s" % rows[0].get("reused_tokens"))
        check("warm repeats report engine reuse > 0",
              all((r.get("reused_tokens") or 0) > 0 for r in rows[1:]),
              "got %s" % [r.get("reused_tokens") for r in rows[1:]])
        check("reused_tokens_p50 > 0", (warm.get("reused_tokens_p50") or 0) > 0)
        check("cache_reuse_engaged is true",
              warm.get("cache_reuse_engaged") is True)
        cold_t = warm.get("ttft_cold_s")
        warm_t = warm.get("ttft_warm_s_p50")
        check("warm p50 TTFT at most half the cold TTFT",
              cold_t and warm_t and warm_t <= cold_t * 0.5,
              "cold=%s warm=%s" % (cold_t, warm_t))
        check("warm speedup > 1.5x",
              (warm.get("warm_ttft_speedup") or 0) > 1.5,
              "got %s" % warm.get("warm_ttft_speedup"))
        check("warm probe exit 0", p3.returncode == 0)

        print("[warm-test] %s (scratch %s)" % ("PASS" if rc == 0 else "FAIL", tmp))
        return rc
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()


if __name__ == "__main__":
    sys.exit(main())
