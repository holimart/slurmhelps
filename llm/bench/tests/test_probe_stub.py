"""Stub-server end-to-end test of the canonical probe (llm/bench/probe.py).

Starts llm/strata/tests/stub_server.py on 127.0.0.1, runs the probe in cold and
warm mode, and asserts the JSONL contract plus deterministic cache reuse. No
GPU, no Strata engine, no model, no Slurm.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "llm" / "bench"
PROBE = BENCH / "probe.py"
UPSTREAM = ROOT / "llm" / "strata" / "strata_probe.py"
STUB = ROOT / "llm" / "strata" / "tests" / "stub_server.py"

COLD_KEYS = {
    "ts", "model", "context", "concurrency", "prefill_tok_s",
    "decode_tok_s_p50", "decode_tok_s_p95", "aggregate_tok_s",
    "ttft_s_p50", "ttft_s_p95", "completion_tokens", "prompt_tokens",
    "engine_metrics", "error", "requests_ok", "requests_failed", "samples",
    "wall_s", "max_tokens", "requests_per_slot", "prompt_target_tokens",
    "prompt_clamped", "max_context", "pad_safety", "calibration",
}



def wait_ready(port, timeout=20.0):
    url = "http://127.0.0.1:%d/v1/models" % port
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if b'"id"' in resp.read(1 << 20):
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("stub server never became ready on port %d" % port)


class ProbeStubTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bench-probe-test-")
        cls.stub = subprocess.Popen(
            [sys.executable, str(STUB), "--port", "0",
             "--token-delay-ms", "5", "--prefill-ms-per-token", "0.2",
             "--cache-reuse"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = cls.stub.stdout.readline().strip()
            if not line.startswith("STUB PORT"):
                raise RuntimeError("stub did not announce a port: %r (stderr: %s)"
                                   % (line, cls.stub.stderr.read()[-500:]))
            cls.port = int(line.split()[-1])
            wait_ready(cls.port)
        except Exception:
            cls.stub.kill()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.stub.terminate()
        try:
            cls.stub.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.stub.kill()

    def run_probe(self, name, extra=None):
        out = Path(self.tmp) / name
        cmd = [sys.executable, str(PROBE),
               "--base-url", "http://127.0.0.1:%d" % self.port,
               "--model", "strata-selftest", "--context", "1024",
               "--concurrency", "1", "--max-context", "1024",
               "--pad-safety", "32", "--max-tokens", "8",
               "--requests-per-slot", "1", "--json-out", str(out)]
        cmd.extend(extra or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [line for line in out.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, proc.stderr)
        return json.loads(lines[0])

    def test_canonical_probe_is_a_byte_copy_of_the_strata_probe(self):
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(sha(PROBE), sha(UPSTREAM),
                         "llm/bench/probe.py must stay an identical copy")

    def test_cold_contract_and_clamp(self):
        cell = self.run_probe("cold.jsonl")
        self.assertEqual(set(cell), COLD_KEYS)
        self.assertIsNone(cell["error"])
        self.assertGreater(cell["decode_tok_s_p50"], 0)
        self.assertTrue(cell["prompt_clamped"])
        self.assertEqual(cell["max_context"], 1024)
        self.assertLessEqual(cell["prompt_tokens"], 1024 - 8)
        self.assertIn("chars_per_token", cell["calibration"])

    def test_warm_mode_reuses_the_prompt_cache(self):
        cell = self.run_probe("warm.jsonl",
                              ["--mode", "warm", "--warm-repeats", "3"])
        self.assertEqual(cell["mode"], "warm")
        for key in ("ttft_cold_s", "ttft_warm_s_p50", "ttft_warm_s_p95",
                    "reused_tokens_p50", "reused_tokens_total",
                    "cache_reuse_engaged", "per_request"):
            self.assertIn(key, cell)
        self.assertTrue(cell["cache_reuse_engaged"])
        self.assertGreater(cell["reused_tokens_p50"], 0)
        self.assertLess(cell["ttft_warm_s_p50"], cell["ttft_cold_s"])
        self.assertGreater(cell["ttft_cold_s"] / max(cell["ttft_warm_s_p50"], 1e-9), 1.5)
        self.assertEqual(len(cell["per_request"]), 4)


if __name__ == "__main__":
    unittest.main()
