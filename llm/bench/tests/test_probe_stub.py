"""Stub-server end-to-end test of the canonical probe (llm/bench/probe.py).

Starts llm/strata/tests/stub_server.py on 127.0.0.1, runs the probe in cold and
warm mode, and asserts the JSONL contract plus deterministic cache reuse. No
GPU, no Strata engine, no model, no Slurm.
"""
from __future__ import annotations

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

# The historical cold cell contract.  Every one of these keys and their
# meaning must survive; the strict client metrics are additive (2026-10-26).
LEGACY_COLD_KEYS = {
    "ts", "model", "context", "concurrency", "prefill_tok_s",
    "decode_tok_s_p50", "decode_tok_s_p95", "aggregate_tok_s",
    "ttft_s_p50", "ttft_s_p95", "completion_tokens", "prompt_tokens",
    "engine_metrics", "error", "requests_ok", "requests_failed", "samples",
    "wall_s", "max_tokens", "requests_per_slot", "prompt_target_tokens",
    "prompt_clamped", "max_context", "pad_safety", "calibration",
}
STRICT_KEYS = {
    "decode_s_p50", "decode_s_p95", "tpot_ms_p50", "tpot_ms_p95",
    "decode_tok_s_strict_p50", "decode_tok_s_strict_p95",
}
COLD_KEYS = LEGACY_COLD_KEYS | STRICT_KEYS

CHAIN_KEYS = {
    "stop", "prev_stop", "prompt_tokens", "reused_tokens", "reuse_fraction",
    "ttft_s", "prefill_s", "prefill_delta_tokens", "decode_s", "tpot_ms",
    "decode_tok_s_strict", "decode_tok_s", "aggregate_tok_s",
    "completion_tokens", "timing_source", "status", "error",
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
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="bench-probe-test-")
        cls.addClassCleanup(cls._tmpdir.cleanup)
        cls.tmp = cls._tmpdir.name
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

    def run_probe_lines(self, name, extra=None, context="1024", max_context="1024"):
        out = Path(self.tmp) / name
        cmd = [sys.executable, str(PROBE),
               "--base-url", "http://127.0.0.1:%d" % self.port,
               "--model", "strata-selftest",
               "--concurrency", "1",
               "--pad-safety", "32", "--max-tokens", "8",
               "--requests-per-slot", "1", "--json-out", str(out)]
        if context is not None:
            cmd.extend(["--context", str(context)])
        if max_context is not None:
            cmd.extend(["--max-context", str(max_context)])
        cmd.extend(extra or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [line for line in out.read_text().splitlines() if line.strip()]
        self.assertTrue(lines, proc.stderr)
        return [json.loads(line) for line in lines]

    def run_probe(self, name, extra=None):
        lines = self.run_probe_lines(name, extra)
        self.assertEqual(len(lines), 1)
        return lines[0]

    def test_canonical_probe_is_the_superset_of_the_frozen_strata_probe(self):
        """llm/bench/probe.py is canonical as of 2026-10-26.

        The upstream copy stays frozen while the V100 campaign runs, so the
        old byte-copy assertion is replaced by two invariants: both files
        exist, and every historical cold key is still emitted by the bench
        probe (checked by test_cold_contract_and_clamp).
        """
        self.assertTrue(PROBE.is_file())
        self.assertTrue(UPSTREAM.is_file())
        self.assertIn("chain", PROBE.read_text())
        # The frozen upstream predates --mode; it must not be silently edited.
        self.assertNotIn("--mode chain", UPSTREAM.read_text())

    def test_cold_contract_and_clamp(self):
        cell = self.run_probe("cold.jsonl")
        self.assertEqual(set(cell), COLD_KEYS)
        self.assertTrue(LEGACY_COLD_KEYS.issubset(cell))
        self.assertIsNone(cell["error"])
        self.assertGreater(cell["decode_tok_s_p50"], 0)
        self.assertGreater(cell["decode_tok_s_strict_p50"], 0)
        self.assertGreater(cell["tpot_ms_p50"], 0)
        self.assertGreater(cell["decode_s_p50"], 0)
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

    def test_chain_mode_grows_the_prefix_and_reuses_the_checkpoint(self):
        stops = [512, 1024, 2048, 4056]
        turns = self.run_probe_lines(
            "chain.jsonl",
            ["--mode", "chain", "--stops", ",".join(str(s) for s in stops),
             "--tokens-per-turn", "8", "--max-context", "4096"],
            context=None)
        self.assertEqual(len(turns), len(stops), turns)
        for i, turn in enumerate(turns):
            self.assertEqual(turn["mode"], "chain")
            self.assertEqual(turn["status"], "ok", turn.get("error"))
            self.assertEqual(turn["stop"], stops[i])
            self.assertEqual(turn["prev_stop"], stops[i - 1] if i else None)
            self.assertTrue(CHAIN_KEYS.issubset(turn), CHAIN_KEYS - set(turn))
            self.assertGreater(turn["prompt_tokens"], 0)
            self.assertGreater(turn["decode_tok_s_strict"], 0)
            self.assertGreater(turn["tpot_ms"], 0)
            self.assertGreater(turn["aggregate_tok_s"], 0)
            self.assertEqual(turn["completion_tokens"], 8)

        # Turn 1 is a fresh conversation: no engine reuse.
        self.assertEqual(turns[0]["reused_tokens"], 0)
        # Every growth turn reuses approximately the previous full prompt and
        # prefills only the delta.
        for i in range(1, len(turns)):
            prev = turns[i - 1]
            cur = turns[i]
            self.assertGreater(cur["reused_tokens"], prev["reused_tokens"])
            self.assertAlmostEqual(cur["reused_tokens"], prev["prompt_tokens"],
                                   delta=8,
                                   msg="reused != previous prompt: %r" % cur)
            self.assertEqual(cur["prefill_delta_tokens"],
                             cur["prompt_tokens"] - cur["reused_tokens"])
            expected_delta = cur["prompt_tokens"] - prev["prompt_tokens"]
            self.assertLessEqual(abs(cur["prefill_delta_tokens"] - expected_delta),
                                 8, cur)
            self.assertLess(cur["prefill_delta_tokens"],
                            cur["prompt_tokens"] * 0.75)
            self.assertGreater(cur["reuse_fraction"], 0.0)

        # The stub keeps the request bodies, so prove the prompt shape directly.
        with urllib.request.urlopen(
                "http://127.0.0.1:%d/debug/state" % self.port, timeout=5) as resp:
            state = json.loads(resp.read().decode("utf-8"))
        texts = [r["text"] for r in state["requests"]][-len(stops):]
        self.assertEqual(len(texts), len(stops))
        for earlier, later in zip(texts, texts[1:]):
            self.assertTrue(later.startswith(earlier),
                            "chain turn is not a prefix extension")


if __name__ == "__main__":
    unittest.main()
