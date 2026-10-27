"""Tests for history scrubbing and output rewriting (no GPU, loopback only)."""

from __future__ import annotations

import http.client
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from slurmhelps.serving.gateway import GatewayServer
from slurmhelps.serving.rewrite import Rewriter, load_rules

URL = ("https://routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com/proxy_temp_file/production/"
       "2026-10-25/trace_x/requestId_y/abc.08946?Expires=1&OSSAccessKeyId=LTAI5tKoG9A3DkwGD635QVZr&Signature=z%3D")
BLOCKED = "https://blocked.invalid/removed-url"


def test_rewrite_text_replaces_urls_hosts_and_keys():
    rw = Rewriter()
    assert rw.rewrite_text(f"see {URL} now") == f"see {BLOCKED} now"
    assert rw.rewrite_text("https://routified.com/0/https://arxiv.org/abs/1") == BLOCKED
    assert rw.rewrite_text("a bizarre aliyuncs proxy") == "a bizarre removed-host proxy"
    assert rw.rewrite_text("key LTAI5tA4qZ9rEj6S26c4vMqE end") == "key REMOVED_KEY end"
    for clean in ("https://arxiv.org/abs/2408.08946", "git checkout router.py", "route planning"):
        assert rw.rewrite_text(clean) == clean
    assert "alibaba" not in BLOCKED and rw.counts["response"] == 4


def test_rewrite_keeps_tool_call_arguments_valid_json():
    rw = Rewriter()
    args = json.dumps({"url": URL, "max_chars": 4000})
    out = rw.rewrite_text(args)
    assert json.loads(out) == {"url": BLOCKED, "max_chars": 4000}


def test_scrub_request_rewrites_history_not_tools():
    rw = Rewriter()
    payload = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "reasoning": f"use {URL}",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "web_fetch", "arguments": json.dumps({"url": URL})}}]},
            {"role": "tool", "tool_call_id": "1", "content": [{"type": "text", "text": f"error for {URL}"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "web_fetch", "description": "aliyun docs"}}],
    }
    rw.scrub_request(payload)
    text = json.dumps(payload["messages"])
    assert "aliyuncs" not in text and "LTAI" not in text and BLOCKED in text
    assert payload["tools"][0]["function"]["description"] == "aliyun docs"
    assert rw.counts["request"] == 3


def _chat_chunks(pieces, field="content", finish=True):
    chunks = [{"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
               "choices": [{"index": 0, "delta": {field: p}, "finish_reason": None}]} for p in pieces]
    if finish:
        chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return chunks


def _collect(chunks, rw, field="content"):
    st = rw.stream()
    out = []
    for c in chunks:
        c = st.chunk(c)
        for ch in c["choices"]:
            d = ch.get("delta", ch)
            out.append(d.get(field, "") or "")
    for c in st.drain():
        for ch in c["choices"]:
            out.append(ch.get("delta", ch).get(field, "") or "")
    return "".join(out)


@pytest.mark.parametrize("seed", range(20))
def test_stream_split_anywhere_is_rewritten(seed):
    text = f"Fetching {URL}\nthen https://arxiv.org/abs/2308.07305 and done."
    rnd = random.Random(seed)
    cuts = sorted(rnd.sample(range(1, len(text)), 25))
    pieces = [text[i:j] for i, j in zip([0] + cuts, cuts + [len(text)])]
    for field in ("content", "reasoning"):
        got = _collect(_chat_chunks(pieces, field), Rewriter(), field)
        assert got == f"Fetching {BLOCKED}\nthen https://arxiv.org/abs/2308.07305 and done."


def test_stream_releases_clean_text_before_finish():
    st = Rewriter().stream()
    first = st.chunk(_chat_chunks(["Hello world, now "], finish=False)[0])
    assert first["choices"][0]["delta"]["content"] == "Hello world, now "
    held = st.chunk(_chat_chunks(["https://rout"], finish=False)[0])
    assert held["choices"][0]["delta"]["content"] == ""


def test_stream_tool_call_arguments_and_flush_on_finish():
    args = json.dumps({"url": URL})
    pieces = [args[i:i + 7] for i in range(0, len(args), 7)]
    st = Rewriter().stream()
    acc = ""
    for p in pieces:
        c = st.chunk({"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": None,
                      "delta": {"tool_calls": [{"index": 0, "function": {"arguments": p}}]}}]})
        acc += c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    fin = st.chunk({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    for call in fin["choices"][0]["delta"].get("tool_calls", []):
        acc += call["function"]["arguments"]
    assert json.loads(acc) == {"url": BLOCKED}


def test_stream_drain_without_finish_and_text_completion():
    st = Rewriter().stream()
    st.chunk({"object": "text_completion", "id": "x", "choices": [{"index": 0, "text": "go https://routify", "finish_reason": None}]})
    out = st.drain()
    assert out and out[0]["choices"][0]["text"] == BLOCKED


def test_load_rules_file(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rules": [{"name": "x", "pattern": r"foo\w*", "replacement": "bar"}]}))
    rw = Rewriter(load_rules(str(f)))
    assert rw.rewrite_text("a fooey b") == "a bar b"
    f.write_text("[]")
    with pytest.raises(ValueError):
        load_rules(str(f))


# -- gateway end to end against a scripted upstream ------------------------------------

class _Upstream:
    def __init__(self, sse_pieces=None, json_text=None):
        bodies = self.bodies = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                return

            def do_POST(self):  # noqa: N802
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if outer.sse_pieces is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for c in _chat_chunks(outer.sse_pieces):
                        self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    raw = json.dumps({"choices": [{"index": 0, "message": {"content": outer.json_text}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

        self.sse_pieces, self.json_text = sse_pieces, json_text
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _post(base, path, payload):
    s = urlsplit(base)
    conn = http.client.HTTPConnection(s.hostname, s.port, timeout=10)
    conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    return r.status, raw


def test_gateway_rewrites_json_and_scrubs_history(tmp_path):
    up = _Upstream(json_text=f"try {URL}")
    log = tmp_path / "rw.jsonl"
    gw = GatewayServer(up.base, rewriter=Rewriter(log_path=str(log)), bad_words=["LTAI5tKoG9A3DkwGD635QVZr"]).start()
    try:
        status, raw = _post(gw.base_url, "/v1/chat/completions",
                            {"model": "m", "messages": [{"role": "assistant", "content": URL}]})
        assert status == 200
        assert json.loads(raw)["choices"][0]["message"]["content"] == f"try {BLOCKED}"
        sent = up.bodies[0]
        assert sent["messages"][0]["content"] == BLOCKED and sent["bad_words"] == ["LTAI5tKoG9A3DkwGD635QVZr"]
        lines = [json.loads(x) for x in log.read_text().splitlines()]
        assert {x["where"] for x in lines} == {"request", "response"}
        assert gw.status()["rewrites"] == {"request": 1, "response": 1}
    finally:
        gw.stop()
        up.stop()


def test_gateway_rewrites_sse_stream():
    text = f"Next {URL} ok"
    up = _Upstream(sse_pieces=[text[i:i + 5] for i in range(0, len(text), 5)])
    gw = GatewayServer(up.base, rewriter=Rewriter()).start()
    try:
        status, raw = _post(gw.base_url, "/v1/chat/completions", {"model": "m", "messages": [], "stream": True})
        assert status == 200
        got = ""
        for line in raw.decode().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:])["choices"]:
                    got += ch["delta"].get("content", "") or ""
        assert got == f"Next {BLOCKED} ok"
        assert raw.decode().rstrip().endswith("data: [DONE]")
    finally:
        gw.stop()
        up.stop()


def test_qwen38_rules_file_matches_builtin_defaults():
    from pathlib import Path

    from slurmhelps.serving.rewrite import default_rules

    path = Path(__file__).resolve().parents[1] / "llm" / "qwen38" / "rewrite_rules.json"
    got = [(r.name, r.regex.pattern, r.replacement) for r in load_rules(str(path))]
    assert got == [(r.name, r.regex.pattern, r.replacement) for r in default_rules()]
    assert all("alibaba" not in r.replacement.lower() for r in load_rules(str(path)))
