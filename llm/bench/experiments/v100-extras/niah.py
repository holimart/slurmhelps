#!/usr/bin/env python3
"""niah.py — needle-in-a-haystack + tool-call canary for Strata (H1).

One-off quality probe for the 2026-10-27 V100 extras.  It deliberately
reuses the canonical benchmark machinery instead of inventing a new contract:

  * tokenizer/char calibration: strata_probe.calibrate_chars_per_token()
  * the same unique-salt cold prompt rule (every repeat is a new document)
  * the same /v1/chat/completions streaming client shape and TTFT definition
    (dispatch -> first content/reasoning/tool delta)
  * the same engine /metrics digest (strata_probe.metrics_digest/raw)
  * one JSON line per request + summary.csv, and the runner's engine logs are
    archived by lib.sh around the server.

It does NOT share stream_one() because grading needs the answer text and tool
calls; that is the only copied code, and it is the same SSE parser.

Prompt shape (recorded verbatim in each JSONL line):

    [document <salt>]
    <filler up to the chosen fraction>
    The access code for cabinet BX-7741 is KITE-9Q7M.
    <remaining filler>
    Question: What is the access code for cabinet BX-7741? Answer with the code only.

The needle sits at 10/50/90 % of the filler characters, each position is
repeated N times with a fresh salt (both repeats cold), and one tool-call
canary per context asks the model to call `read_file` on /etc/hostname.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
STRATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "strata"))
sys.path.insert(0, STRATA)
import strata_probe as sp  # noqa: E402  (canonical calibration/metrics/percentile)

NEEDLE = "The access code for cabinet BX-7741 is KITE-9Q7M."
KEY = "KITE-9Q7M"
QUESTION = ("\n\nQuestion: What is the access code for cabinet BX-7741? "
            "Answer with the code only.")
CORPUS = ("The maintenance log records routine inspections of the north corridor. "
          "Nothing unusual was reported by the night shift. The ventilation "
          "readings stayed inside the usual band. ")
TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string",
                                    "description": "absolute path to read"}},
            "required": ["path"],
        },
    },
}]


def corpus(nchars):
    reps = nchars // len(CORPUS) + 1
    return (CORPUS * reps)[:nchars]


def build_prompt(chars, fraction, salt):
    filler = corpus(chars)
    pos = int(len(filler) * fraction / 100.0)
    return ("[document %s]\n\n%s\n\n%s\n\n%s%s"
            % (salt, filler[:pos], NEEDLE, filler[pos:], QUESTION))


def stream(base, model, body, timeout):
    """SSE chat completion; returns text, tool calls, timings, usage."""
    rec = {"ok": False, "error": None, "ttft_s": None, "total_s": None,
           "prompt_tokens": None, "completion_tokens": None,
           "decode_tok_s": None, "text": "", "reasoning": "",
           "tool_calls": [], "finish_reason": None, "timing_source": None}
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"})
    t0 = time.monotonic()
    t_first = None
    chunks = 0
    usage = None
    timings = None
    parts, reasoning = [], []
    tools = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                if isinstance(chunk.get("timings"), dict):
                    timings = chunk["timings"]
                for ch in chunk.get("choices") or []:
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("reasoning"))
                    tcs = delta.get("tool_calls") or []
                    if (piece or tcs) and t_first is None:
                        t_first = time.monotonic() - t0
                    if piece:
                        chunks += 1
                        if delta.get("reasoning_content") or delta.get("reasoning"):
                            reasoning.append(piece)
                        else:
                            parts.append(piece)
                    for tc in tcs:
                        idx = tc.get("index", 0)
                        slot = tools.setdefault(idx, {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(4096).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            detail = ""
        rec["error"] = "HTTPError %s: %s" % (exc.code, detail[:400])
        rec["total_s"] = time.monotonic() - t0
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["total_s"] = time.monotonic() - t0
        return rec

    rec["total_s"] = time.monotonic() - t0
    rec["ttft_s"] = t_first
    rec["text"] = "".join(parts)
    rec["reasoning"] = "".join(reasoning)
    rec["tool_calls"] = [tools[i] for i in sorted(tools)]
    if usage:
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
    if rec["completion_tokens"] is None:
        rec["completion_tokens"] = chunks
    pred = timings.get("predicted_ms") if isinstance(timings, dict) else None
    if pred and rec["completion_tokens"]:
        rec["decode_tok_s"] = rec["completion_tokens"] / (pred / 1000.0)
        rec["timing_source"] = "engine.timings.predicted_ms"
    elif t_first is not None and rec["completion_tokens"] and rec["total_s"] > t_first:
        rec["decode_tok_s"] = rec["completion_tokens"] / (rec["total_s"] - t_first)
        rec["timing_source"] = "client.stream-window"
    rec["ok"] = bool((parts or reasoning or tools) and rec["completion_tokens"])
    return rec


def ask_needle(base, model, ctx, fraction, repeat, chars, max_tokens, timeout):
    salt = uuid.uuid4().hex[:12]
    prompt = build_prompt(chars, fraction, salt)
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    r = stream(base, model, body, timeout)
    answer = (r["text"] or "").strip()
    tail = answer[-300:]
    correct = KEY.lower() in answer.lower()
    return {"kind": "needle", "context": ctx, "position": fraction,
            "repeat": repeat, "salt": salt, "expected": KEY,
            "answer_tail": tail, "correct": correct,
            "prompt_chars": len(prompt), **r}


def ask_tool(base, model, ctx, chars, timeout):
    salt = uuid.uuid4().hex[:12]
    prompt = ("[document %s]\n\n%s\n\nUse the read_file tool to read the file "
              "/etc/hostname. You must call the tool; do not answer in prose."
              % (salt, corpus(chars)))
    out = {"kind": "tool", "context": ctx, "position": None, "repeat": 1,
           "salt": salt, "expected": "read_file",
           "tool_choice_used": None, "correct": False}
    for choice in ("required", "auto"):
        body = {"model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": TOOLS, "tool_choice": choice,
                "max_tokens": 128, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        r = stream(base, model, body, timeout)
        out.update(r)
        out["tool_choice_used"] = choice
        if r["error"] and "HTTPError 400" in (r["error"] or "") and choice == "required":
            out["required_error"] = r["error"]
            continue
        names = [tc.get("name") for tc in r["tool_calls"]]
        out["tool_names"] = names
        out["tool_arguments"] = [tc.get("arguments") for tc in r["tool_calls"]]
        out["correct"] = any(n == "read_file" for n in names)
        break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="65536,131072,262144")
    ap.add_argument("--positions", default="10,50,90")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--pad-safety", type=int, default=128)
    ap.add_argument("--overhead-tokens", type=int, default=384,
                    help="tokens reserved for the needle/question/header")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--metrics-timeout", type=float, default=15.0)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--metrics-out", default="")
    ap.add_argument("--tool-canary", action="store_true")
    args = ap.parse_args()

    base = sp.chat_root(args.base_url)
    root = sp.host_root(args.base_url)
    cal = sp.calibrate_chars_per_token(base, args.model, args.temperature,
                                       args.timeout, args.max_context)
    cpt = cal.get("chars_per_token") or 4.0
    intercept = cal.get("intercept_tokens") or 0.0
    print("[niah] calibration chars/token=%.3f intercept=%.0f" % (cpt, intercept),
          file=sys.stderr)

    positions = [float(x) for x in args.positions.split(",") if x.strip()]
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    rows = []
    with open(args.json_out, "a") as fh:
        for ctx in contexts:
            budget = min(ctx, args.max_context) - args.max_tokens - args.pad_safety
            target_chars = max(1024, int((budget - args.overhead_tokens - intercept) * cpt))
            before = sp.metrics_digest(root, args.metrics_timeout)
            for frac in positions:
                for rep in range(1, args.repeats + 1):
                    row = ask_needle(base, args.model, ctx, frac, rep,
                                     target_chars, args.max_tokens, args.timeout)
                    fh.write(json.dumps(row, sort_keys=False) + "\n"); fh.flush()
                    rows.append(row)
                    print("[niah] ctx=%d pos=%.0f%% rep=%d ok=%s correct=%s "
                          "prompt=%s ttft=%s dec=%s"
                          % (ctx, frac, rep, row["ok"], row["correct"],
                             row["prompt_tokens"],
                             "%.2f" % row["ttft_s"] if row["ttft_s"] else "-",
                             "%.1f" % row["decode_tok_s"] if row["decode_tok_s"] else "-"),
                          file=sys.stderr)
            if args.tool_canary:
                row = ask_tool(base, args.model, ctx, target_chars, args.timeout)
                fh.write(json.dumps(row, sort_keys=False) + "\n"); fh.flush()
                rows.append(row)
                print("[niah] ctx=%d tool ok=%s correct=%s names=%s choice=%s err=%s"
                      % (ctx, row["ok"], row["correct"],
                         row.get("tool_names"), row.get("tool_choice_used"),
                         (row.get("error") or "")[:120]), file=sys.stderr)
            if args.metrics_out:
                after = sp.metrics_digest(root, args.metrics_timeout)
                with open(args.metrics_out, "a") as mf:
                    mf.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
                                         "context": ctx, "before": before,
                                         "after": after}) + "\n")

    with open(args.summary_out, "w") as sf:
        sf.write("ctx,kind,position,repeat,prompt_tokens,ttft_s,decode_tok_s,"
                 "correct,answer_tail,error\n")
        for r in rows:
            def num(v):
                return "%.3f" % v if isinstance(v, (int, float)) else ""
            tail = (r.get("answer_tail") or "").replace(",", " ").replace("\n", " ")[:200]
            err = (r.get("error") or "").replace(",", " ").replace("\n", " ")[:200]
            sf.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                r["context"], r["kind"], r.get("position", ""), r.get("repeat", ""),
                r.get("prompt_tokens") or "", num(r.get("ttft_s")),
                num(r.get("decode_tok_s")), r.get("correct"), tail, err))

    acc = {}
    for r in rows:
        key = (r["context"], r["kind"])
        a = acc.setdefault(key, [0, 0])
        a[0] += 1
        a[1] += 1 if r.get("correct") else 0
    print("[niah] accuracy:")
    for (ctx, kind), (n, c) in sorted(acc.items()):
        print("  ctx=%d %-6s %d/%d" % (ctx, kind, c, n))


if __name__ == "__main__":
    main()
