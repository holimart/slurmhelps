#!/usr/bin/env python3
"""Bounded N-turn tool loop (reported: 400s/malformed calls grow with session length)."""
import json, sys, re
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe as p
model = sys.argv[1]; N = int(sys.argv[2]); out = sys.argv[3]
tools = [{"type": "function", "function": {"name": "read_chunk", "description": "Read chunk i of a log; returns text and the next index or DONE",
          "parameters": {"type": "object", "properties": {"i": {"type": "integer"}}, "required": ["i"]}}}]
msgs = [{"role": "user", "content": "Read chunks 0,1,2,... one at a time with read_chunk until it says DONE. Then report the total number of chunks. Never stop early."}]
log = []; bad = 0
for turn in range(N):
    s, r, dt = p.call({"model": model, "messages": msgs, "tools": tools, "max_tokens": 800, "temperature": 0})
    if s != 200:
        log.append({"turn": turn, "status": s, "err": str(r)[:300]}); break
    m = r["choices"][0]["message"]; fr = r["choices"][0]["finish_reason"]
    leaked = bool(p.BAD.search(json.dumps(m)))
    tc = m.get("tool_calls") or []
    entry = {"turn": turn, "dt": round(dt, 1), "finish": fr, "n_tc": len(tc), "leaked": leaked,
             "prompt_tokens": r["usage"]["prompt_tokens"], "content": (m.get("content") or "")[:100]}
    for t in tc:
        try: json.loads(t["function"]["arguments"])
        except Exception: entry["bad_args"] = t["function"]["arguments"][:100]; bad += 1
    log.append(entry)
    if not tc: break
    msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tc})
    for t in tc:
        try: i = json.loads(t["function"]["arguments"]).get("i", 0)
        except Exception: i = 0
        entry.setdefault('arg_types', []).append(type(i).__name__)
        try: i = int(i)
        except Exception: i = 0
        body = ("chunk %d: " % i) + ("lorem ipsum dolor sit amet " * 200)
        body += " NEXT=%d" % (i + 1) if i < N - 6 else " DONE total=%d" % (i + 1)
        msgs.append({"role": "tool", "tool_call_id": t["id"], "content": body})
res = {"model": model, "turns": len(log), "bad_args": bad, "leaks": sum(1 for e in log if e.get("leaked")),
       "errors": [e for e in log if "status" in e], "last": log[-3:], "final": msgs[-1] if msgs else None, "log": log}
open(out, "w").write(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k not in ("log", "final")})
