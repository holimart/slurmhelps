#!/usr/bin/env python3
"""Bounded API probes for deepseek-v4.1-flash on Ollama Cloud (OpenAI-compatible /v1).

Checks failure modes reported for DeepSeek V4 in agent harnesses:
  P1 tool call returns structured tool_calls (not DSML/XML in content)
  P2 multi-turn with tool result: with / without reasoning echo -> 400?
  P3 parallel tool calls
  P4 needle-in-haystack at ~N tokens (context honoured, not truncated)
  P5 streaming: finish_reason + tool-call deltas assembled
  P6 stray URLs/IPs/leaked special tokens in outputs
Key is read from OpenCode auth.json (never printed). ~12 requests total.
"""
import json, os, re, sys, time, urllib.request, urllib.error

AUTH = os.path.expanduser("~/.local/share/opencode/auth.json")
KEY = json.load(open(AUTH))["ollama-cloud"]["key"]
BASE = "https://ollama.com/v1"
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["deepseek-v4.1-flash:1017"]
HAY_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
OUT = sys.argv[3] if len(sys.argv) > 3 else ""

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
BAD = re.compile(r"(DSML|<｜|<\|tool|<tool_call>|function_calls|invoke name=)")
URL = re.compile(r"https?://[^\s\"'<>)]+|\b\d{1,3}(?:\.\d{1,3}){3}\b|aliyun\w*|alicdn|\.cn\b", re.I)


def call(body, stream=False, timeout=180):
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                return r.status, json.loads(r.read()), time.time() - t
            chunks = []
            for line in r:
                line = line.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    chunks.append(json.loads(line[5:]))
            return r.status, chunks, time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500], time.time() - t
    except Exception as e:  # timeout etc
        return 0, repr(e), time.time() - t


def msg(resp):
    return resp["choices"][0]["message"], resp["choices"][0]["finish_reason"]


def run(model):
    R = {"model": model}
    base = {"model": model, "max_tokens": 1500, "temperature": 0}
    q = [{"role": "user", "content": "What is the weather in Prague? Use the tool."}]
    # P1
    s, r, dt = call({**base, "messages": q, "tools": TOOLS})
    R["P1"] = {"status": s, "dt": round(dt, 1)}
    if s == 200:
        m, fr = msg(r)
        R["P1"].update(finish=fr, tool_calls=m.get("tool_calls"), content=(m.get("content") or "")[:200],
                       reasoning_keys=[k for k in m if "reason" in k or k == "thinking"],
                       leaked=bool(BAD.search(json.dumps(m))))
        # P2: continue with tool result, with and without reasoning echo
        tc = m.get("tool_calls") or []
        if tc:
            asst = {"role": "assistant", "content": m.get("content") or "", "tool_calls": tc}
            tool = [{"role": "tool", "tool_call_id": t["id"], "content": '{"temp_c": 7, "sky": "rain"}'} for t in tc]
            rc = m.get("reasoning_content") or m.get("reasoning") or ""
            for name, a in (("no_echo", asst), ("echo", {**asst, "reasoning_content": rc, "reasoning": rc})):
                s2, r2, dt2 = call({**base, "messages": q + [a] + tool, "tools": TOOLS})
                d = {"status": s2, "dt": round(dt2, 1)}
                if s2 == 200:
                    m2, fr2 = msg(r2)
                    d.update(finish=fr2, content=(m2.get("content") or "")[:200], leaked=bool(BAD.search(json.dumps(m2))))
                else:
                    d["err"] = r2 if isinstance(r2, str) else json.dumps(r2)[:300]
                R["P2_" + name] = d
        R["P1"]["has_tool_calls"] = bool(tc)
    else:
        R["P1"]["err"] = r
    # P3 parallel
    s, r, dt = call({**base, "tools": TOOLS, "messages": [{"role": "user", "content":
                    "Get the weather for Prague, Brno and Ostrava. Call the tool once per city, all in one turn."}]})
    R["P3"] = {"status": s, "dt": round(dt, 1)}
    if s == 200:
        m, fr = msg(r)
        R["P3"].update(finish=fr, n_calls=len(m.get("tool_calls") or []), leaked=bool(BAD.search(json.dumps(m))),
                       args=[t["function"]["arguments"] for t in (m.get("tool_calls") or [])])
    # P5 streaming tool call
    s, ch, dt = call({**base, "tools": TOOLS, "stream": True, "messages": q}, stream=True)
    R["P5"] = {"status": s, "dt": round(dt, 1)}
    if s == 200:
        names, args, fins, ids = [], {}, set(), []
        for c in ch:
            for choice in c.get("choices", []):
                if choice.get("finish_reason"): fins.add(choice["finish_reason"])
                for t in (choice.get("delta", {}).get("tool_calls") or []):
                    i = t.get("index", 0)
                    if t.get("id"): ids.append(t["id"])
                    f = t.get("function", {})
                    if f.get("name"): names.append(f["name"])
                    args[i] = args.get(i, "") + (f.get("arguments") or "")
        R["P5"].update(finish=sorted(fins), names=names, ids=ids, args=args)
        try: R["P5"]["args_json_ok"] = all(json.loads(v) is not None for v in args.values())
        except Exception: R["P5"]["args_json_ok"] = False
    # P4 needle
    filler = "The quick brown fox jumps over the lazy dog near the old mill. "
    n = HAY_TOKENS // 14
    pos = n // 2
    text = filler * pos + "SECRET-CODE-7431 is the vault passphrase. " + filler * (n - pos)
    s, r, dt = call({**base, "max_tokens": 200, "messages": [{"role": "user", "content":
                    text + "\n\nWhat is the vault passphrase? Answer with the code only."}]}, timeout=300)
    R["P4"] = {"status": s, "dt": round(dt, 1), "hay_tokens_est": HAY_TOKENS}
    if s == 200:
        m, fr = msg(r)
        R["P4"].update(finish=fr, answer=(m.get("content") or "")[:120], ok="7431" in (m.get("content") or ""),
                       usage=r.get("usage"))
    else:
        R["P4"]["err"] = r if isinstance(r, str) else json.dumps(r)[:300]
    # P6 stray URLs in a neutral coding answer
    s, r, dt = call({**base, "max_tokens": 600, "messages": [{"role": "user", "content":
                    "Write a short Python function that downloads a file with requests and saves it. Include install and usage notes."}]})
    R["P6"] = {"status": s, "dt": round(dt, 1)}
    if s == 200:
        c = msg(r)[0].get("content") or ""
        R["P6"].update(urls=sorted(set(URL.findall(c)))[:20], leaked=bool(BAD.search(c)))
    return R


if __name__ == "__main__":
    res = [run(m) for m in MODELS]
    txt = json.dumps(res, indent=2)
    if OUT:
        open(OUT, "w").write(txt + "\n")
    print(txt)
