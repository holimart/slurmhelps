# DeepSeek V4.1 Flash via OpenCode + Ollama Cloud

Bounded replication of the Qwen experiments (OpenCode agent suite from
`llm/qwen35/`, plus direct API probes). Model: `ollama-cloud/deepseek-v4.1-flash:1017`
(no `:1107` tag exists on ollama.com; `:1017` and untagged are the only ones).

| File | Purpose |
|---|---|
| `../qwen35/opencode_tasks.py` | shared suite (fixed: no `--standalone`, `stdin=DEVNULL`, bytes on timeout) |
| `probe.py` | tool-call, reasoning echo, parallel calls, streaming, needle, stray-URL probes |
| `probe_loop.py` | N-turn tool loop (session-growth failures) |
| `experiments/data/` | raw JSON |

## Results (2026-11-03)

- OpenCode suite: **4/4 pass**, 12.7–22.9 s per task.
- Probes (`probe-1017.json`): structured `tool_calls`, no DSML/XML leak, 3 parallel
  calls OK, streamed args valid JSON, finish_reason correct.
- Reasoning echo (reported 400s): **not reproduced**. This endpoint returns no
  reasoning field even with `reasoning_effort=high`; multi-turn with/without echo both 200.
- Needle at 60K and 400K tokens: found (400K → 9 s). Untagged model same tool behaviour.
- 36-turn tool loop to 44K tokens: 0 errors, 0 leaks, 0 malformed args.
- Stray URLs in a neutral coding answer: none.

## Harness bugs found (ours, not the model)

1. `opencode run` hangs forever when stdin is an open pipe (subprocess default):
   all 4 tasks timed out at 240 s with empty output. Fix: `stdin=DEVNULL`.
2. `--standalone` is not a flag in the installed opencode; run exits 1 in 1 s.
3. `subprocess.TimeoutExpired.stdout` is bytes; the original suite silently
   drops partial output on timeout (fixed here).
4. opencode logs `inotify_add_watch ... No space left on device` (watch limit, not disk).
5. Intermittent: in one probe_loop run the model sent `i` as a string for an
   integer parameter (not captured; the rerun was all ints).

## Web findings (Haiku agents; only items marked verified were fetched)

- Most-cited DeepSeek V4 failure: 400 when `reasoning_content` is not passed back with tools
  (docs: api-docs.deepseek.com/guides/thinking_mode). OpenCode #60038 (intermittent, grows with session).
- DSML/XML tool calls in content: OpenCode #50206 / #51077 (OpenCode Go provider, verified).
- Provider-ID flip between two IDs alternately failing (#51077).
- Stream truncation treated as complete on OpenAI-compatible providers: PR #21727 (unmerged).
- Ollama Cloud: no documented rate limits; community claims 5 h / 7 d session limits (unverified).
- Not found: any report of spurious Chinese/raw-IP URLs from DeepSeek.

## Local OpenCode DB URL audit

`scan2.py` (scratchpad) scans assistant text/tool inputs only for raw IPs and
CN-ish hosts. First 100K parts: only legit dev domains. Full 46 GB scan still running.
