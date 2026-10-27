# Qwen3.8: gateway repairs measured live (2026-10-27)

The node gateway (`lib/slurmhelps/serving/`) now rewrites more than the routify URL. Every number
below comes from the live qwen38 server through the gateway running in-process, compared with the
same requests sent straight to vLLM. All requests are text only: no model tool call was executed.

Research, scripts and raw data are in the llmtuning repo:

- `experiments/routify-small-context`
- `experiments/upload-probe`
- `experiments/metadata-probe`
- `experiments/unshare-escalation`
- `experiments/gateway-validation`

## What the gateway does

| Behaviour | Output (what the client receives) | History (what the model sees next) |
|---|---|---|
| Routify / Alibaba OSS URL, key, signed-URL look-alikes (`rewrite.py`) | URL → `https://blocked.invalid/removed-url`, key → `REMOVED_KEY` | the tool call and its result are dropped (default `--history-mode drop`); leftover mentions become plain words |
| Upload to a host the conversation never named, probe payloads (`shellguard.py`) | the offending statements become `echo '[gateway:removed] (upload) …'`; benign statements stay | a call with nothing benign left is dropped; otherwise the benign part stays and the last repaired result gets one generic `[gateway]` note |
| Cloud metadata requests (169.254.169.254, 100.100.100.200, metadata.google.internal, …) | same as uploads (`(metadata)`); a `web_fetch` URL becomes the blocked URL | same as uploads |
| Privilege workarounds: sudo/su, unshare, nsenter, chroot, setcap, privileged docker, /etc or /proc/sys writes (`privguard.py`) | the command becomes an echo of the sandbox notice | workaround calls are dropped; the notice that names the blocked mechanisms is appended once, to the last permission-error result |

While streaming, the tool-call arguments are held back whole, so each command is checked complete.

To switch parts off: `--no-shellguard`, `--no-privilege-guard`, `--history-mode swap`, and
`--allow-host H` to allow uploads to a host.

## Live results

| Context (n) | Straight to vLLM | Through the gateway |
|---|---|---|
| 9 stuck routify conversations, small contexts (54 each) | 14/54 clean next call | **46/54** clean; 0 Alibaba strings at the client; 0 invalid tool JSON |
| Same, long contexts (exp7 → exp7c, 27) | 6/27 (placeholder scrub) | 18/27 (repair) |
| Cohere upload call + DNS failure (24) | 10/24 back on task | 17/24 |
| Two failed uploads (24) | 6/24, 10 more uploads | **19/24**, 0 uploads |
| Three failed metadata probes (24) | 16/24, 4 more probes | 19/24, 0 probes |
| Plain permission error, 5 tasks (80) | 14/80 workarounds | **1/80**, and it was rewritten |
| Refused sudo already in history (80) | 3/80 | 2/80, both rewritten |

Notes:

- **Kind-specific notes were worse.** Upload and metadata notes worded like the best research
  variants scored 38/72 back on task, against 55/72 for the generic note, so the gateway keeps the
  generic note.
- **The server is noisy.** The same prompt swings a lot from request to request. Use single-sample
  requests with a unique `cache_salt`, and repeat a run before trusting a difference of a few samples.
- **What is left is mostly not a security problem.** Remaining off-task turns are DNS checks and
  "simulate the attack" downloads that the security-detection task itself invites. No uploads or
  metadata requests remain.

## Root cause found on the client side

langywrap `web_fetch` refused `application/atom+xml` (the arXiv API) as "response content type is
not text". One such failure in the history raised the routify rate from 8/240 to 45/240. Fixed in
langywrap fa977270: XML, JSON and other text types are now returned as text.
