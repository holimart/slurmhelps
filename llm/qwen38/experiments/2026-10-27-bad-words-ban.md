# Qwen3.8: banning the Alibaba URL strings with vLLM `bad_words` (2026-10-27)

Live server (container vLLM `0.1.dev21830+g8244d1c44`, MTP on) accepts per-request `bad_words` and
enforces it. The cap is 128 token sequences after tokenization; each word counts about twice.

Replay of the batch-fetch trigger contexts (competitionsall
`experiments/qwen38-alibaba-interrogation/exp5_ban.py`, 48 samples per arm):

| Arm | Clean URL | Alibaba URL | Look-alike proxy host |
|---|---|---|---|
| No ban | 23 | 25 | 0 |
| 5 words (`aliyuncs routify LTAI5t proxy_temp_file fastgpttest`) | 32 | 6 | 10 |
| 49 words (separator/case variants) | 23 | 7 | 18 |

Why it leaks: vLLM bans the token sequence of the standalone word (with and without a leading space).
Inside a URL, `.aliyuncs` or `=LTAI5t…` tokenize differently, so the full template still appears. And
the model spells around the ban: `routified.com`, `routIFY.ai`, `aliyuncsl.com`, S3 / googleapis /
trycloudflare / duckdns look-alikes. Those hosts are not on the langywrap sandbox denylist.

Key-only ban (`LTAI5tKoG9A3DkwGD635QVZr`, `LTAI5tA4qZ9rEj6S26c4vMqE` and their `=` forms): 26/48
still emit the Alibaba URL (no change), but the key comes out as `…QVZ` / `…QVZc`: the exact
memorised credential is never sent and no traffic moves to new hosts.

Decision: qwen38 runs vLLM behind the gateway on the node (`bin/serve_with_gateway.sh`, vLLM on
127.0.0.1:PORT+1) with the **key-only** list in `llm/qwen38/bad_words.json`. Wider word bans stay
off because they push part of the traffic from a blocked host to unblocked ones. The sandbox
denylist (langywrap 3a7e4891: `routi[fz]`, `ltai5t`, `ossaccesskeyid`) blocks the rest; the habit
itself needs a weight-level fix (see the llmtuning project). `QWEN38_GATEWAY=0` restores bare vLLM.
