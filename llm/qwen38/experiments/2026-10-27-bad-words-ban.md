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

Decision: the gateway supports `--bad-word` / `--bad-words-file`, but it is **not** enabled for
qwen38. A ban moves part of the traffic from a blocked host to unblocked ones. The habit needs a
weight-level fix (see the llmtuning project).
