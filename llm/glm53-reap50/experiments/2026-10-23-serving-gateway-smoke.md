# 2026-10-23 — serving gateway smoke on 2× A100 (real GPU + mock)

The P0/P1 serving path is implemented in `lib/slurmhelps/serving/` and was
exercised against the real SM80 GLM server on `ampere02`, plus a no-GPU mock
and an OpenCode client.

## Server defaults exercised

```text
model             GLM-5.3-Flash-REAP50-NVFP4-v2
engine            vLLM SM80 fork, TRITON_MLA_SPARSE
GPUs              2× A100 80 GB
partition         ffa-preempt (preemptible, tier 5)
cpus              8
mem               220G
max_model_len     786432  (checkpoint text limit 1,048,576)
max_num_seqs      32
max_num_batched_tokens 8192
KV dtype          bfloat16
block size        128
GPU KV            19.65 GiB/GPU, 1,765,762 tokens
max concurrency at 786432 tokens/request: 2.25×
```

The job ran with `--requeue` and was submitted at `nice=100`; for the smoke it
was lowered with `scontrol update Nice=0` so it would start immediately. The
final defaults remain gentle (`nice=100`, `ffa-preempt`, no `--exclusive`).

## Load-time state

`serve status` reported `queued → starting → loading:weights → warming → ready`
while the HTTP server did not yet exist. Observed:

| Start | Load time |
|---|---:|
| Cold model (98 GiB, Lustre) | **~26 min** (1,559 s) |
| Warm page cache, same node | **~6.25 min** (374.7 s) |

The status file carries `load_elapsed_s`, `load_estimate_s`,
`load_estimate_source`, phase and heartbeat. The cold fallback was 540 s, which
underestimated the I/O-contended cold load; the warm start was measured and
recorded in history.

## 700K context

The checkpoint advertises **1,048,576** text positions. The server was started
with `max_model_len=786432` (768K), leaving headroom for output. Real requests:

| Request | Prompt tokens | Elapsed | Note |
|---|---:|---:|---|
| cold, 690K words | 690,000 | **235.1 s** | prefill ~2,936 tok/s |
| cache hit, 700K words | 700,000 | **5.89 s** | shared 690K cached tokens + 10K suffix |

Both returned HTTP 200 with `finish_reason=length` and one output token. Two
concurrent 700K cache-hit requests also returned 200 (1.92 s and 2.94 s): the
KV cache (1.766M tokens) holds two 700K requests, and `max_num_seqs=32` is not
the limiting factor at that context length. Raw JSON:
[`data/2026-10-23-serving-long-context-2xa100.json`](data/2026-10-23-serving-long-context-2xa100.json).

## HTTP smoke matrix

`tests/serve_smoke.py` ran against:

1. the real server directly (`http://ampere02:8000`);
2. the real server through the loopback gateway with bearer auth;
3. the mock backend in CI (`tests/test_serving_smoke.py`, 10 checks).

Results:

| Check | Real direct | Gateway | Mock |
|---|---|---|---|
| health / models | pass | pass | pass |
| chat non-stream | pass | pass | pass |
| SSE stream | pass | pass | pass |
| cache accounting | warn* | warn* | pass |
| 5 shared prefix, outputs 16..512 | **short 1.85 s, long 43.2 s** | **short 1.84 s, long 43.2 s** | pass |
| C=1/2/4/8 at 500/800 | pass | pass | pass |
| max_model_len rejection | 400 | 400 | 400 |
| auth | skip (no key) | **bad=401, good=200** | pass |
| queue/admission | skip | **32×200, 1×429** | pass |

\* The SM80 fork does not expose `prompt_tokens_details.cached_tokens`; the
cache check falls back to TTFT comparison and reports `warn`, not `fail`. The
700K cache-hit result above is the direct proof of prefix reuse.

Raw: [`data/2026-10-23-serving-smoke-2xa100.json`](data/2026-10-23-serving-smoke-2xa100.json)
and [`data/2026-10-23-serving-gateway-smoke-2xa100.json`](data/2026-10-23-serving-gateway-smoke-2xa100.json).

## OpenCode

- Mock backend: `opencode run --standalone --model local/glm53-reap50` returned
  a text part; the mock recorded the request.
- Real server: OpenCode initially failed with
  `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`;
  the backend now defaults to `--enable-auto-tool-choice --tool-call-parser glm47`
  and OpenCode returned `smoke ok` through the real GLM endpoint.

Example provider config: [`examples/opencode-serve.json`](../../examples/opencode-serve.json).

## What is not built

Raw-KV spill to Lustre is **not** implemented. The SM80 fork has no KV
export/import API; the design keeps `spill.mode=state-only` as the safe path and
gates raw KV behind LMCache/HiCache or a custom serializer plus benchmarks.
Idle-time/size policy and atomic manifests are described in
[`docs/serving-gateway.md`](../../docs/serving-gateway.md), but no fake KV
snapshot is claimed.
