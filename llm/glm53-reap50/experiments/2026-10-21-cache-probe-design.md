# Cache-probe design — tiers, continuations, capacity (2026-10-21)

This is the design for the next measurement round. It exists because the
current context sweep measures **bulk prefill from zero**, not what an agentic
harness sees: a long cached prefix plus a small new continuation, repeated
across many sessions.

## 1. Corrected model

A request is `prefix(P) + suffix(S)`.

- **Miss**: the engine prefills `P+S`; TTFT is a full-prefill time.
- **GPU-tier hit**: KV for `P` is already in HBM; the engine prefills only `S`
  and decodes at context `P+S`.
- **RAM-tier hit**: KV for `P` is in host RAM; the engine copies it to HBM
  (H2D), then prefills `S` and decodes.

A hit does **not** mean “no prefill”:

```text
TTFT ≈ lookup + optional H2D + prefill(S at offset P) + queue
```

And prefill of `S` at offset `P` is not the same as prefill of `S` at zero:
attention/indexer still sees the cached `P`, and decode still pays attention
over `P+S`. Report `TTFT`, `suffix_prefill_tokens_per_s`, and `decode_tokens/s`
separately; never compare a cached `P+128` run with a cold `P+128K` run.

## 2. What can actually be measured on each node

| Node | Engine | GPU prefix tree | RAM tier | Miss | Verdict |
|---|---|---|---|---|---|
| 2× L40S 96 GB | llama.cpp + IQ4_XS | **No** — one active slot KV; prompt cache is host | host prompt cache | yes | Host-cache vs miss only; **no GPU-tree or true RAM-tier measurement** |
| 2× L40S 96 GB | vLLM/SGLang | yes | yes | yes | **No** — FP8 160.6 GiB / NVFP4 98.2 GiB do not fit 96 GB |
| 2× H100 160 GB | SGLang + NVFP4 | yes (RadixAttention) | yes (HiCache) | yes | **Best true-tier option**, if SM90 NVFP4 loads |
| 2× H100 160 GB | vLLM + NVFP4 | yes | via LMCache/CPU offload | yes | Cross-check |
| 2× H100 160 GB | llama.cpp + IQ4_XS/Q4 | no | host prompt cache | yes | Same limitation as 96 GB, but faster |

**Consequence:** the 96 GB run can answer “what does a host prompt-cache hit
cost versus a miss, at different context and continuation lengths, and how many
host-cache branches fit?”, but it cannot answer “how many GPU-tree branches fit”
or “GPU hit vs RAM hit”. Those require the 160 GB SGLang/vLLM run.

## 3. Experiment axes

| Axis | Values |
|---|---|
| Tier | `miss`, `gpu`, `ram` |
| Prefix `P` | 0, 100K, 200K, …, 1M |
| Suffix `S` | 1, 16, 128, 1K, 8K |
| Concurrency `C` | 1, 2, 4, 8 (fit-dependent) |
| Runs | 2–3, median |
| OSL | 128 (fixed for comparability) |

The **cache ladder**: prefill 100K, probe `P+128` (hit) and a mutated-prefix
miss; extend the same prefix to 200K, probe again; continue to 1M. The 1M fill is
the expensive part; after that all 100K cut points are cheap.

**Important:** the ladder is fast for **hits**, not for the initial fill. A
miss at 1M still requires a full 1M prefill. Do not “simulate” a 1M miss by
reusing the cached prefix; mutate the first token or flush the cache.

## 4. Metrics per point

```text
engine, model, checkpoint, engine_version, node, gpu_count
tier, prefix_tokens, suffix_tokens, concurrency, run
ttft_ms {p50,p90,p99}
suffix_prefill_tokens_per_s   # S / suffix-prefill time
bulk_prefill_tokens_per_s     # (P+S)/TTFT for miss only
decode_tokens_per_s           # aggregate and per-request
tpot_ms, e2e_ms
end_to_end_output_tokens_per_s
cache_hit_tokens, cache_hit_rate
h2d_bytes, h2d_ms             # RAM tier
gpu_kv_bytes, ram_kv_bytes, evictions
completed, failed
```

## 5. Script shape

`cache-probe.py` (new, stdlib-only, reuses `bench.py`’s request/summarize):

```text
for tier in tiers:
  for P in prefixes:
    prefix = stable_prefix(P)
    if tier == miss:  flush_cache(); prefix = mutate_first_token(prefix)
    if tier in (gpu, ram): warm(prefix, max_tokens=1)
    for S in suffixes:
      for C in concurrency:
        measure(prefix + suffix(S), concurrency=C, osl=128, runs=R)
```

`cache-sweep.sh` orchestrates:

- **llamacpp**: one server, `-np 1 -c 1048576`, `--cache-prompt`; `miss` uses a
  mutated prefix, `ram` uses the host prompt cache, `gpu` is marked
  **unsupported**. Capacity from the startup `llama_kv_cache` line and
  `--cache-ram`.
- **sglang**: one server, `--max-model-len 1048576`,
  `--enable-prefix-caching` (RadixAttention), optional
  `--enable-hierarchical-cache --hicache-size N`; `miss` calls `/flush_cache`;
  `gpu` is a normal hit; `ram` is produced by filling GPU cache with other
  branches until the target prefix is evicted, then re-requesting it.
- **vllm**: same, with `/reset_prefix_cache` (dev mode) and LMCache/CPU offload.

## 6. Cache-capacity smoketest

Do not only compute from theory; read the engine’s actual pool and then verify:

1. Parse startup logs for `GPU KV cache size: N tokens` (vLLM),
   `max_total_num_tokens` (SGLang), or `llama_kv_cache: size = X MiB
   (N cells)` (llama.cpp).
2. Compute `bytes/token` from the measured KV, not the model card.
3. Fill N branches of length P and count until eviction/OOM.
4. Report, per P (100K … 1M):

```text
max_gpu_tokens, max_gpu_branches, max_ram_tokens, max_ram_branches,
max_concurrent_agents_at_P, eviction_rate, metadata_bytes
```

For 500K branches, the working numbers are ~3 GB FP8 / ~6 GB BF16 / ~1.8 GB q4,
plus ~0.15 GB state/indexer and 0.01–0.3 GB tree metadata. On 96 GB with 82 GB
weights, only ~2–4 FP8 branches fit; on 160 GB with 98 GB NVFP4, ~15–18; on
192 GB with 87 GB Q4, ~30. These are KV-only and must be confirmed against the
engine’s pool.

## 7. Run plan and time

| Run | What | Estimated wall time |
|---|---|---|
| 96 GB llama.cpp, host-cache vs miss | 1M fill + 10 hit probes + 3 misses (100K/400K/1M) | **1.5–2.5 h** |
| 96 GB, only 16K/64K/256K | small smoke | **45–60 min** |
| 160 GB SGLang NVFP4, GPU/RAM/miss | 1M fill + hit ladder + RAM eviction + capacity | **1.5–2 h** |
| 160 GB, model/image download + first startup | one-off | **20–40 min** |

“At once” is not realistic on one node: the 1M fill and misses are serial, and
the 96 GB and 160 GB runs use different engines/nodes. Run them sequentially,
or in parallel on two different nodes if both are free.

## 8. Decision needed before refactoring

The 96 GB run can be implemented immediately with llama.cpp, but it will
**not** produce GPU-tree capacity or GPU-vs-RAM tier numbers. The true-tier
experiment requires the 160 GB node with SGLang/vLLM and NVFP4. If we run only
llama.cpp on both nodes, we get a useful **host-cache vs miss** comparison but
must label the tiers correctly and must not claim GPU-tree results.

The recommended order is:

1. **96 GB llama.cpp probe**, 2 h: host-cache vs miss, continuation lengths,
   concurrency, host-RAM capacity.
2. **160 GB SGLang NVFP4 probe**, 2 h: true GPU hit / RAM hit / miss, GPU and
   RAM tree capacities, concurrent-agent limits.

Only then refactor `bench.py`/`context-sweep.sh` around the confirmed schema.
