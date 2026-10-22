# 2026-10-22 — context sweep + guarded cache probe on 2× L40S

Two measurements run side by side on two free 2× L40S nodes.

## A. Context / concurrency sweep — `uk-g6-05`

IQ4_XS, q8_0 KV, one server per context, one request per slot. `C` is the
number of concurrent requests; all prompts are independent one-shot prompts.

| Context | C | Prefill | Decode | TTFT | End-to-end output | TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 8 | **1,170 tok/s** | **50.8 tok/s** | 22.1 s | 30.0 tok/s | 93.2 ms |
| 16K | 4 | **1,204 tok/s** | **13.8 tok/s** | 40.4 s | 8.9 tok/s | 133.1 ms |
| 64K | 1 | **992 tok/s** | **33.6 tok/s** | 62.2 s | 1.94 tok/s | 30.0 ms |
| 256K | 1 | **517 tok/s** | **22.5 tok/s** | **476.9 s** | 0.27 tok/s | 44.8 ms |

All points completed. 64K C=1 is the shape that fits; 64K C=2 is what OOM'd in
the earlier run.

Raw: [`data/2026-10-22-context-sweep-2x48gb.csv`](data/2026-10-22-context-sweep-2x48gb.csv)
and [`.json`](data/2026-10-22-context-sweep-2x48gb.json).

## B. Guarded cache probe — `uk-g6-06`

IQ4_XS, q4_0 KV, 512/128 batch, one sequence. The guard reads the `n_past`
from the current fill request’s `new prompt` server-log block and aborts if
it would prefill more than **105K** new tokens on top of the restored cache.

| Prefix P | Reused prefix | New prefill | Fill TTFT | Cumulative fill | Hit TTFT | Active repeat | Decode | Inferred miss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100K | 17,143 | **82,857** | 171.4 s | 171.4 s | **0.542 s** | **0.178 s** | 30.3 tok/s | 171.9 s |
| 200K | 104,562 | **95,438** | 311.1 s | 482.5 s | **0.809 s** | **0.294 s** | 24.6 tok/s | 483.3 s |
| 300K | 209,304 | **90,696** | 426.9 s | 909.4 s | **1.086 s** | **0.415 s** | 20.7 tok/s | 910.4 s |

The guard passed every point: the largest new prefill was **95,438 tokens**,
never a full 200K/300K re-prefill. No miss requests were sent; miss TTFT is
`cumulative_fill_s + hit_ttft`.

The 100K reused prefix is only 17,143 because the preflight filled a 16K
prefix; the first ladder step therefore prefilled 82.9K, still under the
105K guard.

Raw: [`data/2026-10-22-cache-probe-2x48gb.csv`](data/2026-10-22-cache-probe-2x48gb.csv)
and [`.json`](data/2026-10-22-cache-probe-2x48gb.json).

## Reading the two together

- **Cache hit TTFT** is 0.54–1.09 s; **active-slot repeat** is 0.18–0.41 s.
  The first hit is host prompt-cache restore; the repeat is the one active GPU
  slot. Both are single-sequence, not a multi-branch tree.
- **Cold fill** is 171–427 s for the same prefixes. The inferred miss is
  171.9/483.3/910.4 s.
- **Decode** remains context-bound: 30.3 → 24.6 → 20.7 tok/s at 100K/200K/300K,
  and 50.8 tok/s at 4K C=8. A cache hit removes prefix prefill, not decode
  attention.
- **256K cold prefill** is 476.9 s at 517 tok/s; end-to-end output is
  0.27 tok/s because prefill dominates.
- **1M is not reachable** on this 96 GB node with IQ4_XS/q4/batch512; 400K
  already OOM’d in compute buffers.

## Still unmeasured

- Multi-branch GPU prefix-tree reuse and capacity.
- RAM-vs-GPU KV tier attribution and H2D timing.
- Multi-agent GPU cache capacity.
- 4× L40 / 160 GB results; those nodes were allocated.
- Vulkan performance (IQ3_M works but 4K TTFT 20.3 s; see the Vulkan note).
