# 2026-10-21 — cache reuse on 2× L40S 48 GB (IQ4_XS)

First cache-reuse probe: a 100K→300K prefix ladder, a 128-token suffix, and a
repeat of the same request to observe active-slot reuse. **No miss requests were
issued.** Miss TTFT is inferred from the cumulative fill time.

## Setup

| | |
|---|---|
| Node | `uk-g6-05`, 2× L40S 48 GB (96 GB total) |
| CPUs / RAM | 32 CPUs, 256 GiB |
| Model | `GLM-5.3-Flash-REAP50-IQ4_XS.gguf` (82.0 GiB) |
| Engine | llama.cpp `glm5-next-reap50-gguf-v1`, CUDA 12.5, sm_89 |
| Server | `-np 1 -c 1048576 -b 512 -ub 128 -fa on -ctk q4_0 -ctv q4_0 --cache-prompt --cache-reuse 256 --no-mmap` |
| Probe | prefill P, request `P + 128`, repeat the same request; OSL 128 |

`q4_0` KV and `-b 512/-ub 128` were used because the 1M context does not fit
with q8_0 KV plus the dense-DSA compute buffers on 96 GB.

## Results

`hit` = first request after the fill (host prompt cache restore);
`repeat` = immediately repeated request (active GPU slot, **one sequence only**).

| Prefix P | Fill TTFT | Cumulative fill | Hit TTFT | Repeat TTFT | Decode | Inferred miss |
|---:|---:|---:|---:|---:|---:|---:|
| 100K | 171.2 s | 171.2 s | **0.54 s** | **0.18 s** | 30.2 tok/s | 171.7 s |
| 200K | 310.2 s | 481.4 s | **0.80 s** | **0.29 s** | 24.5 tok/s | 482.2 s |
| 300K | 424.9 s | 906.3 s | **1.09 s** | **0.42 s** | 20.7 tok/s | 907.4 s |

Preflight at 16K: fill 32.0 s, hit 0.30 s, reuse **106.7×**.

Raw: [`data/2026-10-21-cache-reuse-2x48gb.csv`](data/2026-10-21-cache-reuse-2x48gb.csv)
and [`.json`](data/2026-10-21-cache-reuse-2x48gb.json).

## What this shows

- **Host-cache hit TTFT is 0.5–1.1 s**, versus 171–425 s to fill the same
  prefix. That is a 315–389× reduction, before counting the suffix.
- **The immediate repeat is 2.5–3× faster than the first hit**: 0.18–0.42 s vs
  0.54–1.09 s. The first hit is consistent with a host prompt-cache restore to
  the GPU; the repeat is the active GPU slot. This is the **only GPU-reuse
  number we can measure** in llama.cpp, and it is **one active sequence**.
- Decode falls with context: 30.2 → 24.5 → 20.7 tok/s at 100K/200K/300K.
  A cache hit removes prefix prefill, not decode attention.
- **400K failed**: `cudaMalloc` OOM during flash-attention/compute-buffer
  allocation (`launch_fattn<512,8,8>`). With IQ4_XS, q4_0 KV and 512/128 batch,
  the practical ceiling on this 96 GB node is about **300K** — not 1M.
- Capacity: GPU KV pool = 1,048,576 cells / 3,168 MiB (q4_0); host prompt cache
  size limit = 8,192 MiB. At 950K that host cache is ~1.5 serialized states, but
  it is **not a paged multi-branch tree**.

## Explicitly unmeasured

Printed by the probe and repeated here:

- **GPU prefix-tree reuse** — llama.cpp has one active slot KV, not a
  GPU-resident multi-prefix tree.
- **RAM vs GPU KV tier** — the host prompt cache is serialized state, not a
  paged KV tier; no per-tier counters.
- **Multi-agent GPU cache capacity** — unmeasured.
- **Miss** — inferred from `cumulative_fill_s`, not measured.
- **1M context** — blocked at 400K by compute-buffer OOM on 96 GB.

## 160 GB attempt

A 2× H100 run with a separately built sm_90 binary started on `hopper01`, but
the job was **preempted/requeued** at ~400K before producing results, and the
replacement is pending because both H100 GPUs are allocated. The 160 GB
cache-tier run therefore remains open; it needs a stable 2-hour allocation on
`hopper01` (or a 4× L40 FP8/vLLM run).

## Reproduction

```bash
# 96 GB
LLAMA_CUDA_HOME=/usr/local/cuda-12.5 ./submit.sh \
  --node uk-g6-05 --partition gpu-ffa --account ffa \
  --gpus 2 --cpus 32 --mem 256G --time 02:00:00 \
  --engine llamacpp --variant iq4_xs --no-install --cache-sweep \
  --cache-prefixes "100000 200000 300000" \
  --cache-suffix 128 --cache-osl 128

# 160 GB H100 (sm_90 binary)
LLAMA_CUDA_HOME=/usr/local/cuda-12.5 \
LLAMA_SERVER=$WORK_ROOT/build/llama-h100/bin/llama-server \
CACHE_KV_TYPE=q8_0 CACHE_BATCH=1024 CACHE_UBATCH=256 \
./submit.sh --node hopper01 --partition gpu-ffa-gpulab --account ffa \
  --gpus 2 --cpus 32 --mem 512G --time 02:00:00 \
  --engine llamacpp --variant iq4_xs --no-install --cache-sweep \
  --cache-prefixes "100000 200000 300000 400000 500000 600000 700000 800000 900000 950000" \
  --cache-suffix 128 --cache-osl 128
```
