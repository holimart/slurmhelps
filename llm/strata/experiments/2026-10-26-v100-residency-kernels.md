# 2026-10-26 — V100 residency, maximum context and Volta kernels

Second V100 campaign, run on **volta05** (4× Tesla V100-SXM2-32GB, sm_70,
380 GB RAM, 64 CPUs, `ffa-gpulab`). It extends the 1× V100-32G baseline in
[`2026-10-25-strata-smoke-matrix.md`](2026-10-25-strata-smoke-matrix.md)
(1k/4k/16k, 65k boundary, warm/cold 1k–16k, Coder IQ1_M + IQ3_S). **Nothing
from that baseline was re-run**: the 1× numbers below are quotes from that
document; every 2×/4× number is new.

The questions:

1. can the packs be **fully resident** on 2× and 4× V100-32G, and is that
   provable from the engine, not assumed?
2. how far does the context ladder go (does 128k work on 2×, 256k on 4×)?
3. what does prompt-cache reuse do to TTFT at the maximum contexts?
4. which **opt-in sm_70 kernels** actually help on this exact hardware?

## Setup

```text
node:       volta05 (ffa-gpulab), 4× V100-SXM2-32GB, 32,768 MiB each, sm_70
driver:     580.178.04 (from the baseline capture; not re-queried)
engine:     0.1.50, cuda12 variant (Control selects it on sm_70)
harness:    llm/strata/{bench-matrix,strata_probe,knob-sweep}.sh
packs:      Coder IQ1_M 29.6 GB + 28.8 GB table
            Q2_0       37.6 GB + 28.8 GB table (downloaded 2026-10-26)
            IQ3_S      54.8 GB + 28.8 GB table
```

All runs pin the baseline recipe explicitly (the installed config is shared
mutable state; the runner now passes it with `--set` and verifies the engine's
start line):

```text
strata --serve --pack <private>/packs/<pack> --native <shard1> --ple-gguf <shard2>
  --expert-profile <shipped> --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <private>/mtp/rt --kv int8
  --kv-resident 32768 --max-context <window> [--layer-split auto] --batch 4
```

`--layer-split auto` is added by `strata launch` whenever `--gpu` names more
than one card; it is not a runner flag. Every launch was verified from the
engine's `engine started:` line (archived as `engine-argv.txt`).

Measurement contract (unchanged from the baseline): one server per pack and
job, one cell = C concurrent streams × 3 sequential requests, unique prompt
salt (cold), 128 completion tokens, temperature 0; probe calibrates
chars/token with two `include_usage` requests and clamps to
`window − max_tokens − 128`. `prompt_tokens` in the tables is the real
per-request context (`prompt_tokens / requests_ok` in the JSONL).

Staging: the packs are copied to node-local `/tmp` (206 GB on volta05).
The shared 28.8 GB PLE shard is **single-striped on one OST** and read from
Lustre at only 4–14 MB/s on this day; the successful combined job staged it
once, in parallel with the two 37.6/54.8 GB shard-1 reads on other OSTs, in
51 min. Two earlier attempts with `COPY_MODELS=0` (engine reading the GGUF
straight from Lustre) hit `strata launch`'s fixed 810 s load watchdog; the
real errors and engine start lines are kept under `_attempts/`. On volta05
`/tmp` is a per-job namespace, so a staged shard does not survive the job —
which is why A, B and E were run inside **one allocation** after staging the
shared shard once (and hard-linking it into every pack dir).

## A. Fully resident on 2× V100-32G (64 GB)

### Coder IQ1_M — job 38239040, `--gpus 0,1`

Engine evidence (full lines in `a-2x-coder/coder/engine-facts.txt`):

```text
strata generate: layer split across 2 GPUs: CUDA0, then CUDA1 (split auto)
strata generate: layer split auto: K=24 - predicted 28.4 ms per decode window;
  the caches hold 12288 of 12288 profiled pairs (~100.0% of the routed mass)
strata generate: layer split: CUDA0 runs layers 0-23
strata generate: layer split: CUDA1 runs layers 24-47, expert cache 6144 slots
  (12.35 GiB), 6144 of its 6144 profiled pairs
strata generate: expert cache 6144 slots, 11.07 GiB of VRAM (CUDA0)
strata serve: layer split: 100% of the experts resident, the prompt path's
  streamed ring 96 slots
strata serve: 11308 MiB of VRAM free (CUDA0), 8444 MiB (CUDA1)
```

→ **fully resident**, proven by the engine: both stage caches hold every one
of the 12,288 profiled (layer, expert) pairs, 100 % of the routed mass; the
`served` line says 100 % of experts resident. 23.4 GiB of the 64 GB is expert
cache (11.07 + 12.35 GiB), 1.54 GiB is batch slots; the rest is dense
weights, KV (the engine streams 32,768 of 131,072 cells per QSA layer and
prints 0.77 GiB of pinned K/V) and workspace.

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | real ctx | status |
|---:|--:|---:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 94.01 | 94.17 | 49.35 | 1.20 | 1.21 | 1,037 | ok |
| 1024 | 2 | 30.03 | 43.98 | 45.80 | 1.21 | 2.56 | 1,037 | ok |
| 1024 | 4 | 17.91 | 28.92 | 52.63 | 2.54 | 5.27 | 1,037 | ok |
| 4096 | 1 | 90.86 | 91.60 | 31.14 | 2.73 | 2.73 | 4,106 | ok |
| 4096 | 2 | 21.70 | 42.31 | 29.27 | 2.73 | 5.62 | 4,106 | ok |
| 4096 | 4 | 12.33 | 27.78 | 31.77 | 5.61 | 11.45 | 4,106 | ok |
| 16384 | 1 | 93.68 | 93.77 | 14.43 | 7.54 | 7.56 | 16,414 | ok |
| 16384 | 2 | 18.57 | 87.67 | 10.95 | 17.10 | 19.65 | 16,414 | ok |
| 16384 | 4 | 18.58 | 89.27 | 10.77 | 41.26 | 43.86 | 16,414 | ok |
| 65536 | 1 | 79.00 | 80.52 | 5.13 | 23.50 | 23.57 | 65,487 | ok |
| 65536 | 2 | 11.18 | 77.13 | 3.21 | 72.90 | 72.96 | 65,548 | ok |
| 65536 | 4 | 11.23 | 78.88 | 3.20 | 73.25 | 365.30 | 65,642 | ok |
| 131072 | 1 | 75.90 | 78.84 | 2.40 | 51.84 | 51.99 | 130,835 | ok |

### Q2_0 — job 38240753, `--gpus 0,1`
Engine evidence (`a-2x-q2_0/q2_0/engine-facts.txt`):

```text
strata generate: layer split across 2 GPUs: CUDA0, then CUDA1 (split auto)
strata generate: layer split auto: K=24 ... the caches hold 24576 of 24576
  profiled pairs (~100.0% of the routed mass), best of 46 placements
strata generate: expert cache 12288 slots, 15.82 GiB of VRAM (CUDA0)
strata generate: layer split: CUDA1 runs layers 24-47, expert cache 12288 slots
  (15.82 GiB), 12288 of its 12288 profiled pairs
strata serve: layer split: 100% of the experts resident, the prompt path's
  streamed ring 96 slots
strata serve: 7262 MiB of VRAM free (CUDA0), 5820 MiB (CUDA1)
```

→ **fully resident**. Q2_0 has 512 experts/layer, so the two cards cache
2 × 12,288 = 24,576 (layer, expert) pairs, all of them. Free VRAM is smaller
than for Coder (5.8–7.3 GiB) because Q2_0's expert blobs are bigger.

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | real ctx | status |
|---:|--:|---:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 107.10 | 110.63 | 49.28 | 1.39 | 1.39 | 1,036 | ok |
| 1024 | 2 | 30.28 | 48.79 | 44.49 | 1.42 | 3.01 | 1,036 | ok |
| 1024 | 4 | 17.57 | 32.94 | 49.79 | 2.99 | 6.24 | 1,037 | ok |
| 4096 | 1 | 96.90 | 101.62 | 29.59 | 3.01 | 3.04 | 4,106 | ok |
| 4096 | 2 | 21.58 | 47.38 | 28.14 | 3.03 | 6.25 | 4,105 | ok |
| 4096 | 4 | 12.13 | 31.40 | 30.23 | 6.23 | 12.75 | 4,106 | ok |
| 16384 | 1 | 99.73 | 100.75 | 13.14 | 8.08 | 9.22 | 16,414 | ok |
| 16384 | 2 | 18.14 | 89.94 | 10.59 | 17.65 | 20.40 | 16,414 | ok |
| 16384 | 4 | 18.28 | 95.62 | 10.46 | 17.64 | 95.00 | 16,414 | ok |
| 65536 | 1 | 88.14 | 90.90 | 4.82 | 25.18 | 25.67 | 65,580 | ok |
| 65536 | 2 | 16.56 | 79.81 | 3.05 | 80.41 | 81.01 | 65,549 | ok |
| 65536 | 4 | 17.39 | 80.09 | 2.98 | 80.17 | 333.90 | 65,579 | ok |
| 131072 | 1 | 78.76 | 100.50 | 1.33 | 54.15 | 54.16 | 130,992 | ok |

Q2_0 is the fastest pack at C1 short context (107.1 tok/s at 1k) and keeps
16.6 tok/s at 64k C2, but its larger expert blobs make it the tightest fit
(5.8 GiB free on CUDA1). 128k C1 works (54.2 s TTFT, 78.8 tok/s decode).

### 2× vs the 1× V100 baseline (Coder IQ1_M, same pack and recipe)

The 1× column is the 2026-10-25 baseline (real prompts 1.14–1.23× the
nominal, so TTFT is slightly pessimistic there).

| ctx | C | decode 1× | decode 2× | Δ | agg 1× | agg 2× | Δ | TTFT 1× | TTFT 2× | Δ |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 86.44 | 94.01 | +8.8 % | 40.05 | 49.35 | +23 % | 1.70 | 1.20 | −29 % |
| 1024 | 2 | 24.28 | 30.03 | +24 % | 35.89 | 45.80 | +28 % | 1.73 | 1.21 | −30 % |
| 1024 | 4 | 14.46 | 17.91 | +24 % | 40.60 | 52.63 | +30 % | 3.67 | 2.54 | −31 % |
| 4096 | 1 | 84.16 | 90.86 | +8.0 % | 25.97 | 31.14 | +20 % | 3.38 | 2.73 | −19 % |
| 4096 | 2 | 18.22 | 21.70 | +19 % | 24.14 | 29.27 | +21 % | 3.40 | 2.73 | −20 % |
| 4096 | 4 | 10.34 | 12.33 | +19 % | 26.11 | 31.77 | +22 % | 7.07 | 5.61 | −21 % |
| 16384 | 1 | 86.39 | 93.68 | +8.4 % | 9.55 | 14.43 | +51 % | 12.00 | 7.54 | −37 % |
| 16384 | 2 | 10.62 | 18.57 | +75 % | 8.68 | 10.95 | +26 % | 17.02 | 17.10 | +0.5 % |
| 16384 | 4 | 10.59 | 18.58 | +75 % | 8.64 | 10.77 | +25 % | 31.78 | 41.26 | +30 % |
| 65536 | 1 | — (probe bug) | 79.00 | — | — | 5.13 | — | — | 23.50 | — |
| 131072 | 1 | — | 75.90 | — | — | 2.40 | — | — | 51.84 | — |

The second card buys 8–24 % per-stream decode and 20–30 % aggregate at short
context, and 37 % TTFT at 16k. At 16k C4 the TTFT becomes **worse** (+30 %)
because four 16k prefills now queue behind a 2-stage pipeline. 64k/128k were
unmeasurable on 1× (the old probe overshot the window); 2× serves them and
128k C1 lands at 51.8 s TTFT / 75.9 tok/s.

## B. Fully resident on 4× V100-32G (128 GB) — IQ3_S

Job 38240753 (same allocation as A and E), `--gpus 0,1,2,3`, window 262,144.

Engine evidence (`b-4x-iq3s/iq3s/engine-facts.txt`):

```text
strata generate: layer split across 4 GPUs: CUDA0, then CUDA1,2,3 (split auto)
strata generate: layer split auto: K=12,24,36 ... the caches hold 24576 of
  24576 profiled pairs (~100.0% of the routed mass), best of 15180 placements
strata generate: layer split: CUDA1 runs layers 12-23, expert cache 6144 slots
  (11.33 GiB), 6144 of its 6144 profiled pairs; slot 0 verified
strata generate: layer split: CUDA2 runs layers 24-35, expert cache 6144 slots
  (12.21 GiB), ...; CUDA3 runs layers 36-47 (12.49 GiB), ...
strata serve: layer split: 100% of the experts resident, the prompt path's
  streamed ring 96 slots
strata serve: 11774 MiB free (CUDA0), 11250 (CUDA1), 10350 (CUDA2), 8502 (CUDA3)
```

→ **fully resident with 8.5–11.8 GiB still free on every card**. Splits:
CUDA0 layers 0-11, CUDA1 12-23, CUDA2 24-35, CUDA3 36-47 + head. The 4×
configuration is over-provisioned for IQ3_S at 32 GB/card: it does not need
the ~45 GB of expert cache it uses to be spread thinner anywhere.

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | real ctx | status |
|---:|--:|---:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 88.76 | 94.92 | 44.63 | 1.47 | 1.50 | 1,036 | ok |
| 1024 | 2 | 34.79 | 61.66 | 49.50 | 1.43 | 3.01 | 1,037 | ok |
| 1024 | 4 | 22.12 | 53.08 | 58.06 | 3.03 | 6.49 | 1,037 | ok |
| 4096 | 1 | 96.28 | 97.84 | 28.44 | 3.15 | 3.25 | 4,106 | ok |
| 4096 | 2 | 23.60 | 59.43 | 29.28 | 3.14 | 6.64 | 4,106 | ok |
| 4096 | 4 | 14.11 | 51.77 | 32.52 | 6.43 | 13.45 | 4,106 | ok |
| 16384 | 1 | 100.61 | 102.10 | 15.05 | 7.25 | 7.26 | 16,413 | ok |
| 16384 | 2 | 13.22 | 58.48 | 14.91 | 7.36 | 14.83 | 16,414 | ok |
| 16384 | 4 | 7.40 | 51.35 | 15.75 | 14.90 | 30.08 | 16,414 | ok |
| 65536 | 1 | 82.30 | 88.36 | 6.63 | 18.06 | 18.32 | 65,642 | ok |
| 65536 | 2 | 6.19 | 56.96 | 6.46 | 18.17 | 37.63 | 65,549 | ok |
| 131072 | 1 | 77.90 | 82.06 | 3.68 | 33.51 | 33.72 | 131,177 | ok |
| 131072 | 2 | 3.48 | 55.06 | 3.56 | 33.63 | 68.76 | 131,176 | ok |
| **262144** | 1 | **72.15** | 74.59 | 1.76 | **72.01** | 72.23 | **261,904** | ok |
| **262144** | 2 | 1.69 | 51.06 | 1.72 | 71.50 | 144.91 | 261,749 | ok |

**262,144 tokens works at C1 and C2 on 4× V100.** A 261,904-token prompt is
prefilled in 72.0 s (≈3,640 tok/s) and decodes at 72.2 tok/s. This is the
point of the exercise and it is not close to a fit limit: each card still has
8.5–11.8 GiB free.

Caveat on C2 at long context: the second stream's 262k prefill overlaps the
first stream's decode, so per-stream decode collapses (72.2 → 1.69 tok/s)
while **aggregate throughput stays flat** (1.76 → 1.72 tok/s). The same shape
appears at 131k (77.9 → 3.48 decode, 3.68 → 3.56 aggregate) and 64k
(82.3 → 6.19 decode, 6.63 → 6.46 aggregate). Concurrency does not buy
throughput at max context with a single batch-4 server; it only interleaves.

### 4× vs the 1× V100 baseline (IQ3_S, same pack and recipe)

| ctx | C | decode 1× | decode 4× | Δ | agg 1× | agg 4× | Δ | TTFT 1× | TTFT 4× | Δ |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 76.35 | 88.76 | +16 % | 27.76 | 44.63 | +61 % | 2.86 | 1.47 | −49 % |
| 1024 | 2 | 19.82 | 34.79 | +76 % | 27.13 | 49.50 | +82 % | 2.88 | 1.43 | −50 % |
| 1024 | 4 | 9.15 | 22.12 | +142 % | 29.98 | 58.06 | +94 % | 2.93 | 3.03 | +3 % |
| 4096 | 1 | 89.69 | 96.28 | +7 % | 24.77 | 28.44 | +15 % | 3.72 | 3.15 | −15 % |
| 4096 | 2 | 17.75 | 23.60 | +33 % | 23.01 | 29.28 | +27 % | 3.76 | 3.14 | −16 % |
| 4096 | 4 | 10.07 | 14.11 | +40 % | 24.99 | 32.52 | +30 % | 7.62 | 6.43 | −16 % |
| 16384 | 1 | 81.83 | 100.61 | +23 % | 8.53 | 15.05 | +76 % | 13.52 | 7.25 | −46 % |
| 16384 | 2 | 9.91 | 13.22 | +33 % | 7.73 | 14.91 | +93 % | 20.14 | 7.36 | −63 % |
| 16384 | 4 | 9.96 | 7.40 | −26 % | 7.75 | 15.75 | +103 % | 53.15 | 14.90 | −72 % |
| 65536 | 1 | — (probe bug) | 82.30 | — | — | 6.63 | — | — | 18.06 | — |
| 131072 | 1 | — | 77.90 | — | — | 3.68 | — | — | 33.51 | — |
| 262144 | 1 | — | 72.15 | — | — | 1.76 | — | — | 72.01 | — |

The four cards turn IQ3_S from a streaming pack (1× holds ~22.4 GiB of
experts) into a fully resident one, and that is what moves aggregate at 16k
(+76 % at C1, +103 % at C4) and TTFT (−46 % to −72 %). At 16k C4 per-stream
decode is lower than 1× (7.40 vs 9.96) because four streams share a 4-stage
pipeline; aggregate is still double.

## C. Placement evidence

Every resident configuration was proven from the engine's own start/output
lines (archived `engine-argv.txt`, `engine-facts.txt`, `engine-logs/`), not
assumed:

| config | packs | expert-cache placement | all routed experts in VRAM | free VRAM after load |
|:--|:--|:--|:--|:--|
| 2× V100-32G | Coder IQ1_M | 6,144 slots on CUDA0 (11.07 GiB) + 6,144 on CUDA1 (12.35 GiB) | **yes** — `100% of the experts resident`; 12,288 of 12,288 profiled pairs | 11.3 / 8.4 GiB |
| 2× V100-32G | Q2_0 | 12,288 + 12,288 slots (15.82 GiB each) | **yes** — `100% of the experts resident`; 24,576 of 24,576 pairs | 7.3 / 5.8 GiB |
| 4× V100-32G | IQ3_S | 4 × 6,144 slots (10.82–12.49 GiB) | **yes** — `100% of the experts resident`; 24,576 of 24,576 pairs | 11.8 / 11.3 / 10.4 / 8.5 GiB |

The engine's layer-split batch summary prints `routed entries per window:
VRAM 0.0 (0.0%), PCIe 0.0 (0.0%), CPU 0.0 (0.0%)` and per-request
`hit_rate: null` on this path — that is a bookkeeping artefact of the split,
not a 0 % cache. The authoritative lines are `100% of the experts resident`
and `N of N profiled pairs`, and there are no `expert cache miss`, `PCIe`
(non-probe) or CPU-expert lines anywhere in the three logs. **No engine in
this campaign streamed experts.**

KV is a separate axis: with `--kv-resident 32768` the K/V of the resident
configs hit VRAM on 73–99 % of block reads (0.2–25 MiB read from pinned RAM
per sample); that is the intended streaming KV mode, not expert paging. The
engine prints “32768 of <window> cells per QSA layer in VRAM, the K/V in
0.77 GiB of pinned RAM” at both the 131k and 262k windows, i.e. the same
0.77 GiB resident share is kept while the window grows; the rest is read from
pinned host memory as the 73–99 % hit rates show.

## D. Warm vs cold at the maximum contexts (4× resident IQ3_S)

`--warm-cells 65536:1,131072:1,262144:1 --warm-repeats 3` ran on the **same
server** as B, right after the cold ladder (bench-matrix's integrated warm
pass, which is the same `strata_probe.py --mode warm` that `warm-cache.sh`
drives; it was folded into the B job because a separate `warm-cache.sh` job
would have paid the 30–90 min single-OST staging again). Each slot sends one
saltless prompt and three byte-identical repeats; repeat 1 is the
same-prompt cold reference, repeats 2–4 hit the prompt checkpoint cache.

| ctx | C | cold repeat-1 (s) | warm p50 (s) | warm p95 (s) | speedup | reused tokens p50 |
|---:|--:|---:|---:|---:|---:|---:|
| 65536 | 1 | 16.94 | **1.064** | 1.339 | **15.9×** | 65,535 |
| 131072 | 1 | 31.80 | **1.400** | 2.305 | **22.7×** | 131,008 |
| 262144 | 1 | 68.96 | **3.133** | 3.691 | **22.0×** | 261,611 |

Reuse engaged fully: `reused_tokens_p50 = prompt − 5` at every size
(65,535 / 131,008 / 261,611) and `cache_reuse_engaged=true`. The cold
reference from repeat 1 is 4–6 % faster than the salted cold cell in B
(16.94 vs 18.06 s at 64k, 68.96 vs 72.01 s at 256k), which is the expected
salt/tokenizer difference.

The absolute TTFT at max context falls from **69 s to 3.1 s**: the
checkpoint restore cost barely grows with context (1.06 s at 64k → 3.13 s at
256k), while prefill grows 4×. That is the strongest single result of the
campaign for agent-style workloads that extend a stable prefix.

## E. Volta kernel / flag sweep (1× V100-32G)

Job 38240753 (same allocation, `--gpus 0`, window 65,536), Coder IQ1_M,
`knob-sweep.sh` with one change per variant. The pack is effectively resident
on the card: `expert cache 11773 slots, 22.43 GiB` and `decode expert cache
hit rate: 100.0% (60459 hits / 60480 lookups)`; a single batch window
reported 93.9 % VRAM / 0.6 % PCIe / 5.5 % CPU while the adaptive tier warmed,
then settled at 100 %. So the kernel deltas are not diluted by expert
paging, except for that warm-up window.

Every variant launched and every cell is `ok`; no `skipped-flag`,
`launch-fail` or `skipped-budget` rows. All five opt-in env names exist in
the cuda12 engine binary (`strings` finds `STRATA_SM70_TABLE`,
`STRATA_FUSE_DECODE`, `STRATA_PF_WMMA`, `STRATA_ARGMAX_ROWS_SM70`,
`STRATA_GR_NORM_FAST`), and `variants.csv` records the exact env/flags each
variant used. The engine does not log which opt-in kernel it took, so
engagement is established by the switch being present in the binary plus the
timing signature; it is not a per-kernel log line.

Control: `basefull` (the whole ladder, baseline recipe) and the one-cell
`control`. `control` (1k C1) = **81.43** tok/s, `basefull` (1k C1) =
**81.30** — a −0.2 % host/engine variance check. (Both are ~6 % below the
2026-10-25 Coder 1× baseline of 86.44; that is day-to-day variance, which is
why every delta below is against `basefull` in the same job.)

### Baseline ladder (`basefull`)

| ctx | C | decode p50 | aggregate | TTFT p50 | TTFT p95 |
|---:|--:|---:|---:|---:|---:|
| 1024 | 1 | 81.30 | 43.16 | 1.44 | 1.44 |
| 1024 | 4 | 16.19 | 46.82 | 3.00 | 6.22 |
| 4096 | 1 | 80.61 | 27.74 | 2.99 | 3.00 |
| 4096 | 4 | 11.41 | 29.36 | 6.13 | 12.55 |
| 16384 | 1 | 82.33 | 11.50 | 9.59 | 9.60 |
| 16384 | 4 | 7.40 | 10.64 | 18.81 | 67.18 |

### Delta vs `basefull` (%, positive = better except TTFT)

| variant | what changed | 1k C1 | 1k C4 | 4k C1 | 4k C4 | 16k C1 | mean agg 1k/4k | mean TTFT 1k/4k |
|:--|:--|---:|---:|---:|---:|---:|---:|---:|
| `sm70table` | `STRATA_SM70_TABLE=1` | **+7.0** | +2.7 | **+5.9** | +2.2 | **+6.6** | +1.8 | −0.0 |
| `pfwmma` | `STRATA_PF_WMMA=1` | +1.6 | −0.6 | +9.1 | +2.5 | +2.8 | **+2.2** | −1.7 |
| `grnormfast` | `STRATA_GR_NORM_FAST=1` | +7.7 | +0.5 | +1.6 | +0.1 | +1.5 | +0.6 | −0.0 |
| `argmaxrows` | `STRATA_ARGMAX_ROWS_SM70=1` | +3.1 | −0.3 | +5.3 | +0.0 | +2.2 | +0.6 | −0.3 |
| `fusedecode` | `STRATA_FUSE_DECODE=1` | +0.6 | +0.7 | +1.2 | +0.0 | +7.5 | +0.1 | +0.1 |
| `spec_pure` | `--set spec-min-p=0` | +3.6 | −0.7 | +3.5 | −0.3 | +4.8 | +0.4 | −0.1 |
| `mtpoff` | `--mtp off` | −0.2 | +5.6 | +0.1 | +3.4 | −7.9 | +1.6 | −2.1 |
| `kvres16k` | `--set kv-resident=16384` (engine clamps to 20,480) | +2.5 | +0.3 | −0.3 | +0.4 | −0.1 | +0.5 | −0.7 |
| `kvres64k` | `--set kv-resident=65536` | +3.3 | +0.7 | +2.6 | +1.8 | +5.8 | +1.1 | −0.1 |
| `kvfp16` | `--set kv=fp16` | +0.1 | −1.8 | +0.3 | −3.3 | −1.1 | **−3.2** | **+5.0** |
| `batch8` | `--parallel 8`, C=8 cells | −3.7 | −1.4 | +8.8 | −1.6 | +8.1 | −0.5 | +1.4 |

The numbers are per-cell, not smoothed: a ±2–3 % move at C1 (3 samples) or
on a batched cell is inside run noise. The 16k **C4** cell is excluded from
attribution entirely: its TTFT p50 ranged 18.4–42.3 s across variants with
the same workload (18.81 s in `basefull`), so it is not a stable control.

### What each switch did

- **`STRATA_SM70_TABLE=1` (the three Volta decode kernels together) is the
  clear winner**: +7.0 % at 1k C1, +5.9 % at 4k C1, +6.6 % at 16k C1, +2.2–2.7 %
  at C4 where the batch, not the single-token GPU path, dominates. This
  reproduces the maintainer's “22.8 → 21.2 ms per verify window” on this
  exact card and is worth turning on for single-stream serving.
- **`STRATA_PF_WMMA=1` (FP16 tensor-core prompt experts) helps the prompt
  path**: +2.2 % aggregate and −1.7 % TTFT on the 1k/4k mean, +9.1 % decode
  at 4k C1 (the one decode cell where prompt-expert work is still in flight),
  +2.8 % decode at 16k C1. Note this changes the expert arithmetic (FP16
  activations instead of MMQ q8_1), so it is not bit-identical; the V100 doc
  already warns answers can differ.
- **`STRATA_GR_NORM_FAST=1` / `STRATA_ARGMAX_ROWS_SM70=1`** produce the
  expected small positive decode effect (+2 % class) but only at C1; at C4
  they are inside noise. Both are bitwise per the V100 doc; safe to enable,
  but not a reason to rebuild a service.
- **`STRATA_FUSE_DECODE=1` is invisible end-to-end** (+0.6/+0.7 % at short
  context; +7.5 % only on the unstable 16k C1 sample). Matches the doc's
  “4–16 µs per token, not visible end to end”. Bitwise; harmless, no
  measurable win here.
- **`--set spec-min-p=0`** (accept every draft extension) helps single-stream
  decode by 3.5–4.8 % at C1 but not at C4 (−0.3 to −0.7 %). With four slots
  the wider verify windows cost more than the extra accepted tokens.
- **`--mtp off`**: at 1k/4k it is a wash (the MTP drafter's win is small when
  the GPU is already batch-bound), and at 16k C1 it is clearly worse
  (decode −7.9 %, aggregate −3.6 %, TTFT +3.4 %). Keep the drafter.
- **`--kv fp16` is a loss**: aggregate −3.2 %, TTFT +5.0 % on the mean;
  int8 is the right KV mode on V100 (half the KV traffic, and the int8
  dequant is not the bottleneck).
- **`--kv-resident 16384/65536`**: neutral at ≤16k, as expected — the cells
  never reach the resident window, so only the 64k setting can matter and it
  is +1.1 % (noise). Not a knob to tune at these contexts.
- **`--parallel 8`**: C8 aggregate is 48.66 (1k), 30.07 (4k), 10.57 (16k) vs
  C4's 46.82/29.36/10.64. No throughput gain (memory-bandwidth-bound), while
  per-stream decode halves (7.20/4.64/7.21) and TTFT p95 grows (11.4/22.3/
  167.2 s). Batch 4 is the right shape on one V100.

## Findings, caveats and gap list

### What the campaign establishes

1. **Both 2× configs are fully resident, and 4× is over-provisioned.** The
   engine caches every profiled (layer, expert) pair on every config:
   12,288/12,288 for Coder and 24,576/24,576 for Q2_0 on 2×64 GB, and
   24,576/24,576 for IQ3_S on 4×128 GB, with `100% of the experts resident`
   in all three start lines. IQ3_S leaves **8.5–11.8 GiB free on every card**;
   the 4× configuration does not need all its VRAM. No config streamed
   experts (KV streaming is a separate, intended axis).
2. **Context:** 128k works on 2× for both Coder (51.8 s TTFT, 75.9 tok/s at
   C1) and Q2_0 (54.2 s, 78.8 tok/s). **262,144 tokens works on 4×**: a
   261,904-token prompt prefills in 72.0 s and decodes at 72.2 tok/s at C1
   (aggregate 1.76 tok/s); C2 also completes but aggregate is flat
   (1.72 tok/s) while per-stream decode collapses to 1.69 tok/s.
3. **Warm cache is the biggest TTFT lever at long context**: 64k 16.94 →
   1.06 s (15.9×), 128k 31.80 → 1.40 s (22.7×), **262k 68.96 → 3.13 s
   (22.0×)**, with full reuse (prompt−5 tokens) and `cache_reuse_engaged`.
   Restore cost grows only ~3× while prefill grows 4×.
4. **Scaling:** the second card buys +20–30 % aggregate and −19…−37 % TTFT at
   1k–16k C1; the four cards turn IQ3_S from streaming into resident and buy
   +76 % (16k C1) to +103 % (16k C4) aggregate and −46…−72 % TTFT vs 1×. At
   max context, concurrency does not add throughput: prefill of the next
   stream head-of-line-blocks decode of the previous one.
5. **Volta kernels:** `STRATA_SM70_TABLE=1` is the only unambiguous win
   (+7.0 % 1k C1, +5.9 % 4k C1, +6.6 % 16k C1; +1.8 % aggregate).
   `STRATA_PF_WMMA=1` is a smaller but consistent prompt-path win (+2.2 %
   aggregate, −1.7 % TTFT; it changes expert arithmetic). `GR_NORM_FAST` and
   `ARGMAX_ROWS_SM70` are +2 % class at C1 only. `FUSE_DECODE` is end-to-end
   invisible. `--kv fp16` is worse (−3.2 % aggregate, +5 % TTFT);
   `--parallel 8` adds no throughput; `spec-min-p=0` helps C1 only; MTP should
   stay on (16k C1 −7.9 % decode without it).

### Caveats

- One run per cell, 3 requests per slot. C1 p95 is the slowest of three
  samples. `control` vs `basefull` at 1k C1 differed by only −0.2 %, but the
  same recipe was ~6 % below the 2026-10-25 1× baseline, so **cross-day**
  comparisons carry a ~5–6 % error bar; all kernel deltas are within-job
  against `basefull`.
- The 16k **C4** sweep cell is unstable (TTFT p50 18.4–42.3 s for identical
  work) and is excluded from attribution.
- On a layer split the engine's per-request `hit_rate` is `null` and the
  batch summary prints `routed entries: VRAM 0.0 (0.0%)`; that is a
  bookkeeping artefact. Residency is proven by the `100% of the experts
  resident` and `N of N profiled pairs` lines.
- 262k C2's low per-stream decode is real but is a scheduling effect, not a
  kernel regression.
- The V100 doc says a 2-GPU split “adds expert-cache room, not prompt speed”;
  that was measured without this exact kernel/engine combination. Here 2×
  does improve TTFT and short-context aggregate, but the split is at most
  24 layers/side and the pipeline adds a hand-off per window.
- Q2_0 is a 2-bit pack: these are performance numbers only, not a quality
  comparison with IQ3_S/Coder.
- Staging (single-OST 28.8 GB shard, 4–14 MB/s) dominated wall time; the
  benchmark itself is ~1 h for A+B+warm and ~80 min for E.

### Gap list (not measured)

- **2× IQ3_S** — the 2×/4× resident crossover for the 93 GB pack. 2× has
  64 GB, so IQ3_S would be partly streamed at 128k/256k; how much is unknown.
- **1× Q2_0** — does not fit 32 GB; the streaming share and its TTFT were not
  measured (the 2026-10-25 baseline has no Q2_0 at all).
- **262k at C>2**, and 262k C4 — not attempted; a 512k window was not tried
  (the pack's trained context is 262,144).
- **Warm at C>1** and **warm on 2× Q2_0** — only C1 warm at 64k/128k/256k.
- **Quality at 262k** (needle-in-a-haystack / long-doc QA), and the answer
  differences introduced by `STRATA_PF_WMMA=1` (FP16 expert activations).
- **Power, clocks and thermal throttling** — not sampled; V100-SXM2 power
  headroom may explain part of the day-to-day variance.
- **NVLink/P2P topology** — the engine only printed the 12.4 GB/s host→device
  PCIe probe; cross-card link bandwidth was not measured.
- **`--batch-mtp`, `STRATA_BATCH_MTP[_SERIAL]`, `kv q4_0`/`k8v4`,
  `--vram-reserve-mib`, `--prefill-pipe`** — not part of this sweep.
- **Vision/`mmproj`** — `--vision off` throughout, as in the baseline.

## Jobs and provenance

| job | node | config | wall | outcome |
|--:|:--|:--|--:|:--|
| 38239014 | volta05 | A, 2× Coder, `COPY_MODELS=0` | 0:15:33 | aborted: engine load watchdog (Lustre random reads ~5 MB/s); engine log proves `--layer-split auto` |
| 38239036 | volta05 | A, 2× Q2_0 | 1:38:30 | aborted: 810 s load watchdog after a 75 min single-OST staging; retried |
| 38239040 | volta05 | A, 2× Coder | **0:59:06** | complete, 13/13 cells `ok` |
| 38240250 | volta05 | Q2_0 diagnostic | 0:03:04 | `--warm-cells ""` arg error (harness), no data |
| 38240325 | volta05 | Q2_0 diagnostic, `COPY_MODELS=0` | 0:14:42 | load watchdog; proved the pack loads, storage was the limit |
| 38240592 | volta05 | A, 2× Q2_0 retry | — | cancelled when the one-allocation plan replaced it |
| **38240753** | volta05 | A Q2_0 + B IQ3_S + warm + E knobs | **3:41:51** | complete: 13 + 15 + 3 warm + 78 knob cells, all `ok` |

All jobs: `--nice=100`, partition `ffa-gpulab`, `--nodelist=volta05`.
Job 38239040 used 2 GPUs / 24 CPUs / 200 GB; 38240753 used 4 GPUs / 32 CPUs /
340 GB. The engine in every completed run was
`0.1.50-linux-x86_64-cuda-cuda12` (sm_70), and every launch was verified
against the engine's `engine started:` line (files in each pack dir).

## Artifacts

Raw results (never edited after the run):

```text
/work/ffa/holecem/strata-runs/2026-10-26-v100-resident/
  a-2x-coder/       job 38239040: cells.jsonl, summary.csv, engine-argv.txt,
                    engine-facts.txt, launch-config.json, engine-logs/
  a-2x-q2_0/        job 38240753, Q2_0 2x (same files)
  b-4x-iq3s/        job 38240753, IQ3_S 4x cold + warm (warm-cells.jsonl,
                    summary-warm.csv)
  e-knobs-coder/    job 38240753, knobs.csv, variants.csv,
                    coder/<variant>/{cells.jsonl,summary.csv,engine-*/,...}
  _attempts/        aborted runs with their real errors
  NOTES.md, report.py
/work/ffa/holecem/strata-runs/2026-10-26-v100-knobs/   (CSV view + symlink)
/work/ffa/holecem/strata-runs/2026-10-26-v100-warm/    (warm view + symlink)
```

Harness/doc commits on branch `serve-time-36h`:

- `44bba48` — Q2_0 pack, `--cells`/`--warm-cells`, explicit baseline recipe,
  sm_70 kernel variants.
- `c194099` — parallel staging of the two single-striped shards.
- `aec6f8a` — this document (residency, 256k, warm, kernel sweep).
- `e6b183d` — full knob-sweep appendix and the warm-method note.
- `98ea07b` — doc fixes (expert-cache size, KV wording, baseline prompt ratio).

Q2_0 manifest entries were appended to
`/work/ffa/holecem/models/strata/manifest.txt` (2026-10-26): shard 1 verified
(`69820c02…b199b720`), shard 2 `316b46f3…61e113` by hardlink to the verified
shared inode.

## Cells not run, and why

- **A/Q2_0 131072 C2/C4**: only C1 was in the requested ladder for A; the
  wall budget was kept for B/E.
- **B 262144 C4**: not requested and would only add prefill contention; C2
  already showed aggregate is flat.
- **Warm for Q2_0**: A's server was already stopped when the warm pass was
  defined for the 4× job; would need another 30–90 min of staging.
- **`nospec`** (removing `--spec`): impossible on a native IQ pack (the
  engine refuses to start); already recorded as a launch-fail in the
  2026-10-25 sweep and not re-run.
- **`kvbf16` / `expert-cache max`**: flags do not exist in 0.1.50; recorded
  as `skipped-flag` in 2026-10-25, not repeated.

## Appendix — every knob-sweep cell

Raw values behind the delta table. Each variant's exact engine argv is in
`e-knobs-coder/coder/<variant>/engine-logs/launch/*.json` and is the
baseline argv below plus only the `--set`/env named in `variants.csv`:

```text
strata --serve --pack <private>/packs/coder-iq1_m --native <staged shard1>
  --ple-gguf <staged shard2> --expert-profile expert-profile-coder.bin
  --max-context 65536 --kv int8 --kv-resident 32768 --expert-cache auto
  --prefill auto --ple-io ram --spec 4 --spec-min-p 0.5 --mtp <private>/mtp/rt
  --batch 4
```

| variant | ctx | C | decode p50 | aggregate | TTFT p50 | TTFT p95 |
|:--|---:|--:|---:|---:|---:|---:|
| basefull | 1024 | 1 | 81.30 | 43.16 | 1.44 | 1.44 |
| basefull | 1024 | 4 | 16.19 | 46.82 | 3.00 | 6.22 |
| basefull | 4096 | 1 | 80.61 | 27.74 | 2.99 | 3.00 |
| basefull | 4096 | 4 | 11.41 | 29.36 | 6.13 | 12.55 |
| basefull | 16384 | 1 | 82.33 | 11.50 | 9.59 | 9.60 |
| basefull | 16384 | 4 | 7.40 | 10.64 | 18.81 | 67.18 |
| spec_pure | 1024 | 1 | 84.24 | 43.53 | 1.44 | 1.44 |
| spec_pure | 1024 | 4 | 16.07 | 46.53 | 2.99 | 6.22 |
| spec_pure | 4096 | 1 | 83.46 | 28.20 | 3.01 | 3.01 |
| spec_pure | 4096 | 4 | 11.38 | 29.30 | 6.09 | 12.54 |
| spec_pure | 16384 | 1 | 86.28 | 11.10 | 10.07 | 10.09 |
| spec_pure | 16384 | 4 | 7.40 | 10.54 | 19.02 | 82.48 |
| mtpoff | 1024 | 1 | 81.15 | 43.12 | 1.40 | 1.42 |
| mtpoff | 1024 | 4 | 17.09 | 48.64 | 2.92 | 6.13 |
| mtpoff | 4096 | 1 | 80.67 | 27.77 | 2.94 | 2.95 |
| mtpoff | 4096 | 4 | 11.80 | 30.04 | 6.04 | 12.40 |
| mtpoff | 16384 | 1 | 75.83 | 11.09 | 9.92 | 9.95 |
| mtpoff | 16384 | 4 | 11.00 | 10.18 | 22.63 | 60.55 |
| kvfp16 | 1024 | 1 | 81.36 | 41.91 | 1.49 | 1.51 |
| kvfp16 | 1024 | 4 | 15.90 | 45.72 | 3.11 | 6.47 |
| kvfp16 | 4096 | 1 | 80.88 | 26.84 | 3.19 | 3.19 |
| kvfp16 | 4096 | 4 | 11.03 | 28.05 | 6.51 | 13.28 |
| kvfp16 | 16384 | 1 | 81.40 | 10.34 | 10.91 | 10.92 |
| kvfp16 | 16384 | 4 | 10.13 | 9.28 | 42.31 | 56.14 |
| kvres16k | 1024 | 1 | 83.30 | 43.38 | 1.42 | 1.44 |
| kvres16k | 1024 | 4 | 16.24 | 46.81 | 2.97 | 6.21 |
| kvres16k | 4096 | 1 | 80.37 | 28.21 | 2.98 | 3.00 |
| kvres16k | 4096 | 4 | 11.45 | 29.36 | 6.13 | 12.52 |
| kvres16k | 16384 | 1 | 82.28 | 11.12 | 10.01 | 10.04 |
| kvres16k | 16384 | 4 | 10.95 | 10.12 | 38.77 | 60.97 |
| kvres64k | 1024 | 1 | 83.96 | 42.91 | 1.46 | 1.46 |
| kvres64k | 1024 | 4 | 16.31 | 46.67 | 3.04 | 6.31 |
| kvres64k | 4096 | 1 | 82.70 | 28.75 | 2.96 | 2.97 |
| kvres64k | 4096 | 4 | 11.61 | 29.82 | 6.01 | 12.39 |
| kvres64k | 16384 | 1 | 87.12 | 11.70 | 9.48 | 9.50 |
| kvres64k | 16384 | 4 | 7.36 | 10.76 | 18.42 | 93.54 |
| batch8 | 1024 | 1 | 78.27 | 41.93 | 1.47 | 1.48 |
| batch8 | 1024 | 4 | 15.96 | 45.90 | 3.07 | 6.35 |
| batch8 | 1024 | 8 | 7.20 | 48.66 | 3.12 | 11.37 |
| batch8 | 4096 | 1 | 87.67 | 28.86 | 3.01 | 3.01 |
| batch8 | 4096 | 4 | 11.23 | 28.97 | 6.17 | 12.63 |
| batch8 | 4096 | 8 | 4.64 | 30.07 | 6.24 | 22.32 |
| batch8 | 16384 | 1 | 89.00 | 11.10 | 10.10 | 10.15 |
| batch8 | 16384 | 4 | 10.72 | 9.87 | 26.76 | 78.99 |
| batch8 | 16384 | 8 | 7.21 | 10.57 | 30.70 | 167.15 |
| sm70table | 1024 | 1 | 86.98 | 43.79 | 1.43 | 1.44 |
| sm70table | 1024 | 4 | 16.63 | 47.70 | 3.00 | 6.23 |
| sm70table | 4096 | 1 | 85.38 | 28.55 | 3.00 | 3.00 |
| sm70table | 4096 | 4 | 11.66 | 29.69 | 6.15 | 12.50 |
| sm70table | 16384 | 1 | 87.80 | 11.61 | 9.59 | 9.65 |
| sm70table | 16384 | 4 | 10.90 | 10.15 | 26.30 | 112.27 |
| fusedecode | 1024 | 1 | 81.79 | 42.80 | 1.44 | 1.45 |
| fusedecode | 1024 | 4 | 16.31 | 46.86 | 3.00 | 6.21 |
| fusedecode | 4096 | 1 | 81.55 | 28.20 | 3.00 | 3.02 |
| fusedecode | 4096 | 4 | 11.41 | 29.24 | 6.13 | 12.61 |
| fusedecode | 16384 | 1 | 88.53 | 11.14 | 10.07 | 10.08 |
| fusedecode | 16384 | 4 | 10.85 | 10.03 | 26.38 | 74.24 |
| pfwmma | 1024 | 1 | 82.63 | 43.19 | 1.44 | 1.44 |
| pfwmma | 1024 | 4 | 16.10 | 46.60 | 3.01 | 6.22 |
| pfwmma | 4096 | 1 | 87.92 | 29.50 | 2.88 | 2.89 |
| pfwmma | 4096 | 4 | 11.69 | 30.14 | 5.91 | 12.08 |
| pfwmma | 16384 | 1 | 84.64 | 11.71 | 9.47 | 9.52 |
| pfwmma | 16384 | 4 | 11.55 | 10.62 | 25.27 | 95.05 |
| argmaxrows | 1024 | 1 | 83.83 | 43.44 | 1.44 | 1.44 |
| argmaxrows | 1024 | 4 | 16.14 | 46.73 | 3.00 | 6.23 |
| argmaxrows | 4096 | 1 | 84.89 | 28.37 | 2.97 | 3.00 |
| argmaxrows | 4096 | 4 | 11.41 | 29.30 | 6.10 | 12.54 |
| argmaxrows | 16384 | 1 | 84.14 | 11.48 | 9.63 | 9.63 |
| argmaxrows | 16384 | 4 | 7.39 | 10.56 | 18.86 | 67.37 |
| grnormfast | 1024 | 1 | 87.58 | 43.57 | 1.44 | 1.45 |
| grnormfast | 1024 | 4 | 16.27 | 46.80 | 3.00 | 6.23 |
| grnormfast | 4096 | 1 | 81.92 | 28.14 | 2.99 | 3.01 |
| grnormfast | 4096 | 4 | 11.42 | 29.34 | 6.12 | 12.64 |
| grnormfast | 16384 | 1 | 83.59 | 11.14 | 10.00 | 10.08 |
| grnormfast | 16384 | 4 | 10.86 | 10.05 | 26.44 | 61.25 |
