# 2026-10-25 — Strata smoke matrix, Qwen3.8-Flash-Next GGUFs on L40 + V100

First campaign for the Strata engine harness (`llm/strata/`). Three cluster
configurations completed: **1× L40S**, **2× L40S**, and **1× V100-SXM2-32GB**.
The runner agents' final reports were lost after the runs (connection error),
so this document was consolidated from the raw `summary.csv`, `cells.jsonl`,
launch logs, engine logs and runner notes under
`/work/ffa/holecem/strata-runs/2026-10-25-*`. A fourth configuration on
Vast.ai (2× Titan RTX) is a separate, still-running campaign and is only
reserved as a placeholder below.

## Setup

### Model packs (all configs)

```text
IQ3_S   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
        IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
            54,817,524,224 B  sha256 4c1eb2ce…d2aca3
        IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
            28,800,138,432 B  sha256 316b46f3…61e113  (shared table)
Coder   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
        IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
            29,608,446,496 B  sha256 e11083ba…087fad
        IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
            28,800,138,432 B  sha256 316b46f3…61e113  (same table as IQ3_S)
```

Both shards are mandatory per pack. SHA-256 values are pinned from
`/work/ffa/holecem/src/Strata/setup.py` (2026-09-29 revisions).
`fetch-models.sh --verify` was run against the shared library and reported
**`missing=0 corrupt=0` at 2026-10-25T16:30:49Z**; the fetch manifest records
`status=verified` with the same pinned SHA-256 for all four files. The l40x2
runner cites the same audit; the v100x1 runner's `logs/fetch-verify.log`
records the same four VERIFIED hashes and the same final line. The shared
28.8 GB shard 2 is the same content
under two pack-specific filenames; it was stored once and hardlinked.

`--vision off`, so the mmproj reader was not used.

### l40x1 — 1× NVIDIA L40S 48 GB

```text
node:       uk-g6-05 (gpu-ffa), 1× L40S 46,068 MiB, sm_89
driver:     615.71.09
tmpfs:      378 GB /tmp; 754 GB host RAM
engine:     0.1.50, variant …-linux-x86_64-cuda-cuda13 (engine start line)
             Terminal 0.2.6 / build 0.1.51
jobs:       38236263 (IQ3_S), 38236278 (Coder), supp 38236454/38236455
matrix:     ctx 1024,4096,16384,65536,131072 × C 1,2,4
```

Engine argv actually used (captured in the Coder engine log; IQ3_S is identical
except the pack/shards):

```text
strata --serve --pack <private>/packs/<pack> --native <shard1> --ple-gguf <shard2>
  --expert-profile <...>/expert-profile*.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <private>/mtp/rt --max-context 131072
  --kv int8 --ple-io ram --kv-resident 32768 --batch 4
```

The campaign's bench-matrix CLI was only `--model`, `--contexts`, `--batches`
and `--budget-min`; `bench-matrix.sh` itself passed
`--context`/`--parallel`/`--kv`/`--vision`/`--data-dir` to `strata launch`.
The remaining values above are engine defaults selected by `strata launch`,
not runner CLI flags. The runner additionally used `LD_LIBRARY_PATH` with the
staged `deps/cxx/lib` (EL9 libstdc++ fix) and a private `DATA_DIR` per pack.

Drafter: **present and loaded**. Launch logs: `Preparing the draft layer …
5.2 of 5.2 GB`; engine log: `strata mtp: draft layer loaded, 835 MiB of VRAM
(experts 675, dense 111)`, `draft head over 106299 tokens`, flags
`--spec 4 --spec-min-p 0.5`. The runner-side `DRAFTER-READY` marker never
appeared; that marker was not an engine readiness signal.

### l40x2 — 2× NVIDIA L40S 48 GB

```text
node:       uk-g6-08 (gpu-ffa), 2× L40S, sm_89
driver:     615.71.09 (gpu-ffa image; l40x1's node probe captured it;
            l40x2 logs did not repeat nvidia-smi)
engine:     0.1.50, variant …-linux-x86_64-cuda-cuda13 (engine start line)
jobs:       38236384 (IQ3_S), 38236385 (Coder); earlier attempts 38236146/47 cancelled
matrix:     ctx 1024,4096,16384,65536,131072 × C 1,2,4
```

Engine log confirms the two-card path: `strata generate: layer split across 2
GPUs: CUDA0, then CUDA1 (split auto)`, `layer split auto: K=24 - predicted 9.7
ms per decode window`, `layer split: CUDA0 runs layers 0-23`, `CUDA1 runs
layers 24-47`, `100% of the experts resident`, `26.9 GB/s` PCIe probe. The
harness at run time did not pass `--gpu`; the runner used a shim that injected
`--gpu 0,1`. This is now a first-class `--gpus`/`GPUS_PER_NODE` option (see
harness fixes).

Engine argv and drafter evidence are the same as l40x1, with
`--layer-split auto` and `--max-context 131072`; the engine log again says
`strata mtp: draft layer loaded, 835 MiB of VRAM`.

### v100x1 — 1× NVIDIA Tesla V100-SXM2-32GB

```text
node:       volta05 (ffa-gpulab), 1× V100-SXM2-32GB 32,768 MiB, sm_70
driver:     580.178.04
engine:     0.1.50, variant …-linux-x86_64-cuda-cuda12 (engine start line)
jobs:       38236131 (IQ3_S), 38236434 (Coder); 38236480 diagnostic (leftover)
matrix:     ctx 1024,4096,16384,65536 × C 1,2,4
```

Engine argv actually used (captured from `/proc/<pid>/cmdline`):

```text
strata --serve --pack <shared>/packs/coder-iq1_m --native <shard1>
  --ple-gguf <shard2> --expert-profile <...>/expert-profile-coder.bin
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp <shared>/mtp/rt --max-context 65536 --kv int8 --ple-io ram
  --kv-resident 32768 --batch 4
```

Engine log: `GPU 0: Tesla V100-SXM2-32GB, compute capability 7.0`, `12.4 GB/s
host->device`, `KV streaming: 32768 of 65536 cells per QSA layer in VRAM`,
`expert cache 11773 slots, 22.43 GiB`, `strata mtp: draft layer loaded, 835
MiB`. The cuda12 variant is the one Control selected for sm_70; the cuda13
variant is not usable on Volta.

Drafter: present and loaded in both final packs (`Preparing the draft layer`
in both launch logs; `draft layer loaded` in the captured engine log).

## Matrix contract and methodology

```text
probe:       strata_probe.py (stdlib OpenAI streaming client)
cell:        C concurrent streams × 3 sequential requests
max_tokens:  128
temperature: 0
prompt:      unique salt + deterministic filler, no prompt-cache reuse
sampling:    p50/p95 over per-request samples; aggregate = total completion
             tokens / wall from first dispatch to last termination
TTFT:        dispatch → first content/reasoning/tool delta
decode:      engine llama.cpp-style timings.predicted_ms when present,
             else client stream window
prefill:     prompt_tokens / engine timings.prompt_ms when present
budget:      420 min (final matrices); no skipped-budget cells
status:      ok | fail | server-dead | skipped-budget
```

- **Repeats**: final matrices ran each (ctx, C) cell once, with 3 requests per
  slot (3C samples). A fully failed cell can trigger one server restart and one
  retry; no final-matrix cell consumed a restart. At C1 p95 is the slowest of
  three samples; at C>1 it is over 3C samples. Either way treat it as a tail
  indicator, not a stable percentile.
- **Medians**: `decode_tok_s_p50` is the median over all successful requests in
  the cell; the summary CSV is a projection of the last JSONL line. Where the
  CSV was ambiguous, the JSONL was used for `decode_tok_s_p95`.
- **Context calibration**: the probe was written with a ~4 chars/token
  assumption. The server tokenizer yields ~1.139 tokens per 4-char unit at
  large contexts: measured prompt tokens per request were 1,254 at nominal
  1024 (1.225×; fixed chat-template overhead dominates), 4,726 at 4096
  (1.154×), 18,708 at 16384 (1.142×), and 74,632 at 65536 (1.139×). A nominal
  131072 cell therefore carried ~149,187 real tokens and was rejected by the
  131072-token window. The supplementary runs used a nominal 114000 context,
  which landed at **129,782** (IQ3_S) / **129,781** (Coder) real prompt tokens.
  The harness now measures chars/token with two small `include_usage`
  requests (slope + chat-template intercept), scales the filler, clamps to
  `max_context - max_tokens - pad_safety`
  (default safety 128), and records the calibration and clamp in every JSONL
  cell. Historical numbers below are the uncalibrated ones.
- **Drafter handling**: results were not discarded for the absent
  `DRAFTER-READY` marker. Engine logs prove the MTP drafter loaded in every
  final run; where `/metrics` exposed acceptance it was ~0.77–0.81. The
  l40x2 engine's `/metrics` request records report `hit_rate: null` (the
  layer-split path keeps all experts resident and reports 0 routed misses),
  so no per-cell cache percentage is reconstructed for that config.
- **KV mode**: `--kv int8 --kv-resident 32768`. L40: 32,768 of 131,072 cells
  per QSA layer in VRAM, K/V in 1.55 GiB pinned RAM. V100: 32,768 of 65,536,
  K/V in 0.77 GiB pinned RAM. Block-read VRAM hit was 58–65% at short
  contexts and 96–99% at long contexts; the 2× L40 layer split keeps 100% of
  experts resident.
- **Expert cache**: `--expert-cache auto --batch 4`. l40x1 IQ3_S main cells:
  99.9% (73,341/73,432 and 76,689/76,794); supplementary 99.8–100% (first
  cold sample 95.8%). The l40x1 Coder engine-log snapshot has no
  `decode expert cache hit rate` line, but its C4 batch windows report
  95.3–96.8% of routed entries from VRAM (1.1% PCIe, 3.6% CPU). V100
  per-cell hit rates from the runner's `cells.jsonl.report.tsv`: 98.5–100%.
  l40x2: all experts resident in the layer split, so the metric is not
  comparable (null in JSONL).

## Results

All tables are from `summary.csv`; `decode p95` is from `cells.jsonl`.
`—` means the cell produced no successful request; status is `fail`.

### l40x1 — IQ3_S (1× L40S)

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | status |
|---:|--:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 141.69 | 145.84 | 68.50 | 0.92 | 1.03 | ok |
| 1024 | 2 | 44.15 | 63.69 | 66.40 | 0.91 | 1.86 | ok |
| 1024 | 4 | 27.99 | 49.36 | 79.82 | 1.85 | 3.77 | ok |
| 4096 | 1 | 153.98 | 154.80 | 57.48 | 1.40 | 1.41 | ok |
| 4096 | 2 | 37.31 | 64.05 | 52.54 | 1.41 | 2.89 | ok |
| 4096 | 4 | 22.49 | 45.45 | 60.13 | 2.81 | 5.74 | ok |
| 16384 | 1 | 152.98 | 154.74 | 23.00 | 4.72 | 4.75 | ok |
| 16384 | 2 | 24.74 | 164.14 | 20.41 | 7.57 | 11.17 | ok |
| 16384 | 4 | 24.68 | 167.78 | 20.36 | 11.16 | 36.79 | ok |
| 65536 | 1 | 140.67 | 144.16 | 7.00 | 17.39 | 17.44 | ok |
| 65536 | 2 | 24.11 | 134.33 | 6.69 | 32.95 | 36.66 | ok |
| 65536 | 4 | 24.21 | 136.52 | 6.68 | 32.99 | 148.03 | ok |
| 131072 | 1 | — | — | — | — | — | fail |
| 131072 | 2 | — | — | — | — | — | fail |
| 131072 | 4 | — | — | — | — | — | fail |

Supplementary 128K-class cells (nominal 114000 = 129,782 real prompt tokens;
same server, `--context 131072`):

| nominal ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | prefill tok/s | status |
|---:|--:|---:|---:|---:|---:|---:|---:|:--|
| 114000 | 1 | 121.64 | 125.23 | 4.11 | 30.08 | 30.16 | 4,323 | ok 3/3 |
| 114000 | 2 | 23.32 | 118.68 | 3.99 | 58.45 | 62.43 | 4,088 | ok 6/6 |
| 114000 | 4 | 23.81 | 126.57 | 4.01 | 62.48 | 286.39 | 4,093 | ok 12/12 |

### l40x1 — Coder IQ1_M (1× L40S)

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | status |
|---:|--:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 140.88 | 148.98 | 86.05 | 0.56 | 0.66 | ok |
| 1024 | 2 | 49.84 | 62.76 | 80.21 | 0.57 | 1.18 | ok |
| 1024 | 4 | 31.27 | 44.11 | 97.23 | 1.15 | 2.38 | ok |
| 4096 | 1 | 146.09 | 161.94 | 62.27 | 1.21 | 1.21 | ok |
| 4096 | 2 | 39.40 | 64.01 | 57.09 | 1.20 | 2.47 | ok |
| 4096 | 4 | 23.32 | 42.72 | 64.50 | 2.44 | 4.95 | ok |
| 16384 | 1 | 152.38 | 155.59 | 24.56 | 4.39 | 4.40 | ok |
| 16384 | 2 | 20.91 | 147.91 | 22.02 | 4.44 | 10.55 | ok |
| 16384 | 4 | 20.81 | 145.42 | 21.09 | 18.28 | 28.94 | ok |
| 65536 | 1 | 126.62 | 130.05 | 7.45 | 16.19 | 16.26 | ok |
| 65536 | 2 | 18.67 | 122.14 | 7.11 | 29.10 | 34.38 | ok |
| 65536 | 4 | 18.76 | 122.44 | 7.12 | 29.04 | 137.01 | ok |
| 131072 | 1 | — | — | — | — | — | fail |
| 131072 | 2 | — | — | — | — | — | fail |
| 131072 | 4 | — | — | — | — | — | fail |

Supplementary 128K-class cells (nominal 114000 = 129,781 real tokens):

| nominal ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | prefill tok/s | status |
|---:|--:|---:|---:|---:|---:|---:|---:|:--|
| 114000 | 1 | 117.76 | 123.72 | 4.32 | 28.63 | 28.67 | 4,541 | ok 3/3 |
| 114000 | 2 | 18.29 | 112.50 | 4.18 | 54.01 | 59.41 | 4,281 | ok 6/6 |
| 114000 | 4 | 18.43 | 115.65 | 4.19 | 114.99 | 145.61 | 4,289 | ok 12/12 |

### l40x2 — IQ3_S (2× L40S)

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | status |
|---:|--:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 158.99 | 163.39 | 94.15 | 0.54 | 0.62 | ok |
| 1024 | 2 | 52.75 | 66.86 | 84.53 | 0.54 | 1.13 | ok |
| 1024 | 4 | 34.11 | 51.86 | 107.07 | 1.11 | 2.28 | ok |
| 4096 | 1 | 157.06 | 157.83 | 62.83 | 1.23 | 1.23 | ok |
| 4096 | 2 | 40.48 | 65.72 | 57.18 | 1.24 | 2.53 | ok |
| 4096 | 4 | 24.55 | 49.53 | 66.39 | 2.51 | 5.10 | ok |
| 16384 | 1 | 164.02 | 167.52 | 30.15 | 3.46 | 3.48 | ok |
| 16384 | 2 | 24.95 | 149.10 | 22.54 | 6.66 | 9.41 | ok |
| 16384 | 4 | 24.96 | 152.54 | 22.09 | 9.43 | 30.32 | ok |
| 65536 | 1 | 138.90 | 146.05 | 10.53 | 11.26 | 11.30 | ok |
| 65536 | 2 | 25.83 | 136.24 | 7.38 | 31.34 | 31.37 | ok |
| 65536 | 4 | 25.92 | 139.77 | 7.41 | 31.31 | 157.18 | ok |
| 131072 | 1 | — | — | — | — | — | fail |
| 131072 | 2 | — | — | — | — | — | fail |
| 131072 | 4 | — | — | — | — | — | fail |

### l40x2 — Coder IQ1_M (2× L40S)

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | status |
|---:|--:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 131.40 | 139.25 | 88.07 | 0.46 | 0.55 | ok |
| 1024 | 2 | 51.85 | 66.54 | 86.54 | 0.47 | 0.98 | ok |
| 1024 | 4 | 32.79 | 45.49 | 105.28 | 0.96 | 2.00 | ok |
| 4096 | 1 | 140.43 | 145.42 | 62.74 | 1.12 | 1.13 | ok |
| 4096 | 2 | 40.10 | 64.03 | 58.90 | 1.13 | 2.31 | ok |
| 4096 | 4 | 23.91 | 42.62 | 66.60 | 2.30 | 4.70 | ok |
| 16384 | 1 | 145.95 | 153.31 | 31.61 | 3.19 | 3.21 | ok |
| 16384 | 2 | 25.84 | 139.22 | 23.62 | 6.29 | 8.88 | ok |
| 16384 | 4 | 25.80 | 139.15 | 23.08 | 11.96 | 34.59 | ok |
| 65536 | 1 | 119.86 | 122.94 | 11.31 | 10.32 | 10.32 | ok |
| 65536 | 2 | 26.67 | 118.21 | 7.81 | 29.62 | 29.62 | ok |
| 65536 | 4 | 26.77 | 118.91 | 7.60 | 29.58 | 131.65 | ok |
| 131072 | 1 | — | — | — | — | — | fail |
| 131072 | 2 | — | — | — | — | — | fail |
| 131072 | 4 | — | — | — | — | — | fail |

### v100x1 — IQ3_S (1× V100-SXM2-32GB)

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | status |
|---:|--:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 76.35 | 80.35 | 27.76 | 2.86 | 2.86 | ok |
| 1024 | 2 | 19.82 | 39.71 | 27.13 | 2.88 | 6.00 | ok |
| 1024 | 4 | 9.15 | 27.62 | 29.98 | 2.93 | 12.27 | ok |
| 4096 | 1 | 89.69 | 91.06 | 24.77 | 3.72 | 3.81 | ok |
| 4096 | 2 | 17.75 | 39.51 | 23.01 | 3.76 | 7.75 | ok |
| 4096 | 4 | 10.07 | 27.08 | 24.99 | 7.62 | 15.63 | ok |
| 16384 | 1 | 81.83 | 91.80 | 8.53 | 13.52 | 13.61 | ok |
| 16384 | 2 | 9.91 | 77.59 | 7.73 | 20.14 | 30.31 | ok |
| 16384 | 4 | 9.96 | 79.69 | 7.75 | 53.15 | 69.71 | ok |
| 65536 | 1 | — | — | — | — | — | fail |
| 65536 | 2 | — | — | — | — | — | fail |
| 65536 | 4 | — | — | — | — | — | fail |

### v100x1 — Coder IQ1_M (1× V100-SXM2-32GB)

| ctx | C | decode p50 | decode p95 | aggregate | TTFT p50 | TTFT p95 | status |
|---:|--:|---:|---:|---:|---:|---:|:--|
| 1024 | 1 | 86.44 | 90.83 | 40.05 | 1.70 | 1.73 | ok |
| 1024 | 2 | 24.28 | 38.83 | 35.89 | 1.73 | 3.68 | ok |
| 1024 | 4 | 14.46 | 26.57 | 40.60 | 3.67 | 7.67 | ok |
| 4096 | 1 | 84.16 | 84.34 | 25.97 | 3.38 | 3.39 | ok |
| 4096 | 2 | 18.22 | 38.49 | 24.14 | 3.40 | 7.09 | ok |
| 4096 | 4 | 10.34 | 26.05 | 26.11 | 7.07 | 14.54 | ok |
| 16384 | 1 | 86.39 | 86.59 | 9.55 | 12.00 | 12.08 | ok |
| 16384 | 2 | 10.62 | 79.14 | 8.68 | 17.02 | 26.49 | ok |
| 16384 | 4 | 10.59 | 78.55 | 8.64 | 31.78 | 91.33 | ok |
| 65536 | 1 | — | — | — | — | — | fail |
| 65536 | 2 | — | — | — | — | — | fail |
| 65536 | 4 | — | — | — | — | — | fail |

## Failed cells — actual strings and causes

### L40 131072 × C 1/2/4 (12 cells: 2 packs × 3 C × 2 L40 configs)

`cells.jsonl` records:

```text
all 3 requests failed (first: HTTPError: HTTP Error 400: Bad Request)
all 6 requests failed (first: HTTPError: HTTP Error 400: Bad Request)
all 12 requests failed (first: HTTPError: HTTP Error 400: Bad Request)
```

The engine-side body was captured by the supplementary diagnostic:

```json
{"error": {"type": "invalid_request_error", "message": "request exceeds the
context window: prompt (149187 tokens) leaves no room to answer in the context
(131072); requests are never truncated"}}
```

Cause: probe padding, not memory. The server was up and had already served
the 1K–64K cells from the same launch. Nominal 131072 produced 149,187 real
tokens because the old filler assumed 4 chars/token; the engine never
truncates. The 128K-class supplementary cells above cover the same window at
129.78k real tokens. A separate launch with `--context 132096` produced the
same rejection at 149,187 tokens, confirming the prompt—not the launch
window—was the problem.

### V100 65536 × C 1/2/4 (all six cells, both packs)

`cells.jsonl` records the same client-side error, with counts 3/6/12:

```text
all N requests failed (first: HTTPError: HTTP Error 400: Bad Request)
```

This is a **probe/context-window failure, not OOM**. Evidence:

- The server was launched at `--context 65536` and served every 1K, 4K and
  16K cell; the engine log shows a normal load (`expert cache 11773 slots`,
  `496 MiB of VRAM free`, no allocation failure).
- The same old probe produced **74,632** real prompt tokens for nominal 65536
  on the L40s, and the V100's 16K cells show the same 1.142× ratio
  (18,708 real tokens). The 65K prompt is therefore ~74.6k tokens;
  74,632 + 128 > 65,536, so the request is rejected at admission.
- The leftover diagnostic 38236480 ran nominal **56000** (63,781 real tokens)
  successfully at C1/C2/C4, which brackets the boundary below 65,536. Its
  65536 re-test is not used as a result because the shared probe was updated
  mid-run (see “Pending diagnostic” below).
- The exact V100 engine body was lost with the per-job scratch directory; that
  artifact-loss bug is fixed by copying engine logs before the EXIT trap. The
  L40 body above is the same server code path and the same error shape.

### Nothing else failed

No final-matrix cell was `server-dead` or `skipped-budget`; no final-matrix
restart was consumed. Earlier l40x1 attempts that were cancelled (GLIBCXX,
shared `mtp/rt` race) produced no results and are harness history, not matrix
cells.

### Pending diagnostic (job 38236480)

The leftover `ffa-gpulab strata-bench` job on `volta05` was still running at
the time of writing. It has already written
`/work/ffa/holecem/results/strata-bench-38236480-20261025T171007Z/iq3s/`:
nominal 56000 C1/C2/C4 are `ok` at 68.09/9.16/9.51 decode p50 tok/s with
63,781 real prompt tokens per request. Its 65536 C1 cell also completed
(75.25 decode p50 tok/s) because the mid-run swap to the calibrated probe
made that prompt 59,595 real tokens — below the 65,536 window. That is
useful boundary evidence (the engine serves ~60K tokens at 65K context), but
the cell is not comparable to the original 74,632-token failures and it is
not treated as a matrix result. The job is not part of the three campaigns
and did not write under `strata-runs/`; if it finishes, treat it as
supplementary boundary evidence, not as a fourth matrix.

## Comparison

### 1× vs 2× L40 scaling (aggregate tok/s)

| pack | ctx | C1 1× | C1 2× | Δ C1 | C4 1× | C4 2× | Δ C4 |
|:--|---:|---:|---:|---:|---:|---:|---:|
| IQ3_S | 1024 | 68.50 | 94.15 | +37% | 79.82 | 107.07 | +34% |
| IQ3_S | 4096 | 57.48 | 62.83 | +9% | 60.13 | 66.39 | +10% |
| IQ3_S | 16384 | 23.00 | 30.15 | +31% | 20.36 | 22.09 | +8% |
| IQ3_S | 65536 | 7.00 | 10.53 | +50% | 6.68 | 7.41 | +11% |
| Coder | 1024 | 86.05 | 88.07 | +2% | 97.23 | 105.28 | +8% |
| Coder | 4096 | 62.27 | 62.74 | +1% | 64.50 | 66.60 | +3% |
| Coder | 16384 | 24.56 | 31.61 | +29% | 21.09 | 23.08 | +9% |
| Coder | 65536 | 7.45 | 11.31 | +52% | 7.12 | 7.60 | +7% |

The second card helps most at long context and C1: the layer split keeps all
experts resident and cuts prefill/TTFT (e.g. IQ3_S 65536 TTFT p50 17.39 s →
11.26 s; Coder 16.19 s → 10.32 s). At short context the C1 gain is small for
Coder (+2%, 88.07 vs 86.05 aggregate, within run noise), and C4 gains are
modest because per-stream decode is already batch-bound. Per-stream C1
decode also rises on 2× at some points (IQ3_S 158.99 vs 141.69 at 1K; 164.02
vs 152.98 at 16K), but this is a single run per point, not a scaling law.

### IQ3_S vs Coder IQ1_M

Coder IQ1_M is the smaller, lower-bit pack (29.6 GB shard 1 vs 54.8 GB) and is
a Coder variant, so this is a **performance** comparison only.

| config | C | 1K | 4K | 16K | 65K |
|:--|--:|---:|---:|---:|---:|
| l40x1 | 1 | +26% | +8% | +7% | +6% |
| l40x1 | 4 | +22% | +7% | +4% | +7% |
| l40x2 | 1 | −6% | 0% | +5% | +7% |
| l40x2 | 4 | −2% | 0% | +4% | +3% |
| v100x1 | 1 | +44% | +5% | +12% | n/a |
| v100x1 | 4 | +35% | +4% | +11% | n/a |

(Δ = Coder aggregate vs IQ3_S aggregate.) Coder is better at 1K on 1× L40
and V100, where its smaller working set reduces prefill/cache pressure; on 2×
L40 it is 2–6% lower at 1K but still slightly better at 16K/65K. The two are
roughly equal at 4K. IQ3_S still has the higher per-stream decode at long
context on 1× L40 (140.67 vs 126.62 at 65K C1), so the Coder advantage there
is aggregate/scheduling, not raw decode.

### V100 vs L40

| pack | ctx | C | V100 aggregate | L40 1× aggregate | V100/L40 |
|:--|---:|--:|---:|---:|---:|
| IQ3_S | 1024 | 1 | 27.76 | 68.50 | 0.41× |
| IQ3_S | 4096 | 1 | 24.77 | 57.48 | 0.43× |
| IQ3_S | 16384 | 1 | 8.53 | 23.00 | 0.37× |
| IQ3_S | 1024 | 4 | 29.98 | 79.82 | 0.38× |
| IQ3_S | 4096 | 4 | 24.99 | 60.13 | 0.42× |
| IQ3_S | 16384 | 4 | 7.75 | 20.36 | 0.38× |
| Coder | 1024 | 1 | 40.05 | 86.05 | 0.47× |
| Coder | 4096 | 1 | 25.97 | 62.27 | 0.42× |
| Coder | 16384 | 1 | 9.55 | 24.56 | 0.39× |
| Coder | 1024 | 4 | 40.60 | 97.23 | 0.42× |
| Coder | 4096 | 4 | 26.11 | 64.50 | 0.40× |
| Coder | 16384 | 4 | 8.64 | 21.09 | 0.41× |

V100 is ~2.1–2.7× slower in aggregate and ~1.6–1.9× slower per stream (C1
decode ratio 0.53–0.61). The cuda12 engine, 32 GB card, 12.4 GB/s host→device
link and no layer split all contribute; this is not an ISA-only comparison.
V100 has no 65K matrix result because the old probe overshot the 65,536
window; 65K is a legal engine context on this card.

### vs bw01 FP4 baseline (indicative only)

The Qwen3.8 bw-results baseline on bw01 (2× BW6000 96 GB, vLLM FP4 container,
500–800-token prompts) reported **~120 tok/s at C1 and ~335 tok/s at C3**.
The L40 C1 decode here is 141–159 tok/s (IQ3_S) and 131–152 tok/s (Coder) at
1K–16K, i.e. in the same per-stream range or slightly above; the C4 aggregates
are 80–107 tok/s at 1K, well below the bw01 C3 figure. That gap is not a
model/engine verdict: it compares a different runtime (vLLM FP4 vs Strata
GGUF), different hardware (2× BW6000 vs 1–2× L40), different prompt lengths
and different batch points. Treat the bw01 numbers as context, not a
like-for-like benchmark.

## Harness fixes landed after the campaign

The campaign exposed six confirmed runner bugs; all are fixed in
`llm/strata/` with regression checks where cheap:

1. **`fetch-models.sh` Hub prefixes** — `hf download` needs `IQ3_S/…` and
   `IQ1_M/…`; bare filenames exit 0 without fetching. The script now passes
   prefixed paths, asserts existence and exact size, and fails loudly.
2. **Shared shard 2** — the 28.8 GB table is downloaded once into
   `_shared/Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf` and hardlinked
   into both pack dirs (copy fallback), instead of twice.
3. **`MODEL_NAME` / `GPUS_PER_NODE`** — now real env overrides (unknown values
   fail fast). `--model` / `--gpus` win; `GPUS_PER_NODE=2` becomes `--gpu 0,1`.
   Comma lists in Slurm `--export` are documented as unsafe; use trailing CLI
   args.
4. **Shared `DATA_DIR` race** — each job gets
   `$DATA_DIR/jobs/<job-id>`, seeded with hardlinked read-only `mtp/`/`packs/`
   trees and a copied mutable `mtp/rt/`. If no complete shared tree exists,
   the runner holds `$DATA_DIR/.prepare.lock` across the first launch.
5. **EL9 libstdc++ floor** — a startup precheck checks the selected engine's
   highest `GLIBCXX_*` and either uses the node library or prepends
   `CXX_LIB_DIR` (default the reusable l40x1 `deps/cxx/lib`), otherwise exits
   immediately with the workaround.
6. **Engine logs before cleanup** — `stop_everything` now archives
   `$XDG_STATE_HOME/strata/` into `OUTDIR/<model>/engine-logs/` before the
   scratch `rm -rf`.

Additionally, `strata_probe.py` now calibrates chars/token with two small
`include_usage` requests, clamps prompts to
`max_context - max_tokens - pad_safety`, and records
`prompt_target_tokens`, `prompt_clamped`, `max_context`, `pad_safety` and the
calibration in every JSONL cell. On a future 131072 run the top cell lands at
~130.8k real tokens (with `max_tokens=128`, safety 128) instead of 149,187.

Self-test evidence (login node, 2026-10-25): `bench-matrix.sh --self-test`
passes 4/4 stub cells (decode exactly 100 tok/s, aggregate doubles C1→C2),
records calibration 4.007 chars/token, clamps the nominal 1024 cell to 864
prompt tokens, archives a fake engine log, and verifies that a private `rt/`
write does not touch the shared tree. `tests/test_fetch_models.sh` passes with
fake `hf`/`sha256sum`: prefixed paths, exactly one shard-2 download, hardlinks
into both packs, exact sizes, a clean `--verify` audit, and a loud failure when
`hf` creates nothing.

## Gotchas / caveats

- One run per point; no statistical sweep. Node-to-node and run-to-run noise
  is visible in the C1 aggregates (e.g. Coder 4K 62.27 vs 62.74 on 1×/2×).
- TTFT includes admission queueing at C>1; C4 p95 TTFT is dominated by the
  last stream (e.g. 148 s at 65K on l40x1).
- Aggregate at C1 includes prefill gaps between the 3 sequential requests, so
  it is lower than the per-stream decode number by construction.
- The l40x1 IQ3_S internal engine log survived only as excerpts; the Coder
  engine log is the complete captured argv/log. The v100 Coder engine log was
  captured live; the IQ3_S v100 engine log was not.
- The drafter was loaded in all final runs, but the runner-side
  `DRAFTER-READY` marker was never written. Do not use that marker as a
  correctness gate; grep the engine log for `draft layer loaded`.
- The 2× L40 run used a shim to inject `--gpu 0,1`; the harness now does this
  natively. The 2× node's driver was not re-queried (same gpu-ffa image as
  l40x1, which reported 615.71.09).
- The 131072 and V100 65536 failures are historical probe-calibration
  artifacts. Re-running with the fixed probe will produce a real (clamped)
  prompt rather than a 400.

## Vast.ai 2x Titan RTX — pending (separate run)

Placeholder. The Vast.ai 2× Titan RTX campaign is a separate run and is **not**
part of this document's matrices, comparisons or artifacts. Do not fold its
numbers into the L40/V100 tables; add a separate section when that run's
runner reports.

## Artifacts

```text
/work/ffa/holecem/strata-runs/2026-10-25-l40x1/
  FINAL-REPORT.md, NOTES.md
  iq3s/results-copy/{summary-iq3s.csv,cells-iq3s.jsonl}
  coder/results-copy/{summary-coder.csv,cells-coder.jsonl}
  coder/engine-logs/iq1_m-128k.engine.log
  supp/iq3s/, supp/coder/ (129.78k cells, launch logs)
  logs/ (job outputs, fetch verify, drafter poll)
/work/ffa/holecem/strata-runs/2026-10-25-l40x2/
  {iq3s,coder}/{summary.csv,cells.jsonl}
  diag/engine-iq3s-live.log, diag/engine-crash-38236333.log
  logs/, notes.md
/work/ffa/holecem/strata-runs/2026-10-25-v100x1/
  {iq3s-matrix,coder-matrix}/{summary-*.csv,cells-*.jsonl}
  logs/ (job outputs, live-capture engine log/cmdline)
  scripts/ (cells_report.py, poll_drafter.sh, watch_job.sh)
```

## Reproduce

```bash
cd /work/ffa/holecem/slurmhelps-fleet

# plumbing, no GPU/network/model:
llm/strata/bench-matrix.sh --self-test
llm/strata/tests/test_fetch_models.sh

# 1× L40 (both packs; contexts are CLI args, not --export):
sbatch --nice=100 llm/strata/strata-bench.sbatch \
  --model both --contexts 1024,4096,16384,65536,131072 --batches 1,2,4

# 2× L40:
sbatch --nice=100 --nodelist=<two-L40-node> llm/strata/strata-bench.sbatch \
  --model iq3s --gpus 0,1 \
  --contexts 1024,4096,16384,65536,131072 --batches 1,2,4

# V100 (cuda12 auto-selected; 65536 is now clamped by the probe):
sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta05 \
  --gres=gpu:1 --mem=190G --cpus-per-task=24 \
  llm/strata/strata-bench.sbatch --model both \
  --contexts 1024,4096,16384,65536 --batches 1,2,4
```
