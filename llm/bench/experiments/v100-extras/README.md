# 2026-10-27 V100 extras — high/medium-value follow-ups

One 11 h allocation on **volta05** (4× V100-SXM2-32GB), sequential. The
orchestrator is `run-session.sh`; it stages each pack once (the shared 28.8 GB
PLE shard is single-striped and reads at 4–14 MB/s from Lustre), then runs the
experiments below in order. Results land in
`/work/ffa/holecem/strata-runs/2026-10-27-v100-extras/<item>/`.

## Unified vs one-off

| item | classification | why / where |
|:--|:--|:--|
| **H1** quality (NIAH + tool canary) | **one-off** | no quality mode exists in benchctl. `niah.py` inherits the canonical calibration, unique-salt cold rule, SSE/TTFT client, `/metrics` digest and JSONL+summary contract. |
| **H2** sustained 1024 | **unified** | `matrix` with `max_tokens=1024`; recipe `strata-v100-sustained-1024`. Executed as a probe cell on the shared long-context server (the orchestrator avoids a second 5-min load); the probe invocation is exactly the matrix runner's. |
| **H3** multi-conversation warm | **one-off** | the warm runner measures one conversation per slot. `multiconv.py` reuses its calibrated saltless-after-first-turn prompts, `stream_one`, per-request `reused` claim and conversation-cache block. |
| **H4** 2× IQ3_S | **unified** | `matrix` on `--gpus 0,1`; recipe `strata-v100-resident-2x-iq3s`. |
| **H5** chain on 4× | **unified** | existing `strata-chain-v100-4x` (`chain` runner / `chain-cache.sh`); previously pending, now run. |
| **H6** power / tokens-per-joule | **one-off sampler around a unified cell** | no power mode. `power-sampler.sh` + `power-report.py` bracket the H2 262k matrix cell; idle baseline 30 s before the server. |
| **M7** KV dtypes at 262k | **unified** | `knobs` with `kvq4_0`, `kvk8v4`; recipe `strata-v100-kv-dtypes-262k`. |
| **M8** prefill chunks at 131k | **unified** | `knobs` with `pf4096/pf8192/pf16384` and `cpushare_auto`; recipe `strata-v100-prefill-chunks-131k`. CPU share is documented as off for layer splits and `--batch`; the variant records the engine's own line. |
| **M9** batch-MTP / lookup (Coder) | **unified** | `knobs` with `batchmtp`, `lookup4`; recipe `strata-v100-batchmtp-lookup-coder`. |
| **M10** NVLink/P2P | **one-off diagnostic** | `p2p.sh` collects `nvidia-smi topo`/`nvlink` and runs a P2P test only if one exists; otherwise topology + the engine's PCIe probe are the evidence. |
| **M11** long-run stability | **one-off loop on the canonical probe** | no stability mode. Alternates cold and warm cells at 131k/262k for up to ~2 h, with power and RAM sampling. |
| **M12** cold-start/load time | **derived** | `timing.csv` records Lustre staging and launch→ready per phase from the same job. |

One-offs are allowed here because they live in the repo, reuse the canonical
probe/calibration/metrics, write one JSONL line per request plus `summary.csv`,
and are surrounded by the same engine-log/argv archive (`lib.sh`).

## Principles inherited (all one-offs)

- **Same tokenizer calibration**: `strata_probe.calibrate_chars_per_token()`
  (two-point chars/token + chat-template intercept); no 4-chars/token guess.
- **Cold prompts are unique-salted**; only an explicitly warm experiment may
  reuse a prompt. NIAH repeats use a fresh document each time.
- **Same TTFT definition**: dispatch → first content/reasoning/tool delta.
- **Same decode definition**: engine `timings.predicted_ms` when present
  (strict generation window), else the client stream window; TPOT = 1000/rate.
- **Same metrics**: `strata_probe.metrics_raw/metrics_digest`,
  `conversation_cache_of`, `claim_engine_request` for `reused`.
- **Same evidence**: each server block archives `engine-argv.txt`,
  `engine-facts.txt`, `launch-config.json`, `engine-logs/` before scratch
  cleanup; `timing.csv` records staging/load.
- **Real errors**: a failed cell records the exception/HTTP body and the run
  continues; nothing is fabricated.
- **Caveat carried forward**: the 16k C4 cell is unstable; M9 does not
  attribute from it alone.

## Scripts

```text
lib.sh             shared server lifecycle (launch recipe, private data dir,
                   ready wait, recipe verification, log archive)
niah.py            H1: needle at 10/50/90 %, 2 cold repeats,
                   tool-call canary (read_file) at each context
multiconv.py       H3: N distinct conversations, cold turn then interleaved
                   warm rounds; per-slot TTFT/reused/evictions/budget
power-sampler.sh   1 Hz per-GPU power/clocks/temp/util/VRAM
power-report.py    tokens/J, tokens/€ (EUR 0.20/kWh), idle baseline
p2p.sh             M10 topology/NVLink/P2P evidence
run-session.sh     the one-allocation orchestrator (this is what sbatch runs)
```

Recipes (benchctl): `strata-v100-resident-2x-iq3s`,
`strata-v100-sustained-1024`, `strata-v100-kv-dtypes-262k`,
`strata-v100-prefill-chunks-131k`, `strata-v100-batchmtp-lookup-coder`.
`strata-chain-v100-4x` is the existing chain recipe.

## 2026-10-28 — remaining R1–R3 (one allocation, `run-remaining.sh`)

Follow-up after the 2026-10-27 extras: the three questions that the first
session could not close. One allocation on **volta05** (4× V100-SXM2-32GB),
packs staged once to node-local `/tmp`, results in
`/work/ffa/holecem/strata-runs/2026-10-28-v100-remaining/`.

| item | question | method |
|:--|:--|:--|
| **R1** | quality at **262,144** for Coder IQ1_M and Q2_0 (IQ3_S was already 6/6 + 1/1) | `niah.py --contexts 262144` unchanged: 3 positions × 2 cold repeats + tool canary, calibrated, clamped. 2× first (`--gpus 0,1`); on a launch failure the exact engine error is captured and the pack retried on 4×. |
| **R2** | is the 155 GB/s/GPU NVLink actually used, or is the layer split confined to PCIe? | `p2p.sh` (topology/NVLink/`nvidia-smi nvlink --help`), `p2pbench.c` (driver-API `cuMemcpyPeerAsync` 256 MiB 0→1 and 0→2 vs host-staged), and `memcpy-trace.c` (LD_PRELOAD shim) on three 4× layer-split 32k cells: default, traced, and `STRATA_PEER_NO_P2P=1`. |
| **R3** | do the 262k HTTP 400 probe-window rejections disappear with the calibrated/clamped probe? | `stability.py` 45 min at C1, alternating a unique-salt cold 262k turn with a byte-identical warm turn, one JSON line per request (TTFT, tokens, reused, exact error), power + RAM per iteration. |

New scripts:

```text
p2pbench.c         driver API (dlopen libcuda) peer/HBM/PCIe bandwidth bench;
                   gcc only — the node's nvcc is CUDA 13.4 and emits no sm_70
memcpy-trace.c     LD_PRELOAD shim for cudaMemcpy{,Async,Peer,PeerAsync} and
                   peer-access calls; one TSV row per call (transport evidence)
trace-summary.py   aggregates the TSV by (function, kind) inside a time window
stability.py       R3 calibrated cold/warm 262k loop, per-request errors/drift
run-remaining.sh   the R1–R3 orchestrator (this is what sbatch runs)
```

```bash
sbatch --nice=100 llm/bench/experiments/v100-extras/run-remaining.sh
```

Node-local discipline: the `#SBATCH --output` is `/tmp/strata-v100-rem-%j.out`,
every engine log and pack is under `/tmp/v100-rem-$JOB`, and `publish()` copies
only small JSONL/CSV/JSON/TXT summaries to Lustre after each phase. The runner
never reads a live file from `/work`.

## Run

```bash
sbatch --nice=100 \
  llm/bench/experiments/v100-extras/run-session.sh
```

The `#SBATCH` block pins `ffa-gpulab`, `volta05`, `gpu:4`, 32 CPUs, 340 G and
11 h. The orchestrator stops opening phases when fewer than the phase's
minimum minutes remain and records the skip in `skipped.tsv`.

## Output map

```text
h1-quality/{iq3s,coder,q2_0}/   cells.jsonl, summary.csv, metrics.jsonl
h2-sustained/                   cells.jsonl (1024-token cells)
h3-multiconv/<ctx>/             cells.jsonl, summary.csv, cache.json
h4-2x-iq3s/                     matrix cells + engine facts
h5-chain-4x/                    per-turn JSONL/summary + engine facts
h6-power/                       idle.csv, cell-262k.csv, power-262k.json
m7-kv-262k/                     knobs.csv, variants.csv, per-variant cells
m8-prefill-131k/                same
m9-batchmtp-lookup-coder/       same
m10-topology/                   topology.txt, nvlink.txt, p2p.txt
m11-stability/                  cells.jsonl, power.csv, ram.csv
longctx-4x/                     the shared H1/H2/H3 server logs/argv/facts
timing.csv, skipped.tsv, session.log
```
