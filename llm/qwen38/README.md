# Qwen3.8-Flash-Next on 2× BW6000 96 GB

Target: **Qwen3.8-Flash-Next, 4-bit FP4**, 2× NVIDIA BW6000 96 GB (192 GB
total, native Blackwell FP4), 262K native context.

The open checkpoint is experimental; the full FP8 checkpoint is ~172.8 GiB and
does not leave useful KV headroom on 2×96 GB. The 4-bit local FP4 checkpoint is
~96 GiB.

## Model

```text
nvidia/Qwen3.8-Flash-Next-FP4-Local
$WORK_ROOT/models/qwen38-flash-next-fp4-local
```

The model has a 125B main model, 51B N-gram/PLE table, and 4B MTP head; only
6B parameters activate per token. The PLE table can be host/CPU offloaded.

## Serve

```bash
cd $WORK_ROOT/slurmhelps-fleet

llm/qwen38/serve.sh up
llm/qwen38/serve.sh status --json
llm/qwen38/serve.sh wait
llm/qwen38/serve.sh endpoint
llm/qwen38/serve.sh down
```

The default job is:

```text
2× BW6000 96 GB
TP2, 262144 context
BF16 KV
FFA-preempt, 8 CPUs, 220 GB, 12 h, nice=100
prefix cache + chunked prefill
qwen3 reasoning/tool parsers
PLE CPU offload enabled
```

The backend is `serve-container.sh`. It runs the community Blackwell container
`ghcr.io/local-inference-lab/vllm:release-10152026` from
`$WORK_ROOT/images/qwen38-vllm.sif`; a stock vLLM cannot load the checkpoint's
compressed NVFP4 scales.

## Connect from outside

The compute server has no public listener. `http://127.0.0.1:8010/v1` in the
examples is the **client end of an SSH `-L` tunnel**, not a port that exists on
a laptop by itself. `serve.sh up` submits/follows the Slurm job; it does not
create or maintain an SSH tunnel.

From a laptop, one hop through the pinned login node is enough:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8010:bw01:8010 "$USER@hpc-head.troja.mff.cuni.cz"
curl -s http://127.0.0.1:8010/v1/models
```

From the login node itself:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8010:bw01:8010 bw01
```

If the job is rescheduled onto a different compute node, recreate the tunnel
using the new node from `serve.sh status --json` or `serve.sh endpoint`. A
tunnel targets `host:port`, not a Slurm job ID, so an already open tunnel can
survive a job restart only while the node and port stay the same.

## Deployments and the login-node router

| deployment | config | GPUs | node port | state dir | status |
|---|---|---|---|---|---|
| `bw` | `config-bw.json` | 2× BW6000 96 GB, bw01 | 8010 | `.slurmhelps-serve/qwen38-flash-next` | production |
| `hopper` | `config-hopper.json` | 2× H100 80 GB, hopper01 (`--gres=gpu:H100:2`) | 8020 | `.slurmhelps-serve/qwen38-flash-next-hopper` | production (reserved for Qwen) |
| `strata-volta` | `config-strata-volta.json` | 4× V100-SXM2 32 GB, volta05 (`--gres=gpu:V100:4`), `ffa-gpulab`, `--nice=100000` | 8030 (Strata on 8031 behind the node gateway) | `.slurmhelps-serve/qwen38-flash-next-strata-volta` | lowest-priority fallback (operator 2026-11-17); not yet smoke-tested as a deployment |

`serve.sh` selects a deployment with `--target bw|hopper|strata-volta`
(default `bw`, which behaves exactly as before). The deployments are meant to
run at the same time, so that at least one is up whenever possible.

**strata-volta** serves the IQ3_S GGUF (`models/strata/iq3s`, 54.8 GB + 28.8 GB
PLE table) with the Strata engine (cuda12 build, sm_70) on all four V100s of
volta05 at 262,144 context, `--kv int8`, `--parallel 4`, MTP on: the recipe
measured in `../strata/experiments/2026-10-27-v100-extras.md` (needle 6/6 and
tool 1/1 at 262k). `serve-strata.sh` stages the pack to the job's node-local
`/tmp` (re-staged every job: `/tmp` is a per-job namespace on volta05, 22-90
min from Lustre), seeds a private engine data dir per job (a concurrent
`strata launch` cannot strip `--mtp`), starts Strata on loopback `$PORT+1`,
and starts the node gateway on `$PORT` only once `/v1/models` lists the model
(Strata's own `/health` answers 200 while still loading). SIGTERM stops the
engine with `strata launch --stop`. The serving config has no node-list
field; `--gres=gpu:V100:4` pins the job to volta05, the only node with four
V100s (the other volta nodes have 1-2 cards). Memory is 360 GB of
volta05's 380 GB (raised from 320 GB on 2026-11-17 for the parked-conversation
cache below): the ~84 GB staged pack counts against the job if `/tmp` is
tmpfs, plus the PLE table in RAM (`--ple-io ram`, 28.8 GB), pinned KV and the
drafter; the 262k soak saw 274 GB `MemAvailable` on the node at its lowest.

**Parked-conversation cache.** Between requests Strata parks each
conversation's KV in RAM so the next request reuses the prefix instead of
re-reading it. Its defaults (`--conversation-cache-mib` up to 8192,
`--conversation-cache-slots` 4) hold only 2-3 agent conversations of
100-145k tokens: with seven ADGFEM lanes on it (2026-11-17) every tool call
re-read ~100k prompt tokens (25-35 s each, 126 evictions) and no lane finished
a turn in 45 min. The deployment now sets 192 GiB and 48 slots
(`STRATA_CONV_CACHE_MIB`, `STRATA_CONV_CACHE_SLOTS`) and `--parallel 8`. The
engine itself keeps ~104 GiB resident.

**What "blocked" meant.** hopper01 is blocked for the ADGFEM experiments, so
that those do not clash with the Qwen deployments: the H100s are reserved for
Qwen, like bw01. It does not mean the deployment cannot run. The hopper target
uses the upstream vLLM nightly container (`serve-container-hopper.sh`, Marlin
W4A16 MoE experts, SM90, no b12x kernels) and served on 2026-11-07 (KV cache
1.9 M tokens, 7.3x concurrency at 262,144 tokens; see
`experiments/data/2026-11-07-qwen38-hopper`). The feasibility note
[`experiments/2026-11-06-qwen38-hopper.md`](experiments/2026-11-06-qwen38-hopper.md)
is about the earlier b12x container only, whose NVFP4-CSF experts run on
SM120/121 only, and is superseded for hopper by the nightly container.

The **router** runs on the login node at `127.0.0.1:18000`, bound to loopback
only. It discovers every deployment under `.slurmhelps-serve/*/state.json`;
`llm/models.json` supplies metadata, ports and guards, and deployments missing
from the catalogue are found too. A deployment counts as ready when its state
is `ready` with a fresh heartbeat.

- **Routing:** each request goes by its OpenAI `model` field.
  - `qwen38-flash-next` goes to `qwen38-flash-next` (bw) first, then to
    `qwen38-flash-next-hopper`, then to `qwen38-flash-next-strata-volta`.
    Change the order with `--prefer`; map extra names to a served model with
    `--alias NAME=MODEL`. With `--balance least-inflight` the deployment with
    the fewest requests in flight takes each request and preference order
    breaks ties, so the Strata fallback is used only when bw and hopper are
    busier or down.
  - Strata reports the id `qwen3.8-flash-next-iq3_s`. Its `models.json` entry
    has `"rewrite_model": true`, so the router treats it as a replica of
    `qwen38-flash-next` and rewrites the request's `model` to Strata's id.
  - Other names go to whichever deployment serves them. Unknown names get 404
    `model_not_found`.
  - `/v1/models` is the union of the ready deployments' models.
  - `/health` returns 200 when anything is ready and 503 `"no ready
    deployment"` otherwise.
  - `/router/status` lists every deployment found, with its state and the
    reason it is not ready.
- **Failover:** a connection failure skips that deployment for 30 s and the
  router tries the next one.
- **Authentication:** the router adds no API key and passes `Authorization`
  headers through.

**Guards.** For a model with `guards` in `models.json` (qwen38:
`bad_words.json` and `rewrite_rules.json`), the router applies the node
gateway's guards in-process, reusing the gateway code:

- **Requests:** bad_words are merged in and the history is scrubbed, with
  shellguard and privguard repairs.
- **Responses:** JSON and SSE are rewritten; streams stay streams.

A deployment started with `QWEN38_GATEWAY=0` is therefore still guarded.
Strata has no `bad_words` sampling parameter, so the strata-volta entry sets
`"request_bad_words": false`: the router sends it no `bad_words` field (a
client's own is dropped too), while the history scrub and the response
rewrite still apply. The
guards are idempotent, so they also run when the node gateway is on, and tests
check that the result is the same as applying them once. Router rewrites are
logged to `.slurmhelps-serve/router/rewrites-<model>.jsonl`.

```bash
llm/qwen38/router.sh start    # background; pid/log in .slurmhelps-serve/router/
llm/qwen38/router.sh status   # active deployment + per-deployment reason
llm/qwen38/router.sh stop
```

To connect a client, tunnel to the router on the login node instead of to a
compute node. The tunnel then survives preemption, requeue and a switch between
deployments:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8010:127.0.0.1:18000 "$USER@hpc-head.troja.mff.cuni.cz"
curl -s http://127.0.0.1:8010/v1/models     # served name: qwen38-flash-next
```

On the login node itself, use `http://127.0.0.1:18000/v1` directly.

### Operator rules

1. Every Qwen job is preemptible and lowest priority: partition `ffa-preempt`
   only, `--requeue`, `--nice=100`. Never `gpu-ffa`, `gpu-ffa-preempt`,
   `gpu-long` and so on. Exception: the Strata fallback
   `qwen38-flash-next-strata-volta` runs on volta05 in `ffa-gpulab` at
   `--nice=100000` (operator 2026-11-17), lowest priority, preemptible
   (`--requeue`, 12 h partition maximum).
2. Never submit to the L40 nodes (ampere01/03, fuuk-*, uk-g6-*, mff-a2-19,
   ucjf-a4-01). They are reserved for experiments. Hopper jobs are pinned with
   `--gres=gpu:H100:2`; hopper01 is the only H100 node.
3. bw01 (Blackwell) and hopper01 (H100) stay reserved for Qwen; ADGFEM
   experiments do not run on them.
4. Do not modify, cancel or resubmit a running or pending deployment job to
   make room for another one. If a job pends for RAM or GPU, report it; do not
   raise its priority.
5. Git: never force-push, never re-add an `upstream` remote, no git hooks.
   Commit with path-scoped commits on a branch.

## Measure

Once ready:

```bash
BASE_URL=http://bw01:8010 OUT=llm/qwen38/experiments/data/2026-10-24-qwen38-bw \
  llm/qwen38/measure.sh
```

This runs:

- 10-check HTTP smoke at 500/800 tokens, C=1/2/4/8
- 500- and 800-token batched sweeps at C=1/2/4/8/16/32
- context sweep 4K C8, 16K C4, 64K C2, 256K C1
- prefix-cache ladder 50K/100K/200K

Outputs go to `experiments/data/<name>/`.

## Status

Working on 2× BW6000 using the community Blackwell container. The 11 GiB SIF
is built and reused; JIT/autotune caches are persistent under
`{work_root}/cache/qwen38/qwen38-vllm-release-10152026`.

Measured: 2.64M KV tokens, 10.08× at 262K, 4K/C8 353.6 output tok/s, 256K
TTFT 15.9 s, 200K cache-hit TTFT 0.61 s. Full results:
[`experiments/2026-10-24-qwen38-bw-results.md`](experiments/2026-10-24-qwen38-bw-results.md).

OpenCode passed **4/4** local agent tasks against this endpoint:
[`experiments/2026-10-24-opencode-qwen38-flash-next.md`](experiments/2026-10-24-opencode-qwen38-flash-next.md).

`pull-container.sh` now exits without rebuilding when the SIF exists; set
`QWEN38_FORCE_PULL=1` to rebuild. The first boot's 43-minute JIT/autotune is
cached for subsequent starts.

## Scope

- 262K native context is the measurement target. YaRN/1M is a separate
  experiment and can degrade quality.
- The 4× L40 route is not equivalent: no 4× node is currently free, and
  Qwen3.8 with PLE on SM89 is unverified.
- The 2× A100 route is possible at 4-bit but tighter; 2×96 GB BW6000 is the
  preferred host.
