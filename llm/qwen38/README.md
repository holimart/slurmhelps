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

## Two deployments and the login-node router

| deployment | config | GPUs | node port | state dir | status |
|---|---|---|---|---|---|
| `bw` | `config-bw.json` | 2× BW6000 96 GB, bw01 | 8010 | `.slurmhelps-serve/qwen38-flash-next` | production |
| `hopper` | `config-hopper.json` | 2× H100 80 GB, hopper01 (`--gres=gpu:H100:2`) | 8020 | `.slurmhelps-serve/qwen38-flash-next-hopper` | **blocked** |

`serve.sh` selects a deployment with `--target bw|hopper` (default `bw`, which
behaves exactly as before). Hopper `up` is refused: this container loads the
NVFP4-CSF experts and the NVFP4 PLE table only through b12x kernels, which run
on SM120/121 only. See
[`experiments/2026-11-06-qwen38-hopper.md`](experiments/2026-11-06-qwen38-hopper.md)
for the evidence and what an SM90 build would need.

The **router** runs on the login node, bound to loopback only, at
`127.0.0.1:18000`. It reads both state files. Each request goes to the first
deployment, in the order `bw`, then `hopper`, that is `ready` with a fresh
heartbeat. A connection failure skips that deployment for 30 s, and the router
fails over to the next one.

- **Proxied traffic:** all OpenAI paths, including SSE streaming,
  `/v1/models` and `/health`. Responses carry `X-Slurmhelps-Deployment`.
- **No ready deployment:** the router answers `503 "no ready deployment"`.
- **Router status:** `/router/status` reports the router's own view.
- **Authentication:** the router adds no API key of its own and passes
  `Authorization` headers through.

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
   `gpu-long` and so on.
2. Never submit to the L40 nodes (ampere01/03, fuuk-*, uk-g6-*, mff-a2-19,
   ucjf-a4-01). They are reserved for experiments. Hopper jobs are pinned with
   `--gres=gpu:H100:2`; hopper01 is the only H100 node.
3. bw01 (Blackwell) stays reserved for Qwen.
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
