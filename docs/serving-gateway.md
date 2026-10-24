# Serving gateway — master design

Status: P0/P1 implemented and exercised on the real 2× A100 server; see
[`llm/glm53-reap50/experiments/2026-10-23-serving-gateway-smoke.md`](../llm/glm53-reap50/experiments/2026-10-23-serving-gateway-smoke.md).
Raw-KV Lustre spill remains gated behind engine support and is not implemented;
the default is state-only/no raw-KV spill.

## Implementation map (as built)

Links point at the code that implements each part of this design.

| Design area | Implementation | Tests / evidence |
|---|---|---|
| CLI and process lifecycle | [`bin/slurmhelps-serve`](../bin/slurmhelps-serve), [`lib/slurmhelps/serving/cli.py`](../lib/slurmhelps/serving/cli.py) | [`tests/test_serving_cli.py`](../tests/test_serving_cli.py) |
| Config, defaults, Slurm argv | [`config.py`](../lib/slurmhelps/serving/config.py) | [`test_serving_config.py`](../tests/test_serving_config.py) |
| Atomic state, phases, load history | [`state.py`](../lib/slurmhelps/serving/state.py) | [`test_serving_state.py`](../tests/test_serving_state.py) |
| Login-side submit/poll/status | [`controller.py`](../lib/slurmhelps/serving/controller.py) | [`test_serving_controller.py`](../tests/test_serving_controller.py) |
| Compute-side worker and SIGTERM | [`worker.py`](../lib/slurmhelps/serving/worker.py) | [`test_serving_worker.py`](../tests/test_serving_worker.py) |
| SSH `-L` and loopback relay | [`tunnel.py`](../lib/slurmhelps/serving/tunnel.py) | [`test_serving_tunnel.py`](../tests/test_serving_tunnel.py) |
| OpenAI gateway/auth/admission | [`gateway.py`](../lib/slurmhelps/serving/gateway.py) | [`test_serving_gateway.py`](../tests/test_serving_gateway.py) |
| Mock GPU/OpenAI backend | [`mock.py`](../lib/slurmhelps/serving/mock.py) | [`test_serving_mock.py`](../tests/test_serving_mock.py), [`test_serving_smoke.py`](../tests/test_serving_smoke.py) |
| OpenCode integration | [`examples/opencode-serve.json`](../examples/opencode-serve.json) | [`test_serving_opencode.py`](../tests/test_serving_opencode.py) |
| Real GLM A100 backend | [`llm/glm53-reap50/serve-server.sh`](../llm/glm53-reap50/serve-server.sh) | [`2026-10-23-serving-gateway-smoke.md`](../llm/glm53-reap50/experiments/2026-10-23-serving-gateway-smoke.md) |
| HTTP smoke matrix | [`tests/serve_smoke.py`](../tests/serve_smoke.py) | real and mock JSON under [`experiments/data`](../llm/glm53-reap50/experiments/data) |
| Idle/state-only spill | [`spill.py`](../lib/slurmhelps/serving/spill.py) | [`test_serving_spill.py`](../tests/test_serving_spill.py) |

### Named components / apps

| Name | What it is |
|---|---|
| `slurmhelps-serve` | the operator CLI (`up`, `status`, `wait`, `endpoint`, `tunnel`, `relay`, `gateway`, `spill`, `down`, `mock`, `smoke`, `doctor`) |
| `slurmhelps.serving.controller` | login-side controller for one Slurm allocation |
| `slurmhelps.serving.worker` | compute-side process/state supervisor |
| `slurmhelps.serving.gateway` | optional loopback OpenAI proxy with auth/admission |
| `slurmhelps.serving.mock` | deterministic no-GPU OpenAI backend |
| `serve-server.sh` | GLM-5.3 REAP50 SM80 vLLM backend |
| `serve_smoke.py` | 10-check HTTP smoke matrix |
| `opencode-serve.json` | OpenCode provider configuration |

### Public APIs and environment

HTTP:

- `GET /health`
- `GET /status` (gateway-local; controller status is also available on the login node)
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /metrics` (where the backend provides it)

Worker/backend environment (parameterizable):

```text
PORT, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
GPU_MEM_UTIL, KV_DTYPE, BLOCK_SIZE, ENABLE_CHUNKED_PREFILL,
ENABLE_TOOL_CALLS, TOOL_CALL_PARSER, WORK_ROOT
```

The API key is not passed through these variables by the CLI. `key_env` names
the operator's environment variable (`SLURMHELPS_API_KEY` by default); no value
is stored in the repo or state.

## Goal

Run one defined model on a defined Slurm GPU allocation and make it available
as a normal OpenAI-compatible endpoint through an SSH-only path:

```text
OpenCode / harness / curl
        │  (SSH tunnel or SSH ProxyJump)
        ▼
login node: slurmhelps serve controller + status + optional loopback relay
        │  (Slurm job, no public listener)
        ▼
compute node: model server + continuous batching + paged KV cache
        │
        ├── /work/.../serve/state.json       status, phases, heartbeat
        └── /work/.../serve/kv-spill/        optional idle snapshot
```

The job is **preemptible by default**: it runs in the lowest-priority
preemptible partition, requests only the resources it needs, and can be
requeued. Anyone with a higher-priority partition can take the GPUs. Same-tier
jobs cannot preempt it under `preempt/partition_prio`; that requires a cluster
policy change, not a user flag.

Non-negotiables:

- no credentials, tokens, private keys or host passwords in the repo;
- no public bind on the login or compute node by default;
- no request is admitted if it would exceed KV capacity or the queue limit;
- a cache hit is never allowed to evict KV blocks that an active request still
  references;
- preemption must leave a truthful status and a retryable endpoint, never a
  silent hang.

## 1. Process model

### Controller (login node)

`slurmhelps serve up` runs or starts:

- `serve-controller`: validates config, submits one Slurm job, writes
  `state.json`, follows `squeue`/`sacct`, and reports the current endpoint or
  an SSH `-L` hint. It does **not** keep an SSH tunnel or relay process alive;
  those are separate operator-managed commands.
- `serve-status`: reads `state.json` plus Slurm state; works even when the
  model HTTP server is not up yet.
- `serve-relay`: optional loopback TCP relay on the login node for compute
  nodes that are reachable only from the login node. It binds
  `127.0.0.1:<relay-port>` and forwards to `<compute-node>:<server-port>`.
- `serve-tunnel`: client-side helper that opens `ssh -L` (optionally
  `-J login`) to the relay or directly to the compute node.

The controller takes an exclusive lock on the serve directory so two
controllers cannot fight over one job.

### Worker (compute node)

`serve-worker` is the Slurm job body. It:

1. writes phases to `state.json` (`starting`, `loading:weights`,
   `loading:kv`, `warming`, `ready`, `draining`, `stopped`);
2. starts the engine (vLLM fork for SM80 A100, or stock vLLM for H100);
3. waits for `/health`, records KV capacity and max concurrency;
4. heartbeats every few seconds;
5. traps `SIGTERM`, stops admission, persists status, and exits/requeues;
6. optionally spills idle cache state after the idle timeout.

The HTTP server binds `127.0.0.1` or the compute node's cluster interface,
never a public address.

## 2. Configuration

All values are parameterizable. The defaults below are the researched
GLM-5.3-Flash REAP50 / 2× A100 starting point, not claims of optimality.

```json
{
  "serving": {
    "model": "glm53-reap50-nvfp4",
    "engine": "vllm-sm80",
    "gpus": 2,
    "gpu_type": "A100",
    "partition": "ffa-preempt",
    "account": "from cluster.account",
    "cpus": 8,
    "mem": "220G",
    "time": "04:00:00",
    "requeue": true,
    "nice": 100,
    "exclusive": false,

    "max_model_len": 786432,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.94,
    "kv_cache_dtype": "bfloat16",
    "block_size": 128,

    "host_kv_ram_gib": 96,
    "queue_limit": 64,
    "coalesce_ms": 0,

    "api": {
      "bind": "127.0.0.1",
      "port": 8000,
      "key_env": "SLURMHELPS_API_KEY"
    },
    "tunnel": {
      "via": "cluster.ssh",
      "relay_port": 18000,
      "bind": "127.0.0.1"
    },
    "spill": {
      "enabled": false,
      "idle_seconds": 600,
      "dir": "{work_root}/cache/glm53-reap50/kv",
      "max_gib": 128,
      "mode": "state-only"
    }
  }
}
```

Notes:

- `gpus` maps to `--gres=gpu:a100:<n>`; `gpu_type` is validated against
  `scontrol show node`/`sinfo`, not hardcoded in code.
- `partition` defaults to the cheapest preemptible partition with a matching
  GPU; `ffa-preempt` on this cluster. `account` comes from the project config
  or `SLURM_ACCOUNT`.
- `max_num_seqs` is the engine's maximum active sequences;
  `max_num_batched_tokens` is the per-iteration token budget. Both are
  required because they cap different things.
- `max_model_len` defaults to **786,432** tokens (768K). The checkpoint's
  text limit is **1,048,576**, and the measured A100 KV capacity is ~1.95M
  tokens, so one 700K request fits and two 700K requests fit only with
  careful admission. This is a capacity limit, not a promise that 700K
  prefill is fast: measured cold 690K prefill was **235 s**, while a 700K
  request sharing the cached prefix returned in **5.9 s**.
- `host_kv_ram_gib` is a **host** cache budget, separate from GPU KV. It must
  fit inside `mem` together with the model and worker overhead.
- `coalesce_ms` defaults to **0**. A multi-second batching window would add
  seconds to single-request TTFT and is not appropriate for interactive
  serving. See §4.
- `spill.mode = state-only` is the safe default: journal request/session state
  and warm prompt text, not raw KV. Raw-KV disk spill needs an engine
  extension and is a separate verification project (§5).
- Secrets are read from the environment or a `0600` file outside the repo.
  `key_env` names the variable; the value is never written to
  `slurmhelps.json`, `state.json`, Slurm arguments, or logs.
- If `key_env` is unset, the server is loopback-only and SSH is the security
  boundary.

## 3. Status and load-time honesty

`serve status --json` is authoritative and must work while the model is still
loading. It reads:

- `state.json` on the shared filesystem (worker-written);
- the job id's `squeue`/`sacct` state;
- the last N entries of `load-history.jsonl`.

Example:

```json
{
  "schema": 1,
  "state": "loading",
  "job_id": "38084003",
  "node": "ampere02",
  "phase": "loading:weights",
  "progress": 0.42,
  "load_elapsed_s": 240,
  "load_estimate_s": 570,
  "load_estimate_source": "cold-p50",
  "endpoint": null,
  "last_heartbeat": "2026-10-23T14:00:00Z",
  "capacity": null,
  "active_requests": 0,
  "queued_requests": 0
}
```

State machine:

```text
queued -> starting -> loading -> warming -> ready
   |         |           |          |
   +---------+-----------+----------+--> failed
   ready -> draining -> stopped
   any   -> preempted -> requeued -> starting
```

Load estimate:

- record per-phase durations on every start:
  `container_start`, `weight_load`, `kv_init`, `warmup`;
- cold start is measured at **9–10 min** for the 98 GiB checkpoint on A100;
- if the same node and the page-cache marker still match the model mtime,
  report a warm estimate instead;
- if a KV snapshot is being restored, add the restore estimate;
- report a range/confidence rather than a false exact ETA; update it as phases
  complete;
- on requeue, preserve the old phase history and mark the new attempt.

`serve wait` prints phase changes and the estimate; `serve status` returns a
non-zero exit code for `queued/loading/failed` so a harness can branch without
parsing text. `serve endpoint` prints the SSH command and loopback URL only
when `state=ready`.

## 4. Batching, cache hits, and mixed request lifetimes

### What the engine already does

vLLM/SGLang-style engines use **continuous batching**, not a fixed batch:

- the scheduler runs every iteration (milliseconds);
- each sequence has its own paged KV block table;
- a sequence joins when admitted, decodes one token per iteration, and leaves
  as soon as it emits its stop condition;
- another sequence can join while others are decoding;
- chunked prefill prevents one long prompt from monopolising the iteration
  budget;
- prefix-cache blocks are reference-counted and shared until no request needs
  them.

So there is **no need to wait for a fixed “five-request” window**. Five
requests arriving within a few seconds are admitted as they arrive; the engine
batches whichever are runnable at each iteration. A short request is removed
when it finishes; it does not wait for a longer one.

### The five-cache-hit example

Assume five requests share a 500-token system prefix, have unique suffix
prompts, and hit the prefix cache:

1. The engine maps the shared prefix blocks into each request's block table.
2. Each request prefills only its unique suffix (if any). If the prefix is in
   GPU memory, this is a short prefill. If it is in host RAM, the engine must
   copy the referenced blocks to GPU first; that is a block copy, not a
   token-by-token “stream” of attention.
3. All five enter the decode batch. A step produces one token for each active
   sequence.
4. Request 3 has a 16-token answer and finishes first. It is returned and its
   block references are dropped. Requests 1, 2, 4, 5 keep decoding in the next
   batch; the batch shrinks smoothly.
5. When a new request arrives, it is admitted into a later iteration; it does
   not restart the batch.

Thus “the first one finishes because it had a shorter fill” is normal: each
request has its own completion event. Production engines avoid a
batch-wide barrier; that is the main reason continuous batching beats static
batching for mixed output lengths.

### Why `coalesce_ms = 0` is the default

A batching window is a latency bet:

- a 2 s window would make a single request wait 2 s for a batch that may never
  come; unacceptable for interactive use;
- vLLM's iteration already gives natural micro-batching (milliseconds);
- if a burst arrives, continuous batching picks it up at the next iteration;
- a small explicit window (5–20 ms) can help only if the client is known to
  send synchronized bursts and the extra TTFT is acceptable.

Implement `coalesce_ms` as a parameter, but default it to 0. If verification
shows a win for a specific agent workload, use a small adaptive window under
load, never a multi-second one.

### Mixed prefill and decode

A long cold prefill can still raise inter-token latency for decoders. The
controls are:

- `max_num_batched_tokens`: cap prefill chunks per iteration;
- `--enable-chunked-prefill`: required;
- scheduler preference for decode when both are runnable;
- admission control: hold very large prefills until the decoder queue is
  below a threshold.

This needs measurement with a 256K cold prefill alongside several 500-token
cached decoders; see §9.

## 5. KV cache and spill hierarchy

Token cost on the measured A100 configuration is about **24.7 KiB/token**
across both GPUs (44.9 GiB for 1.95M tokens). Keep the tiers explicit:

| Tier | What | Default |
|---|---|---|
| L0 | GPU paged prefix cache | 22.45 GiB/GPU, ~1.95M tokens |
| L1 | host RAM prefix/session cache | 96 GiB, optional |
| L2 | Lustre idle snapshot | disabled by default |
| L3 | recompute from prompt text | always available |

L0 is the measured vLLM prefix cache. L1 is not free: host-to-device block
copies add latency, and the engine must support CPU offload/swap. L2 is not a
stock vLLM feature in the SM80 fork; enabling raw-KV disk spill requires an
extension such as LMCache/SGLang HiCache or a custom block serializer, and
must be benchmarked before it is called production.

Idle spill policy:

- trigger only after `spill.idle_seconds` **and** zero active requests;
- a new request cancels or defers the snapshot;
- snapshot only immutable, unreferenced prefix blocks plus a manifest
  (model hash, tokenizer hash, block size, dtype, layout, prefix hashes,
  lengths, checksums);
- write atomically: `*.tmp` → fsync → rename; never overwrite the last good
  manifest until the new one is complete;
- cap by `spill.max_gib`; do not let an idle spill fill Lustre;
- restore only if the model/tokenizer/engine layout matches; otherwise
  discard and recompute;
- if the engine cannot dump/restore KV, use `state-only`: journal the token
  prefixes and session metadata, then recompute. At 500–800 tokens recompute
  is cheap; at 100K it is ~30 s and at 300K ~96 s on the measured A100 path.

“Once every ten minutes if there are no connections” should be
**change-driven**: if nothing changed and no requests arrived, a second
identical snapshot is pointless. Snapshot on the first idle transition, then
only when the cache changed or a longer interval elapsed.

LRU retains common prefixes and the most recent sessions; never evict blocks
referenced by an active request. Use prefix hashes over token IDs, not raw
text, and include model/tokenizer identity in the key.

### Disk spill research (design only — not wired)

The SM80 fork is less closed than the first run suggested:

- `from vllm.config import KVTransferConfig` **works** in
  `vllm-glm53-sm80.sif`;
- the connector factory registers **`LMCacheConnectorV1`** and
  **`LMCacheMPConnector`**;
- the `lmcache` Python package is **not installed** in the image.

So raw-KV spill is not an engine-API dead end. It is an integration task:

1. Build an image (or overlay/venv) containing a compatible `lmcache`
   version. The vLLM/LMCache versions must match; do not assume the latest
   LMCache works with this older fork.
2. Configure local disk offload (LMCache docs):
   `LMCACHE_USE_EXPERIMENTAL=True`, `chunk_size: 256`,
   `local_disk: "file:///work/.../kv"`, `local_cpu: true`,
   `max_local_cpu_size`, `max_local_disk_size`, `LMCACHE_CONFIG_FILE`.
3. Start vLLM with:

   ```text
   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
   ```

4. Verify with repeated long prompts that the LMCache hit-token counter rises
   and TTFT falls; then benchmark restart/requeue.

Why this is the preferred route over a custom serializer:

- LMCache uses **256-token chunks stored as separate files**, not one giant
  KV blob; it writes asynchronously, evicts by LRU and can prefetch from disk
  to CPU/GPU.
- A preemption does not need a whole-job freeze: completed chunks are already
  durable; only the last incomplete chunk is lost. This directly addresses
  “the job was killed, can the next start reuse work?”
- It has a CPU tier and disk tier, matching the L1/L2 design above.
- It avoids inventing a block-table/tensor format that the fork may change
  underneath us.

Caveats:

- LMCache chunks are keyed by token/model identity; changing weights,
  tokenizer, block size, KV dtype or TP layout invalidates them.
- Local disk files are only node-local unless the directory is on Lustre;
  a requeue to another node must use the same shared path.
- LMCache is not measured here. The fork’s compatibility is unproven until a
  build-and-replay test succeeds.
- SGLang HiCache is the other mature design (L1 GPU / L2 host / L3 file,
  Mooncake, HF3FS, NIXL, write-through/back policies). It is not usable for
  this A100 model because SGLang has no SM80/SM89 GLM-5.3 sparse-DSA backend.
- vLLM’s newer shortcut (`--kv-offloading-backend lmcache
  --kv-offloading-size ...`) is not assumed to exist in this fork; use the
  explicit `KVTransferConfig` only after checking `vllm serve --help`.

References:

- LMCache local disk: <https://docs.lmcache.ai/kv_cache/local_storage.html>
- vLLM + LMCache examples and `LMCacheConnectorV1`:
  <https://docs.vllm.ai/en/stable/examples/disaggregated/lmcache/>
- SGLang HiCache design:
  <https://docs.sglang.ai/advanced_features/hicache_design.html>

### Lustre characteristics and measured rates

`/work` is Lustre with **3 OSTs**; the default directory stripe is
`stripe_count=1`, `stripe_size=1 MiB`. A single big file therefore lands on
one OST. `lfs df` showed ~92% full at the time of the run, so 45 GiB spill
files need a cap and cleanup.

We did **not** get a clean raw one-file read/write benchmark: the 8 GiB `dd`
test was interrupted by the storage-quota/overlimit intervention and the
partial file was deleted. Do not quote a raw GB/s number from that attempt.

What we do have is **end-to-end model-load throughput**, which is a lower
bound and includes container/GPU/engine work:

| Path | Bytes | Observed | Effective rate |
|---|---:|---:|---:|
| Cold model load from Lustre | 98 GiB | 1,559 s | **~64 MiB/s = 0.063 GiB/s** |
| Warm page-cache model load | 98 GiB | 374.7 s | **~274 MiB/s = 0.268 GiB/s** |
| Earlier clean A100 load | 98 GiB | ~540 s | **~186 MiB/s = 0.182 GiB/s** |

These are not raw `dd` numbers. For spill planning, use a range rather than a
single claim: on this `/work` layout a single-stripe file may be far below the
filesystem aggregate; multiple chunk files (LMCache’s native layout) or an
explicitly striped directory (`lfs setstripe -c 3`) should be used before any
GB/s claim. At 100 MiB/s a 17 GiB 700K KV snapshot takes ~170 s; at 500 MiB/s
it takes ~34 s. Compare that to the measured 235 s cold 700K prefill and 5.9 s
GPU-cache hit before deciding whether disk spill helps.

## 6. SSH access and helpers

No public port is opened.

### Client on a laptop

```bash
slurmhelps serve endpoint          # wait until ready; prints node:port and job id
slurmhelps serve tunnel --local-port 8000
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

`serve tunnel` prefers:

1. direct `ssh -L 8000:<node>:<port> <node>` if the client can reach compute;
2. otherwise `ssh -J <cluster.ssh> -L 8000:<node>:<port> <node>`;
3. otherwise an SSH session to the login node running `serve relay`, then a
   second `ssh -L 8000:127.0.0.1:<relay-port> <login>`.

The third mode is the robust one when compute nodes are reachable only from
the login node.

`serve tunnel` execs `ssh -L` and stays in the foreground; `serve relay` stays
in the foreground on the login node. Neither is started automatically by
`serve up`, and neither is automatically recreated after a preemption. When
the allocation moves to another node, stop the old helper and rerun it with
the new node from `serve status --json`.

### Login-node relay

```bash
slurmhelps serve relay --listen 127.0.0.1:18000 --target "$(slurmhelps serve endpoint --field node):8000"
```

It is a dumb TCP relay, loopback-only, and logs connection counts, not
payloads. It stops when the job stops or the controller restarts it.

### Connecting other engines and clients

The transport contract is deliberately generic:

- the backend must listen on the configured port and expose `GET /health`
  plus OpenAI-compatible `GET /v1/models`, `POST /v1/completions` and
  `POST /v1/chat/completions`;
- the worker writes the phase/state file and starts whatever command is
  supplied; it does not contain model-specific flags;
- the gateway/tunnel only speak HTTP, so OpenCode, an OpenCode harness, curl,
  or another OpenAI client use the same endpoint.

The default backend is `llm/glm53-reap50/serve-server.sh` (SM80 vLLM fork).
To use a different backend, pass a command after `--`:

```bash
slurmhelps-serve up --yes --backend-command bash /path/to/engine.sh
```

The command must honor `PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `GPU_MEM_UTIL` and `KV_DTYPE` from the environment
(or accept equivalent flags). The worker still provides state, heartbeat,
load history and SIGTERM handling; engine-specific KV-capacity parsing is
best-effort.

Any OpenAI client can then use:

```bash
slurmhelps-serve endpoint
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
# export OPENAI_API_KEY=...   # only if the gateway has auth enabled
```

For OpenCode, copy [`examples/opencode-serve.json`](../examples/opencode-serve.json)
to the OpenCode config and point `baseURL` at the endpoint.

### API

- `GET /health` — liveness only;
- `GET /status` — same state as the controller, including load phase/ETA;
- `GET /v1/models` — OpenAI-compatible;
- `POST /v1/completions`, `POST /v1/chat/completions` — OpenAI-compatible;
- `GET /metrics` — Prometheus: queue depth, active sequences, batch size,
  TTFT/ITL histograms, cache hits, KV blocks, preemptions, spill status;
- `X-Request-Id` echoed; `Retry-After` on 429/503.

Auth is optional:

- loopback/SSH-only: no application key required;
- if `SLURMHELPS_API_KEY` is set, require
  `Authorization: Bearer <key>`, constant-time compare;
- never log the header or write the key to state;
- the key is read from env or a `0600` file; it is not placed in
  `sbatch --export` because that is visible in Slurm job metadata.

## 7. Preemption and “gentle” defaults

Default job shape:

```bash
#SBATCH -p ffa-preempt
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --nice=100
```

Do not use `--exclusive`; do not reserve both GPUs for an idle server beyond
the requested time. On `SIGTERM`:

1. stop admitting new requests;
2. return 503 with `Retry-After` for queued requests;
3. allow in-flight requests a short drain;
4. write `state=preempted` and any allowed snapshot;
5. exit; Slurm requeues if `--requeue`;
6. controller waits for the new allocation and updates `state=starting`; the
   operator must recreate the SSH tunnel/relay if the compute node changed
   (the controller does not manage those processes).

If a hard guarantee is required, only an **admin-created advanced
reservation** provides it; there is no user-facing `--no-preempt` flag on this
cluster.

## 8. Failure and admission policy

- `queue_limit`: maximum waiting requests; beyond it return 429.
- `max_num_seqs`: engine active limit; the proxy should not admit more than
  the engine can hold.
- `max_model_len`: reject prompts whose prompt + max output exceeds it.
- KV admission: estimate prompt blocks + decode growth; if not available,
  queue or reject; never OOM the engine.
- Backend crash: state `failed`; controller may restart once if
  `restart_on_failure`; clients retry with idempotency key.
- Lost heartbeat: status becomes `unknown` after N missed beats, not
  `ready`.
- Stale `state.json` from a previous job is never trusted without matching
  job id and boot/heartbeat generation.

## 9. Verification plan

The production question is not “does batching work?” but “does the optionality
hold under mixed cache hits, output lengths, and preemption?” Tests should run
on the real A100 path and be repeatable.

| # | Test | Pass condition |
|---|---|---|
| 1 | Cold start, warm page-cache start, snapshot restore | status phases correct; ETA error reported honestly; no false ready |
| 2 | C=1,2,4,8,16,32 at 500/800-token cached prompts | no OOM; per-stream and aggregate tok/s reported; knee identified |
| 3 | Five shared-prefix requests, outputs 16/64/512/2048/8192 | short request returns before long; no batch-wide stall; ITL p99 bounded |
| 4 | 4 cached short + 1 cold 64K/256K prefill | decode ITL degradation measured; chunked prefill prevents full stall |
| 5 | Prefix cache hit vs cold, same output | identical tokens/logprobs (within tolerance); cache key includes model/tokenizer |
| 6 | Host-RAM cache restore | restore latency and throughput measured; no active-block eviction |
| 7 | Idle spill after 600 s | starts only idle; cancels on arrival; atomic manifest; bounded size |
| 8 | Preempt with higher-priority filler | state transitions; client 503/retry; requeue; tunnel recovers |
| 9 | Tunnel via login relay / ProxyJump | OpenCode/curl reaches API; no public listener |
| 10 | Auth absent/present | loopback policy; key never in logs/state |

Instrumentation must record per request: arrival, admission, TTFT, first
decode, finish, output tokens, cache hit tokens, KV blocks, and engine batch
size. Aggregate numbers alone cannot answer the short-vs-long question.

The earlier `bench.py` and `paged_probe.py` are the starting clients; a
`serve_probe.py` should add arrival timestamps, per-request completion, and
mixed-output schedules.

## 10. Delivery phases

1. **P0 — controller/status/tunnel — DONE**: `serve up/status/wait/endpoint/
   tunnel/down`, state machine, load history, preemptible Slurm defaults.
2. **P1 — gateway — DONE**: OpenAI passthrough, SSE, bearer auth, admission,
   request IDs, optional `coalesce_ms=0`; real 32-way + 429 test.
3. **P2 — cache — NOT DONE**: host-RAM tier via engine-supported swap/LMCache
   is still a research step; no host-KV capacity claim.
4. **P3 — spill — PARTIAL**: state-only manifest is implemented; raw-KV Lustre
   snapshot remains gated behind an engine serializer/benchmark.
5. **P4 — hardening — PARTIAL**: preemption/state/auth/tunnel/load history
   covered by tests and one real run; long soak and preemption drills remain.

## 11. Implementation notes from the real run

- Commits: [`4132131`](https://github.com/holimart/slurmhelps/commit/4132131)
  (implementation) and [`fa260a4`](https://github.com/holimart/slurmhelps/commit/fa260a4)
  (engine/client connection contract). The later doc/implementation map is
  committed on top of these.
- Real backend: [`serve-server.sh`](../llm/glm53-reap50/serve-server.sh) with
  `vllm-glm53-sm80.sif`, the `glm4v.py` prefix patch and the checkpoint
  dense-MLP quant ignore. It uses `TRITON_MLA_SPARSE`, BF16 KV, chunked prefill
  and `--enable-auto-tool-choice --tool-call-parser glm47` for OpenCode.
- `mem=220G` is deliberate: 260G did not fit beside the existing CPU jobs on
  `ampere02`; 220G did. The node had 760G total.
- `config.env` defaults `PORT=30000` and Blackwell FP8 KV. The generated
  worker command exports `PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
  `MAX_NUM_BATCHED_TOKENS`, `GPU_MEM_UTIL` and `KV_DTYPE` so the serving config
  wins.
- `nice=100` correctly kept the job gentle but made it wait; the smoke used
  `scontrol update JobId=... Nice=0` to start it. Defaults remain `nice=100`.
- Measured model startup: **1,559 s cold from Lustre under contention** and
  **374.7 s warm page cache on the same node**. The earlier clean A100 weight
  load was ~9 min. Warm page cache is node-local and can be lost on requeue.
- Measured KV: **1,765,762 tokens**, **2.25×** at 786,432. Cold 690K prefill
  **235 s**; 700K cache hit **5.89 s**; two concurrent 700K cache hits
  **1.92 s / 2.94 s**.
- Full test suite: **280 passed**. The real 10-check HTTP matrix passed
  directly and through the gateway; auth was 401/200 and admission was
  32×200 + 1×429. OpenCode passed against both mock and real endpoints.
- The serving job was stopped after the run; `ampere02` GPUs were returned.
- Raw-KV spill was not faked. Only the state-only manifest and idle policy are
  implemented; the SM80 fork has no KV export/import API.

### Reproduce the smoke

```bash
cd $WORK_ROOT/slurmhelps-fleet
PYTHONPATH=lib python3 -m pytest -q

# after the server is ready:
PYTHONPATH=lib python3 tests/serve_smoke.py \
  --base-url http://ampere02:8000 --model glm53-reap50 \
  --contexts '500 800' --concurrency '1 2 4 8' \
  --model-max-len 786432
```

## Open questions to settle by measurement

- Does the SM80 vLLM fork expose CPU swap/host prefix cache, or does L1 need
  LMCache/SGLang?
- What is the real warm-load time after preemption on the same node?
- At 500–800 tokens, what `max_num_seqs`/`max_num_batched_tokens` maximises
  aggregate throughput without hurting p99 ITL?
- Does a 5–20 ms coalescing window help a real OpenCode burst, or is it pure
  added latency?
- How much does a 256K cold prefill raise ITL for cached decoders?
- What is the actual Lustre write/read rate for a 45 GiB snapshot, and is raw
  KV spill worth it versus state-only recompute?
