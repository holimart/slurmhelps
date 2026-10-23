# Serving gateway — master design

Status: design + verification plan. No production server is implemented by this
document. The measured A100/H100 results are in
[`llm/glm53-reap50/experiments/`](../llm/glm53-reap50/experiments/).

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
  `state.json`, follows `squeue`/`sacct`, and restarts/recreates the tunnel
  after requeue.
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
    "mem": "260G",
    "time": "04:00:00",
    "requeue": true,
    "nice": 100,
    "exclusive": false,

    "max_model_len": 327680,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.94,
    "kv_cache_dtype": "bfloat16",
    "block_size": 128,

    "host_kv_ram_gib": 128,
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
| L1 | host RAM prefix/session cache | 128 GiB, optional |
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

### Login-node relay

```bash
slurmhelps serve relay --listen 127.0.0.1:18000 --target "$(slurmhelps serve endpoint --field node):8000"
```

It is a dumb TCP relay, loopback-only, and logs connection counts, not
payloads. It stops when the job stops or the controller restarts it.

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
#SBATCH --mem=260G
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
6. controller waits for the new allocation, updates `state=starting`, and
   re-establishes the tunnel/relay.

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

1. **P0 — controller/status/tunnel**: `serve up/status/wait/endpoint/tunnel/
   down`, state machine, load ETA from history, Slurm submit with preemptible
   defaults. No custom proxy.
2. **P1 — gateway**: OpenAI passthrough, queue limit, request IDs, metrics,
   optional `coalesce_ms=0`; verify continuous batching behavior.
3. **P2 — cache**: host-RAM tier via engine-supported swap/LMCache; measure
   restore and capacity; only then enable.
4. **P3 — spill**: state-only journal first; raw-KV Lustre snapshot only after
   a separate benchmark; never on by default.
5. **P4 — hardening**: preemption drills, stale-state detection, auth, tunnel
   recovery, load ETA calibration.

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
