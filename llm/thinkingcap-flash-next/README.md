# ThinkingCap Qwen3.8-Flash-Next on 2× A100 80 GB

This is the A100 counterpart of the Blackwell
[`qwen38-flash-next`](../qwen38/README.md) deployment. It serves
**BottleCap AI's ThinkingCap fine-tune** of Qwen3.8-Flash-Next with stock vLLM:

```text
bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4
$WORK_ROOT/models/thinkingcap-flash-next-nvfp4
```

The checkpoint is ~132 GB:

| Component | Precision |
|---|---|
| Routed experts, including the MTP head | NVFP4 W4A16, compressed-tensors, group 16 |
| N-gram/PLE table | FP8 E4M3, one per-tensor scale (~51 GB) |
| Attention, linear attention, shared experts, router, vision tower, `lm_head` | BF16 |

It is the closest ThinkingCap build to the quality of the Strata
`IQ3_S` Qwen3.8-Flash-Next pack: same base architecture, same 262K context,
but ThinkingCap's shorter-thinking post-training.

## Hardware

| Item | Value |
|---|---|
| Node | `ampere02` (the only A100 node) |
| GPUs | 2× A100 80 GB, TP2 |
| Partition | `ffa-preempt` |
| Nice | 100 |
| Job time | 36 h, requeue enabled |
| Context | 262,144 native |
| Max concurrent sequences | 2 |
| KV cache | BF16 |

A100 has no native FP4 tensor cores. vLLM runs the NVFP4 experts through its
**Marlin W4A16** kernel. This works but is slower than the BW6000 path; it is
an experimental A100 deployment, not a Blackwell-equivalent throughput claim.

## Fetch the model

The checkpoint is public; no HF token is required.

```bash
cd $WORK_ROOT/slurmhelps-fleet
llm/thinkingcap-flash-next/pull-model.sh
```

The download is ~132 GB. To refresh:

```bash
THINKINGCAP_FORCE_PULL=1 llm/thinkingcap-flash-next/pull-model.sh
```

The revision is pinned in `pull-model.sh` (`df3922b...`).

## Serve

```bash
cd $WORK_ROOT/slurmhelps-fleet

llm/thinkingcap-flash-next/serve.sh up --yes
llm/thinkingcap-flash-next/serve.sh status --json
llm/thinkingcap-flash-next/serve.sh wait
llm/thinkingcap-flash-next/serve.sh endpoint
llm/thinkingcap-flash-next/serve.sh down
```

The controller config is [`config-a100.json`](config-a100.json); the backend is
[`serve-server.sh`](serve-server.sh).

`nice=100` is deliberate. For a short smoke you can lower it after submission:

```bash
scontrol update JobId=<jobid> Nice=0
```

### MTP and vision

The checkpoint keeps the MTP head. MTP is **off by default** on A100 to keep
the first boot simple and the KV/activation budget predictable:

```bash
MTP=1 llm/thinkingcap-flash-next/serve.sh up --yes
```

The vision tower is included. To run text-only and save a little memory:

```bash
LANGUAGE_MODEL_ONLY=1 llm/thinkingcap-flash-next/serve.sh up --yes
```

If the 262K/2-sequence configuration OOMs during startup, lower the context
first:

```bash
# edit config-a100.json: max_model_len 131072, max_num_seqs 4
```

or pass the same change through the controller config before `up`.

## Connect from outside

The compute server has no public listener. `http://127.0.0.1:8011/v1` is the
**client end of an SSH `-L` tunnel**, not a port on your laptop.

From a laptop, one hop through the pinned login node is enough:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8011:ampere02:8011 "$USER@hpc-head.troja.mff.cuni.cz"
curl -s http://127.0.0.1:8011/v1/models
```

From the login node itself:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8011:ampere02:8011 ampere02
```

`ampere02` is the only A100 node, so a requeue normally stays there. Still
check the node after a restart:

```bash
llm/thinkingcap-flash-next/serve.sh status --json
llm/thinkingcap-flash-next/serve.sh endpoint
```

A tunnel targets `host:port`, not a Slurm job ID. If the node changes, recreate
the tunnel with the new node from `status --json`.

## OpenCode provider

Copy [`examples/opencode-thinkingcap-flash-next.json`](../../examples/opencode-thinkingcap-flash-next.json)
into your OpenCode config, or add:

```json
"slurmhelps-thinkingcap": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "slurmhelps ThinkingCap Flash-Next (A100)",
  "options": {
    "baseURL": "http://127.0.0.1:8011/v1",
    "apiKey": "{env:SLURMHELPS_API_KEY}"
  },
  "models": {
    "thinkingcap-flash-next": {
      "name": "ThinkingCap Qwen3.8-Flash-Next NVFP4 (A100)"
    }
  }
}
```

Then:

```bash
opencode models | grep thinkingcap
opencode run --model slurmhelps-thinkingcap/thinkingcap-flash-next \
  --auto 'Inspect this repository and run its tests.'
```

If `SLURMHELPS_API_KEY` is unset, the server is loopback-only and a placeholder
key is enough for local clients.

## Expected limits

- Weights: ~122.6 GiB (132 GB decimal).
- Total A100 VRAM: 160 GB; at `gpu_memory_utilization=0.95`, about 19 GiB is
  left for KV and activations across both GPUs.
- With BF16 KV, that is roughly 600–700K KV tokens — enough for two 262K
  sequences, not for the 2.65M-token capacity of the BW6000 deployment.
- `max_num_seqs=2` is therefore intentional.

## A100 caveat and fallback

The ThinkingCap NVFP4 checkpoint is validated by BottleCap on **2× RTX PRO
6000 Blackwell**, not A100. The A100 path relies on vLLM's Marlin W4A16
fallback for compressed-tensors NVFP4. If vLLM aborts during startup with a
missing FP4 Marlin kernel / `no kernel image` error, the A100 NVFP4 route is
not usable as-is.

In that case use the GGUF ThinkingCap build instead:

```text
bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-GGUF  IQ4_XS (95.5 GB)
```

That runs under llama.cpp, or can be tested under Strata on `fuuk-15`
(4× L40 48 GB, 192 GB total). It is not the same vLLM endpoint and is not a
drop-in replacement for this job.

## Scope

- This is a vLLM/compressed-tensors job, not a Strata pack.
- No speculative decoding by default; `MTP=1` enables the checkpoint's own MTP
  head with 3 draft tokens.
- The model is ThinkingCap's NVFP4 build, not the NVIDIA FP4-Local checkpoint
  used on `bw01`.
- No image/video benchmark is claimed until the A100 endpoint is measured.
