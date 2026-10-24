# Qwen3.8-Flash-Next on 2× BW6000 96 GB

Target: **Qwen3.8-Flash-Next, 4-bit FP4**, 2× NVIDIA BW6000 96 GB (192 GB
total, native Blackwell FP4), 262K native context.

The open checkpoint is experimental; the full FP8 checkpoint is ~172.8 GiB and
does not leave useful KV headroom on 2×96 GB. The 4-bit local FP4 checkpoint is
~96 GiB.

## Model

```text
nvidia/Qwen3.8-Flash-Next-FP4-Local
/work/ffa/holecem/models/qwen38-flash-next-fp4-local
```

The model has a 125B main model, 51B N-gram/PLE table, and 4B MTP head; only
6B parameters activate per token. The PLE table can be host/CPU offloaded.

## Serve

```bash
cd /work/ffa/holecem/slurmhelps-fleet

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
FFA-preempt, 16 CPUs, 300 GB, 2 h
prefix cache + chunked prefill
qwen3 reasoning/tool parsers
PLE CPU offload enabled
```

The backend is `serve-server.sh`. It uses `modelopt` quantization and requires
the newer Qwen4/PLE-capable vLLM environment; a plain older vLLM may fail to
load the mixed FP4/PLE checkpoint.

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

## Current blocker (2026-10-24)

Stock vLLM 0.32 fails on the checkpoint's compressed NVFP4 scale format:

```text
w2_weight_scale.nvfp4_csf_exceptions
```

The model card requires the Blackwell community container
`ghcr.io/local-inference-lab/vllm:release-10152026`. The Apptainer pull was
started with `/work` cache/tmp, but extraction did not finish (killed after
~37 min). No SIF and no speed numbers yet. See
[`experiments/2026-10-24-qwen38-bw-setup.md`](experiments/2026-10-24-qwen38-bw-setup.md).

## Scope

- 262K native context is the measurement target. YaRN/1M is a separate
  experiment and can degrade quality.
- The 4× L40 route is not equivalent: no 4× node is currently free, and
  Qwen3.8 with PLE on SM89 is unverified.
- The 2× A100 route is possible at 4-bit but tighter; 2×96 GB BW6000 is the
  preferred host.
