# LLM serving catalogue

The model catalogue is [`models.json`](models.json). Downstream consumers
should resolve a model by name instead of hardcoding paths, ports or launcher
scripts.

Defaults:

| Target | Model |
|---|---|
| A100 | `qwen35-122b-a10b` |
| Blackwell | `qwen38-flash-next` |
| GLM/700K | `glm53-reap50` |

Each catalogue entry names its controller config, backend, GPU class, context,
endpoint, OpenCode provider/model and status.

## Qwen3.8-Flash-Next on Blackwell

```bash
# SIF is already built; this is a no-op unless QWEN38_FORCE_PULL=1
llm/qwen38/pull-container.sh

llm/qwen38/serve.sh up
llm/qwen38/serve.sh status --json
llm/qwen38/serve.sh wait
llm/qwen38/serve.sh endpoint
llm/qwen38/serve.sh down
```

The controller config is [`qwen38/config-bw.json`](qwen38/config-bw.json).
The backend is [`qwen38/serve-container.sh`](qwen38/serve-container.sh).

### Cache policy

The expensive parts are deliberately reused:

- **SIF**: `pull-container.sh` exits without pulling when
  `images/qwen38-vllm.sif` already exists. Set `QWEN38_FORCE_PULL=1` to rebuild.
- **JIT / autotune / CUDA graphs**: `serve-container.sh` binds a persistent
  cache:
  ```text
  {work_root}/cache/qwen38/qwen38-vllm-release-10152026/  -> /cache
  ```
  It holds Triton, TorchInductor, CUDA, vLLM, B12X and XDG caches. A restart
  should reuse compiled kernels instead of redoing the 43-minute first boot.
- **Cache invalidation**: change `QWEN38_CACHE_KEY` when the container image,
  model revision or kernel environment changes. This prevents stale kernels
  from being used after an upgrade.
- **Model**: `{work_root}/models/qwen38-flash-next-fp4-local` is a local
  checkpoint; no Hugging Face download is attempted at startup
  (`HF_HUB_OFFLINE=1`).
- **SIF build scratch**: `pull-container.sh` extracts on node-local tmpfs and
  copies only the finished SIF to `/work`; never extract the layer tree on
  Lustre.

## Qwen3.5 on A100

```bash
llm/qwen35/serve.sh up --yes       # 122B, 2x A100
llm/qwen35/serve.sh status --json
llm/qwen35/serve.sh down
```

35B is explicit:

```bash
slurmhelps-serve --config llm/qwen35/config-35b.json up --yes \
  --backend-command bash llm/qwen35/serve-35b.sh
```

## Downstream consumers

Any OpenAI-compatible client can use the catalogue endpoint:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8010/v1
```

OpenCode provider examples:

- [`examples/opencode-qwen35-122b.json`](../examples/opencode-qwen35-122b.json)
- [`examples/opencode-qwen38-flash-next.json`](../examples/opencode-qwen38-flash-next.json)
- [`examples/opencode-serve.json`](../examples/opencode-serve.json)

The API key name is `SLURMHELPS_API_KEY`; no credential is stored in the repo.
