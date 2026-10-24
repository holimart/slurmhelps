# 2026-10-24 — OpenCode 4/4 on Qwen3.8-Flash-Next (2× BW6000)

The Blackwell endpoint was tested with the real OpenCode CLI (not a curl
smoke), using the same local task suite as the Qwen3.5 run.

## Serving

The controller was already `ready` on `bw01`; no restart was needed.

```bash
cd $WORK_ROOT/slurmhelps-fleet
llm/qwen38/serve.sh status --json
# state=ready, endpoint=http://127.0.0.1:8010/v1, node=bw01
```

The workstation reached the loopback-bound server through the login node:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8010:bw01:8010 troja
curl -s http://127.0.0.1:8010/v1/models
```

## OpenCode provider

[`examples/opencode-qwen38-flash-next.json`](../../../examples/opencode-qwen38-flash-next.json)
points at `http://127.0.0.1:8010/v1`; `SLURMHELPS_API_KEY` is unset on this
endpoint, so a placeholder key is enough.

## Local OpenCode benchmark

```bash
XDG_CONFIG_HOME=<provider config dir> \
python3 llm/qwen35/opencode_tasks.py \
  --model slurmhelps-qwen38/qwen38-flash-next \
  --timeout 600 \
  --work-root /tmp/opencode-bench \
  --output llm/qwen38/experiments/data/2026-10-24-opencode-qwen38-flash-next.json
```

Result: **4/4 passed**.

| Task | Result | Time |
|---|---|---:|
| repo_search | PASS (`calc.py:5`) | 16.2 s |
| tool_use_fix | PASS (4 unittest tests OK) | 16.7 s |
| coding | PASS (4 unittest tests OK) | 15.1 s |
| multi_step | PASS (4 unittest tests OK) | 14.0 s |

The model used `bash`, `read`, `edit`, `write`, `glob` and `grep` tools and
ran `python3 -m unittest discover -s tests` itself. Raw events:
[`data/2026-10-24-opencode-qwen38-flash-next.json`](data/2026-10-24-opencode-qwen38-flash-next.json).

## Notes

- OpenCode 1.18.35 removed the `run --standalone` flag; the harness now calls
  plain `opencode run`.
- The measured 262K context is untouched by these short (~11K token) tasks.
