# Qwen3.5 on A100 — serving and local agent benchmark

Two checkpoints, both 4-bit compressed-tensors / WNA16 Marlin:

| Variant | Checkpoint | GPUs | Port |
|---|---|---|---:|
| 35B-A3B | `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | 1× A100 | 8001 |
| 122B-A10B | `cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit` | 2× A100 | 8002 |

They declare native context **262,144**; they are not 700K replacements.

## Serving through slurmhelps

The default entry point is `llm/qwen35/serve.sh` (122B, 2× A100):

```bash
# 122B, two A100s
llm/qwen35/serve.sh up --yes
llm/qwen35/serve.sh status --json
llm/qwen35/serve.sh wait --timeout 3600
llm/qwen35/serve.sh endpoint
llm/qwen35/serve.sh down
```

Equivalent explicit form:

```bash
slurmhelps-serve --config llm/qwen35/config-122b.json up --yes \
  --backend-command bash $WORK_ROOT/slurmhelps-fleet/llm/qwen35/serve-122b.sh
```

# 35B, one A100 (not the default)
slurmhelps-serve --config llm/qwen35/config-35b.json up --yes \
  --backend-command bash $WORK_ROOT/slurmhelps-fleet/llm/qwen35/serve-35b.sh
```

`nice=100` is the default; for a short smoke, lower it after submission:

```bash
scontrol update JobId=<jobid> Nice=0
```

The backend sets `CC/CXX`, `CUDA_HOME`, `JIT_NVCC_COMPILER`, enables
`--disable-custom-all-reduce` (required for TP2; otherwise the workers can spin
in NCCL), and enables `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
for OpenCode.

## OpenCode provider

`~/.config/opencode/opencode.json` can have:

```json
"slurmhelps-qwen": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "slurmhelps qwen A100",
  "options": {"baseURL": "http://ampere02:8002/v1", "apiKey": "dummy"},
  "models": {"qwen35-122b-a10b": {"name": "Qwen3.5-122B-A10B (A100)"}}
}
```

Then:

```bash
opencode models | grep qwen35
opencode run --standalone --format json \
  --model slurmhelps-qwen/qwen35-122b-a10b \
  --auto 'Inspect this repository and run its tests.'
```

## Local agent benchmark

`opencode_tasks.py` runs four disposable-repo tasks through the real OpenCode
CLI:

1. **repo_search** — use repository search, write `calc.py:<line>` to `ANSWER.txt`.
2. **tool_use_fix** — run unittest, find/fix a bug without editing tests.
3. **coding** — implement `parse_ranges`, run unittest.
4. **multi_step** — search, fix, run tests, report the exact command/result.

Run:

```bash
cd $WORK_ROOT/slurmhelps-fleet
python3 llm/qwen35/opencode_tasks.py \
  --model slurmhelps-qwen/qwen35-122b-a10b \
  --timeout 600 \
  --output llm/qwen35/experiments/data/2026-10-23-opencode-qwen35-122b.json
```

This is deliberately local and cheap; it tests the agent loop, tool permissions,
repository search, editing and test execution. It is not a substitute for a
public leaderboard.

## Public benchmarks worth running

| Benchmark | What it measures | Local cost |
|---|---|---|
| **BFCL** (`ShishirPatil/gorilla`) | function/tool selection and argument AST | low–medium; dataset + evaluator |
| **RepoBench / RepoBench-P** | repository-level retrieval and code completion | low; HF dataset |
| **SWE-bench Lite/Verified** | real repo issue repair with tests | high; Docker + repo images |
| **Aider Polyglot** | multi-language code editing | medium; exercise harness |
| **Terminal-Bench** | shell/tool-using agents | medium–high; task containers |
| **τ-bench** | multi-turn tool-use with simulated users | medium |
| **BigCodeBench / LiveCodeBench** | code generation and execution | medium |

For this hardware, run **BFCL + RepoBench + the local OpenCode suite** first;
SWE-bench is the expensive follow-up because each task needs a container and a
repository snapshot.

## Known result

The A100 smoke runs passed all 10 HTTP checks:

- 35B: **164.4 output tok/s at C=8, 500-token context**
- 122B: **110.6 output tok/s at C=8, 500-token context**
- 35B KV 2.287M tokens; 122B KV 2.562M tokens
- shared-prefix short requests returned in ~1.6 s vs ~22–29 s for long ones

See [`experiments/2026-10-23-smoke-2xa100.md`](experiments/2026-10-23-smoke-2xa100.md).
The full OpenCode agent suite (repo_search, tool_use_fix, coding, multi_step)
passed **4/4**; see
[`experiments/2026-10-23-opencode-qwen35-122b.md`](experiments/2026-10-23-opencode-qwen35-122b.md).
