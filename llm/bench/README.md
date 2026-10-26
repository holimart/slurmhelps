# `llm/bench` — one interface over every measurement we run

`benchctl` is a stdlib Python 3 CLI plus four thin bash adapters. It does not
implement a benchmark. It validates **recipes**, resolves site paths, prints
exactly what would be submitted, submits it with `--nice=100`, and projects
the result artifacts back into tables.

The backend scripts remain the source of truth for now:

```text
llm/strata/{bench-matrix,knob-sweep,warm-cache,chain-cache}.sh
                                                      Strata matrix / knobs / warm / chain
llm/qwen38/{serve,measure}.sh + serve-container.sh    vLLM FP4 on Blackwell
llm/qwen35/{serve,serve-server}.sh                    vLLM AWQ on A100
llm/glm53-reap50/{bench,context-sweep,cache-sweep,submit,install}.sh
llm/cloud/vast/*                                      rented-host campaign
```

Nothing under those directories was edited. `benchctl` composes them.

## Quick start

```bash
cd $WORK_ROOT/slurmhelps-fleet

llm/bench/benchctl recipes                 # list/validate all recipes
llm/bench/benchctl plan --recipe strata-l40x1
llm/bench/benchctl run  --recipe strata-l40x1 --dry-run
llm/bench/benchctl run  --recipe strata-l40x1
llm/bench/benchctl status
llm/bench/benchctl doctor --backend strata
llm/bench/benchctl report --experiment 2026-10-25-strata-smoke-matrix
```

Every command that makes sense as data has `--json`. `plan`, `run --dry-run`
and `prepare --dry-run` have no side effects: the adapters are invoked with
their own `--dry-run`, which prints the final command instead of executing it.

```text
recipe      strata-l40x1  (strata/matrix)
experiment  2026-10-25-strata-smoke-matrix
model       iq3s — Qwen3.8-Flash-Next IQ3_S (54.8 GB experts + 28.8 GB shared PLE)
hardware    1x L40S 48GB gpu:1 @uk-g6-05 gpu-ffa 16 cpus 256G 11:30:00
mode        cold
cells       15: 1024 x C1,2,4; 4096 x C1,2,4; ...; 131072 x C1,2,4
probe       max_context=131072 max_tokens=128 requests_per_slot=3 pad_safety=128
outdir      /work/ffa/holecem/strata-runs/strata-l40x1-20261026T113005Z
budget      420 min
estimate    wall 1h15m (2026-10-25 l40x1 matrix wall (both packs); ...)
submit      sbatch --nice=100 ... llm/strata/strata-bench.sbatch --model iq3s ...
```

## Commands

| command | what it does |
|---|---|
| `doctor [--backend all\|strata\|vllm\|glm53\|cloud] [--offline] [--deep-sha] [--strict]` | driver/CUDA visibility, GLIBCXX floor, engine + pack presence, node/partition state, disk/RAM, Vast credit and a live offer per cloud recipe (hourly price + `internet_down_cost_per_tb`), and the known footguns. `--offline` skips the Vast API. Missing optional backends are warnings, not failures. |
| `plan --recipe NAME\|FILE` | validate + print backend, model, hardware, cells, window, probe options, outdir, declared wall/cost, and the exact adapter/sbatch command. No side effects. |
| `prepare --backend strata\|vllm\|glm53 --model PACK` | delegates to `install-engine.sh`, `fetch-models.sh`, `pull-container.sh`, `hf download`, `glm53 install.sh`. Heavy image builds/compiled installs are held on the login node unless `--allow-heavy`. `--dry-run` prints. `--backend cloud --recipe NAME --instance ID` stages the campaign scripts and runs setup on the rented host. |
| `run --recipe NAME\|FILE [--dry-run]` | submit the job(s) with `--nice=100` (strata) or run the documented flow. A Vast recipe without `--yes` only plans. Client benchmarks (`measure`/`context`/`cache`) refuse the login node unless `--allow-heavy`. |
| `status [--all] [--json]` | our Slurm jobs, newest result dirs (cells/summary rows/status counts), last experiment docs. |
| `import [--dataset PATH] [--dry-run]` | hands **new** result dirs to `opportunityscan/modelsquantdb/scripts/import_strata_runs.py`; the dataset importer stays the only writer. State lives in `$WORK_ROOT/bench/state/import.json`. |
| `report --experiment DATE-SLUG [--append]` | assembles markdown tables from the recipes' evidence dirs and appends the recipes' caveats verbatim. `--append` is restricted to `llm/bench/experiments/` so hand-written campaign docs are never rewritten. |
| `vast plan\|rent\|stage\|run\|pull\|destroy` | passthrough to `llm/cloud/vast/vast.py`; `--recipe NAME` resolves to the recipe file. Mutations need `--yes`; `--dry-run` prints the request/rsync/ssh line. |
| `recipes`, `config` | validation table and the resolved site configuration. |

Site values live in `llm/bench/config.json` (work root, output roots, model
paths, Slurm account/nice, Vast defaults). Override the file with
`BENCH_CONFIG`, or only the root with `BENCH_WORK_ROOT`. No cluster path is
hardcoded in `benchlib/*.py`.

## Recipes

Sixteen recipes cover what was actually run (plus clearly-marked new
definitions). Each is validated against `schema.json`; a typo or an impossible
cell/stop fails before submission.

| recipe | backend/runner | hardware | what it reproduces |
|---|---|---|---|
| `strata-l40x1` | strata/matrix, iq3s | 1x L40S, uk-g6-05 | 1k-128k x C1/2/4 cold matrix |
| `strata-l40x2` | strata/matrix, iq3s | 2x L40S, uk-g6-08 | same ladder in layer-split mode |
| `strata-v100x1` | strata/matrix, iq3s | 1x V100-32G, volta05 | 1k-64k x C1/2/4 (cuda12) |
| `strata-v100-resident-2x` | strata/matrix, q2_0 | 2x V100-32G | fully-resident Q2_0, 1k-128k |
| `strata-v100-resident-4x` | strata/matrix, iq3s | 4x V100-32G | resident IQ3_S 1k-256k + warm 64k/128k/256k |
| `strata-knobs` | strata/knobs, iq3s | 1x L40S, uk-g6-07 | one-knob variants at 1k/4k/16k |
| `strata-warm` | strata/warm, iq3s | 1x L40S, uk-g6-06 | warm prompt-cache TTFT |
| `strata-chain-l40` | strata/chain, iq3s | 1x L40S, uk-g6-02 | checkpoint chain 1k→100k→200k→262144, 512 tokens/turn |
| `strata-chain-v100-4x` | strata/chain, iq3s | 4x V100-32G | **pending / not run**: same chain on resident 4x V100 |
| `strata-sustained-512` | strata/matrix, iq3s | 1x L40S | **new, not yet run**: 512-token answers |
| `qwen38-bw-fp4` | vllm/measure | 2x BW6000, bw01 | Qwen3.8 FP4 matrix against the container server |
| `qwen35-ampere02` | vllm/measure | 2x A100, ampere02 | Qwen3.5-122B smoke matrix |
| `glm53-cache-sweep` | glm53/cache | 2x L40 | llama.cpp prefix-cache ladder 100k-950k |
| `glm53-context-sweep` | glm53/context | 2x L40 | 4k-1M context points |
| `vast-titan` | vast/cloud | 2x Titan RTX | 2026-10-25 rented campaign (measured) |
| `vast-3090` | vast/cloud | 1x RTX 3090 | aborted campaign; plan only, no cells |

`output.subdir` is `{name}-{stamp}` so a rerun never overwrites evidence. The
historical directory is named in `evidence` and is what `report` reads.

## Examples per backend

### Strata (Slurm)

```bash
# what would run (no GPU, no submission):
llm/bench/benchctl plan --recipe strata-v100x1

# submit (always --nice=100 inside adapters/strata.sh):
llm/bench/benchctl run --recipe strata-v100x1

# override by editing the recipe (hardware.nodelist/time) or call the adapter:
llm/bench/adapters/strata.sh --runner matrix --model coder \
  --cells "1024:1,4096:4" --gpus 0 --partition gpu-ffa --nodelist uk-g6-05 \
  --gres gpu:1 --cpus 16 --mem 256G --time 11:30:00 --outdir /tmp/demo
```

The adapter chooses `strata-bench.sbatch`, `knob-sweep.sbatch`,
`warm-cache.sbatch` or `chain-cache.sbatch`, passes resources on the sbatch
command line (never via `--export`), and `--self-test` runs the runner's stub
test directly.

**Checkpoint chain.** `strata/chain` measures prefix growth instead of
refilling each context from zero. One fixed salt and one conversation slot;
each stop's prompt is the previous prompt plus appended filler, so turns 2..N
should reuse the previous checkpoint and prefill only the delta. The probe
writes one JSONL line per turn (`mode:"chain"`) with `stop`, `prev_stop`,
`prompt_tokens`, engine `reused_tokens`, `reuse_fraction`, `ttft_s`,
`prefill_s`, `prefill_delta_tokens`, strict `decode_s`/`tpot_ms`/
`decode_tok_s_strict`, `decode_tok_s`, `aggregate_tok_s` and `status`. The
recipe's `probe.stops` is a strictly-increasing comma list; every stop is
clamped to `max_context - tokens_per_turn - pad_safety`.

```bash
llm/bench/benchctl plan      --recipe strata-chain-l40
llm/bench/benchctl run       --recipe strata-chain-l40 --dry-run
llm/bench/adapters/strata.sh --runner chain --model iq3s \
  --stops 1024,100000,200000,262144 --tokens-per-turn 512 \
  --max-context 262144 --node uk-g6-02 --outdir /tmp/chain
```

### vLLM (Qwen3.8 Blackwell / Qwen3.5 A100)

```bash
# serve (resources come from llm/qwen3*/config-*.json):
llm/bench/adapters/vllm.sh --runner serve --flavour qwen38
llm/bench/adapters/vllm.sh --runner serve --flavour qwen35-122b

# measure an already-running server:
llm/bench/benchctl run --recipe qwen38-bw-fp4 --allow-heavy
llm/bench/benchctl run --recipe qwen35-ampere02 --allow-heavy

# local agent tasks (Qwen3.5 only):
llm/bench/adapters/vllm.sh --runner tasks --flavour qwen35-122b \
  --outdir /work/ffa/holecem/results/opencode-$(date -u +%Y%m%dT%H%M%SZ)
```

`qwen38` has `measure.sh`. `qwen35` does not, so the adapter runs the shared
`tests/serve_smoke.py` matrix that produced the 2026-10-23 numbers.

### GLM-5.3 REAP50

```bash
llm/bench/benchctl run --recipe glm53-cache-sweep      # client+server; needs an allocation
llm/bench/benchctl run --recipe glm53-context-sweep

# the generic sbatch wrapper stays reachable:
llm/bench/adapters/glm53.sh --runner submit --engine llamacpp --variant iq4_xs \
  --node uk-g6-05 --partition gpu-ffa --gres gpu:2 --cpus 32 --mem 256G \
  --time 04:00:00 --wait
```

`context`/`cache` start their own llama.cpp server; the CLI refuses them on the
login node. `submit` delegates to `submit.sh`, which owns the comma-safe
`--export` encoding.

### Vast.ai

```bash
# live offers + cost (needs VAST_API_KEY):
llm/bench/benchctl vast plan --recipe vast-titan
llm/bench/benchctl vast rent --offer-id 56177428 --label strata-titan --yes
llm/bench/benchctl vast stage --instance 57698991 --recipe vast-titan --yes
llm/bench/benchctl vast run   --instance 57698991 --script /root/run-matrix.sh --yes
llm/bench/benchctl vast pull  --instance 57698991 --dest /tmp/titan-results
llm/bench/benchctl vast destroy --instance 57698991 --yes
```

The client filters on driver, CUDA, reliability **and**
`internet_down_cost_per_tb <= 5` before showing a price. `plan` prints GPU cost
and the download cost separately. `destroy` verifies the instance list after
the DELETE.

### Results

```bash
llm/bench/benchctl status --json
llm/bench/benchctl import --dry-run
llm/bench/benchctl report --experiment 2026-10-26-v100-residency-kernels
llm/bench/benchctl report --experiment 2026-10-25-vast-titanx2 --append
```

## Migration table

| old command | new command | notes |
|---|---|---|
| `llm/strata/bench-matrix.sh …` | `llm/bench/benchctl run --recipe strata-l40x1` | recipe holds cells/hardware; adapter still calls the script |
| `sbatch … llm/strata/strata-bench.sbatch …` | same, via `benchctl run` | `--nice=100` is always added by the adapter |
| `llm/strata/knob-sweep.sh …` | `benchctl run --recipe strata-knobs` | variant list in `options.variants` |
| `llm/strata/warm-cache.sh …` | `benchctl run --recipe strata-warm` | probe `--mode warm` unchanged |
| `llm/strata/chain-cache.sh …` | `benchctl run --recipe strata-chain-l40` | probe `--mode chain --stops … --tokens-per-turn …` |
| `llm/strata/install-engine.sh` | `benchctl prepare --backend strata --model all` | idempotent; skips when both components exist |
| `llm/strata/fetch-models.sh --only iq3s` | `benchctl prepare --backend strata --model iq3s` | size check first; `--submit` still available |
| `python3 llm/strata/strata_probe.py …` | `python3 llm/bench/probe.py …` | canonical probe; upstream stays frozen at the pre-chain copy |
| `llm/qwen38/serve.sh up` | `adapters/vllm.sh --runner serve --flavour qwen38` | resources from `config-bw.json` |
| `BASE_URL=… llm/qwen38/measure.sh` | `benchctl run --recipe qwen38-bw-fp4 --allow-heavy` | server must already be up |
| `llm/qwen35/serve.sh up` | `adapters/vllm.sh --runner serve --flavour qwen35-122b` | 35B: `--flavour qwen35-35b` |
| `python3 tests/serve_smoke.py …` | `benchctl run --recipe qwen35-ampere02 --allow-heavy` | adapter wraps the same script |
| `llm/glm53-reap50/bench.sh` | `adapters/glm53.sh --runner bench …` | client-only against a live server |
| `llm/glm53-reap50/context-sweep.sh` | `benchctl run --recipe glm53-context-sweep` | needs an allocation |
| `llm/glm53-reap50/cache-sweep.sh` | `benchctl run --recipe glm53-cache-sweep` | needs an allocation |
| `llm/glm53-reap50/submit.sh` | `adapters/glm53.sh --runner submit …` | submit.sh still builds the sbatch line |
| `llm/glm53-reap50/install.sh` | `benchctl prepare --backend glm53 --model iq4_xs` | held on login unless `--allow-heavy` |
| curl + `PUT /asks/`, `DELETE /instances/` | `benchctl vast plan/rent/stage/run/pull/destroy` | filter + teardown verification built in |
| `/work/ffa/holecem/strata-runs/*` eyeballing | `benchctl status`, `benchctl report` | read-only projection |
| `modelsquantdb/scripts/import_strata_runs.py` | `benchctl import` | tracks which dirs were already imported |

## Discrepancies between the old scripts (and how the adapter translates)

These are deliberate; the adapters do not "fix" the runners.

1. **Matrix window is derived, not passed.** `bench-matrix.sh` chooses the
   server window from `max(cells)`; it has no `--max-context`. A recipe's
   `probe.max_context` is used for validation and documentation only. The
   `warm` and `knobs` runners do accept `--max-context`, and the adapter passes
   it there.
2. **Cell syntax differs.** Strata uses `1024:1,4096:4`; GLM context-sweep uses
   `4096:8 16384:4`. Recipes store the native syntax; adapters never rewrite
   the list.
3. **`--export` comma trap.** `submit.sh` encodes lists with `+`; the strata
   sbatch files take trailing argv. The adapters never put a comma list in
   `--export`.
4. **Qwen serving resources are in JSON, not flags.** `config-bw.json`,
   `config-122b.json` and `config-35b.json` own partition/nodelist/gres/time.
   Passing `--partition/--nodelist` to the vLLM adapter prints an informational
   note; it cannot override the serving controller.
5. **Qwen3.5 has no `measure.sh`.** The adapter uses the shared
   `tests/serve_smoke.py`, which is what produced the 2026-10-23 result files.
6. **GLM `submit.sh` builds its own sbatch.** Its `--export` list is long and
   comma-sensitive; re-implementing it in benchctl would be a rewrite. The
   adapter's dry-run prints the `submit.sh` invocation; run
   `submit.sh --dry-run` to see the exact sbatch line.
7. **Historical output layouts differ.** L40 matrices ended up under
   `results-copy/`; Vast under `results/<pack>/`; knobs under
   `<pack>/<variant>/`. `report` scans depth-first, prefers root-level
   summaries, and skips `attic/`, `_attempts/`, `data/`, `deps/` and
   `engine-logs/`.
8. **`llm/strata/strata_probe.py` is frozen; `llm/bench/probe.py` is
   canonical.** They were byte-identical until 2026-10-26, when chain mode and
   strict client timing were added to the bench copy while a V100 campaign was
   still using the upstream file. The bench probe keeps the upstream cold
   request shape and every legacy JSONL field; the extra cold cell fields are
   the additive strict p50/p95 metrics. The byte-copy test was replaced by a
   "canonical probe has chain + calibration" test; do not silently edit the
   frozen upstream.
9. **`strata --version` can block on the login node.** It was observed hanging
   for >20 s with no GPU. `doctor` reads component directory names instead.
10. **Vast tooling was never in the repo.** The campaign `tools/vast.py` was a
    14-byte `404: Not Found`; the run dirs are historical evidence and are not
    modified. `llm/cloud/vast/vast.py` is a fresh stdlib client with the
    lessons encoded.
11. **The `DRAFTER-READY` marker is not a readiness signal.** `doctor` checks
    `_strata-data/mtp/rt` and the shared install config instead; runners verify
    the engine start line.
12. **`llm/strata/experiments/2026-10-26-v100-run.sh` stages one allocation.**
    The 2x/4x recipes each run alone and will pay the Lustre PLE read again.
    This is called out in their caveats.

## Tests

All tests are stdlib `unittest` (the cluster system Python has no pytest) and
use stubs/fakes: no GPU, no model load, no network, no real `sbatch`.

```bash
cd $WORK_ROOT/slurmhelps-fleet
python3 -m unittest discover -s llm/bench/tests -v
```

| file | proves |
|---|---|
| `test_recipes.py` | all 16 recipes validate; typos, bad cells, unknown backends, GPU index beyond `gres`, cell > window, bad chain stops, cloud without offer fail fast; `plan` for every recipe creates no files |
| `test_adapters_dry_run.py` | every adapter and every recipe has a no-submit `--dry-run`; a fake `sbatch` proves the real strata submit path passes `--nice=100`; both `bench-matrix.sh` and `chain-cache.sh` self-tests pass end-to-end through the adapter |
| `test_probe_stub.py` | canonical probe has chain+calibration; legacy cold contract + clamp + strict p50/p95; warm reuse drops TTFT; chain prompts are prefix extensions, `reused` grows, delta prefill shrinks, strict decode/TPOT emitted |
| `test_doctor.py` | doctor JSON is self-consistent (unique ids, valid statuses, a fix for every warn/fail), required checks exist, backend filtering works, `--strict` exits 0 |
| `test_import.py` | new-dir detection, `--dry-run` does not import, state file updates once, second run is empty |
| `test_vast.py` | the $53/TB and driver-570 offers are rejected; plan separates GPU/download cost; rent/destroy/stage dry-runs print without network; destroy refuses without `--yes` |

## Remaining gaps

- **Quality evaluation** is not a mode. OpenCode tasks exist only for Qwen3.5
  (`opencode_tasks.py`); no shared quality harness, no BFCL/RepoBench runner.
- **AMD/ROCm** is not modelled: recipes assume CUDA (sm_70/sm_89/sm_120) and
  the Strata cuda12/cuda13 components.
- **Checkpoint chain is now expressible** as a first-class runner/mode
  (`strata-chain-*`, `probe.py --mode chain`), but **multi-recipe scheduling**
  is still not wrapped: `bin/hpc_chain.sh` can serialize arbitrary job lines,
  but there is no `benchctl chain` that orders recipes and waits for each
  result dir. The two chain recipes are separate submissions (L40 measured,
  4x V100 pending).
- **`vllm.sh` serve does not wait/verify readiness**; `serve.sh wait` remains a
  separate call. A `run --recipe qwen38-serve` flow would need its own recipe.
- **Report cannot infer provenance** beyond recipe evidence paths; it does not
  reconcile a rerun's JSONL with the historical table, by design.
- **Vast `stage`/`run` need an explicit `--instance`/`--ssh`**; no stateful
  instance registry is kept between commands (only `instance.json` written by
  `rent --outdir`).
- `doctor` cannot read per-node driver/CUDA from the login node; the runners'
  precheck remains authoritative.

Suggested next step: add a `chain` recipe kind that lists other recipe names
and a `--wait`/`--stop-on-fail` scheduler over `squeue`, so a multi-pack night
(e.g. V100 A→B→E) is one command with one result root. That is the smallest
step that removes the remaining manual sequencing from
`llm/strata/experiments/2026-10-26-v100-run.sh`.
