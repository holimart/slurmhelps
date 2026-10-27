# 2026-10-27 — AMD MI210 campaign: Qwen3.8-Flash-Next on 2× gfx90a

The unmeasured AMD point in the serving matrix. **r50** is a PCIe node with
2× **AMD Instinct MI210 64 GB** (gfx90a, CDNA2), 384 EPYC 9654 CPUs, 1.5 TB
RAM, ROCm 6.2. This campaign measures a **compiled-for-gfx90a llama.cpp HIP
build tuned for decode**, not the Strata engine used on the CUDA rigs, and
derives the CDNA roofline efficiency, tokens/J and €/Mtok.

MI50 (gfx906) and MI100 (gfx910) are **not** part of this cluster and are not
measured here. Their numbers in the report are paper/spec estimates and are
marked **unvalidated**; only MI210 is a measured AMD point.

## Cluster finding — which partition accepts the account

`--test-only` accepts both partitions, but only `ffa-preempt` actually runs
the account's jobs:

| partition | `AllowAccounts` | test-only | real submit |
|:--|:--|:--|:--|
| `math` | `math` | "to start at …" | **PENDING forever, `Reason=PartitionConfig`** |
| `ffa-preempt` | `ALL` | "to start at …" | **runs** (12 h / 36 h max, preemptible) |

All jobs below therefore use `--partition=ffa-preempt --nice=100`. `r50` is
listed in `math`, `ffa-preempt` and `ffa-checkpoint`; `math` is for the math
account, not `ffa`.

## Node facts (probe job)

```text
host        r50.hpc.troja.mff.cuni.cz, Rocky Linux 9.2, kernel 5.14.0-687
CPU         2× AMD EPYC 9654 (96C/192T each), 384 logical CPUs, 2 NUMA nodes
RAM         1510 GiB visible, 1498 GiB available at probe
GPU         2× AMD Instinct MI210, gfx90a:sramecc+:xnack-, 104 CUs, wave64
VRAM        68,702,699,520 B = 63.98 GiB usable per card (64 GB class)
ROCm        HIP 6.2.41133-dd7f95766, AMD clang 18.0.0git (roc-6.2.0)
rocm-smi    2.3.0+5ac86f6 (lib 7.3.0), amd-smi 24.6.2
topology    GPU0↔GPU1 = PCIE, 2 hops, weight 40, no xGMI; both under NUMA 1
            (PCIe Gen4 ×16 per card; this is the PCIe MI210 SKU, not OAM)
idle        41 W/card, sclk 800 MHz, mclk 1600 MHz, 38 °C edge
limits      300 W max power, 99 °C edge / 105 °C hotspot slowdown
scratch     /tmp and /dev/shm are both 756 GiB tmpfs (RAM-backed)
```

`/dev/shm` is used for model staging: it is RAM, not Lustre, and 1.5 TB is
enough for all three packs (208 GB) plus the measurement state.

## Source and build

Source: upstream **llama.cpp b11785** (`8baabd53b77d1c48c6ef8d3b620436292f3f6231`),
shallow clone at `/work/ffa/holecem/src/llama-hip-b11785`. The architecture is
present and the PLE graph input exists at the exact symbol the campaign asked
for:

```text
src/models/qwen4exp.cpp:1197: class llm_graph_input_qwen4exp_ple
src/llama-arch.cpp:          LLM_ARCH_QWEN4EXP -> "qwen4exp"
```

`gguf` metadata of the IQ3_S pack confirms the match:

```text
general.architecture = qwen4exp
block_count 48, embedding 2560, heads 24, kv heads 2, key/value 256
full_attention_interval 4, indexer.head_count 4, indexer.top_k 2048
ple.layers [1], ple.ngram_size 3, ple.heads_per_ngram 8, ple.conv_kernel 4
per_layer_token_embd.weight = IQ4_NL [160, 320001536]  (28.8 GB shard-2 table)
```

Build (inside the `build.sbatch` job, never on the login node):

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DAMDGPU_TARGETS=gfx90a \
  -DGGML_HIP=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release -j 64
```

Flags checked in-tree before use:

| flag | state | note |
|:--|:--|:--|
| `GGML_HIP` | present | HIP backend |
| `GGML_HIP_MMQ_MFMA` | present (default ON) | CDNA2 MFMA MMQ; explicitly ON |
| `GGML_HIP_GRAPHS` | present (default ON) | HIP graphs |
| `GGML_HIP_NO_VMM` | present (default ON) | no VMM on ROCm 6.2 |
| `GGML_CUDA_FA` | present (default ON) | flash attention |
| `GGML_CUDA_FA_ALL_QUANTS` | present | needed for q8_0 KV variant |
| `GGML_HIP_ROCWMMA_FATTN` | **absent** | not in b11785's `ggml/CMakeLists.txt` or `ggml/src/ggml-hip/CMakeLists.txt`; dropped, not silently passed |

Compiler recorded by the job: HIP 6.2.41133, clang 18.0.0git, cmake 3.31.8,
ninja 1.10.2. Binaries are persisted at
`/work/ffa/holecem/build/llama-hip-b11785-gfx90a/` (bin + lib).

## Hard rule — no live logs on Lustre

Job logs are written to node-local `/tmp` or `/dev/shm`. Only `squeue`/`sacct`
are polled while a job runs. After a job ends, small summaries (`summary.csv`,
cell JSONL, head/tail of server logs, `rocminfo.txt`, build log) are copied to
`/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210/`. Nothing greps or tails a
Lustre file that a running job may still be writing.

## Models (pinned, size-checked; SHA-256 verified in `manifest.txt`)

| pack | shard 1 | shard 2 (shared PLE table) | bytes/token active |
|:--|---:|---:|---:|
| IQ3_S | 54,817,524,224 | 28,800,138,432 | 2.63 GB (6 B active @ 3.51 bpw) |
| Coder IQ1_M | 29,608,446,496 | 28,800,138,432 | ~1.4 GB (6 B @ 1.9 bpw) |
| Q2_0 | 37,623,740,192 | 28,800,138,432 | ~1.8 GB (6 B @ 2.4 bpw) |

The campaign checks exact shard sizes before launch; full SHA-256 was already
done when the packs were fetched (`models/strata/manifest.txt`) and is not
re-read from Lustre in a timed job.

## Measurement protocol

- **Resident**: `-ngl 99`, f16 KV, flash attention on, 2-card layer split
  (`-sm layer`, default). The server/bench logs record `offloaded N/48` and
  per-device buffer sizes; `rocm-smi --showmeminfo vram` is captured at load.
- **Prefill / decode**: `llama-bench` separately, `-p <ctx> -n 0` for pp and
  `-p 0 -n 128 -d <ctx>` for tg at depth, JSON output.
- **TTFT / cells**: a running `llama-server` (`-c 262144 -np 1`), stdlib client
  `probe_client.py`; TTFT = server `prompt_ms` + one `predicted_per_token_ms`;
  the actual `prompt_n` from the server is reported, never the target.
- **Concurrency**: `-np 2` (2×131 k slots), two simultaneous requests at
  4 k and 16 k; per-stream and aggregate decode from the overlap window.
- **KV variant**: `-ctk q8_0 -ctv q8_0` at 64 k/262 k.
- **Power**: `power_sampler.py` reads amdgpu hwmon at 1 Hz (power, sclk,
  temp, energy counter), 30 s idle baseline, windows around the 16 k and
  262 k cells; `power-report.py` computes cell and decode-only tokens/J.

## Scripts

```text
probe.sbatch        rocminfo/rocm-smi/topology/scratch/compiler facts
build.sbatch        HIP build + HBM float4 baseline
validate.sbatch     smoke: qwen4exp load, bench pp/tg, server, q8_0 KV
knobs.sbatch        decode knobs, the 4096 state-restore fault, 262k fit
run-campaign.sh     the 11 h measurement job (packs sequential)
strata-hip.sbatch   optional Strata HIP attempt for gfx90a (clean refusal)
probe_client.py     stdlib llama-server client, exact token calibration
power_sampler.py    1 Hz hwmon sampler
power-report.py     energy, tokens/J, €/Mtok
summarize.py        cells -> summary.csv
report.py           summary/power -> the markdown tables in the doc
active-bytes.py     per-token active bytes + roofline from the GGUF
hbm-bw.cpp          float4 HBM copy/triad/read baseline
```

## Run

```bash
cd $WORK_ROOT/slurmhelps-fleet
sbatch --nice=100 llm/bench/experiments/amd-mi210/probe.sbatch
sbatch --nice=100 llm/bench/experiments/amd-mi210/build.sbatch
sbatch --nice=100 llm/bench/experiments/amd-mi210/validate.sbatch
sbatch --nice=100 llm/bench/experiments/amd-mi210/run-campaign.sh
```

Every script pins `--partition=ffa-preempt --nodelist=r50 --gres=gpu:2` and
`--nice=100`. The run job asks for 1200 G because `/dev/shm` pages are charged
to the job's memory cgroup.

## Output map

```text
probe/            rocminfo.txt, rocm-smi-*.txt, probe.log
build/            build.log, build-make.log, cmake-command.txt, CMakeCache.txt, build-facts.txt
validate/         validate-facts.txt, hbm-bw.txt, bench-*.json, cells-*.jsonl
knobs/            tg variants (baseline, PLE-GPU, row split, FA off), 262k fit
cells/            bench-<pack>-{pp,tg}<ctx>.json, server-<pack>-{f16,q8kv,np2}.jsonl
power/            power-<job>.csv, cell-16k.json, cell-262k.json
logs/             head/tail of server and bench logs, session.log
residency-*.txt   VRAM at 262k; the lazy PLE table is host-mapped
summary.csv       flattened pp/tg/TTFT table
active-bytes-*.json  active bytes/token + roofline, per pack
strata-hip/       configure refusal + logs (no gfx90a binary)
timing.csv        staging seconds
skipped.tsv       any cell skipped for time
```

## Caveats

- **Not Strata.** The CUDA comparison rows in the parent doc were measured
  with the Strata engine; this row is llama.cpp b11785 HIP. Engine differences
  are part of the comparison, and are called out rather than hidden.
- **PCIe MI210, no xGMI.** The 2-card layer split has no peer fabric; only the
  small activation hand-off crosses PCIe Gen4 ×16.
- **ECC on** (`sramecc+`); the HBM baseline is the measured achievable
  bandwidth (read 1.336 TB/s), not the 1.6 TB/s paper peak. Both are reported.
- **Power is GPU package power**, not wall power; CPUs, fans and PSU losses
  are excluded. €/Mtok uses €0.20/kWh and **no capex**; the capex assumption
  is stated separately in the report and is not folded into the €/Mtok.
- **One allocation.** Cells are single-run unless repeats are noted; a
  preemption can truncate the last pack (the script records skips in
  `skipped.tsv` and never fabricates a row).
- **MI50/MI100 are unvalidated estimates.** They are not in this cluster;
  ROCm ≥7 does not support gfx906. Do not import their rows as measurements.
- **No Strata point on gfx90a.** Strata's HIP backend refuses gfx90a at CMake
  time (wave32 RDNA only); the MI210 row is llama.cpp b11785 HIP. Do not read
  it as the engine's ceiling on this card.
- **Hybrid KV state restore is broken in b11785.** llama-bench repeats log
  `failed to restore kv cache`; all cells are single-rep. One `-r 2` run at an
  exact 4096 depth also produced a GPU memory access fault and three 53 GB
  `gpucore.*` files (deleted). Use `ctx−1` and watch Lustre quota.
