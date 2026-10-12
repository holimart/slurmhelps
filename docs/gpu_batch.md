# HPC helpers — generic Slurm/GPU batch (`gpu_batch.py`)

Project-agnostic, stdlib-only helpers for probing idle GPU nodes and submitting
GPU batch jobs to **any** Slurm cluster over ssh. Nothing here knows a specific
cluster, account, directory or partition — all of it is passed in.

## Probe

```bash
python lib/slurmhelps/gpu_batch.py probe --ssh user@login --account myacct \
    --exclude-node-prefix badnode --exclude-gpu V100 \
    --gpu-rank T600=1 --gpu-rank V100=10 \
    --partition-rank short=1 --partition-rank long=5
# --json for machine output
```

Runs a read-only `sinfo -h -N -o "%P|%N|%G|%t|%a"` and keeps only:

* `idle` nodes,
* with a `gpu:` GRES,
* whose node does not start with any `--exclude-node-prefix`,
* whose GPU model does not contain any `--exclude-gpu`,
* whose partition is permitted (`--allowed-partition` allow-list, else the
  partition's `AllowAccounts` must be `ALL` or include `--account`).

De-duplicated per node (keeping the cheapest), then sorted cheapest-first by
GPU rank, partition rank, node. Text output is one
`partition:node [gpu, state]` per line.

## Run

```bash
python lib/slurmhelps/gpu_batch.py run \
    --ssh user@login --account myacct \
    --workdir /scratch/env --remote-tmp /scratch/tmp \
    --local-batch ./batch --remote-name job1 \
    --run-cmd 'python run.py --out {output}' \
    --setup-cmd 'bash setup.sh' \
    --gres gpu:1 --cpus 4 --mem 16G --minutes 10 \
    --results-name results.jsonl --yes
```

`run` selects a GPU (same rules as `probe`, or honors `--partition`/`--nodelist`),
stages `--local-batch` to `<remote-tmp>/<remote-name>/` over rsync, generates the
sbatch script (tokens `{batch_dir}`, `{output}`, `{workdir}` substituted into
`--setup-cmd`/`--run-cmd`), submits with `sbatch --parsable`, polls `squeue`
until the job leaves, reads the final `sacct` state, rsyncs `<results-name>`
back to `--results-local` (default `./<remote-name>-<ts>.jsonl`) and removes the
remote directory.

Without `--yes` (or with `--dry-run`) it prints the plan and submits nothing.

Remote directories are created only under `--remote-tmp` / `--workdir`; nothing
is ever written to `$HOME`.

## Python API

```python
from slurmhelps.gpu_batch import probe_gpus, select_gpu, submit_run

nodes = probe_gpus("user@login", account="myacct", exclude_gpu=("V100",))
path = submit_run(
    "user@login",
    account="myacct",
    workdir="/scratch/env",
    remote_tmp="/scratch/tmp",
    local_batch="./batch",
    remote_name="job1",
    run_cmd="python run.py --out {output}",
    yes=True,
)
```

`run_ssh` / `run_local` are the single external-process choke points and are
monkeypatched by the network-free tests in `tests/test_gpu_batch.py`.
