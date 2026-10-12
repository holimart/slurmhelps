# slurmhelps

Cluster job tooling shared by projects: Slurm GPU jobs over ssh (Troja first), a job chain, a
mirror mount, a LAN GPU workstation runner, and the lane GPU queue that feeds both. Everything
cluster-, user- and host-specific comes from the **calling project's** config; nothing here
hardcodes a cluster.

Check it out as a sibling of the project (`<projects>/slurmhelps`) and call it as
`../slurmhelps/...` from the project root.

## Layout

| Path | What |
|------|------|
| `bin/hpc_run.sh` | one Slurm GPU job: cheapest idle allowed GPU, node/GPU exclusions, caps, one-job limit, node-local writes, failure write-up (`{output}`, `{output2}`) |
| `bin/hpc_chain.sh` | runs `JOB<TAB>MINUTES<TAB>RUNCMD` lines one after another (paired `GPU &&& CPU` lines), bridge hook when idle; `--dry-run` |
| `bin/hpc_bridge_gpu_queue.py` | runs the oldest pending lane gpu_queue job on the cluster when the chain is idle |
| `bin/gpu_queue.py` | the lane GPU queue (`submit` / `status` / `log` for lanes, `run` for the steward's runner) |
| `bin/gpu_run.sh` | one job on the LAN GPU workstation (rsync scripts + data, flock, thread caps) |
| `bin/hpc_minutes.py` | suggest `--time` per job type from sacct history |
| `bin/mirror_mount.sh` | sshfs mount of the cluster mirror |
| `lib/hpc_partial.py` | `PartialResults` checkpointing for job code (`from hpc_partial import PartialResults`) and the wrapper's `finalize-failure` |
| `lib/slurmhelps/config.py` | project config loader (`show`, `get KEY`, `shell`) |
| `lib/slurmhelps/gpu_batch.py` | generic probe + sbatch/poll/fetch engine used by `hpc_run.sh` (stdlib only; was langywrap `scripts/hpchelpers`) |
| `lib/slurmhelps/slurm_gpu.py`, `slurm_gpu_cli.py` | on-demand GPU jobs from `fleet.slurm.gpu` in lanes.json (was `langywrap jobs gpu`) |
| `docs/` | `config.md`, `gpu_batch.md`, `hpc_partial.md` |
| `examples/slurmhelps.json` | whiteboxchallenge's config (Troja + LAN 1080 Ti) |

## Configuration

`<project>/slurmhelps.json`, or a top-level `"slurmhelps"` block in `<project>/.langywrap/lanes.json`.
See `docs/config.md` and `examples/slurmhelps.json`. The project root is `$SLURMHELPS_PROJECT`, else
the nearest directory above the cwd holding the config, else the git toplevel.

```bash
python3 ../slurmhelps/lib/slurmhelps/config.py show
```

## hpc_partial on PYTHONPATH

Job code imports `hpc_partial`. `bin/hpc_run.sh` rsyncs this checkout to `cluster.slurmhelps_remote`
before every real submit, puts `<remote>/lib` on the job's `PYTHONPATH`, and installs a
`slurmhelps.pth` naming it into the mirror venv (so commands that set `PYTHONPATH=scripts` inline still
import it). `bin/gpu_run.sh` does the same on the LAN host. Locally, install the `.pth` once per venv:

```bash
../slurmhelps/bin/install_pth.sh .venv
```

## Tests

```bash
python -m pytest            # network-free; needs pytest and click
```
