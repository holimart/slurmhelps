# slurmhelps config

Read by `lib/slurmhelps/config.py` from `<project>/slurmhelps.json` (or the `"slurmhelps"` block of
`<project>/.langywrap/lanes.json`). Missing keys take the defaults in `config.py` (`DEFAULTS`).
Keys starting with `_` are comments.

| Key | Used by | Meaning |
|-----|---------|---------|
| `cluster.ssh` | hpc_run, hpc_chain, bridge, hpc_minutes, mirror_mount | `user@login-host` |
| `cluster.user` | squeue/sacct `-u` | default: user part of `ssh` |
| `cluster.account` | hpc_run | Slurm account (partition AllowAccounts filter) |
| `cluster.mirror` | hpc_run, bridge | project mirror on the cluster; jobs `cd` here |
| `cluster.remote_tmp` | hpc_run | staging dir for batch dirs (never `$HOME`) |
| `cluster.slurmhelps_remote` | hpc_run | where this checkout is rsynced; `<it>/lib` goes on PYTHONPATH and into `<mirror>/<venv>` as `slurmhelps.pth` |
| `cluster.venv` | hpc_run, bridge | mirror-relative venv |
| `cluster.job_prefix` | all cluster tools | our job names start with it (whitebox: `wb-`) |
| `cluster.max_jobs` | hpc_run, hpc_chain | at most this many of our jobs queued/running |
| `cluster.gres` / `cpus` / `mem` | hpc_run | per-job caps |
| `cluster.threads` | hpc_run, bridge | OMP/MKL threads |
| `cluster.env` | hpc_run | extra `KEY=VALUE` exported in the job setup (e.g. a CuPy pool limit) |
| `cluster.allowed_partitions`, `partition_rank` | hpc_run | allow-list and cost order |
| `cluster.exclude_node_prefixes`, `exclude_nodes`, `exclude_gpus`, `gpu_rank` | hpc_run | node/GPU filters (probe AND `#SBATCH --exclude`). On Troja `fuuk` must be excluded: hpc_run refuses a Troja config without it |
| `cluster.poll_slack_s` | hpc_run | poll timeout = minutes*60 + this |
| `cluster.sacct_since` | hpc_minutes | default `sacct -S` |
| `hpc_dir` | hpc_run, hpc_chain, bridge | project-relative dir for job dirs, chain logs, `bridge.enabled` |
| `mount.remote`, `mount.mountpoint` | mirror_mount | sshfs source (default `ssh:mirror`) and mountpoint |
| `lan_gpu.host`, `base`, `ssh_key`, `known_hosts`, `threads`, `gpu` | gpu_run | LAN GPU workstation |
| `queue.dir` | gpu_queue, bridge | project-relative queue dir |
| `queue.bridge_note`, `bridge_default_minutes`, `bridge_max_sync_gb` | bridge | bridge behaviour |
