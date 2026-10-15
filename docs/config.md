# slurmhelps config

Read by `lib/slurmhelps/config.py` from `<project>/slurmhelps.json` (or the `"slurmhelps"` block of
`<project>/.langywrap/lanes.json`). Missing keys take the defaults in `config.py` (`DEFAULTS`).
Keys starting with `_` are comments.

| Key | Used by | Meaning |
|-----|---------|---------|
| `cluster.ssh` | hpc_run, hpc_chain, bridge, hpc_minutes, mirror_mount, code_sync | `user@login-host` |
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
| `mounts.<name>.remote`, `.mountpoint`, `.mode` | mirror_mount NAME | named sshfs mounts (`user@host:/dir`, `~/mnt/x`, `rw`/`ro`, default `rw`); default `{}`. Example below |
| `lan_gpu.host`, `base`, `ssh_key`, `known_hosts`, `threads`, `gpu` | gpu_run | LAN GPU workstation |
| `queue.dir` | gpu_queue, bridge | project-relative queue dir |
| `queue.bridge_note`, `bridge_default_minutes`, `bridge_max_sync_gb` | bridge | bridge behaviour |

## Named mounts (`mounts`)

`bin/mirror_mount.sh NAME [mount|status|umount [--lazy]]` mounts one entry over user-space sshfs;
`bin/mirror_mount.sh --list` prints them. Workstation side only: refused when `SLURM_JOB_ID` is
set, `sbatch` is on `PATH`, or the hostname matches `site.cluster_host_glob`. Named mounts never
pass `allow_other`, refuse a non-empty mountpoint, and `"mode": "ro"` mounts read-only.

```json
"mounts": {
  "whitebox":     {"remote": "holecem@hpc.troja.mff.cuni.cz:/work/ffa/holecem/whitebox",
                   "mountpoint": "~/mnt/troja-whitebox", "mode": "rw"},
  "workset-view": {"remote": "holecem@hpc-head.troja.mff.cuni.cz:/work/ffa/holecem/images/view",
                   "mountpoint": "~/mnt/troja-workset-view", "mode": "ro"}
}
```

The workset view is a squashfuse mount (`workset.sh view`) that exists on ONE login node only, so
its `remote` must name that pinned host (`site.login_host_pinned`, `hpc-head` on Troja), not the
round-robin `hpc.troja.mff.cuni.cz`. After the view is re-mounted on the cluster, remount the
sshfs side too.

## bootstrap

Read by `bootstrap/*.sh` (disaster recovery: `rebuild-all.sh`, `checkouts.sh`,
`toolchain-install.sh`, `state-init.sh`, `lock-versions.sh`). After a total
loss the project checkout is gone, so the same sections are kept as a standalone
file and passed with `--config` (`examples/adgfem.bootstrap.json`). Values may use
`{slurmhelps}` (this checkout); relative paths are project-relative.

| Key | Used by | Meaning |
|-----|---------|---------|
| `bootstrap.project_name` | all | checkout dir `<site.work_root>/<name>`; default: basename of the project root (set it in a `--config` file) |
| `bootstrap.origin`, `upstream`, `branch` | checkouts | clone `origin` at `branch`; `upstream` becomes a second remote; `gc.auto=0` is set |
| `bootstrap.lane_worktree_dir` | checkouts | lane worktrees inside the checkout (default `.lane-worktrees`) |
| `bootstrap.worktrees_file` | checkouts | `checkouts.sh snapshot` output (`git worktree list --porcelain`); lane worktrees in it are recreated when their branch/commit is on origin. Empty: `lane/<id>` for the lanes of `lanes.json` that exist on origin |
| `bootstrap.langywrap_origin`, `langywrap_branch` | checkouts | the shared langywrap checkout (`site.langywrap_checkout`) |
| `bootstrap.langywrap_submodules` | checkouts, toolchain-install | submodules to init (default `pi`, `rtk`); `rtk` there makes `--with-rust auto` install cargo |
| `bootstrap.pi_upstream` | checkouts | where pi's release tag (`versions.lock` `PI_RELEASE_TAG`) is fetched |
| `bootstrap.lock_file` | all | pinned versions (default `{slurmhelps}/bootstrap/versions.lock`) |
| `bootstrap.env_local` | toolchain-install | lines of `<work_root>/config/env.local.sh` (project extras sourced by `env/env.sh`; written only if absent) |
| `bootstrap.bin_shims` | toolchain-install | `{"name": "target"}`: `<work_root>/bin/<name>` exec shims |

## site (cluster-side paths)

Read by the cluster-side tools through `lib/common.sh` (`config.py site-shell` →
`SH_*` shell variables): `fleet/`, `image/`, `bootstrap/`, `maint/`. Values may use
`{work_root}` and `{user}`; `toolchain_profile` may use `{slurmhelps}` (this checkout).

| Key | Shell var | Meaning |
|-----|-----------|---------|
| `site.work_root` | `SH_WORK_ROOT` | per-user root on the shared parallel FS (Troja: `/work/ffa/{user}`); never `/home` |
| `site.agents_dir` | `SH_AGENTS_DIR` (`AGENTS_DIR`) | fleet logs + state: `alloc.env`, `logs/`, store snapshots |
| `site.langywrap_checkout` | `SH_LANGYWRAP_CHECKOUT` | the shared langywrap checkout + `.venv` (never in an image) |
| `site.env_sh` | `SH_ENV_SH` | toolchain + caches profile sourced by every fleet process (`env/env.sh` via its shim) |
| `site.user_bin`, `images_dir` | `SH_USER_BIN`, `SH_IMAGES_DIR` | user bin dir; squashfs images |
| `site.fleet_home`, `pi_agent_dir`, `codex_home`, `xdg_cache_home` | `SH_FLEET_HOME`, ... | HOME and agent homes of fleet processes |
| `site.node_tmp` | `SH_NODE_TMP` | node-local tmp (Troja: job-private tmpfs, RAM, wiped at job end) |
| `site.slurm_bin` | `SH_SLURM_BIN` | appended to PATH when missing |
| `site.partition_long`, `partition_short` | `SH_PARTITION_LONG`, `SH_PARTITION_SHORT` | fleet partition; helper-job partition |
| `site.short_time`, `short_cpus`, `short_mem` | `SH_SHORT_*` | helper-job footprint (`sh_srun_short`) |
| `site.toolchain_profile` | `SH_TOOLCHAIN_PROFILE` | site toolchain shell profile (`env/toolchain-mffsrv.sh` on Troja) |
| `site.toolchain_dirs` | `SH_TOOLCHAIN_DIRS` | user toolchain dirs relative to `work_root` (the toolchain image) |
| `site.node_version` | `SH_NODE_VERSION` | node under `<work_root>/node/<version>` |
| `site.mpi_env` | `SH_MPI_ENV` | exported for every fleet process after `env.sh` (Troja: `OMPI_MCA_pml=ob1`, the UCX SIGFPE landmine) |
| `site.login_host_pinned` | `SH_LOGIN_HOST_PINNED` | the login node holding FUSE views (`workset.sh view`) |
| `site.cluster_host_glob` | `SH_CLUSTER_HOST_GLOB` | hostnames that mean "on the cluster" (workstation tools refuse there) |

## fleet (the renewable allocation)

Read by `fleet/ly`, `fleet/supervisor.sbatch` and the lane launchers. Environment variables with the
historical names (`AGENTS_PARTITION`, `AGENTS_TIME`, `AGENTS_CPUS`, `AGENTS_MEM`,
`AGENTS_RENEW_SIGNAL`, `AGENTS_JOBNAME`, `AGENT_PROJECT`, `LANE_MODEL`, ...) still win, and
`ly ensure partition=... cpus=...` overrides per call.

| Key | Meaning |
|-----|---------|
| `fleet.project` | the langywrap project the daemon serves (default: the project root) |
| `fleet.jobname` | allocation job name (default `agent-supervisor`) |
| `fleet.partition`, `time`, `cpus`, `mem` | footprint (`--ntasks=cpus --cpus-per-task=1`); partition defaults to `site.partition_long` |
| `fleet.renew_signal` | seconds before the limit: `USR1` -> final spill -> self-requeue |
| `fleet.extra_writable` | colon list added to the lane sandbox writable roots (default `{work_root}/home`) |
| `fleet.work_checkout_var` | project-named alias of `SH_WORK_CHECKOUT` in node-local mode (ADGFEM: `ADGFEM_WORK_CHECKOUT`) |
| `fleet.lane_model`, `lane_prompt_file`, `lane_exclude` | `start-lanes.sh`: model, prompt template (`{lane}`, `{model}`), lanes left out of the default set |

## workset (image defaults)

Defaults for `image/workset.py`; the project's `.langywrap/lanes.json` `fleet.workset` block
overrides them key by key and is the only place `enabled` is read. Placeholders: `{project}`,
`{project_name}`, `{work_root}`, `{images}`, `{node_tmp}`, `{uid}`. Keys and defaults: see
`DEFAULTS` in `image/workset.py` and `image/README.md`.

### Per-job GPU memory admission

Queue submit accepts optional `--min-vram-gib`, `--min-free-vram-gib`, and
`--cupy-pool-limit-gib`. Values use GiB (2^30 bytes), must be finite positive
numbers, and are stored in `gpu_memory`. The pool requires a free-memory guard
and cannot exceed it. Minimum total VRAM is a lower bound, so a free requirement
may exceed it; routing uses the maximum requirement. For example, a 48GB-class
L40 job can request `--min-vram-gib 44 --min-free-vram-gib 42
--cupy-pool-limit-gib 40`. Advertised 48GB is not 48GiB.

The LAN runner filters incompatible pending rows before atomic claim and scans
past them without changing priority. It probes again immediately before claiming,
then checks actual total/free memory inside the existing remote job flock.
The cluster bridge preserves its existing lane fairness and claim protocol, but
constrained rows require `queue.bridge_gpu_vram_gib` to describe a conservative
minimum capacity across **all** GPUs allowed by the cluster routing config.
The default is `null` (unknown), so constrained rows are not bridged until that
capacity is explicitly configured. This field does not select a GPU model or
change cluster limits. Keep allowed/excluded GPU routing consistent with it.
Allocated jobs recheck actual memory before executing the model command.

If available memory drops after routing, the execution guard exits 78 and the
job finishes once; it is not automatically retried. Guard observations and the
effective requested pool are logged. Per-job pool settings export
`EVAL_CUPY_POOL_LIMIT_GB`, which supported evaluators must honor; arbitrary child
programs must implement that contract themselves. Explicit contradictory
`--cupy-pool-limit-gb` command arguments reject at submit. No host-global pool is
raised. Jobs without metadata retain previous execution behavior. Existing
cluster/LAN concurrency limits remain unchanged.

For declared pools, both submission and allocated execution reject known immediate
`env EVAL_CUPY_POOL_LIMIT_GB=...` overrides unless they match the declaration.
Matching numeric values are allowed and argv is never silently rewritten.
Known `--cupy-pool-limit-gb` overrides receive the same check at both boundaries,
including persisted/manual job records. Environment-clearing/unsetting or
unrecognized immediate `env` options fail closed for a declared pool. This does
not parse the interior of `bash -c`, Python code, or scripts: those frozen runners
remain responsible for honoring the pool contract and asserting their effective
backend limit. Memory admission is not a sandbox for arbitrary user code.
