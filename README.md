# slurmhelps

Cluster job tooling shared by projects: Slurm GPU jobs over ssh (Troja first), a job chain, a
mirror mount, a LAN GPU workstation runner, and the lane GPU queue that feeds both; and, on the
cluster side, the renewable Slurm allocation that hosts a langywrap agent fleet (`fleet/`), its
squashfs working-set images (`image/`), disaster-recovery bootstrap (`bootstrap/`), maintenance
helpers (`maint/`) and the user toolchain profile (`env/`). Everything cluster-, user- and
host-specific comes from the **calling project's** config; nothing here hardcodes a cluster.

Where things run:

| Area | Runs on | Config read through |
|------|---------|---------------------|
| `bin/`, `lib/named_mounts.sh` | the operator's workstation (ssh to the cluster) | `lib/slurmhelps/config.py` |
| `fleet/`, `image/`, `maint/` | the cluster: login node (bridge, `view`, `status`) and Slurm jobs (everything heavy) | `lib/common.sh` (`config.py site-shell`) |
| `bootstrap/` | a Slurm job on the cluster, after a total loss | `lib/common.sh` + `--config FILE` |
| `env/` | sourced by every shell on the cluster via `<work_root>/config/env.sh` | `site` section |

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
| `bin/mirror_mount.sh` | sshfs mount of the cluster mirror; named mounts from `mounts` (`--list`, `NAME [mount\|status\|umount]`) — workstation side |
| `bin/code_sync.sh` | rsync job code to the cluster and verify every file's sha256 there — workstation side |
| `lib/named_mounts.sh` | named-mount logic sourced by `mirror_mount.sh` (and the cluster-node refusal for `code_sync.sh`) |
| `lib/hpc_partial.py` | `PartialResults` checkpointing for job code (`from hpc_partial import PartialResults`) and the wrapper's `finalize-failure` |
| `lib/slurmhelps/config.py` | project config loader (`show`, `get KEY`, `shell`) |
| `lib/slurmhelps/gpu_batch.py` | generic probe + sbatch/poll/fetch engine used by `hpc_run.sh` (stdlib only; was langywrap `scripts/hpchelpers`) |
| `lib/slurmhelps/slurm_gpu.py`, `slurm_gpu_cli.py` | on-demand GPU jobs from `fleet.slurm.gpu` in lanes.json (was `langywrap jobs gpu`) |
| `lib/common.sh` | cluster-side config loader (`sh_load`, `sh_fleet_env`): `site` + `fleet` sections as `SH_*` / `AGENTS_*` variables |
| `fleet/ly` | login-side bridge into the fleet allocation (`status`, `ensure`, `up`, `down`, `renew`, `exec`, `lanes`, `start-lane`, `spill`, `store`, `workset`, `hot-reload`) |
| `fleet/supervisor.sbatch` | the renewable allocation: daemon, store stage-in/snapshots, node-local or workset mode, final spill on every exit path |
| `fleet/nodelocal.py`, `fleet/store-sync.py` | node-local git clone working set (`fleet.node_local_checkout`); node-local session store sync |
| `fleet/lane-common.sh`, `start-lanes.sh`, `start-all-lanes.sh`, `start-one-lane.sh`, `label-lanes.sh` | lane launchers (inside the allocation) |
| `fleet/login-node-guard.sh`, `stray-check.py`, `index-hygiene.py`, `steward-tick.sh`, `prepare-project.sh` | steward checks |
| `fleet/test-nodelocal.sh`, `fleet/test_index_hygiene.py` | fleet tests (Slurm job) |
| `image/workset.sh`, `image/workset.py`, `image/toolchain.sh` | squashfs working-set + toolchain images: build, stage-in, hold/enter/exec, spill, view, writeback (`image/README.md`) |
| `image/test-workset.sh` | image tests (Slurm job: needs user namespaces, squashfs tools, FUSE) |
| `bootstrap/` | disaster recovery: `rebuild-all.sh`, `toolchain-install.sh`, `checkouts.sh`, `state-init.sh`, `versions.lock` (`bootstrap/README.md`) |
| `maint/` | `install-shims.sh`, `land.sh`, `salvage.sh`, `wt-audit.sh`, `conflicts.sh`, `trash-bulky.sh`, `check-home.sh`, `fuse-bench.sh`, `toolchain-size.sh` |
| `env/env.sh`, `env/toolchain-mffsrv.sh` | user toolchain + caches profile; Troja's MPI/MKL toolchain profile |
| `tests/` | pytest suite (workstation tools; network-free) + `tests/run-cluster-tests.sh` (cluster suites, in Slurm) |
| `docs/` | `config.md`, `gpu_batch.md`, `hpc_partial.md` |
| `examples/slurmhelps.json` | whiteboxchallenge's config (Troja + LAN 1080 Ti) |
| `examples/adgfem.bootstrap.json`, `examples/adgfem.worktrees` | ADGFEM's standalone disaster-recovery config + worktree snapshot |

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

<!-- BEGIN workstation-side (mirror_mount named mounts, code_sync) -->
## Workstation-side

These run on the operator's machine, never on the cluster, where `/work` is already local. Named
mounts and `code_sync.sh` refuse to run when `SLURM_JOB_ID` is set, when `sbatch` is on `PATH`, or
when the hostname matches `site.cluster_host_glob`. The no-name mirror mount behaves exactly as
before.

```bash
../slurmhelps/bin/mirror_mount.sh                      # the project mirror (mount.remote / mount.mountpoint)
../slurmhelps/bin/mirror_mount.sh --list               # named mounts from slurmhelps.json "mounts"
../slurmhelps/bin/mirror_mount.sh workset-view         # mount (no-op if already mounted)
../slurmhelps/bin/mirror_mount.sh workset-view status  # 0 mounted, 1 not mounted, 2 stale
../slurmhelps/bin/mirror_mount.sh workset-view umount [--lazy]
../slurmhelps/bin/code_sync.sh ./job /work/ffa/$USER/jobs/job1 [--delete] [--exclude P] [--dry-run]
```

- **Named mounts** are `"mounts": {"NAME": {"remote": "user@host:/dir", "mountpoint": "~/mnt/x",
  "mode": "rw"|"ro"}}`; see `docs/config.md`.
  - They use user-space sshfs (`PATH`, else `~/.local/bin/sshfs`, or `$SSHFS`) and the system
    `fusermount3`.
  - Options: `reconnect`, `ServerAliveInterval=15`, `ServerAliveCountMax=3`, `idmap=user`,
    `dir_cache` (5 s for rw, 60 s for ro), and `ro` for read-only mounts. `SH_SSHFS_OPTS`
    adds options; `allow_other` is always refused.
  - A non-empty mountpoint is refused.
- **No-sudo sshfs** (Debian/Ubuntu; the system `fuse3` package provides `fusermount3` and
  `libfuse3`):
  ```bash
  cd "$(mktemp -d)" && apt-get download sshfs && dpkg -x sshfs_*.deb ~/.local/pkg-sshfs
  mkdir -p ~/.local/bin && ln -sf ~/.local/pkg-sshfs/usr/bin/sshfs ~/.local/bin/sshfs
  ```
- **Workset view.** `workset.sh view` squashfuse-mounts the newest working-set image at
  `images/view` on ONE login node. Its `mounts` entry must therefore name that pinned host
  (`site.login_host_pinned`, `hpc-head` on Troja), not the round-robin alias. Remount the sshfs
  side after the view is refreshed.
- **code_sync.sh** sends job code to the cluster:
  1. It builds a local sha256 manifest of exactly the files rsync sends; `.git/` is excluded by
     default.
  2. It runs `rsync -a` to `cluster.ssh` (or `--host`).
  3. It runs `sha256sum -c` of that manifest on the cluster.

  Exit codes: 0 verified, 1 error, 3 checksum mismatch. Use mounts to read and write outputs in
  place; use `code_sync.sh` for code a job runs.
- **Tests:** `tests/shell/test-remote.sh` uses stubs for sshfs, fusermount3, ssh and rsync, and
  the pytest suite runs it through `tests/test_remote_shell.py`. A real workstation mount cannot
  be tested from the cluster.
<!-- END workstation-side -->

<!-- BEGIN cluster-side (fleet, image, bootstrap, maint, env) -->
## Cluster-side: adopting slurmhelps for a langywrap fleet

1. **Check out slurmhelps on the cluster** as a git checkout of its own (on Troja
   `/work/ffa/$USER/slurmhelps-dev` today; see "Deploy target" below). Pin it in the project's
   `.langywrap/lanes.json`: `"fleet": {"slurmhelps": {"path": "<checkout>", "ref": "<commit>"}}`.
   `ly info` warns when the checkout's HEAD differs from the pinned ref.
2. **Write the project config** (`<project>/slurmhelps.json`): the `site` section (paths,
   partitions, `mpi_env`), the `fleet` section (footprint, lane prompt), optionally `workset`
   defaults and `bootstrap`. See `docs/config.md`.
3. **Keep thin shims in the project** so existing commands keep working, e.g.
   `<project>/cluster/agents/ly`:
   ```bash
   #!/usr/bin/env bash
   export SLURMHELPS_PROJECT="${SLURMHELPS_PROJECT:-$(cd "$(dirname "$0")/../.." && pwd)}"
   exec bash "${SLURMHELPS_HOME:-$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["fleet"]["slurmhelps"]["path"])' "$SLURMHELPS_PROJECT/.langywrap/lanes.json")}/fleet/ly" "$@"
   ```
4. **Once per user**: `maint/install-shims.sh --apply` points `<work_root>/config/env.sh`,
   `<work_root>/bin/*` and old fleet script copies at slurmhelps.
5. `ly ensure`, `ly status`, `ly doctor` (from the login node; everything heavy runs in the allocation).

Working-set images (`fleet.workset`): `image/README.md`. Node-local git clone
(`fleet.node_local_checkout`): `fleet/README.md`. Both off: the fleet works on the shared
filesystem as before. If both are enabled the workset wins and the supervisor logs a WARN.

**Deploy target.** `bin/hpc_run.sh` rsyncs the calling workstation's slurmhelps checkout to
`cluster.slurmhelps_remote` with `--delete` before every GPU submit. That directory is therefore a
*deploy mirror* of some workstation's copy: never develop in it, and never point a fleet's
`fleet.slurmhelps.path` at it (a submit from another project would replace the fleet's code
mid-run). Keep the fleet on its own git checkout.
<!-- END cluster-side -->

## Tests

```bash
python -m pytest            # workstation side; network-free; needs pytest and click
bash tests/run-cluster-tests.sh   # cluster side: submits ONE short Slurm job running
                                  # image/test-workset.sh, fleet/test-nodelocal.sh,
                                  # fleet/test_index_hygiene.py, tests/shell/test-remote.sh
```
