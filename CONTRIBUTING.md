# Contributing to slurmhelps

slurmhelps is shared by several projects (whiteboxchallenge, langywrap, ADGFEM, ...) and by
several agent sessions. Keep it generic and keep everyone's entry points working.

## Layout rules

| Area | Runs on | Rule |
|------|---------|------|
| `bin/`, `lib/named_mounts.sh` | the operator's **workstation** (ssh, sshfs, rsync to the cluster) | must refuse on a cluster node where it makes no sense (`SLURM_JOB_ID`, `sbatch` on PATH, `site.cluster_host_glob`) |
| `fleet/`, `image/`, `maint/`, `env/` | the **cluster**: login node for light commands, Slurm jobs for anything heavy | never heavy work on a login node: builds, tests, image builds, tree walks go through `srun`/`sbatch` |
| `llm/` | a GPU Slurm job (install, serve, benchmark) | model-serving helpers; no site paths in code; downloads/builds/serving run in an allocation |
| `bootstrap/` | a Slurm job on the cluster after a total loss | idempotent; writes only below `--prefix`; never copies secrets |
| `lib/slurmhelps/` | both | stdlib only, Python 3.9+ (the cluster's system python) |
| `docs/`, `examples/` | — | every config key is documented in `docs/config.md` |

Entry points that other projects call (`bin/*`, `fleet/ly`, `fleet/supervisor.sbatch`,
`image/workset.sh`, `image/toolchain.sh`) are API: do not move or rename them; add options instead.

## Configuration, not code

- No project names, user names, hosts, partitions or absolute paths in code. They live in the
  calling project's config: `<project>/slurmhelps.json` or the `"slurmhelps"` block of its
  `.langywrap/lanes.json` (`docs/config.md`). Cluster-side shell code reads it through
  `lib/common.sh` (`sh_load`); workstation tools through `lib/slurmhelps/config.py`.
- Fleet behaviour switches (`fleet.workset`, `fleet.node_local_checkout`, `fleet.session_store`)
  stay in the project's `lanes.json` `fleet` block.
- Site landmines (e.g. Troja's `OMPI_MCA_pml=ob1`) go in `site.mpi_env` / the site toolchain
  profile under `env/`, with a comment saying why.
- The fleet tests grep the cluster-side code for site paths; keep them passing.

## A project pins the slurmhelps it uses

In the project's `.langywrap/lanes.json`:

```json
"fleet": {"slurmhelps": {"path": "/work/ffa/<user>/<slurmhelps checkout>", "ref": "<commit>"}}
```

`ly info` warns when the checkout's HEAD is not the pinned ref. Bump `ref` in the project in the
same change that adopts a new slurmhelps behaviour, after the tests below pass.

Do not point `path` at a `cluster.slurmhelps_remote` directory: `bin/hpc_run.sh` rsyncs a
workstation copy over it with `--delete` before every GPU submit.

## Tests

- Workstation side: `python -m pytest` (network-free).
- Cluster side: `bash tests/run-cluster-tests.sh` (one short Slurm job; runs
  `image/test-workset.sh`, `fleet/test-nodelocal.sh`, `fleet/test_index_hygiene.py`,
  `tests/shell/test-remote.sh`). Tests use fake trees under node-local tmp only: never `/home`,
  never the live `/work` trees.
- Report exact pass/fail counts in the commit message or PR.

## Git

- Small, focused commits; say which project needed the change.
- **Always `git fetch origin` and rebase onto (or merge) `origin/master` before pushing**, then
  re-run the tests. Other sessions push here too; never force-push `master`.
- On shared filesystems keep `gc.auto=0` in the checkout and do maintenance deliberately.
