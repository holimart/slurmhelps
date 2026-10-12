# Working-set images (`fleet.workset`)

`workset.sh` packs a fleet's working set into ONE squashfs file on the shared filesystem and runs
the fleet on an unpacked copy in node-local tmp, bound back over the ORIGINAL absolute paths in a
private mount namespace. `toolchain.sh` does the same for the user toolchain.

## Why

Lane tool calls on the Lustre checkout had a p95 of 30-43 s; a synced 4 KB write on Lustre takes
0.2-3.8 s, on node tmp ~0 s. The working set is ~300k small files (project checkout incl. `.git`,
lane worktrees, `.langywrap` runtime, agent homes). Moving it as one file is one sequential read
(~2.3 GB in 4.7 s on Troja), and binding it back at the same absolute paths keeps every absolute
path valid: venv shebangs, `HOME`, `PI_CODING_AGENT_DIR`, git worktree `gitdir:` links.

## Layout at run time

```
<images>/<project>.sqfs (+ .manifest.json, .prev)      shared filesystem
<images>/toolchain.sqfs
<node_tmp>/workset-<uid>/root/<abs path of each tree>   unpacked data image (tmpfs = RAM)
<node_tmp>/workset-<uid>/toolchain-root/<abs path>      unpacked toolchain image
<node_tmp>/workset-<uid>/ns.env                         NS_PID of the namespace holder
<node_tmp>/workset-cache-<uid>/{uv,npm,pip,pycache,xdg} package caches inside a workset
```

Inside the namespace (`workset.sh hold` creates it, `exec` joins it):

1. every `live_binds` dir (default `{project}/runs`, results) is first held by a bind mount while
   it is still the shared-filesystem dir;
2. every tree (`trees`, then `toolchain_dirs`) is bound from node tmp over its absolute path;
3. the held live dirs are bound back on top, so results are written straight to the shared
   filesystem and stay visible to jobs on other nodes.

The namespace is created by `workset.py` itself (ctypes `unshare(CLONE_NEWUSER|CLONE_NEWNS)`, uid
mapped to itself, binds while the namespace still grants the capability, then `exec`): processes
run as the user's own uid without capabilities. `unshare -c` + `mount(8)` does not work for this
(`mount` refuses a non-root uid). `exec` joins with `nsenter -t NS_PID -U -m
--preserve-credentials`, also from another `srun --overlap` step (verified on Troja); if the holder
is gone it falls back to a fresh, equivalent namespace (same binds of the same node-tmp tree).

Secrets (`secrets`, e.g. `auth.json`) are never in an image: copied into node tmp at stage-in,
written back (atomically, only when changed) at every spill.

## Commands

```bash
W=$SLURMHELPS_HOME/image/workset.sh
$W --project P status                    # images, stage, holder, last spill (login node OK)
$W --project P build --from-lustre       # image from the Lustre originals (Slurm job, fleet DOWN)
$SLURMHELPS_HOME/image/toolchain.sh --project P build
$W --project P stage-in                  # copy + unsquashfs + verify (supervisor does this)
$W --project P hold &                    # namespace holder (supervisor does this)
$W --project P exec -- CMD               # run CMD in the holder's namespace (ly exec does this)
$W --project P spill [--mode manual]     # new image from node tmp (.new + fsync + rename, .prev kept)
$W --project P view [--umount]           # squashfuse the newest image at <images>/view (login node)
$W --project P writeback [--apply]       # cut-back: rsync the image's trees onto the Lustre originals
$W --project P unpack DEST               # inspect an image
```

Exit codes: 0 ok, 1 error, 2 build/spill failed with the published image untouched, 3 refused /
busy / no image yet.

## Spill

1. secrets written back; 2. `--pre` hook (the supervisor passes the session-store snapshot);
3. scan of the node-local tree (excludes below); 4. every `*.sqlite3` in the tree is copied with
the SQLite backup API and bound over the live file for mksquashfs only (consistent DBs, WAL
content included, `-wal`/`-shm` excluded); 5. `mksquashfs` (periodic: zstd level 3; final: lz4)
into node tmp; 6. verify counts (`unsquashfs -lls` vs the scan, small drift tolerated while lanes
run); 7. publish: copy as `<image>.new`, fsync, `unsquashfs -s`, hard-link the old image to
`.prev`, rename. A failing, timed-out or deadline-missed spill never replaces the published image.

Default excludes (`workset.excludes`, plus `extra_excludes`): `.log/`,
`.langywrap/{activity-spool,artifacts,runtime-trash}/`, `.langywrap/.langypi-session.*`,
`.langywrap/*.lock`, `runtime.sqlite3*`, `sessions.sqlite3*`, `*.sqlite3-{wal,shm,journal}`,
`*.sock` (and every socket/device by type), `index.lock`, `.git/*.lock`, `*.telemetry.jsonl`,
`.cache/`, `.trash/`, `uv/cache/`. Pattern rules: trailing `/` = directories only; no `/` = basename
at any depth; with `/` = any component-aligned suffix (`*` crosses `/`); leading `/` = absolute.

## Supervisor integration

`fleet/supervisor.sbatch` reads `fleet.workset` (`workset.sh config`). When enabled:
stage-in (before the store stage-in) -> namespace holder -> `alloc.env` gets `WORKSET=1`,
`WORKSET_SH`, `WORKSET_PROJECT`, `WORKSET_STAGE`, `WORKSET_STATE`, `WORKSET_NS_PID`,
`WORKSET_IMAGE` -> store stage-in, snapshot loop and the daemon run through `workset.sh exec` ->
periodic spill every `spill_interval_s` -> ONE final spill on USR1 (renewal, budget
`final_budget_usr1_s`), TERM (`final_budget_term_s`, KillWait), INT (`ly down`), normal end and any
other exit (`final_budget_end_s`), after quiescing the daemon and every process in the namespace.
`ly exec`, `ly lanes`, `ly steward`, `ly start-lane` join the namespace; `ly spill [status]` and
`ly workset ...` run the engine outside it. No image yet (stage-in exit 3): runs on the shared
filesystem as before. Any other stage-in failure is FATAL (no lanes on a stale tree).

## Enabling (operator)

1. Fleet down (`ly down`). Build the images in a short job:
   `srun -p <short> -c 16 --mem=48G --time=110 bash $W --project P build --from-lustre`, then
   `toolchain.sh --project P build`. Check `$W --project P status`.
2. Set `fleet.workset.enabled: true` in the project's lanes.json (and `node_local_checkout.enabled:
   false`; if both are on, the workset wins and the supervisor logs a WARN).
3. `ly up`; `ly status`; `ly exec 'readlink /proc/self/ns/mnt; ls <project>'`; `ly spill`.
4. From then on the Lustre copy of the trees is STALE: edit the project only through `ly exec`
   (or git push/pull), read the newest state with `workset.sh view`.

To go back: `ly down` (final spill), `workset.sh writeback` (dry run) then `--apply` in a short
job, set `enabled: false`.

## Memory

Node tmp is RAM on Troja (job-private tmpfs counted against `--mem`): unpacked data image +
toolchain + one compressed spill copy must fit next to the fleet's own memory.

## Tests

`image/test-workset.sh /tmp/<dir>` (in a Slurm job): build, toolchain, stage-in, hold, exec from
another `srun --overlap` step, live binds, spill content and excludes, SQLite WAL capture, secret
write-back, failed/timed-out spills never replacing the image, supervisor TERM/USR1/end final
spills, requeue restore, workset-over-node-local precedence, view, writeback.
