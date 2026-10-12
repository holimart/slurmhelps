# langywrap fleet in a renewable Slurm allocation — `ly` bridge, supervisor, lanes

**Source of truth for these scripts is slurmhelps `fleet/`.** A project adopts them with its
`slurmhelps.json` (`site` + `fleet` sections, `docs/config.md`), a pin in its lanes.json
(`fleet.slurmhelps`), and thin shims (e.g. `fleet/ly` exec'ing `fleet/ly`). Runtime state
and logs live on the shared filesystem (`site.agents_dir`), never in `$HOME`. The project
checkout is also there, so SQLite/WAL/SHM and supervisor-owned runtime directories stay as real
paths inside `.langywrap`; Langywrap rejects symlinked private runtime directories.
`prepare-project.sh` verifies that placement. Langywrap itself has no Slurm code; this is an outer
execution substrate that moves the supervisor + lane fleet into a renewable Slurm allocation while
the steward stays on the login node.

In the examples below `$P` is the project, `$W` the per-user root (`site.work_root`), and `ly`
is the project's shim (or `$SLURMHELPS_HOME/fleet/ly` with `SLURMHELPS_PROJECT=$P`).

## Why an allocation

Every langypi lane is a `pi --mode rpc` child spawned by the **langywrap
session supervisor daemon** (`lib/langywrap/sessions/subagents.py:599`, via the
detached owner `sessions/attempt_runtime.py`). The daemon owns its lanes through
same-host mechanisms — `os.killpg` process groups, `/proc/<pid>` identity, the
dead-owner reaper — and its control socket is node-local
(`$XDG_RUNTIME_DIR/langywrap-<uid>/sessions-*.sock`). Therefore the daemon and
all lane children must live on **one host**, and that host should be a compute
node, not the login node.

`lanes.json` is unchanged: it is a declarative registry (what the fleet is), not
a placement mechanism. Here it simply runs inside the allocation.

## Architecture

```
LOGIN NODE (persistent)                     SLURM ALLOCATION (renewable)
┌───────────────────────────┐  srun --jobid ┌──────────────────────────────────┐
│ Claude Code STEWARD       │──────────────▶│ langywrap supervisor daemon      │
│ (steward-langypi-loop)    │  --overlap    │  └─ langypi lanes (pi --mode rpc) │
│  allocation lifecycle     │               │     all on this compute node      │
│  renew / requeue          │  shared /work │                                   │
│  dispatch, push           │◀─────────────▶│  state: .langywrap/* -> /work     │
└───────────────────────────┘               └──────────────────────────────────┘
```

## Files

| File | Role |
|---|---|
| `supervisor.sbatch` | the renewable allocation: publishes `state/alloc.env`, starts the daemon, self-requeues on `USR1` (`fleet.renew_signal` before the limit) and exits on `TERM` (preemption → Slurm requeue); node-local or workset mode |
| `ly` | login-side bridge: `ly info / status / ensure / langywrap / lanes / steward / langypi / exec / doctor / up / down / renew / logs / store / spill / workset / hot-reload` |
| `lane-common.sh` | sourced by the launchers: enforces the daemon's environment (`HOME` on `/work`, `XDG_RUNTIME_DIR` and `LANGYWRAP_SESSIONS_DB_DIR` from `state/alloc.env`) and the release/abandon-stale logic |
| `start-lanes.sh` | the one lane launcher: `[--abandon-stale] [lane...]`; default is all worker lanes, `steward` only when named |
| `start-all-lanes.sh`, `start-one-lane.sh`, `label-lanes.sh` | thin entry points over `start-lanes.sh` / `lane-common.sh` |
| `prepare-project.sh` | verifies runtime and git coordination paths are real, project-local paths under `site.work_root` |
| (project) `slurmhelps.json` | `fleet` section: partition/time/size/project/lane prompt; `site` section: paths, MPI env |
| `nodelocal.py` | node-local working set (`fleet.node_local_checkout`): stage-in, periodic/final/manual spill, `MANIFEST.json`; see below |
| `test-nodelocal.sh` | harness for `nodelocal.py` + `supervisor.sbatch` + `ly spill` (stubs, toy repo, fake daemon; run it in a Slurm job) |
| `../image/workset.sh` | working-set images (`fleet.workset`): see `image/README.md` |

`ly langywrap` and `ly langypi` preserve each argument across the remote shell;
pass a session message as one quoted argument. `ly exec` remains an explicit
remote-shell interface for scripts.

Run the tracked steward checklist from the allocation with
`ly exec 'bash "$SLURMHELPS_HOME/fleet/steward-tick.sh"'`. The helper refuses to run
outside Slurm, reports each probe's actual exit status, and returns nonzero if
any probe fails. Do not invoke the checklist directly from the login node.

New allocations and the `ly` CLI use the Langywrap checkout and editable venv at
`site.langywrap_checkout` (`LANGYWRAP_CHECKOUT` can override it). An already
running daemon keeps its original interpreter and import path until it exits;
do not treat a pull in the new checkout as proof that the active process loaded
that code.

## Quick start

```bash
ly info                                               # paths, pinned slurmhelps ref, alloc.env
bash $SLURMHELPS_HOME/fleet/prepare-project.sh $P     # verify runtime placement
ly ensure                                             # reuse a suitable alloc, else submit
ly status                                             # daemon pids + sessions
ly doctor                                             # langypi readiness
ly langywrap lanes report --project $P
ly lanes start <lane> --project $P --prompt "..." --request-key "$(date +%s)"
ly renew                                              # requeue in place (same JobID)
ly down                                               # final spill (node-local/workset), then cancel
```

Start lanes from the login node through the allocation, never directly:

```bash
ly start-lane <lane> [<lane>...]          # through start-lanes.sh inside the allocation
ly start-lane steward                     # the steward (excluded from the default set: fleet.lane_exclude)
ly start-lane --abandon-stale <lane>...   # auto-abandon stale open turns
```

A lane whose detach is refused by an open delivered turn prints the exact
`langywrap lanes abandon-turn …` remedy. `--abandon-stale` runs it automatically,
and only when the lane has no live attempt (abandon-turn refuses otherwise). The
launcher exits 1 and prints a RELEASE/START table if any lane did not start.

`ly ensure [partition=… time=… cpus=… mem=… project=…]` implements the
preferred flow: **reuse a RUNNING allocation if it is on the right partition
with at least the requested CPUs; otherwise submit one.** Because the store lock
allows only one daemon per project, an under-sized running allocation is
*cancelled and replaced* (not run alongside). Defaults come from the config's `fleet` section.

### Temporary Slurm control failures

`ly` retries read-only `squeue`/`scontrol show job` queries indefinitely by
default with bounded exponential backoff (2, 4, 8, 16, then 30-second waits).
`in_alloc` streams
normal task output and retries `srun` only for diagnostics that establish the
step never started, such as a transient socket refusal. A local `Operation not
permitted` denial is reported immediately because waiting cannot change the
execution policy. Other command
failures are returned as-is because the command may already have run. Tune the
query/step retry budget with `LY_SLURM_RETRIES`,
`LY_SLURM_RETRY_BASE_SECONDS`, and `LY_SLURM_RETRY_MAX_SECONDS` when needed.
Set `LY_SLURM_RETRIES` to a positive number to cap retries for an invocation.
The default patiently retries transient Slurm transport failures until the query
succeeds or a proven not-started `srun` step can be launched. This does not retry payload failures or
ambiguous commands that may already have run. Interrupt the waiting command
explicitly if the operator needs to stop waiting.

After retries are exhausted, inspect `ly status` and the command's target state
before repeating a state-changing action. In particular, do not repeat `ly up`,
`ensure`, `renew`, `down`, or session `send`/`steer`/`interrupt` based only on a
lost response. A temporarily empty queue query is not evidence that the
allocation ended; confirm its recorded JobID with `scontrol` before replacing it.

Supervisor CLI calls can also wait while SQLite checkpoints the shared session
store. Keep and poll the same command handle when the runner reports one. A timeout
from the outer bridge leaves a state-changing operation's outcome unknown; inspect
the target session, attempt, and artifacts before retrying. For `session supervise`,
use its operation timeout (for example `--timeout 300`) and inspect the returned
`outcome`/`state_version`; a second call is appropriate only if the recorded session
still needs recovery.

When a command reports `session supervisor readiness timed out` with a spawned PID,
inspect that PID and the recorded daemon/socket before another control call. Wait
for the spawned process to exit, then recheck session and command state. A listening
socket does not prove the API is responsive; do not spawn another supervisor or
repeat the mutation while its outcome remains unknown.

### SQLite runtime placement and I/O waits

`prepare-project.sh <project>` verifies that the project and its runtime
directories, SQLite/WAL/SHM files, Pi transcripts, artifacts, reports and
`.git/langywrap` coordination store are real paths under `/work`. Keep these
paths inside the work-side checkout: symlinking private supervisor directories
causes startup to fail. Preserve any prior runtime separately; never merge
SQLite databases.

If a remote CLI stays live with no output, inspect that process with
`ly exec 'ps -p PID -o pid,etime,state,wchan:32,args'`. `D` means the process is in
uninterruptible kernel I/O. Keep polling the same command handle; do not kill it,
restart the daemon, or launch another query against the same SQLite store. Inspect
the process's `/proc/PID/syscall` and `/proc/PID/fd`, plus the sizes and timestamps
of `sessions.sqlite3` and its WAL/SHM files, and `.git/langywrap/lane-turns.sqlite3`
and its sidecars. A CLI can be blocked on the lane-turns database while the
sessions database is responsive. If descriptors point at either store, treat this
as a SQLite/filesystem wait. Wait until the live readers and writers are settled
before profiling or relocating either database.

`langywrap session db-profile <database> --budget-seconds 30` is read-only; treat
`UNMEASURED` as no result, not as an empty or healthy store. Before replacing a
daemon, require `langywrap session restart-daemon --dry-run --timeout 120` to report
a measured inventory. `database is locked`/`UNMEASURED` means keep the existing
daemon and wait for the store to recover.

## Session store on node-local tmpfs

(langywrap master now also has a built-in `ram-sqlite` store profile with its own node-local
snapshots, `store.node_local`; `store-sync.py` stays until a project switches to it.)

Why: on 2026-10-10 the session DB on Lustre grew a ~1 GB WAL that was never
reset (the daemon and lanes never left SQLite a reader-free moment), and every
`-shm` lookup paid Lustre latency. The fleet stalled for over an hour
(`.langywrap/audits/store-stall-2026-10-10T1004Z.md`).

Where: the daemon and every `ly`-run CLI open the store in `$STORE_LOCAL`,
`<site.node_tmp>/langywrap-store-<uid>/<project>`, through `LANGYWRAP_SESSIONS_DB_DIR`
(recorded in `state/alloc.env`). Before the daemon starts,
`store-sync.py stage-in` migrates a legacy regular-file store from the checkout
(parking it under `$STORE_PERSIST/legacy-<ts>/`) or restores the newest snapshot.
`$STORE_PERSIST` is `<site.agents_dir>/state/runtime/<project>/store`. It
is refreshed every `AGENTS_STORE_SNAPSHOT_SECS` (default 600) and once more on
renew, TERM and normal exit. No symlinks are created; langywrap refuses them, so
`stage-in` fails on a symlinked store in the checkout.

Inspect: `ly store` prints paths, sizes and mtimes; `ly store snapshot` copies
the local store to `/work` now.

Risk: the live copy is in node tmpfs. If the node dies hard, session state written
since the last snapshot (up to 10 minutes) is lost. A restart on another node
restores the newest snapshot.

## Node-local working set (`fleet.node_local_checkout`)

Why: lane tool calls on the Lustre checkout have a p95 of 30-43 s; a synced
4 KB write on `/work` takes 0.2-3.8 s (median 0.9 s) versus 0 s on `/tmp`.

Switch: `.langywrap/lanes.json` -> `fleet.node_local_checkout`
`{enabled, root, spill_interval_s, spill_root, ...}` (other keys and their
defaults: `DEFAULTS` in `nodelocal.py`). `supervisor.sbatch` reads it from the
`/work` checkout at start. With `enabled: false` (or the block absent) the batch
script runs exactly the commands it ran before; the only addition is the one
`python3 nodelocal.py config` read.

When enabled:

* **Stage-in** (before the store stage-in and the daemon): `nodelocal.py stage-in`
  builds `/tmp/<project>-<uid>/<project>`, a `git clone --shared` of the `/work` repo
  (objects borrowed through `alternates`, so `/work` must keep `gc.auto=0`; the
  remote `work` is `/work`, `origin` is the GitHub URL). Source:
  the newest spill under `spill_root` if one exists, else the `/work`
  checkout (first cut-over: an exact copy of the tracked, untracked and ignored
  files and the git index of the main checkout and every `.lane-worktrees/*`).
  `.git/langywrap` (lane-turns.sqlite3) is copied with the SQLite backup API.
  `runs/` of every tree is a symlink to the same path on `/work` (the tracked
  `runs/**` fixtures get `skip-worktree`). Then `PROJECT` (daemon `--serve`,
  `cd`, `alloc.env PROJECT`, hence `ly --chdir` and `start-lanes.sh`) is the
  node-local root; `WORK_PROJECT` stays `$P`.
  `LANGYWRAP_PI_TELEMETRY_FILE` moves to `/tmp/langywrap-telemetry-<uid>/`.
  Exit 3 (refused, no spill exists) falls back to `/work`; any other failure is
  FATAL so lanes never run on a tree older than the spill.
* **Periodic spill** every `spill_interval_s` (default 300) to
  `<site.agents_dir>/state/runtime/<project>/spill/<jobid>/`:
  every branch and worktree HEAD is pushed into the `/work` repo under
  `refs/nodelocal/<jobid>/` (a `commits.bundle` if that push fails); each tree's
  non-clean files (modified, untracked, ignored) are copied incrementally with
  the path lists and git index that make the restore exact; the fleet branch is
  fast-forwarded in whichever direction is possible (`/work` checkout <->
  node-local), other branches are fast-forwarded into `/work` and new `/work`
  branches (the steward's `prep/*`) appear in the clone; the runtime paths in
  `mirror_to_work` (`lane-reports`, `turn-gate-reports`, `turn-gates`,
  `gpu-jobs.ndjson`) are copied into the `/work` checkout. `MANIFEST.json`
  (status, git HEAD, per-tree dirty paths, sizes, steps, reasons) is written
  last, atomically.
* **FAILSAFE final spill**: USR1 (renewal), TERM/INT (time limit, `scancel`,
  preemption), the daemon ending (clean exit or restart budget exhausted) and any
  other `exit` after stage-in all run `nl_finalize` exactly once BEFORE the
  script returns: stop the spill and snapshot loops, quiesce (TERM the daemon's
  process group, then TERM/KILL lane processes left in this job's `step_batch`
  cgroup or with a cwd inside the node-local root), then one spill with a
  deadline (TERM: 25 s of KillWait=30 s; USR1: 3000 s of the 3600 s warning;
  INT, end: 1800 s). Steps run most valuable first: commits, dirty trees
  (main, then the most recently active worktrees), `.git/langywrap`,
  telemetry, runs, mirror, branch sync, fsync, and LAST the store snapshot
  (48-58 s for the 7.5 GB store on 2026-10-11, so it is skipped when less than
  `store_min_s`=90 s remain; the periodic snapshot stands). A step that misses
  the deadline is skipped and the manifest says `partial` with the reason. The
  final spill ignores a second TERM. Measured on Lustre (login node, 21 trees
  of real size): a final spill after a periodic one took 10.4 s, the first
  full spill 39.5 s.
* **`ly down`** (node-local only) first sends INT to the batch shell
  (`scancel --batch --signal=INT`, no KILL follows), waits up to
  `LY_DOWN_GRACE` (1800 s) for the final spill and exit, then cancels as
  before. A bare `scancel` is TERM: 25 s, no store snapshot.
* **Base layer**: a tree imported from `/work` at the first cut-over keeps its
  `/work` copy as the layer under the spill. A file unchanged since the import
  is listed in the spill but read from `/work` on restore, so the ~330k runtime
  files under `.langywrap` (activity-spool, artifacts, pi-sessions) are never
  copied twice. Do not delete or clean untracked/ignored files of the `/work`
  checkout or of `/work/.../.lane-worktrees/*` while the node-local mode is on.
* **Restore**: the next stage-in (a requeue of the same JobID or a new job; /tmp
  is `job_container/tmpfs`, wiped at job end) clones `/work`, takes the branch
  tips from `refs/nodelocal/<jobid>/` (and the bundle), checks out each spilled
  worktree at its spilled HEAD, overlays its spilled files (spill copy, else
  base layer), removes the files it had deleted and installs its index. A
  `/work` main-checkout file modified after the spill (the steward's edit) wins
  over the spilled copy. A new job renames the spill directory it restored from
  to `spill/<its jobid>/` (one lineage, so its first spill stays incremental);
  its manifest keeps the old `jobid` until the new job writes one.

Steward:

* `ly spill` spills now (status `manual`, idempotent, waits for a running
  periodic spill); `ly spill status` prints the last manifest.
* Lane commits reach the `/work` checkout's fleet branch (`fleet.branch`) by fast-forward at each
  spill. A commit made directly in the `/work` checkout is fast-forwarded into
  the clone only while the clone has not committed since; otherwise the manifest
  says `diverged` and the steward merges `refs/nodelocal/<jobid>/heads/<fleet.branch>`
  on `/work` (the next spill then fast-forwards the clone).
* Uncommitted lane state is readable on the login node under
  `spill/<jobid>/trees/<tree>/` (at most `spill_interval_s` old).
* Anything submitted to OTHER nodes (sbatch grid/GPU jobs) must reference `/work`
  paths: `runs/` already resolves there; a solver binary built in a node-local
  worktree must be copied under `runs/` (or built from the `/work` checkout)
  before such a job uses it. `SH_WORK_CHECKOUT` (and the alias named by `fleet.work_checkout_var`) holds the `/work` path.

Harness: `fleet/test-nodelocal.sh <scratch-dir>` (in a short Slurm job; `tests/run-cluster-tests.sh`).

## Renewal (verified on the cluster)

- `--signal=B:USR1@300` → the trap calls `scontrol requeue $SLURM_JOB_ID`. The
  **same JobID** returns (`Restarts` increments, ~2 min `cred_expire` backoff)
  and re-executes the batch script, which restarts the daemon from shared
  `/work` state. Proven: gen 0 → USR1 → `Restarts=1` → gen 1.
  **Critical:** `--requeue` alone does *not* renew — a normal `exit 0` is
  `COMPLETED` (Restarts=0). The USR1 handler must call `scontrol requeue`
  explicitly.
- `ffa-preempt`/`ffa-checkpoint` are `PreemptMode=REQUEUE`: a higher-tier job
  sends SIGTERM first; the TERM trap exits and `--requeue` brings the same
  JobID back.
- **Nothing saves a job that reaches its time limit** — renew before it. Only an
  admin can extend a running job's `TimeLimit`.
- Successor-chain alternative (new JobID each generation, robust to SIGKILL):
  `sbatch --dependency=afterany:$SLURM_JOB_ID supervisor.sbatch` from a
  `USR2`/`TERM` trap. Not enabled by default.

## Lane read/write access

A lane's sandbox writable roots come from `dispatch.extra_writable_roots` in the
project's `lanes.json` (the supervisor strips `EXECWRAP_EXTRA_WRITABLE_PATHS`
before overlaying them). a project's fleet defaults typically grant the
langywrap and project checkouts. `supervisor.sbatch` additionally exports
`EXECWRAP_EXTRA_WRITABLE_PATHS` (colon-separated) as a fallback.


## Constraints discovered

- `srun --jobid=<running job> --overlap` works from the login node and needs no
  ssh (direct `ssh <compute>` is denied by `pam_slurm_adopt` unless you hold a
  job there).
- `$HOME` and `/work` are shared Lustre and visible on compute; `/tmp` and
  `$XDG_RUNTIME_DIR` are node-local.
- Compute nodes have full network (pypi/github OK) and the toolchain
  (`node`, `langypi`, `bwrap`).
- `sbatch` does not expand variables in `#SBATCH` lines — the `/work` output
  paths in `supervisor.sbatch` are literal.
- Editing langywrap source while a fleet runs does **not** force a drain: the
  daemon hot-reloads via the no-drain control handoff (see "Hot reload" below).
  It only falls back to a drain when the handoff preflight is not clean, so gate
  the edit on `FREE-PATH: YES`.

## Hot reload (no drain)

langywrap's supervisor watches the loaded package tree (5s scan, 10s quiet) and,
when it changes, **re-execs itself into the new code in place** — same pid,
attempts handed to detached owners, **no drain** — provided two things hold:

- `persistent_attempts` is on. It is compiled into
  `langywrap.sessions.supervisor._main` (`persistent_attempts=True`), so any
  daemon started as `-m langywrap.sessions.supervisor --serve` has it. A daemon
  whose ping omits it, or reports it `false`, drains by construction and must be
  restarted.
- `_handoff_preflight` is clean: no unowned live attempt and no process-local
  subagent / workflow / MCP / tool-channel grant. A plain pi lane between turns
  holds none of these, so the free path holds.

The steward's `drain-status` reports the verdict. Drive the whole thing with:

    ly hot-reload              # guard -> git pull -> prove the in-place re-exec
    ly hot-reload --no-pull    # only wait for an already-pending reload

`ly hot-reload` refuses unless the daemon reads `FREE-PATH: YES`, pulls the
langywrap checkout, then proves the re-exec by watching the reload budget
(`reload_budget.left`, `20 -> 19`) — the successor re-execs with
`LANGYWRAP_SUPERVISOR_SOURCE_RELOADS` decremented, which is the signature of an
in-place `execv`, not a restart. It finishes by listing the sessions so you can
confirm every lane survived. `ly hot-reload --no-pull` waits up to 120 seconds
for an already-pending reload and exits nonzero if it cannot observe the budget
change; inspect `session ping` before retrying an unconfirmed handoff.
The bridge checks `git pull`'s own exit code and stops on failure. If Git reports
an existing `index.lock`, first verify that no Git process is active; remove
only a confirmed stale lock, then rerun the guarded hot-reload. If the checkout
advances but the reload budget does not, the bridge returns failure after its
proof window; leave the allocation running and retry `ly hot-reload --no-pull`
after checking `session ping`.

Verified 2026-10-09: pulling 33 pi-harden commits, the daemon re-exec'd at
18:02:51 (same pid, budget 20 -> 19, `drain: None`) and all 12 lanes stayed
`running`; `source_reload` then read `current`.

Fallback: if the preflight is not clean the daemon does **not** force it.
`_reload_drain_decision` either stops claiming and waits for the drainable
attempts to settle (the drain), or, for a structural blocker, keeps claiming and
stays on the loaded sources ("blocked"). The current pi-harden line reconciles
ghost/dead owners before counting blockers, so a stale `starting` row no longer
wedges a reload. The reload budget (`LANGYWRAP_SUPERVISOR_SOURCE_RELOADS`,
default 20) is a safety valve against an infinite reload loop; when it hits 0 the
daemon stops self-reloading and needs a fresh start — the budget resets then.

## Working-set images (`fleet.workset`)

The alternative to the node-local clone: the whole working set as one squashfs image, unpacked to
node tmp and bound back at its absolute paths in a private mount namespace. See `image/README.md`.
If both are enabled the workset wins (WARN in alloc.log).

## Known gaps / next steps

- The `ly` bridge is a manual step. langywrap's own doctrine says a manual step
  is a missing feature — the natural follow-up is a `fleet.slurm` placement
  backend inside langywrap so lanes spawn via the allocation without the bridge.
- Multi-node fleets: split by a `fleet.slurm.nodes` key; not implemented.
