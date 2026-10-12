#!/usr/bin/env bash
#SBATCH --job-name=agent-supervisor
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --open-mode=append
# =============================================================================
#  agent-supervisor — the renewable Slurm allocation that hosts the langywrap
#  session supervisor daemon and therefore every langypi lane process.
# =============================================================================
#  Source of truth: slurmhelps fleet/supervisor.sbatch. Submit it ONLY through
#  `ly up` / `ly ensure`: they pass account/partition/time/ntasks/mem/signal/
#  output/error from the project's slurmhelps config ("fleet" and "site"
#  sections, docs/config.md) and export SLURMHELPS_HOME + SLURMHELPS_PROJECT,
#  which this script needs because sbatch runs a spooled copy of it.
#  Runtime state + logs: $AGENTS_DIR (config site.agents_dir; never in $HOME).
#
#  The login node runs NO lane. It runs the steward (Claude Code) and the `ly`
#  bridge (fleet/ly), which reach the daemon inside this allocation
#  with `srun --jobid=$SLURM_JOB_ID --overlap`.
#
#  Renewal: `--signal=B:USR1@$AGENTS_RENEW_SIGNAL` (3600 s) fires before the limit; the trap exits,
#  which makes the batch script requeue the SAME JobID (verified: Restarts
#  increments and the generation restarts from shared /work state). Preemption
#  on ffa-preempt/ffa-checkpoint (PreemptMode=REQUEUE) sends SIGTERM -> exit ->
#  Slurm requeues.
#
#  AGENT_PROJECT (a langywrap project dir; config fleet.project, default the
#  project root) selects which .langywrap store this allocation serves.
#  AGENT_ROOTS (colon-separated) lists extra trees the daemon/lanes may write;
#  exported as EXECWRAP_EXTRA_WRITABLE_PATHS for the lane sandbox.
#
#  Workset image (fleet.workset in .langywrap/lanes.json, engine
#  image/workset.py, image/README.md). When enabled, the project's working set
#  is unpacked from a squashfs image into node-local tmp and bound back over
#  its absolute /work paths inside a private mount namespace (unshare -c -m);
#  the daemon and everything that touches project paths run in it
#  (`workset.sh exec`). Live Lustre dirs (fleet.workset.live_binds, e.g. runs/)
#  are bound back on top. A spill every spill_interval_s and one FINAL spill on
#  every exit path write a new image (.new + fsync + rename, .prev kept).
#  Mutually exclusive with the node-local checkout: if both are enabled the
#  workset wins and a WARN is logged.
#
#  Node-local working set (fleet.node_local_checkout in .langywrap/lanes.json,
#  engine fleet/nodelocal.py, README "Node-local working set"). When
#  enabled, the daemon and lanes work in a clone on node-local /tmp
#  (job_container/tmpfs: wiped when the job ends) instead of the Lustre
#  checkout. Stage-in builds it before the daemon starts (from the newest spill,
#  else from /work); a spill every spill_interval_s copies commits, uncommitted
#  and runtime state to /work; every exit path (USR1 renewal, TERM/INT, normal
#  end, crash budget exhausted, any other exit) runs ONE final spill before the
#  script returns. When disabled, every command below runs exactly as before.
# =============================================================================
set -u

if [ -z "${SLURMHELPS_HOME:-}" ] || [ ! -f "$SLURMHELPS_HOME/lib/common.sh" ]; then
  echo "FATAL SLURMHELPS_HOME unset or wrong (${SLURMHELPS_HOME:-}); submit through ly up / ly ensure" >&2
  exit 1
fi
# shellcheck source=../lib/common.sh
. "$SLURMHELPS_HOME/lib/common.sh"
sh_load || { echo "FATAL slurmhelps configuration did not load" >&2; exit 1; }

AGENTS="$AGENTS_DIR"
STATE="$AGENTS/state"
LOG="$AGENTS/logs"
PROJECT="${AGENT_PROJECT:?AGENT_PROJECT unset (config fleet.project)}"
LW_CHECKOUT="$LANGYWRAP_CHECKOUT"
LW_PY="$LW_CHECKOUT/.venv/bin/python"
mkdir -p "$STATE" "$LOG"

log() { echo "$(date -u +%FT%TZ) [gen=${SLURM_RESTART_COUNT:-0}] $*" >> "$LOG/alloc.log"; }

# Node-local working set switch, read from the /work checkout's lanes.json.
# `config` never fails: an unreadable registry prints NL_ENABLED=0.
WORK_PROJECT="$PROJECT"
NL_SRC="$SH_NODELOCAL_PY"
NL_ENABLED=0
NL_READY=0
if [ -f "$NL_SRC" ]; then
  eval "$(python3 "$NL_SRC" config --lanes "$WORK_PROJECT/.langywrap/lanes.json" 2>>"$LOG/alloc.log")" || NL_ENABLED=0
fi

# Workset image switch (fleet.workset). `config` never fails either: an
# unreadable block prints WS_ENABLED=0.
WS_SH="$SH_WORKSET"
WS_ENABLED=0
WS_READY=0
WS_RUN=()
if [ -f "$WS_SH" ]; then
  eval "$(bash "$WS_SH" --project "$WORK_PROJECT" config 2>>"$LOG/alloc.log")" || WS_ENABLED=0
fi
if [ "$WS_ENABLED" = 1 ] && [ "$NL_ENABLED" = 1 ]; then
  echo "$(date -u +%FT%TZ) [gen=${SLURM_RESTART_COUNT:-0}] WARN fleet.workset and fleet.node_local_checkout are both enabled; the workset wins, node-local is disabled for this run" >> "$LOG/alloc.log"
  NL_ENABLED=0
fi

# end_time_of JOBID — Slurm's EndTime for JOBID. scontrol -o prints one line with
# space-separated Key=Value pairs; match the exact key (an awk -F= split on the
# whole line returned "<StartTime> EndTime", the bug this replaces).
end_time_of() {
  local info e=""
  info="$(scontrol show job -o "$1" 2>/dev/null)" || info=""
  e="$(printf '%s\n' "$info" | tr ' ' '\n' | awk -F= '$1=="EndTime"{print $2; exit}')"
  printf '%s' "${e:-UNKNOWN}"
}

# Written on every start, so a requeue/renewal (same or new run) refreshes END.
publish_alloc() {
  local tmp
  tmp="$(mktemp "$STATE/alloc.env.XXXXXX")"
  {
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "HOST=$(hostname -f)"
    echo "NODELIST=$SLURM_JOB_NODELIST"
    echo "PARTITION=$SLURM_JOB_PARTITION"
    echo "GEN=${SLURM_RESTART_COUNT:-0}"
    echo "END=$(end_time_of "$SLURM_JOB_ID")"
    echo "PROJECT=$PROJECT"
    echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
    echo "LANGYWRAP_SESSIONS_DB_DIR=$STORE_LOCAL"
    echo "STORE_PERSIST=$STORE_PERSIST"
    echo "STORE_SYNC=$STORE_SYNC"
    echo "START=$(date -u +%FT%TZ)"
    if [ "$WS_READY" = 1 ]; then
      echo "WORKSET=1"
      echo "WORKSET_SH=$WS_SH"
      echo "WORKSET_PROJECT=$WORK_PROJECT"
      echo "WORKSET_STAGE=$WS_STAGE"
      echo "WORKSET_STATE=$WS_STATE"
      echo "WORKSET_NS_PID=$WS_NS_PID"
      echo "WORKSET_IMAGE=$WS_IMAGE"
    fi
    if [ "$NL_READY" = 1 ]; then
      echo "NODE_LOCAL=1"
      echo "WORK_PROJECT=$WORK_PROJECT"
      echo "NODELOCAL_PY=$NL_PY"
      echo "NODELOCAL=$NL_FROZEN"
      echo "NODELOCAL_STATE=$NL_STATE"
      echo "SPILL_DIR=$NL_SPILL_ROOT/$SLURM_JOB_ID"
    fi
  } > "$tmp"
  mv -f "$tmp" "$STATE/alloc.env"
}

# On renewal (USR1, fired before the time limit): requeue the SAME JobID
# explicitly. `--requeue` alone only covers preemption/node failure -- a normal
# exit is COMPLETED, not requeued (verified on this cluster). On preemption
# (TERM) we just exit and let --requeue bring us back.
renew() {
  log "USR1 -> requeue same JobID"
  if [ "$WS_READY" = 1 ]; then
    [ "${WS_FINALIZING:-0}" = 1 ] && { log "USR1 during the final workset spill; ignored"; return 0; }
    ws_finalize USR1 "$WS_BUDGET_USR1"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if [ "$NL_READY" = 1 ]; then
    [ "${NL_FINALIZING:-0}" = 1 ] && { log "USR1 during the final spill; ignored"; return 0; }
    nl_finalize USR1 "$NL_BUDGET_USR1"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if [ -n "${DAEMON_PID:-}" ]; then
    kill -TERM "-$DAEMON_PID" 2>/dev/null || kill -TERM "$DAEMON_PID" 2>/dev/null || true
  fi
  store_final
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
shutdown() {
  log "$1 -> graceful exit (Slurm --requeue restarts this JobID on preemption)"
  if [ "$WS_READY" = 1 ]; then
    [ "${WS_FINALIZING:-0}" = 1 ] && { log "$1 during the final workset spill; ignored"; return 0; }
    if [ "$1" = INT ]; then ws_finalize "$1" "$WS_BUDGET_END"; else ws_finalize "$1" "$WS_BUDGET_TERM"; fi
    exit 0
  fi
  if [ "$NL_READY" = 1 ]; then
    [ "${NL_FINALIZING:-0}" = 1 ] && { log "$1 during the final spill; ignored"; return 0; }
    # TERM comes from Slurm (time limit, scancel, preemption): KILL follows
    # after KillWait. INT comes from `ly down` (scancel --batch --signal=INT):
    # nothing follows, so the final spill gets the long budget.
    if [ "$1" = INT ]; then nl_finalize "$1" "$NL_BUDGET_END"; else nl_finalize "$1" "$NL_BUDGET_TERM"; fi
    exit 0
  fi
  if [ -n "${DAEMON_PID:-}" ]; then
    kill -TERM "-$DAEMON_PID" 2>/dev/null || kill -TERM "$DAEMON_PID" 2>/dev/null || true
  fi
  store_final
  exit 0
}

# Periodic consistent copy of the node-local store to /work (backup API, so it
# is safe while the daemon writes). Failures are logged and retried next tick.
store_snapshot_loop() {
  while sleep "$STORE_SNAPSHOT_SECS"; do
    ${WS_RUN[@]+"${WS_RUN[@]}"} "$LW_PY" "$STORE_SYNC" snapshot "$PROJECT" "$STORE_LOCAL" "$STORE_PERSIST" >> "$LOG/alloc.log" 2>&1 \
      || log "WARN store snapshot failed; retrying in ${STORE_SNAPSHOT_SECS}s"
  done
}

# Stop the periodic snapshot, then take one last copy. Called after the daemon
# is signalled (renew/shutdown) and after it has exited (normal end).
store_final() {
  if [ -n "${SNAP_PID:-}" ]; then
    kill "$SNAP_PID" 2>/dev/null || true
    wait "$SNAP_PID" 2>/dev/null || true
    SNAP_PID=""
  fi
  sleep 5  # let the daemon flush after SIGTERM; the snapshot is consistent either way
  if ${WS_RUN[@]+"${WS_RUN[@]}"} "$LW_PY" "$STORE_SYNC" snapshot "$PROJECT" "$STORE_LOCAL" "$STORE_PERSIST" >> "$LOG/alloc.log" 2>&1; then
    log "final store snapshot done"
  else
    log "WARN final store snapshot FAILED; the /work copy may be up to ${STORE_SNAPSHOT_SECS}s old"
  fi
}

# ---- node-local working set: periodic spill, quiesce, FINAL spill ----------
# Defined unconditionally, used only when NL_READY=1 (stage-in succeeded).
nl_spill_loop() {
  while sleep "$NL_SPILL_INTERVAL"; do
    timeout -k 10 "$((NL_SPILL_INTERVAL * 2))" "$NL_PY" "$NL_FROZEN" spill --state "$NL_STATE" --mode periodic >> "$LOG/alloc.log" 2>&1 \
      || log "WARN node-local periodic spill rc=$? (2=partial, 3=busy); next try in ${NL_SPILL_INTERVAL}s"
  done
}

# nl_alive PID — true while PID runs (a zombie child counts as gone).
nl_alive() {
  local st
  st="$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -c1)"
  [ -n "$st" ] && [ "$st" != Z ]
}

# nl_quiesce DEADLINE — stop writers before the final spill: the daemon through
# the existing path (TERM to its process group), then any lane process that
# outlived it (detached attempt owners and pi children survive a daemon exit by
# design). Lane processes = this batch step's cgroup (only when it is really
# this job's step_batch cgroup) plus anything whose cwd is inside the
# node-local root. Never this shell. Bounded by DEADLINE.
nl_quiesce() {
  local deadline="$1" q i pids=() p cg cwd
  if [ -n "${DAEMON_PID:-}" ]; then
    kill -TERM "-$DAEMON_PID" 2>/dev/null || kill -TERM "$DAEMON_PID" 2>/dev/null || true
    q=$(( (deadline - $(date +%s)) / 4 )); [ "$q" -le 30 ] || q=30
    i=0
    while [ "$i" -lt "$q" ] && nl_alive "$DAEMON_PID"; do sleep 1; i=$((i + 1)); done
    if nl_alive "$DAEMON_PID"; then
      log "node-local: daemon pid=$DAEMON_PID still up after ${q}s; KILL"
      kill -KILL "-$DAEMON_PID" 2>/dev/null || kill -KILL "$DAEMON_PID" 2>/dev/null || true
    fi
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  [ "$NL_QUIESCE_KILL" = 1 ] || return 0
  cg="$(awk -F: '$1=="0"{print $3; exit}' /proc/self/cgroup 2>/dev/null)"
  if [ -n "$cg" ] && [[ "$cg" == *"job_${SLURM_JOB_ID}/step_batch"* ]] && [ -r "/sys/fs/cgroup$cg/cgroup.procs" ]; then
    mapfile -t -O "${#pids[@]}" pids < "/sys/fs/cgroup$cg/cgroup.procs"
  fi
  for p in /proc/[0-9]*; do
    [ -O "$p" ] || continue
    cwd="$(readlink "$p/cwd" 2>/dev/null)" || continue
    case "$cwd/" in "$NL_ROOT"/*|"$NL_BASE"/*) pids+=("${p#/proc/}") ;; esac
  done
  local victims=()
  for p in ${pids[@]+"${pids[@]}"}; do
    [ "$p" = "$$" ] || [ "$p" = "$BASHPID" ] && continue
    victims+=("$p")
  done
  [ "${#victims[@]}" -gt 0 ] || return 0
  log "node-local: quiescing ${#victims[@]} lane process(es): TERM"
  kill -TERM "${victims[@]}" 2>/dev/null || true
  i=0
  while [ "$i" -lt 3 ]; do
    local any=0
    for p in "${victims[@]}"; do nl_alive "$p" && any=1; done
    [ "$any" = 0 ] && break
    sleep 1; i=$((i + 1))
  done
  kill -KILL "${victims[@]}" 2>/dev/null || true
}

# nl_finalize WHY BUDGET_S — the FAILSAFE: stop the loops, quiesce, then one
# final spill (git refs, dirty trees, .git/langywrap, store snapshot, telemetry,
# runs, branch sync, MANIFEST.json). Runs at most once; the caller exits after.
nl_finalize() {
  local why="$1" budget="$2" t0 deadline rc
  NL_FINALIZING=1
  t0="$(date +%s)"; deadline=$((t0 + budget))
  log "node-local: FINAL spill start (reason=$why, budget ${budget}s)"
  if [ -n "${NL_LOOP_PID:-}" ]; then
    pkill -TERM -P "$NL_LOOP_PID" 2>/dev/null || true
    kill "$NL_LOOP_PID" 2>/dev/null || true
    wait "$NL_LOOP_PID" 2>/dev/null || true
    NL_LOOP_PID=""
  fi
  if [ -n "${SNAP_PID:-}" ]; then
    kill "$SNAP_PID" 2>/dev/null || true
    wait "$SNAP_PID" 2>/dev/null || true
    SNAP_PID=""
  fi
  nl_quiesce "$deadline"
  log "node-local: writers stopped after $(( $(date +%s) - t0 ))s; spilling"
  rc=0
  timeout -k 5 "$(( deadline - $(date +%s) + 10 ))" "$NL_PY" "$NL_FROZEN" spill --state "$NL_STATE" \
    --mode final --reason "$why" --deadline "$deadline" \
    --store "$LW_PY" "$STORE_SYNC" "$PROJECT" "$STORE_LOCAL" "$STORE_PERSIST" >> "$LOG/alloc.log" 2>&1 || rc=$?
  if [ "$rc" != 0 ] && [ "$rc" != 2 ]; then
    # The spill engine itself failed: leave a manifest that says so.
    local m="$NL_SPILL_ROOT/$SLURM_JOB_ID/MANIFEST.json"
    mkdir -p "$(dirname "$m")" && printf '{"schema":"slurmhelps-nodelocal/v1","status":"partial","mode":"final","reason":"%s","reasons":["spill engine rc=%s; see alloc.log"],"jobid":"%s","gen":"%s","written_at":"%s","written_at_epoch":%s,"spill_id":"%s.%s.engine-failed","trees":[]}\n' \
      "$why" "$rc" "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$(date -u +%FT%TZ)" "$(date +%s)" "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" > "$m.tmp$$" \
      && mv -f "$m.tmp$$" "$m"
    log "ERROR node-local final spill engine rc=$rc; wrote a partial manifest"
  fi
  log "node-local: FINAL spill done rc=$rc (0=final, 2=partial) in $(( $(date +%s) - t0 ))s; manifest $NL_SPILL_ROOT/$SLURM_JOB_ID/MANIFEST.json"
  NL_FINAL_DONE=1
}
# Any exit after a successful stage-in that did not already spill (FATAL
# paths, an unexpected `exit`) still spills.
nl_on_exit() {
  local rc=$?
  if [ "$NL_READY" = 1 ] && [ "${NL_FINAL_DONE:-0}" != 1 ] && [ "${NL_FINALIZING:-0}" != 1 ]; then
    log "node-local: EXIT (rc=$rc) without a final spill; spilling now"
    nl_finalize "exit-rc$rc" "$NL_BUDGET_END"
  fi
}

# ---- workset image: periodic spill, quiesce, FINAL spill --------------------
# Defined unconditionally, used only when WS_READY=1 (stage-in succeeded and
# the namespace holder is up).
ws_spill_pre() {  # the store snapshot, as one shell word list for `spill --pre`
  printf '%q ' "$LW_PY" "$STORE_SYNC" snapshot "$PROJECT" "$STORE_LOCAL" "$STORE_PERSIST"
}
ws_spill_loop() {
  while sleep "$WS_SPILL_INTERVAL"; do
    timeout -k 10 "$((WS_SPILL_INTERVAL * 2))" bash "$WS_SH" --project "$WORK_PROJECT" spill --mode periodic \
      --pre "$(ws_spill_pre)" >> "$LOG/alloc.log" 2>&1 \
      || log "WARN workset periodic spill rc=$? (3=busy); the previous image stands; next try in ${WS_SPILL_INTERVAL}s"
  done
}

# ws_ns_pids — processes in the workset mount namespace, except the holder.
ws_ns_pids() {
  local ns p
  ns="$(readlink "/proc/$WS_NS_PID/ns/mnt" 2>/dev/null)" || return 0
  [ -n "$ns" ] || return 0
  for p in /proc/[0-9]*; do
    [ -O "$p" ] || continue
    [ "${p#/proc/}" = "$WS_NS_PID" ] && continue
    [ "$(readlink "$p/ns/mnt" 2>/dev/null)" = "$ns" ] && echo "${p#/proc/}"
  done
}

# ws_quiesce DEADLINE — stop writers before the final spill: the daemon via
# its process group (as before), then every other process still inside the
# workset namespace (detached attempt owners and pi children outlive a daemon
# exit by design). Never the holder: the spill does not need the namespace,
# but `ly exec` diagnostics during the spill do.
ws_quiesce() {
  local deadline="$1" q i victims p any
  if [ -n "${DAEMON_PID:-}" ]; then
    kill -TERM "-$DAEMON_PID" 2>/dev/null || kill -TERM "$DAEMON_PID" 2>/dev/null || true
    q=$(( (deadline - $(date +%s)) / 4 )); [ "$q" -le 30 ] || q=30
    i=0
    while [ "$i" -lt "$q" ] && nl_alive "$DAEMON_PID"; do sleep 1; i=$((i + 1)); done
    if nl_alive "$DAEMON_PID"; then
      log "workset: daemon pid=$DAEMON_PID still up after ${q}s; KILL"
      kill -KILL "-$DAEMON_PID" 2>/dev/null || kill -KILL "$DAEMON_PID" 2>/dev/null || true
    fi
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  mapfile -t victims < <(ws_ns_pids)
  [ "${#victims[@]}" -gt 0 ] || return 0
  log "workset: quiescing ${#victims[@]} process(es) in the workset namespace: TERM"
  kill -TERM "${victims[@]}" 2>/dev/null || true
  i=0
  while [ "$i" -lt 3 ]; do
    any=0
    for p in "${victims[@]}"; do nl_alive "$p" && any=1; done
    [ "$any" = 0 ] && break
    sleep 1; i=$((i + 1))
  done
  kill -KILL "${victims[@]}" 2>/dev/null || true
}

# ws_finalize WHY BUDGET_S — the FAILSAFE: stop the loops, quiesce, then ONE
# final spill (store snapshot first, then the image). A spill that does not
# finish in time never replaces the published image: the previous one stands.
ws_finalize() {
  local why="$1" budget="$2" t0 deadline rc left lp
  WS_FINALIZING=1
  t0="$(date +%s)"; deadline=$((t0 + budget))
  log "workset: FINAL spill start (reason=$why, budget ${budget}s)"
  for lp in "${WS_LOOP_PID:-}" "${SNAP_PID:-}"; do
    [ -n "$lp" ] || continue
    pkill -TERM -P "$lp" 2>/dev/null || true
    kill "$lp" 2>/dev/null || true
    wait "$lp" 2>/dev/null || true
  done
  WS_LOOP_PID=""; SNAP_PID=""
  ws_quiesce "$deadline"
  left=$(( deadline - $(date +%s) ))
  log "workset: writers stopped after $(( $(date +%s) - t0 ))s; spilling (${left}s left)"
  rc=0
  timeout -k 5 "$(( left > 1 ? left : 1 ))" bash "$WS_SH" --project "$WORK_PROJECT" spill --mode final \
    --reason "$why" --deadline "$deadline" --pre "$(ws_spill_pre)" >> "$LOG/alloc.log" 2>&1 || rc=$?
  if [ "$rc" = 0 ]; then
    log "workset: FINAL spill done rc=0 in $(( $(date +%s) - t0 ))s"
  else
    log "ERROR workset FINAL spill rc=$rc after $(( $(date +%s) - t0 ))s; the published image is the previous spill (see the workset lines above)"
  fi
  kill -TERM "$WS_NS_PID" 2>/dev/null || true
  WS_FINAL_DONE=1
}
ws_on_exit() {
  local rc=$?
  if [ "$WS_READY" = 1 ] && [ "${WS_FINAL_DONE:-0}" != 1 ] && [ "${WS_FINALIZING:-0}" != 1 ]; then
    log "workset: EXIT (rc=$rc) without a final spill; spilling now"
    ws_finalize "exit-rc$rc" "$WS_BUDGET_END"
  fi
}

trap 'renew' USR1
trap 'shutdown TERM' TERM
trap 'shutdown INT' INT

# A PRIVATE runtime dir the daemon can trust. langywrap requires every owner-
# registry path component it finally opens to be owned by us with mode 0700
# (attempt_owner._private). On compute nodes /run/user/<uid> is root-owned, so we
# cannot use it. Use a node-local dir under /tmp that we create 0700, and point
# XDG_RUNTIME_DIR at it for the daemon AND for every client (ly reads it back
# from state/alloc.env). Set BEFORE publish_alloc so alloc.env records it.
RT_BASE="$SH_NODE_TMP/xdg-langywrap-$(id -u)"
install -d -m 0700 "$RT_BASE"
export XDG_RUNTIME_DIR="$RT_BASE"
RT="$RT_BASE/langywrap-$(id -u)"
install -d -m 0700 "$RT"
rm -f "$RT"/sessions-*.sock "$RT"/*.capability "$RT"/pi-tools-*.sock 2>/dev/null || true
rm -rf "$RT_BASE"/attempt-owners-* 2>/dev/null || true

# Session store on node-local tmpfs (README: "Session store on node-local
# tmpfs"). The WAL on Lustre stalled the whole fleet on 2026-10-10, so the
# daemon and lanes open the DB in STORE_LOCAL. store-sync.py stages it in before
# the daemon starts, snapshots it to STORE_PERSIST (/work) every
# STORE_SNAPSHOT_SECS, and once more on exit. Defined before publish_alloc so
# alloc.env records it.
STORE_SYNC="$SH_STORE_SYNC"
# langywrap's own node-local store (lanes.json store.node_local) replaces
# store-sync.py: two persisted copies could diverge, and a store-sync restore
# at stage-in would win over langywrap's newer snapshot. fleet.session_store.engine
# ("auto" default, "store-sync", "langywrap") overrides the detection.
STORE_ENGINE="$(python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("store-sync"); raise SystemExit
e = ((d.get("fleet") or {}).get("session_store") or {}).get("engine", "auto")
if e == "auto":
    e = "langywrap" if isinstance((d.get("store") or {}).get("node_local"), dict) else "store-sync"
print(e)' "$WORK_PROJECT/.langywrap/lanes.json" 2>/dev/null)"
if [ "$STORE_ENGINE" = langywrap ] && [ "$STORE_SYNC" = "$SLURMHELPS_HOME/fleet/store-sync.py" ]; then
  STORE_SYNC="$SLURMHELPS_HOME/fleet/store-noop.py"
fi
STORE_LOCAL="$SH_NODE_TMP/langywrap-store-$(id -u)/$(basename "$PROJECT")"
STORE_PERSIST="$STATE/runtime/$(basename "$PROJECT")/store"
STORE_SNAPSHOT_SECS="${AGENTS_STORE_SNAPSHOT_SECS:-600}"
install -d -m 0700 "$SH_NODE_TMP/langywrap-store-$(id -u)"
export LANGYWRAP_SESSIONS_DB_DIR="$STORE_LOCAL"
SNAP_PID=""

publish_alloc
log "allocation up on $(hostname -f); project=$PROJECT"
log "runtime dir: $RT (XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR)"

# Toolchain + caches on /work; langywrap CLI from its /work-backed venv
# (lib/common.sh sh_fleet_env: config site.env_sh, site.fleet_home, mpi_env...).
sh_fleet_env

# Workset stage-in, BEFORE anything opens the project: copy the images to
# node-local tmp, unpack and verify them, then start the namespace holder that
# binds the tree over its /work paths (live dirs back on top). Exit 3 = no
# image published yet (nothing to lose): run on /work as before. Any other
# failure is FATAL: lanes on the stale Lustre copy would fork the history the
# image holds.
if [ "$WS_ENABLED" = 1 ]; then
  log "workset: stage-in of $WS_IMAGE into $WS_STAGE"
  ws_rc=0
  bash "$WS_SH" --project "$WORK_PROJECT" stage-in >> "$LOG/alloc.log" 2>&1 || ws_rc=$?
  if [ "$ws_rc" = 0 ]; then
    rm -f "$WS_STATE"
    bash "$WS_SH" --project "$WORK_PROJECT" hold >> "$LOG/alloc.log" 2>&1 < /dev/null &
    WS_HOLD_PID=$!
    i=0
    while [ "$i" -lt 120 ] && ! grep -q '^NS_PID=' "$WS_STATE" 2>/dev/null && kill -0 "$WS_HOLD_PID" 2>/dev/null; do
      sleep 0.5; i=$((i + 1))
    done
    WS_NS_PID="$(sh_alloc_value "$WS_STATE" NS_PID)"
    if [ -n "$WS_NS_PID" ] && kill -0 "$WS_NS_PID" 2>/dev/null; then
      WS_READY=1
      WS_RUN=(bash "$WS_SH" --project "$WORK_PROJECT" exec --)
      trap 'ws_on_exit' EXIT
      export WORKSET_ACTIVE=1
      sh_workset_cache_env
      publish_alloc
      log "workset: namespace holder pid=$WS_NS_PID; project $PROJECT served from $WS_STAGE; spill every ${WS_SPILL_INTERVAL}s to $WS_IMAGE"
    else
      log "FATAL workset namespace holder did not come up (pid=${WS_HOLD_PID}); see the workset lines above"
      kill "$WS_HOLD_PID" 2>/dev/null || true
      exit 1
    fi
  elif [ "$ws_rc" = 3 ]; then
    log "WARN workset enabled but no image is published yet ($WS_IMAGE); running on $WORK_PROJECT as before. Build one: workset.sh build --from-lustre"
    WS_ENABLED=0
  else
    log "FATAL workset stage-in failed (rc=$ws_rc); not starting lanes on the stale /work copy. See the workset lines above."
    exit 1
  fi
fi

# Node-local stage-in, BEFORE anything opens the project: build the clone and
# its lane worktrees from the newest spill (else from /work), then point the
# daemon and lanes at it. Exit 3 = refused with nothing to lose (no spill
# exists): run on /work as before. Any other failure is FATAL: running lanes
# on an older /work tree would fork the history that the spill holds.
if [ "$NL_ENABLED" = 1 ]; then
  NL_PY="$LW_PY"; [ -x "$NL_PY" ] || NL_PY="$(command -v python3)"
  NL_STATE="$NL_BASE/state.json"
  NL_FROZEN="$NL_BASE/bin/nodelocal.py"
  log "node-local: stage-in into $NL_ROOT (spill root $NL_SPILL_ROOT)"
  nl_rc=0
  "$NL_PY" "$NL_SRC" stage-in --lanes "$WORK_PROJECT/.langywrap/lanes.json" --work "$WORK_PROJECT" \
    --jobid "$SLURM_JOB_ID" --gen "${SLURM_RESTART_COUNT:-0}" --state "$NL_STATE" >> "$LOG/alloc.log" 2>&1 || nl_rc=$?
  if [ "$nl_rc" = 0 ]; then
    NL_READY=1
    PROJECT="$NL_ROOT"
    trap 'nl_on_exit' EXIT
    export LANGYWRAP_PI_TELEMETRY_FILE="$NL_TELEMETRY_DIR/langypi.telemetry.jsonl"
    export SH_WORK_CHECKOUT="$WORK_PROJECT"
    # Project-named alias (config fleet.work_checkout_var).
    [ -n "${SH_WORK_CHECKOUT_VAR:-}" ] && export "$SH_WORK_CHECKOUT_VAR=$WORK_PROJECT"
    AGENT_ROOTS="${AGENT_ROOTS:-$PROJECT:$LW_CHECKOUT${SH_EXTRA_WRITABLE:+:$SH_EXTRA_WRITABLE}}:$WORK_PROJECT:$NL_TELEMETRY_DIR"
    publish_alloc
    log "node-local: project=$PROJECT (work checkout $WORK_PROJECT); spill every ${NL_SPILL_INTERVAL}s to $NL_SPILL_ROOT/$SLURM_JOB_ID"
  elif [ "$nl_rc" = 3 ]; then
    log "WARN node-local stage-in refused and no spill exists; running on $WORK_PROJECT as before"
    NL_ENABLED=0
  else
    log "FATAL node-local stage-in failed (rc=$nl_rc) while a spill exists; not starting lanes on an older tree. See the nodelocal lines above."
    exit 1
  fi
fi

# Lanes may read/write the project and the langywrap checkout. execwrap adds
# these to the lane sandbox's writable roots. Space-separated per execwrap.
# The session DB/WAL/SHM live in $STORE_LOCAL (node-local tmpfs); $STORE_PERSIST
# is the /work snapshot. Session artifacts stay in the project checkout.
ROOTS="${AGENT_ROOTS:-$PROJECT:$LW_CHECKOUT${SH_EXTRA_WRITABLE:+:$SH_EXTRA_WRITABLE}}:$STORE_LOCAL:$STORE_PERSIST"
export EXECWRAP_EXTRA_WRITABLE_PATHS="${EXECWRAP_EXTRA_WRITABLE_PATHS:-${ROOTS//:/ }}"

cd "$PROJECT" || { log "FATAL project missing: $PROJECT"; exit 1; }
if [ ! -x "$LW_PY" ]; then
  log "FATAL langywrap venv python missing: $LW_PY"
  exit 1
fi

# Migrating a legacy (regular-file) store is only safe when no other allocation
# still runs a daemon on it.
others="$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t R,CG -o '%i' | grep -vx "$SLURM_JOB_ID")"
if [ -n "$others" ] && [ -f "$PROJECT/.langywrap/sessions.sqlite3" ] && [ ! -L "$PROJECT/.langywrap/sessions.sqlite3" ]; then
  log "FATAL legacy store still served by job(s) $others; not migrating"
  exit 1
fi
# Stage the node-local store in (legacy migration or restore from the snapshot)
# before the daemon starts. Fatal on failure: a daemon on an unstaged store would
# silently create an empty one.
if ! ${WS_RUN[@]+"${WS_RUN[@]}"} "$LW_PY" "$STORE_SYNC" stage-in "$PROJECT" "$STORE_LOCAL" "$STORE_PERSIST" >> "$LOG/alloc.log" 2>&1; then
  log "FATAL store stage-in failed; see the store-sync lines above in alloc.log"
  exit 1
fi
store_snapshot_loop &
SNAP_PID=$!
if [ "$NL_READY" = 1 ]; then
  nl_spill_loop &
  NL_LOOP_PID=$!
  log "node-local: periodic spill loop pid=$NL_LOOP_PID"
fi
if [ "$WS_READY" = 1 ]; then
  ws_spill_loop &
  WS_LOOP_PID=$!
  log "workset: periodic spill loop pid=$WS_LOOP_PID"
fi
log "store: $STORE_LOCAL (engine ${STORE_ENGINE:-store-sync}: $(basename "$STORE_SYNC"); persist $STORE_PERSIST, snapshot every ${STORE_SNAPSHOT_SECS}s, pid=$SNAP_PID)"
# Start the daemon in its own session/process group and run it as the batch
# script's CHILD so `wait` blocks here and the signal traps above can fire.
# (An srun step would be killed when srun returns; a fully detached process
# would survive this script but complicate renewal.)
start_daemon() {
  # Under a workset the daemon enters the namespace (nsenter execs in place,
  # so DAEMON_PID is the daemon and its process group stays signalable).
  setsid ${WS_RUN[@]+"${WS_RUN[@]}"} "$LW_PY" -m langywrap.sessions.supervisor --serve "$PROJECT" \
    >> "$LOG/daemon.log" 2>&1 < /dev/null &
  DAEMON_PID=$!
  sleep 4
  if kill -0 "$DAEMON_PID" 2>/dev/null; then
    log "daemon up pid=$DAEMON_PID"
  else
    log "daemon NOT up; see $LOG/daemon.log"
  fi
}

# In-place restart on an unexpected daemon exit (crash, kill, lock or re-exec
# failure). Restarting here keeps the node-local store and this allocation; a
# new allocation would cost a ~30 min store stage-in and stop every lane.
# Exit 0 is the deliberate path: serve_forever returned after a `shutdown`
# request. `ly down` scancels the job, which arrives as TERM and hits the trap.
# The budget is bounded so a crash loop still ends the job.
MAX_RESTARTS="${AGENTS_MAX_RESTARTS:-5}"
RESTART_WINDOW="${AGENTS_RESTART_WINDOW_SECS:-3600}"
RESTART_BACKOFF="${AGENTS_RESTART_BACKOFF_SECS:-10}"
RESTART_BACKOFF_MAX=300
RESTART_TIMES=()
start_daemon
while :; do
  rc=0
  wait "$DAEMON_PID" || rc=$?   # a trap (USR1/TERM/INT) exits from inside wait
  if [ "$rc" -eq 0 ]; then
    log "daemon exited cleanly (rc=0, deliberate shutdown); allocation ending"
    break
  fi
  now="$(date +%s)"
  kept=()
  for t in ${RESTART_TIMES[@]+"${RESTART_TIMES[@]}"}; do
    [ $((now - t)) -lt "$RESTART_WINDOW" ] && kept+=("$t")
  done
  RESTART_TIMES=(${kept[@]+"${kept[@]}"})
  n="${#RESTART_TIMES[@]}"
  if [ "$n" -ge "$MAX_RESTARTS" ]; then
    log "ERROR daemon exited rc=$rc; restart budget exhausted ($MAX_RESTARTS per ${RESTART_WINDOW}s); allocation ending"
    break
  fi
  delay=$(( RESTART_BACKOFF << n ))
  [ "$delay" -le "$RESTART_BACKOFF_MAX" ] || delay="$RESTART_BACKOFF_MAX"
  RESTART_TIMES+=("$now")
  log "WARN daemon exited rc=$rc (pid=$DAEMON_PID); restart $((n + 1))/$MAX_RESTARTS (window ${RESTART_WINDOW}s) after ${delay}s backoff"
  # Snapshot first so a restart never resumes from a stale /work copy. The
  # snapshot loop keeps running; it is not stopped here.
  ${WS_RUN[@]+"${WS_RUN[@]}"} "$LW_PY" "$STORE_SYNC" snapshot "$PROJECT" "$STORE_LOCAL" "$STORE_PERSIST" >> "$LOG/alloc.log" 2>&1 \
    || log "WARN pre-restart store snapshot failed; restarting anyway"
  sleep "$delay" & wait $!   # wait, not a bare sleep, so the traps fire during backoff
  start_daemon
done
log "daemon loop ended; allocation ending"
if [ "$WS_READY" = 1 ]; then
  ws_finalize end "$WS_BUDGET_END"
elif [ "$NL_READY" = 1 ]; then
  nl_finalize end "$NL_BUDGET_END"
else
  store_final
fi
