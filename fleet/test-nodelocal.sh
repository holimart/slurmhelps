#!/usr/bin/env bash
# =============================================================================
#  test-nodelocal.sh SCRATCH_DIR — self-contained harness for the node-local
#  working set (nodelocal.py + supervisor.sbatch + ly spill).
# =============================================================================
#  No MPI, no Slurm, no langywrap daemon (run it in a short Slurm job anyway:
#  never heavy work on a login node). Everything lives under SCRATCH_DIR (never
#  /home, never the live /work tree): a toy "/work" repo with lane worktrees,
#  stub srun/scontrol/squeue/sbatch/scancel, a fake langywrap venv python whose
#  `-m langywrap.sessions.supervisor --serve` is a fake daemon that spawns a
#  detached fake lane. supervisor.sbatch and ly run IN PLACE from this
#  slurmhelps checkout; a fake site profile and project env point every path
#  into SCRATCH_DIR (the fleet code is checked for hard-coded site paths).
#
#  Cases
#   a  stage-in (first cut-over) builds the clone + worktrees from /work
#   b  the periodic spill writes MANIFEST.json
#   c  TERM, USR1 and daemon-crash-budget-exhausted each end with a final spill
#      (status "final") written BEFORE the batch script exits
#   d  a failing commit push still spills the uncommitted state -> "partial"
#   e  the next stage-in restores the spill (also `ly spill` -> "manual")
#   f  enabled=false: daemon on the /work checkout, no clone; ly sbatch-args
#
#  Usage: fleet/test-nodelocal.sh /path/to/empty/scratch/dir
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHHOME="$(cd "$HERE/.." && pwd)"
H="${1:?usage: test-nodelocal.sh SCRATCH_DIR}"
case "$H" in /home/*) echo "refusing a scratch dir under /home" >&2; exit 2 ;; esac
mkdir -p "$H"; H="$(cd "$H" && pwd -P)"
rm -rf "$H"/{work,agents,tmp,stubs,langywrap,home,cache,config,wbin,run,spill,origin.git,work-worktrees,f-*} 2>/dev/null
mkdir -p "$H"/{stubs,run,config,wbin,home,cache,tmp}
: > "$H/config/env.sh"
UIDN="$(id -u)"
WORK="$H/work/adgfem"
NLROOT="$H/tmp/adgfem-$UIDN/adgfem"
NLBASE="$H/tmp/adgfem-$UIDN"
SPILL="$H/spill"
ALOG="$H/agents/logs/alloc.log"
PASS=0; FAIL=0; RESULTS=()
ok()   { PASS=$((PASS + 1)); RESULTS+=("PASS $*"); echo "PASS $*"; }
bad()  { FAIL=$((FAIL + 1)); RESULTS+=("FAIL $*"); echo "FAIL $*"; }
check() { local what="$1"; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
# wait_for SECONDS CMD... — bounded poll
wait_for() { local t="$1" i=0; shift; until "$@"; do sleep 0.5; i=$((i + 1)); [ "$i" -ge $((t * 2)) ] && return 1; done; return 0; }
mjson() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {"d": d}))' "$@"; }

# ---------------------------------------------------------------- stubs
cat > "$H/stubs/scontrol" <<EOF
#!/usr/bin/env bash
echo "scontrol \$*" >> "$H/run/trace.log"
case "\$1 \${2:-}" in
  "show job") echo "JobId=\${4:-\$3} JobState=RUNNING StartTime=2026-10-11T00:00:00 EndTime=2026-10-18T00:00:00 Partition=ffa-long" ;;
  "show config") echo "KillWait                = 30 sec" ;;
  "requeue "*) echo "requeue \$2 \$(date +%s.%N)" >> "$H/run/requeue.log" ;;
esac
exit 0
EOF
cat > "$H/stubs/squeue" <<EOF
#!/usr/bin/env bash
echo "squeue \$*" >> "$H/run/trace.log"
if [ "\${2:-}" = -j ] && [ -f "$H/run/batch-\$3.pid" ]; then
  kill -0 "\$(cat "$H/run/batch-\$3.pid")" 2>/dev/null && echo RUNNING
  exit 0
fi
[ -f "$H/run/squeue.out" ] && cat "$H/run/squeue.out"
exit 0
EOF
cat > "$H/stubs/scancel" <<EOF
#!/usr/bin/env bash
echo "scancel \$*" >> "$H/run/trace.log"
if [ "\$1" = --batch ] && [ "\$2" = --signal=INT ] && [ -f "$H/run/batch-\$3.pid" ]; then
  kill -INT "\$(cat "$H/run/batch-\$3.pid")"; exit \$?
fi
echo "stub scancel refused" >&2; exit 1
EOF
for c in sbatch; do
  printf '#!/usr/bin/env bash\necho "%s $*" >> "%s/run/trace.log"\necho "stub %s refused" >&2\nexit 1\n' "$c" "$H" "$c" > "$H/stubs/$c"
done
# srun --jobid=J --overlap -n1 --chdir P CMD... -> run CMD in P, locally
cat > "$H/stubs/srun" <<EOF
#!/usr/bin/env bash
echo "srun \$*" >> "$H/run/trace.log"
dir=.
while [ \$# -gt 0 ]; do
  case "\$1" in --chdir) dir="\$2"; shift 2 ;; --*|-n1) shift ;; *) break ;; esac
done
cd "\$dir" && exec "\$@"
EOF
cat > "$H/stubs/python3" <<EOF
#!/usr/bin/env bash
echo "python3 \$*" >> "$H/run/trace.log"
exec /usr/bin/python3 "\$@"
EOF
chmod +x "$H"/stubs/*

# fake langywrap venv: python -m langywrap.sessions.supervisor --serve P is the fake daemon
mkdir -p "$H/langywrap/.venv/bin"
cat > "$H/langywrap/.venv/bin/python" <<EOF
#!/usr/bin/env bash
echo "lwpy \$*" >> "$H/run/trace.log"
if [ "\${1:-}" = -m ] && [ "\${2:-}" = langywrap.sessions.supervisor ]; then
  exec bash "$H/run/fake-daemon.sh" "\$@"
fi
exec /usr/bin/python3 "\$@"
EOF
chmod +x "$H/langywrap/.venv/bin/python"
cat > "$H/run/fake-daemon.sh" <<EOF
#!/usr/bin/env bash
# fake supervisor daemon: record how it was started, spawn a detached fake lane
# in the project, then idle until TERM or a control word (bounded: 300 s).
proj="\$4"
n=\$(ls "$H/run"/daemon-*.env 2>/dev/null | wc -l)
{ echo "argv=\$*"; echo "cwd=\$(pwd -P)";
  env | grep -E '^(LANGYWRAP_|EXECWRAP_EXTRA|XDG_RUNTIME_DIR|ADGFEM_|HOME=|PI_CODING)' | sort; } > "$H/run/daemon-\$n.env"
echo \$\$ > "$H/run/daemon.pid"
trap 'echo "daemon TERM" >> "$H/run/events.log"; exit 0' TERM
ctl="\$(cat "$H/run/daemon.ctl" 2>/dev/null)"
if [ "\$ctl" = crash ]; then sleep 1; echo "daemon crash" >> "$H/run/events.log"; exit 1; fi
if [ -d "\$proj/.lane-worktrees/harden" ] && [ "\$ctl" != nolane ]; then
  ( cd "\$proj/.lane-worktrees/harden" && setsid bash -c 'echo \$\$ > "$H/run/lane.pid"; i=0; while [ \$i -lt 1500 ]; do echo tick >> lane-heartbeat.log; sleep 0.2; i=\$((i+1)); done' ) &
fi
i=0
while [ \$i -lt 300 ]; do
  [ "\$(cat "$H/run/daemon.ctl" 2>/dev/null)" = exit0 ] && { echo "daemon exit0" >> "$H/run/events.log"; exit 0; }
  sleep 1 & wait \$!
  i=\$((i + 1))
done
EOF

# ---------------------------------------------------------------- toy /work repo
mk_toy_repo() {
  rm -rf "$H/work" "$H/origin.git" "$H/work-worktrees" "$SPILL" "$H/tmp" "$H/agents"
  mkdir -p "$WORK" "$H/tmp"
  git init -q --bare "$H/origin.git"
  git -C "$WORK" init -q -b mh-research
  git -C "$WORK" config user.name harness; git -C "$WORK" config user.email harness@example.invalid
  git -C "$WORK" config gc.auto 0
  git -C "$WORK" remote add origin "$H/origin.git"
  mkdir -p "$WORK"/{cluster/agents,.langywrap,docs,SRC_MPI,runs/gt-accuracy}
  cat > "$WORK/cluster/agents/store-sync.py" <<'PY'
import sys, time, pathlib
cmd, project, local, persist = sys.argv[1:5]
print(f"store-sync(stub): {cmd} {project}", flush=True)
pathlib.Path(local).mkdir(parents=True, exist_ok=True)
if cmd == "snapshot":
    p = pathlib.Path(persist); p.mkdir(parents=True, exist_ok=True)
    (p / "snapshot.txt").write_text(time.strftime("%FT%TZ"))
PY
  printf '/runs/\n/.lane-worktrees/\n.langywrap/worktrees/\n*.o\n/SRC_MPI/Adgfem\n' > "$WORK/.gitignore"
  echo "doc v1" > "$WORK/docs/a.md"; echo "doc b" > "$WORK/docs/b.md"
  echo "program x" > "$WORK/SRC_MPI/x.f90"
  echo "fixture" > "$WORK/runs/gt-accuracy/fixture.txt"
  lanes_json "${1:-true}" > "$WORK/.langywrap/lanes.json"
  git -C "$WORK" add -A; git -C "$WORK" add -f runs/gt-accuracy/fixture.txt
  git -C "$WORK" commit -q -m init
  git -C "$WORK" push -q origin mh-research
  git -C "$WORK" branch lane/harden; git -C "$WORK" branch prep/x
  git -C "$WORK" worktree add -q "$WORK/.lane-worktrees/harden" lane/harden
  git -C "$WORK" worktree add -q --detach "$WORK/.lane-worktrees/count-only" HEAD
  mkdir -p "$H/work-worktrees"; git -C "$WORK" worktree add -q "$H/work-worktrees/prep-x" prep/x
  # uncommitted state on /work: modified, staged, untracked, ignored build product, runtime
  echo "doc v1 + uncommitted edit on /work" > "$WORK/docs/a.md"
  echo "staged new file" > "$WORK/docs/staged.md"; git -C "$WORK" add docs/staged.md
  echo "untracked note" > "$WORK/docs/untracked.md"
  mkdir -p "$WORK/.langywrap/lane-reports" "$WORK/.langywrap/pi-sessions"
  echo '{"lane":"harden"}' > "$WORK/.langywrap/lane-reports/r1.json"
  echo '{"turn":1}' > "$WORK/.langywrap/pi-sessions/s1.jsonl"
  echo "legacy" > "$WORK/.langywrap/sessions.sqlite3"   # must never be copied
  # lane runtime spools/logs: NEVER_COPY in nodelocal.py, neither direction
  mkdir -p "$WORK/.log" "$WORK/.langywrap/activity-spool/v1/aa" "$WORK/.langywrap/artifacts/v1/aa" "$WORK/.trash/old"
  echo x > "$WORK/.log/cmd.log"; echo x > "$WORK/.langywrap/activity-spool/v1/aa/a.log"
  echo x > "$WORK/.langywrap/artifacts/v1/aa/b"; echo x > "$WORK/.langywrap/.langypi-session.Ab12"
  echo x > "$WORK/.langywrap/session-supervisor.log"; echo x > "$WORK/.trash/old/z"
  mkdir -p "$WORK/.lane-worktrees/harden/.log" "$WORK/.lane-worktrees/harden/SRC_MPI/.trash"
  echo x > "$WORK/.lane-worktrees/harden/.log/l.log"; echo x > "$WORK/.lane-worktrees/harden/SRC_MPI/.trash/q"
  echo "lane dirty" > "$WORK/.lane-worktrees/harden/SRC_MPI/x.f90"
  echo "BINARY" > "$WORK/.lane-worktrees/harden/SRC_MPI/Adgfem"; echo "obj" > "$WORK/.lane-worktrees/harden/SRC_MPI/x.o"
  mkdir -p "$WORK/.lane-worktrees/harden/runs/out"; echo "lane run output" > "$WORK/.lane-worktrees/harden/runs/out/log.txt"
  mkdir -p "$WORK/.git/langywrap"
  python3 -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute("create table t(x)"); c.execute("insert into t values (42)"); c.commit()' "$WORK/.git/langywrap/lane-turns.sqlite3"
  echo '{"ledger":1}' > "$WORK/.git/langywrap/lanes-ensure-ledger.json"
  # a DB above BIG_DB_BYTES (64 MiB) takes the late, budgeted copy path
  python3 -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute("create table b(x)"); c.execute("insert into b values (?)", (b"z" * (66 * 2**20),)); c.commit()' "$WORK/.git/langywrap/big.sqlite3"
}

lanes_json() {
  python3 - "$1" "$H" <<'PY'
import json, sys
enabled = sys.argv[1] == "true"
H = sys.argv[2]
print(json.dumps({"schema": "lanes/v1", "slurmhelps": {
    "site": {"work_root": H, "agents_dir": H + "/agents", "langywrap_checkout": H + "/langywrap",
             "env_sh": H + "/config/env.sh", "user_bin": H + "/wbin", "fleet_home": H + "/home/compute",
             "pi_agent_dir": H + "/home/pi/.pi/agent", "codex_home": H + "/home/codex",
             "xdg_cache_home": H + "/cache/xdg", "node_tmp": H + "/tmp", "images_dir": H + "/images",
             "mpi_env": {"OMPI_MCA_pml": "ob1"}},
    "fleet": {"partition": "fake-long", "time": "7-00:00:00", "renew_signal": 3600, "cpus": 40, "mem": "100G",
              "work_checkout_var": "ADGFEM_WORK_CHECKOUT"}},
    "fleet": {"branch": "mh-research", "node_local_checkout": {
    "enabled": enabled, "root": H + "/tmp/adgfem-{uid}/adgfem", "spill_interval_s": 10,
    "spill_root": H + "/spill", "telemetry_dir": H + "/tmp/langywrap-telemetry-{uid}", "min_free_gib": 0.1,
    "final_budget_term_s": 25, "final_budget_usr1_s": 120, "final_budget_end_s": 120, "store_min_s": 5}},
    "lanes": [{"id": "harden"}, {"id": "steward"}]}, indent=2))
PY
}

# the slurmhelps config lives in the toy project's lanes.json ("slurmhelps"
# block, see lanes_json); the override file only points store-sync at the stub
write_config() {
  printf 'SH_STORE_SYNC=%s\n' "$WORK/cluster/agents/store-sync.py" > "$H/agents.env"
}
SHENV=(SLURMHELPS_HOME="$SHHOME" SLURMHELPS_PROJECT="$WORK" SLURMHELPS_PROJECT_ENV="$H/agents.env")
# ly as the steward runs it (from the "login node", through the srun stub)
ly() { ( cd "$H/run" && env PATH="$H/stubs:$PATH" "${SHENV[@]}" bash "$HERE/ly" "$@" ); }

# run_sbatch SCRIPT JOBID GEN [extra env...] — background; prints the pid
run_sbatch() {
  local script="$1" jid="$2" gen="$3"; shift 3
  ( cd "$H/run" && exec env -i PATH="$H/stubs:/usr/bin:/bin" USER="$USER" LOGNAME="$USER" "${SHENV[@]}" \
      SLURM_JOB_ID="$jid" SLURM_RESTART_COUNT="$gen" SLURM_JOB_NAME=agent-supervisor \
      SLURM_JOB_NODELIST=fake-node SLURM_JOB_PARTITION=ffa-long TMPDIR="$H/tmp" "$@" \
      setsid bash "$script" > "$H/run/sbatch-$jid-$gen.out" 2>&1 ) &
  echo $! > "$H/run/batch-$jid.pid"
  echo $!
}
sbatch_done() { ! kill -0 "$1" 2>/dev/null; }
alloc_has() { grep -q "^$1=$2\$" "$H/agents/state/alloc.env" 2>/dev/null; }
manifest_status() { mjson "$SPILL/$1/MANIFEST.json" 'd["status"]' 2>/dev/null; }
manifest_is() { [ "$(manifest_status "$1")" = "$2" ]; }
daemon_up() { [ -s "$H/run/daemon.pid" ] && kill -0 "$(cat "$H/run/daemon.pid")" 2>/dev/null; }
lane_up() { [ -s "$H/run/lane.pid" ] && kill -0 "$(cat "$H/run/lane.pid")" 2>/dev/null; }
wipe_tmp() { rm -rf "$H/tmp"; mkdir -p "$H/tmp"; }
reset_run() { rm -f "$H/run"/{daemon-*.env,daemon.pid,daemon.ctl,lane.pid,events.log,requeue.log,squeue.out}; }
kill_leftovers() {
  local p
  for f in "$H/run/lane.pid" "$H/run/daemon.pid"; do
    [ -s "$f" ] && p="$(cat "$f")" && kill -KILL "$p" 2>/dev/null
  done
  return 0
}

for f in supervisor.sbatch ly start-lanes.sh label-lanes.sh lane-common.sh start-all-lanes.sh start-one-lane.sh prepare-project.sh login-node-guard.sh steward-tick.sh; do
  bash -n "$HERE/$f" || { echo "bash -n failed: $f"; exit 1; }
done
python3 -m py_compile "$HERE/nodelocal.py" "$HERE/store-sync.py" "$HERE/stray-check.py" || exit 1

check "0: fleet code has no hard-coded site paths" bash -c "! grep -nE '/work/ffa|holecem|ffa-long|ffa-short' '$HERE'/*.sh '$HERE'/ly '$HERE'/supervisor.sbatch '$HERE'/*.py '$SHHOME/lib/common.sh' | grep -v '^[^:]*:[0-9]*:[[:space:]]*#' | grep -v test-nodelocal.sh"

mk_toy_repo true
write_config
cp "$HERE/supervisor.sbatch" "$H/run/supervisor.sbatch"   # sbatch runs a spooled copy too

echo "== case a: stage-in (first cut-over from /work) + case b: periodic spill =="
reset_run
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900001 0)"
wait_for 90 alloc_has NODE_LOCAL 1 || bad "a: alloc.env never published NODE_LOCAL=1"
wait_for 30 daemon_up || bad "a: fake daemon never started"
check "a: node-local clone exists on branch mh-research" test "$(git -C "$NLROOT" symbolic-ref --short HEAD 2>/dev/null)" = mh-research
check "a: alloc.env PROJECT is the node-local root" alloc_has PROJECT "$NLROOT"
check "a: alloc.env keeps WORK_PROJECT" alloc_has WORK_PROJECT "$WORK"
check "a: uncommitted edit copied" grep -q "uncommitted edit on /work" "$NLROOT/docs/a.md"
check "a: staged file still staged" bash -c "git -C '$NLROOT' diff --cached --name-only | grep -qx docs/staged.md"
check "a: untracked file copied" test -f "$NLROOT/docs/untracked.md"
check "a: runtime .langywrap copied" test -f "$NLROOT/.langywrap/lane-reports/r1.json"
check "a: legacy store file NOT copied" test ! -e "$NLROOT/.langywrap/sessions.sqlite3"
check "a: runs is a symlink to /work runs" test "$(readlink "$NLROOT/runs")" = "$WORK/runs"
check "a: git status shows no runs noise" bash -c "! git -C '$NLROOT' status --porcelain | grep -q runs"
check "a: lane worktree harden on lane/harden" test "$(git -C "$NLROOT/.lane-worktrees/harden" symbolic-ref --short HEAD 2>/dev/null)" = lane/harden
check "a: lane worktree is a worktree of the node-local repo" bash -c "git -C '$NLROOT' worktree list | grep -q '$NLROOT/.lane-worktrees/harden'"
check "a: lane dirty + ignored build product copied" bash -c "grep -q 'lane dirty' '$NLROOT/.lane-worktrees/harden/SRC_MPI/x.f90' && test -f '$NLROOT/.lane-worktrees/harden/SRC_MPI/Adgfem'"
check "a: lane runs -> /work lane runs" test "$(readlink "$NLROOT/.lane-worktrees/harden/runs")" = "$WORK/.lane-worktrees/harden/runs"
check "a: count-only detached" bash -c "! git -C '$NLROOT/.lane-worktrees/count-only' symbolic-ref -q HEAD"
check "a: steward worktree outside the checkout not imported" test ! -e "$NLROOT/work-worktrees"
check "a: .git/langywrap sqlite copied" test "$(python3 -c 'import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute("select x from t").fetchone()[0])' "$NLROOT/.git/langywrap/lane-turns.sqlite3" 2>/dev/null)" = 42
check "a: objects shared with /work (alternates)" grep -q "$WORK/.git/objects" "$NLROOT/.git/objects/info/alternates"
check "a: daemon serves the node-local root" grep -q "argv=-m langywrap.sessions.supervisor --serve $NLROOT\$" "$H/run/daemon-0.env"
check "a: daemon cwd is the node-local root" grep -qx "cwd=$NLROOT" "$H/run/daemon-0.env"
check "a: telemetry file moved to node-local tmp" grep -qx "LANGYWRAP_PI_TELEMETRY_FILE=$H/tmp/langywrap-telemetry-$UIDN/langypi.telemetry.jsonl" "$H/run/daemon-0.env"
check "a: lane sandbox may write node-local root, /work checkout, telemetry" bash -c "grep '^EXECWRAP_EXTRA_WRITABLE_PATHS=' '$H/run/daemon-0.env' | grep -q '$NLROOT ' && grep '^EXECWRAP_EXTRA_WRITABLE_PATHS=' '$H/run/daemon-0.env' | grep -q ' $WORK ' && grep '^EXECWRAP_EXTRA_WRITABLE_PATHS=' '$H/run/daemon-0.env' | grep -q 'langywrap-telemetry'"
check "a: session store dir unchanged (basename adgfem)" grep -qx "LANGYWRAP_SESSIONS_DB_DIR=$H/tmp/langywrap-store-$UIDN/adgfem" "$H/run/daemon-0.env"
for x in .log .langywrap/activity-spool .langywrap/artifacts .langywrap/.langypi-session.Ab12 .langywrap/session-supervisor.log .trash .lane-worktrees/harden/.log .lane-worktrees/harden/SRC_MPI/.trash; do
  check "a: never-copy $x NOT imported" test ! -e "$NLROOT/$x"
done
wait_for 20 lane_up || bad "a: fake lane never started"

# a lane commits and leaves uncommitted work (cwd outside the root: quiesce kills by cwd)
echo "lane commit" > "$NLROOT/docs/lane.md"; git -C "$NLROOT" add docs/lane.md; git -C "$NLROOT" -c user.name=lane -c user.email=l@x commit -q -m "lane: commit 1" -- docs/lane.md
C1="$(git -C "$NLROOT" rev-parse HEAD)"
echo "lane uncommitted v1" > "$NLROOT/docs/b.md"
echo "harden commit" >> "$NLROOT/.lane-worktrees/harden/SRC_MPI/y.f90"; git -C "$NLROOT/.lane-worktrees/harden" add SRC_MPI/y.f90
git -C "$NLROOT/.lane-worktrees/harden" -c user.name=lane -c user.email=l@x commit -q -m "harden: y" -- SRC_MPI/y.f90
H1="$(git -C "$NLROOT/.lane-worktrees/harden" rev-parse HEAD)"

# node-side runtime spools/logs written after the import must never be spilled
mkdir -p "$NLROOT/.log" "$NLROOT/.langywrap/activity-spool/v1/bb" "$NLROOT/.langywrap/artifacts/v1/bb" "$NLROOT/.trash" "$NLROOT/.lane-worktrees/harden/.log"
echo n > "$NLROOT/.log/n.log"; echo n > "$NLROOT/.langywrap/activity-spool/v1/bb/n"; echo n > "$NLROOT/.langywrap/artifacts/v1/bb/n"
echo n > "$NLROOT/.langywrap/.langypi-session.Zz9"; echo n > "$NLROOT/.trash/n"; echo n > "$NLROOT/.lane-worktrees/harden/.log/n.log"

wait_for 45 manifest_is 900001 periodic || bad "b: no periodic manifest within 45 s"
M="$SPILL/900001/MANIFEST.json"
check "b: periodic MANIFEST.json status=periodic" manifest_is 900001 periodic
check "b: manifest records git HEAD" test "$(mjson "$M" 'd["git_head"]')" = "$C1"
check "b: manifest lists dirty paths" bash -c "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); t=[x for x in d[\"trees\"] if x[\"name\"]==\"main\"][0]; assert \"docs/b.md\" in t[\"dirty_paths\"] and \"docs/untracked.md\" in t[\"dirty_paths\"]' '$M'"
check "b: manifest has sizes" bash -c "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d[\"sizes\"][\"dirty_bytes\"] > 0' '$M'"
check "b: commit pushed to /work refs/nodelocal" test "$(git -C "$WORK" rev-parse -q --verify refs/nodelocal/900001/heads/mh-research)" = "$C1"
check "b: /work mh-research fast-forwarded (dirty /work edit kept)" bash -c "test \"\$(git -C '$WORK' rev-parse mh-research)\" = '$C1' && grep -q 'uncommitted edit on /work' '$WORK/docs/a.md'"
check "b: file unchanged since the import is NOT copied (base layer = /work)" bash -c "test ! -e '$SPILL/900001/trees/main/docs/untracked.md' && tr '\\0' '\\n' < '$SPILL/900001/lists/main.overlay' | grep -qx docs/untracked.md"
check "b: changed file IS copied" test -f "$SPILL/900001/trees/main/docs/b.md"
check "b: never-copy paths absent from every spill tree and list" bash -c "! find '$SPILL/900001/trees' \\( -name .log -o -name activity-spool -o -name artifacts -o -name .trash -o -name '.langypi-session.*' \\) | grep -q . && ! cat '$SPILL'/900001/lists/* 2>/dev/null | tr '\\0' '\\n' | grep -qE '(^|/)(\\.log|\\.trash)(/|\$)|activity-spool|langywrap/artifacts|langypi-session'"
check "b: big .git/langywrap DB copied by the late step" bash -c "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert [s for s in d[\"steps\"] if s[\"name\"]==\"gitdir-langywrap-db\" and s[\"ok\"]]' '$M' && test -s '$SPILL/900001/gitdir-langywrap/big.sqlite3'"
check "b: runtime lane-reports mirrored to /work" test -f "$WORK/.langywrap/lane-reports/r1.json"
check "b: lane branch published to /work refs/nodelocal" test "$(git -C "$WORK" rev-parse -q --verify refs/nodelocal/900001/heads/lane/harden)" = "$H1"

echo "== case c1: TERM -> final spill before exit =="
echo "lane uncommitted v2 (after periodic)" > "$NLROOT/docs/b.md"
echo "new untracked after periodic" > "$NLROOT/docs/late.md"
LANEPID="$(cat "$H/run/lane.pid")"
T_SIG="$(date +%s.%N)"
kill -TERM "$PID"
wait_for 60 sbatch_done "$PID" || bad "c1: batch script did not exit within 60 s"
T_EXIT="$(date +%s.%N)"
check "c1: manifest status=final" manifest_is 900001 final
check "c1: manifest reason TERM" test "$(mjson "$M" 'd["reason"]')" = TERM
check "c1: manifest written before the batch script exited" python3 -c "import json,sys; d=json.load(open('$M')); sys.exit(0 if $T_SIG <= d['written_at_epoch'] <= $T_EXIT else 1)"
check "c1: final spill took < 25 s (KillWait 30 s)" python3 -c "import json,sys; d=json.load(open('$M')); sys.exit(0 if d['duration_s'] < 25 else 1)"
check "c1: late uncommitted edit is in the spill" grep -q "after periodic" "$SPILL/900001/trees/main/docs/b.md"
check "c1: late untracked file is in the spill" test -f "$SPILL/900001/trees/main/docs/late.md"
check "c1: store snapshot taken in the final spill" bash -c "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert [s for s in d[\"steps\"] if s[\"name\"]==\"store\" and s[\"ok\"]]' '$M'"
check "c1: fake lane quiesced (killed) before the spill" bash -c "! kill -0 $LANEPID 2>/dev/null"
check "c1: alloc.log: FINAL spill done logged" grep -q "node-local: FINAL spill done rc=0" "$ALOG"
check "c1: no requeue on TERM" test ! -s "$H/run/requeue.log"
cp "$M" "$H/run/manifest-c1.json"

echo "== case e1 + c2: requeue (gen 1) restores the spill; USR1 -> final spill + requeue =="
reset_run
wipe_tmp   # job_container/tmpfs: a requeued job starts with an empty /tmp
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900001 1)"
wait_for 90 daemon_up || bad "e1: fake daemon never started (gen 1)"
check "e1: stage-in restored from the spill" grep -q "restoring from spill $SPILL/900001 (status=final" "$ALOG"
check "e1: committed lane work restored" test "$(git -C "$NLROOT" rev-parse HEAD)" = "$C1"
check "e1: uncommitted edit restored" grep -q "after periodic" "$NLROOT/docs/b.md"
check "e1: late untracked file restored" test -f "$NLROOT/docs/late.md"
check "e1: base-layer file restored from /work" grep -qx "untracked note" "$NLROOT/docs/untracked.md"
check "e1: base-layer runtime file restored" test -f "$NLROOT/.langywrap/pi-sessions/s1.jsonl"
check "e1: staged state restored" bash -c "git -C '$NLROOT' diff --cached --name-only | grep -qx docs/staged.md"
check "e1: lane worktree restored at its commit" test "$(git -C "$NLROOT/.lane-worktrees/harden" rev-parse HEAD)" = "$H1"
check "e1: lane ignored build product restored" test -f "$NLROOT/.lane-worktrees/harden/SRC_MPI/Adgfem"
check "e1: .git/langywrap restored" test -f "$NLROOT/.git/langywrap/lanes-ensure-ledger.json"
check "e1: big .git/langywrap DB restored" test "$(python3 -c 'import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute("select length(x) from b").fetchone()[0])' "$NLROOT/.git/langywrap/big.sqlite3" 2>/dev/null)" = $((66 * 1048576))
check "e1: clean tracked files intact" grep -qx "program x" "$NLROOT/SRC_MPI/x.f90"
# ly spill (manual) from the "login node" through the srun stub
echo 900001 > "$H/run/squeue.out"
echo "manual edit" > "$NLROOT/docs/manual.md"
ly spill > "$H/run/ly-spill.out" 2>&1
check "e1: ly spill -> manifest status=manual" manifest_is 900001 manual
check "e1: ly spill copied the new file" test -f "$SPILL/900001/trees/main/docs/manual.md"
ly spill >> "$H/run/ly-spill.out" 2>&1
check "e1: ly spill is idempotent (second run manual again)" manifest_is 900001 manual
rm -f "$H/run/squeue.out"
echo "usr1 edit" > "$NLROOT/docs/usr1.md"
T_SIG="$(date +%s.%N)"
kill -USR1 "$PID"
wait_for 90 sbatch_done "$PID" || bad "c2: batch script did not exit within 90 s after USR1"
T_EXIT="$(date +%s.%N)"
check "c2: manifest status=final after USR1" manifest_is 900001 final
check "c2: manifest reason USR1" test "$(mjson "$M" 'd["reason"]')" = USR1
check "c2: manifest written before exit" python3 -c "import json,sys; d=json.load(open('$M')); sys.exit(0 if $T_SIG <= d['written_at_epoch'] <= $T_EXIT else 1)"
check "c2: USR1 edit spilled" test -f "$SPILL/900001/trees/main/docs/usr1.md"
check "c2: requeue issued AFTER the manifest was written" python3 -c "import json,sys; d=json.load(open('$M')); t=float(open('$H/run/requeue.log').read().split()[-1]); sys.exit(0 if t > d['written_at_epoch'] else 1)"

echo "== case c3: daemon crash budget exhausted (gen 2) -> final spill =="
reset_run
echo crash > "$H/run/daemon.ctl"
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900001 2 AGENTS_MAX_RESTARTS=1 AGENTS_RESTART_BACKOFF_SECS=1)"
wait_for 120 sbatch_done "$PID" || bad "c3: batch script did not exit within 120 s"
check "c3: same-job tree reused (warm) when /tmp survived" grep -q "nodelocal: warm: " "$ALOG"
check "c3: restart budget exhausted logged" grep -q "restart budget exhausted" "$ALOG"
check "c3: manifest status=final" manifest_is 900001 final
check "c3: manifest reason end" test "$(mjson "$M" 'd["reason"]')" = end
check "c3: restored then re-spilled the USR1 edit" test -f "$SPILL/900001/trees/main/docs/usr1.md"
check "c3: spill gen recorded" test "$(mjson "$M" 'd["gen"]')" = 2

echo "== case d: failing commit push -> uncommitted state spilled, status=partial =="
reset_run
wipe_tmp
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900002 0)"
wait_for 90 daemon_up || bad "d: fake daemon never started"
check "d: new job restored from job 900001's spill" grep -q "restoring from spill $SPILL/900001" "$ALOG"
check "d: spill lineage renamed 900001 -> 900002" bash -c "test ! -e '$SPILL/900001' && grep -q 'spill lineage: $SPILL/900001 -> $SPILL/900002' '$ALOG'"
# break pushes into /work refs/nodelocal/900002 (a file where git needs a directory)
mkdir -p "$WORK/.git/refs/nodelocal"; : > "$WORK/.git/refs/nodelocal/900002"
echo "d commit" > "$NLROOT/docs/d.md"; git -C "$NLROOT" add docs/d.md; git -C "$NLROOT" -c user.name=lane -c user.email=l@x commit -q -m "lane: d" -- docs/d.md
D1="$(git -C "$NLROOT" rev-parse HEAD)"
echo "d uncommitted" > "$NLROOT/docs/d-dirty.md"
kill -TERM "$PID"
wait_for 60 sbatch_done "$PID" || bad "d: batch script did not exit within 60 s"
MD="$SPILL/900002/MANIFEST.json"
check "d: manifest status=partial" manifest_is 900002 partial
check "d: reasons name the failed push" bash -c "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any(r.startswith(\"git-refs\") for r in d[\"reasons\"])' '$MD'"
check "d: uncommitted file spilled anyway" test -f "$SPILL/900002/trees/main/docs/d-dirty.md"
check "d: unpushed commit kept in commits.bundle" bash -c "git -C '$NLROOT' bundle list-heads '$SPILL/900002/commits.bundle' | grep -q '$D1'"
check "d: index spilled" test -f "$SPILL/900002/index/main.index"
check "d: alloc.log says partial" grep -q "FINAL spill done rc=2" "$ALOG"
rm -f "$WORK/.git/refs/nodelocal/900002"

echo "== case e2: next stage-in restores a PARTIAL spill whose commit never reached /work =="
reset_run
wipe_tmp
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900003 0)"
wait_for 90 daemon_up || bad "e2: fake daemon never started"
check "e2: restored from the partial spill 900002" grep -q "restoring from spill $SPILL/900002 (status=partial" "$ALOG"
check "e2: base-layer file still restored after two lineage renames" grep -qx "untracked note" "$NLROOT/docs/untracked.md"
check "e2: lane ignored build product (base layer of the lane tree) restored" test -f "$NLROOT/.lane-worktrees/harden/SRC_MPI/Adgfem"
check "e2: uncommitted file restored" test -f "$NLROOT/docs/d-dirty.md"
check "e2: unpushed commit restored from the bundle" test "$(git -C "$NLROOT" rev-parse HEAD)" = "$D1"
check "e2: its file is back" grep -qx "d commit" "$NLROOT/docs/d.md"
check "e2: index restored (no rebuild note)" bash -c "! grep -q 'main: index named missing objects' '$ALOG'"
echo exit0 > "$H/run/daemon.ctl"
wait_for 60 sbatch_done "$PID" || bad "e2: batch script did not exit after daemon exit0"
check "e2: normal end -> final" manifest_is 900003 final
kill_leftovers

echo "== case c4: ly down (node-local) -> INT to the batch shell -> final spill with the long budget =="
reset_run
wipe_tmp
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900004 0)"
wait_for 90 daemon_up || bad "c4: fake daemon never started"
echo 900004 > "$H/run/squeue.out"
echo "down edit" > "$NLROOT/docs/down.md"
( export LY_DOWN_GRACE=60; ly down ) > "$H/run/ly-down.out" 2>&1
check "c4: ly down returned after the batch shell finished" sbatch_done "$PID"
check "c4: manifest status=final, reason INT" bash -c "[ \"\$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d[\"status\"], d[\"reason\"])' '$SPILL/900004/MANIFEST.json')\" = 'final INT' ]"
check "c4: INT used the long budget (120 s here, not 25 s)" grep -q "FINAL spill start (reason=INT, budget 120s)" "$ALOG"
check "c4: edit spilled" test -f "$SPILL/900004/trees/main/docs/down.md"
check "c4: ly down sent INT to the batch shell, no plain scancel" bash -c "grep -q '^scancel --batch --signal=INT 900004' '$H/run/trace.log' && ! grep -q '^scancel 900004' '$H/run/trace.log'"
rm -f "$H/run/squeue.out"
kill_leftovers

cp "$ALOG" "$H/alloc-enabled-cases.log"
echo "== case f: enabled=false -> no node-local clone, daemon on the /work checkout =="
mk_toy_repo false
reset_run; rm -f "$H/run/trace.log"
PID="$(run_sbatch "$H/run/supervisor.sbatch" 900009 0)"
wait_for 60 daemon_up || bad "f: daemon never started"
check "f: daemon serves the /work checkout" grep -q "argv=-m langywrap.sessions.supervisor --serve $WORK\$" "$H/run/daemon-0.env"
check "f: alloc.env has no NODE_LOCAL / WORKSET" bash -c "! grep -qE '^(NODE_LOCAL|WORKSET)=' '$H/agents/state/alloc.env'"
check "f: alloc.env records PROJECT and STORE_SYNC" bash -c "grep -q '^PROJECT=$WORK\$' '$H/agents/state/alloc.env' && grep -q '^STORE_SYNC=' '$H/agents/state/alloc.env'"
echo exit0 > "$H/run/daemon.ctl"
wait_for 60 sbatch_done "$PID" || bad "f: did not exit"
check "f: no node-local clone, no spill" bash -c "test ! -e '$NLROOT' && test ! -e '$SPILL'"
SBA="$(ly sbatch-args 2>&1)"
check "f: ly sbatch-args: footprint from the project env, paths from the site profile" bash -c "grep -q -- '--partition=fake-long' <<<'$SBA' && grep -q -- '--output=$H/agents/logs/sup-%j.out' <<<'$SBA' && grep -q 'SLURMHELPS_HOME=$SHHOME' <<<'$SBA'"

kill_leftovers
echo
echo "== summary: $PASS passed, $FAIL failed =="
printf '%s\n' "${RESULTS[@]}" | grep '^FAIL' || true
[ "$FAIL" -eq 0 ]
