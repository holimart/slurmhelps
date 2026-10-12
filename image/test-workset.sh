#!/usr/bin/env bash
# =============================================================================
#  test-workset.sh SCRATCH_DIR — harness for the workset images
#  (image/workset.py + fleet/supervisor.sbatch + fleet/ly).
# =============================================================================
#  Needs a node with unprivileged user+mount namespaces (unshare -c -m),
#  nsenter, mksquashfs/unsquashfs and squashfuse: run it in a Slurm job (a
#  compute node), with SCRATCH_DIR on node-local /tmp. Everything lives under
#  SCRATCH_DIR: a fake "Lustre" prefix $P (project git repo with a lane
#  worktree, live runs/ dir with tracked + untracked files, agent home with a
#  secret, a toolchain dir), a fake node tmp, stub Slurm commands, a fake
#  langywrap venv whose supervisor daemon records what it sees and spawns a
#  detached fake lane.
#
#  Cases
#   A  build from the fake Lustre trees: excludes, tracked files of the live
#      dir kept, secrets never in the image, manifest + verified counts
#   B  toolchain image build
#   C  stage-in unpacks + verifies, copies secrets in
#   D  hold publishes the namespace; exec from a SEPARATE process (an
#      `srun --overlap` step when run inside Slurm) joins it: tmp tree visible
#      at the /work path, live runs/ writes land on the "Lustre" side, toolchain
#      from the image, caches on node tmp
#   E  spill: includes a commit made inside, SQLite WAL content (backup API),
#      excludes live logs/spools/locks/sockets, writes the secret back, runs
#      --pre; .prev rotated
#   F  a failing / timed-out / deadline-missed spill never replaces the image
#   G  build refuses to overwrite a spilled image without --from-lustre
#   H  supervisor.sbatch with fleet.workset enabled: daemon inside the
#      namespace, ly exec joins it, ly spill, TERM -> final spill before exit,
#      requeue stage-in restores the spilled state, USR1 -> final + requeue,
#      workset wins over node_local_checkout
#   I  view (squashfuse) and writeback (dry run, apply)
#
#  Usage: image/test-workset.sh /tmp/some-empty-dir
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHHOME="$(cd "$HERE/.." && pwd)"
H="${1:?usage: test-workset.sh SCRATCH_DIR}"
case "$H" in /home/*|/work/*) echo "refusing a scratch dir on /home or /work; use node-local tmp" >&2; exit 2 ;; esac
mkdir -p "$H"; H="$(cd "$H" && pwd -P)"
# a stale holder/fuse mount from an earlier run
for f in "$H"/node/workset-*/ns.env; do [ -f "$f" ] && kill "$(sed -n 's/^NS_PID=//p' "$f")" 2>/dev/null; done
fusermount -u "$H/lustre/images/view" 2>/dev/null
rm -rf "$H"/{lustre,node,run,stubs,agents.env} 2>/dev/null
mkdir -p "$H"/{lustre,node,run,stubs}
UIDN="$(id -u)"
P="$H/lustre"                     # fake Lustre prefix = SH_WORK_ROOT
PROJ="$P/proj"
STAGE="$H/node/workset-$UIDN"
ROOT="$STAGE/root"
IMG="$P/images/proj.sqfs"
WS="$HERE/workset.sh"
PASS=0; FAIL=0; RESULTS=()
ok()   { PASS=$((PASS + 1)); RESULTS+=("PASS $*"); echo "PASS $*"; }
bad()  { FAIL=$((FAIL + 1)); RESULTS+=("FAIL $*"); echo "FAIL $*"; }
check() { local what="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$what"; else bad "$what"; fi; }
wait_for() { local t="$1" i=0; shift; until "$@" >/dev/null 2>&1; do sleep 0.5; i=$((i + 1)); [ "$i" -ge $((t * 2)) ] && return 1; done; return 0; }
mjson() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {"d": d}))' "$@"; }
listing() { unsquashfs -lls -d R "$1" 2>/dev/null | awk '{print $NF}' | sed 's#^R##'; }
in_img() { listing "$IMG" | grep -x -- "$1" > /dev/null; }   # no -q: an early grep exit + pipefail = false FAIL
not_in_img() { ! listing "$IMG" | grep -x -- "$1" > /dev/null; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

# ---------------------------------------------------------------- config
# The project's slurmhelps config ($PROJ/slurmhelps.json, written below) holds
# the "site" paths; the override file only points store-sync at a stub.
printf 'SH_STORE_SYNC=%s\n' "$H/run/store-sync-stub.py" > "$H/agents.env"
export SLURMHELPS_HOME="$SHHOME" SLURMHELPS_PROJECT="$PROJ" SLURMHELPS_PROJECT_ENV="$H/agents.env"
SHENV=(SLURMHELPS_HOME="$SHHOME" SLURMHELPS_PROJECT="$PROJ" SLURMHELPS_PROJECT_ENV="$H/agents.env")
write_shconfig() {
  python3 - "$H" "$P" > "$PROJ/slurmhelps.json" <<'PY'
import json, sys
H, P = sys.argv[1], sys.argv[2]
print(json.dumps({"site": {
    "work_root": P, "agents_dir": P + "/agents", "langywrap_checkout": P + "/langywrap",
    "env_sh": P + "/config/env.sh", "user_bin": P + "/bin", "fleet_home": P + "/home/compute",
    "pi_agent_dir": P + "/home/pi/.pi/agent", "codex_home": P + "/home/codex",
    "xdg_cache_home": P + "/cache/xdg", "node_tmp": H + "/node", "images_dir": P + "/images",
    "toolchain_dirs": ["tc"], "mpi_env": {"OMPI_MCA_pml": "ob1"}},
    "fleet": {"partition": "fake-long", "cpus": 4, "mem": "8G"},
    "workset": {"spill_interval_s": 3600}}, indent=2))
PY
}

lanes_json() {  # lanes_json WORKSET_ENABLED NODE_LOCAL_ENABLED
  python3 - "$1" "$2" "$H" <<'PY'
import json, sys
ws, nl, H = sys.argv[1] == "true", sys.argv[2] == "true", sys.argv[3]
print(json.dumps({"schema": "lanes/v1", "fleet": {
    "branch": "main",
    "workset": {"enabled": ws, "trees": ["{project}", "{work_root}/home/pi"],
                "live_binds": ["{project}/runs"],
                "secrets": ["{work_root}/home/pi/.pi/agent/auth.json"],
                "extra_excludes": ["{project}/big-scratch/"],
                "spill_interval_s": 3600, "final_budget_term_s": 25,
                "final_budget_usr1_s": 120, "final_budget_end_s": 120},
    "node_local_checkout": {"enabled": nl, "root": H + "/node/proj-{uid}/proj", "spill_root": H + "/lustre/nlspill"},
}, "lanes": [{"id": "harden"}, {"id": "steward"}]}, indent=2))
PY
}

# ---------------------------------------------------------------- fake Lustre trees
mkdir -p "$PROJ" "$P/home/pi/.pi/agent" "$P/tc/bin" "$P/config" "$P/agents" "$P/langywrap/.venv/bin"
: > "$P/config/env.sh"
git -C "$PROJ" init -q -b main
git -C "$PROJ" config user.name t; git -C "$PROJ" config user.email t@example.invalid; git -C "$PROJ" config gc.auto 0
mkdir -p "$PROJ"/{.langywrap,docs,runs/fixtures}
lanes_json true false > "$PROJ/.langywrap/lanes.json"
write_shconfig
printf '/runs/\n/.lane-worktrees/\n.log/\n' > "$PROJ/.gitignore"
echo "readme" > "$PROJ/README"; echo "doc" > "$PROJ/docs/a.md"
echo "tracked fixture" > "$PROJ/runs/fixtures/tracked.txt"
git -C "$PROJ" add -A; git -C "$PROJ" add -f runs/fixtures/tracked.txt
git -C "$PROJ" commit -q -m init
git -C "$PROJ" branch lane/harden
git -C "$PROJ" worktree add -q "$PROJ/.lane-worktrees/harden" lane/harden
echo "big result" > "$PROJ/runs/result.dat"; mkdir -p "$PROJ/runs/case1"; echo r > "$PROJ/runs/case1/out"
mkdir -p "$PROJ/.log" "$PROJ/.langywrap/activity-spool/v1" "$PROJ/.lane-worktrees/harden/.log" "$PROJ/big-scratch"
echo x > "$PROJ/.log/old.log"; echo x > "$PROJ/.langywrap/activity-spool/v1/a"; echo x > "$PROJ/.lane-worktrees/harden/.log/l"
echo x > "$PROJ/big-scratch/blob"
echo legacy > "$PROJ/.langywrap/sessions.sqlite3"; echo x > "$PROJ/.langywrap/runtime.sqlite3-wal"
echo x > "$PROJ/.langywrap/session-supervisor.lock"; : > "$PROJ/.git/index.lock.keep"
python3 -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1])' "$PROJ/.langywrap/pi-tools.sock"
mkdir -p "$PROJ/.git/langywrap"
echo '{"auth":"v1"}' > "$P/home/pi/.pi/agent/auth.json"; chmod 600 "$P/home/pi/.pi/agent/auth.json"
echo '{"s":1}' > "$P/home/pi/.pi/agent/settings.json"
printf '#!/bin/sh\necho tool-v1\n' > "$P/tc/bin/mytool"; chmod +x "$P/tc/bin/mytool"
mkdir -p "$P/uv/cache"

# ---------------------------------------------------------------- stubs + fake venv
cat > "$H/stubs/scontrol" <<EOF
#!/usr/bin/env bash
echo "scontrol \$*" >> "$H/run/trace.log"
case "\$1 \${2:-}" in
  "show job") echo "JobId=\${4:-\$3} JobState=RUNNING EndTime=2026-10-18T00:00:00 Partition=fake-long" ;;
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
exit 1
EOF
printf '#!/usr/bin/env bash\necho "sbatch $*" >> "%s/run/trace.log"\nexit 1\n' "$H" > "$H/stubs/sbatch"
cat > "$H/stubs/srun" <<EOF
#!/usr/bin/env bash
echo "srun \$*" >> "$H/run/trace.log"
dir=.
while [ \$# -gt 0 ]; do
  case "\$1" in --chdir) dir="\$2"; shift 2 ;; --*|-n1) shift ;; *) break ;; esac
done
cd "\$dir" && exec "\$@"
EOF
chmod +x "$H"/stubs/*
cat > "$H/run/store-sync-stub.py" <<'PY'
import sys, time, pathlib
cmd, project, local, persist = sys.argv[1:5]
print(f"store-sync(stub): {cmd} {project}", flush=True)
pathlib.Path(local).mkdir(parents=True, exist_ok=True)
if cmd == "snapshot":
    p = pathlib.Path(persist); p.mkdir(parents=True, exist_ok=True)
    (p / "snapshot.txt").write_text(time.strftime("%FT%TZ"))
PY
cat > "$P/langywrap/.venv/bin/python" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = -m ] && [ "\${2:-}" = langywrap.sessions.supervisor ]; then
  exec bash "$H/run/fake-daemon.sh" "\$@"
fi
exec /usr/bin/python3 "\$@"
EOF
chmod +x "$P/langywrap/.venv/bin/python"
cat > "$H/run/fake-daemon.sh" <<EOF
#!/usr/bin/env bash
proj="\$4"
n=\$(ls "$H/run"/daemon-*.env 2>/dev/null | wc -l)
{ echo "argv=\$*"; echo "cwd=\$(pwd -P)"; echo "mntns=\$(readlink /proc/self/ns/mnt)";
  [ -e "\$proj/.workset-seen" ] && echo "seen=tmp-tree" || echo "seen=lustre";
  [ -e "\$proj/lane-late.txt" ] && echo "late=present" || echo "late=absent";
  env | grep -E '^(WORKSET_ACTIVE|UV_CACHE_DIR|LANGYWRAP_SESSIONS_DB_DIR|HOME|OMPI_MCA_pml)=' | sort; } > "$H/run/daemon-\$n.env"
echo \$\$ > "$H/run/daemon.pid"
trap 'echo "daemon TERM" >> "$H/run/events.log"; exit 0' TERM
( cd "\$proj" && setsid bash -c 'echo \$\$ > "$H/run/lane.pid"; i=0; while [ \$i -lt 1500 ]; do echo tick >> "\$0/.log/hb.log"; echo "\$i" > "\$0/lane-work.txt"; sleep 0.2; i=\$((i+1)); done' "\$proj" ) &
i=0
while [ \$i -lt 300 ]; do
  [ "\$(cat "$H/run/daemon.ctl" 2>/dev/null)" = exit0 ] && { echo "daemon exit0" >> "$H/run/events.log"; exit 0; }
  sleep 1 & wait \$!
  i=\$((i + 1))
done
EOF

# ws ARGS... — workset.sh for this test project
ws() { bash "$WS" --project "$PROJ" "$@"; }
# from_other_process CMD... — run CMD as a process that is NOT a descendant of
# the holder: a separate srun step when inside Slurm, else a new session
OVERLAP=0
from_other_process() {
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1 && [ "${WS_TEST_NO_OVERLAP:-0}" != 1 ]; then
    srun --jobid="$SLURM_JOB_ID" --overlap -n1 -N1 env -u BASH_ENV "$@"
  else
    setsid "$@" < /dev/null
  fi
}
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1 && [ "${WS_TEST_NO_OVERLAP:-0}" != 1 ]; then OVERLAP=1; fi
in_ns() { from_other_process bash "$WS" --project "$PROJ" exec --no-fallback -- bash -c "$1"; }

python3 -m py_compile "$HERE/workset.py" || exit 1
bash -n "$WS" "$HERE/toolchain.sh" || exit 1
check "0: image code has no hard-coded site paths" bash -c "! grep -nE '/work/ffa|holecem|ffa-long|ffa-short' '$HERE/workset.py' '$HERE/workset.sh' '$HERE/toolchain.sh'"

echo "== A: build =="
ws build > "$H/run/build.out" 2>&1; rc=$?
check "A: build rc=0" test "$rc" = 0
[ "$rc" = 0 ] || cat "$H/run/build.out"
check "A: image published" test -s "$IMG"
check "A: manifest kind=build" test "$(mjson "$IMG.manifest.json" 'd["kind"]')" = build
check "A: tracked file kept" in_img "$PROJ/README"
check "A: git dir kept" in_img "$PROJ/.git/HEAD"
check "A: lane worktree kept" in_img "$PROJ/.lane-worktrees/harden/README"
check "A: agent home kept" in_img "$P/home/pi/.pi/agent/settings.json"
check "A: live dir: tracked file kept" in_img "$PROJ/runs/fixtures/tracked.txt"
check "A: live dir: results excluded" not_in_img "$PROJ/runs/result.dat"
check "A: live dir: result dirs excluded" not_in_img "$PROJ/runs/case1"
check "A: secret NOT in image" not_in_img "$P/home/pi/.pi/agent/auth.json"
check "A: .log excluded (main + worktree)" bash -c "! unsquashfs -lls -d R '$IMG' | grep -q '/\\.log'"
check "A: activity-spool excluded" not_in_img "$PROJ/.langywrap/activity-spool"
check "A: session/runtime stores excluded" bash -c "! unsquashfs -lls -d R '$IMG' | grep -qE 'sessions\\.sqlite3|runtime\\.sqlite3'"
check "A: lock files excluded" not_in_img "$PROJ/.langywrap/session-supervisor.lock"
check "A: socket excluded" not_in_img "$PROJ/.langywrap/pi-tools.sock"
check "A: absolute extra exclude" not_in_img "$PROJ/big-scratch"
check "A: embedded manifest" in_img "/.workset/manifest.json"
check "A: manifest counts == image" python3 -c "import json,sys; d=json.load(open('$IMG.manifest.json')); sys.exit(0 if d['image']['files']==d['scan']['files']+1 else 1)"

echo "== B: toolchain build =="
bash "$HERE/toolchain.sh" --project "$PROJ" build > "$H/run/tc-build.out" 2>&1
check "B: toolchain image built" test -s "$P/images/toolchain.sqfs"
check "B: holds the tool" bash -c "unsquashfs -lls -d R '$P/images/toolchain.sqfs' | grep -q 'tc/bin/mytool'"
printf '#!/bin/sh\necho tool-lustre-changed\n' > "$P/tc/bin/mytool"

echo "== C: stage-in =="
ws stage-in > "$H/run/stagein.out" 2>&1; rc=$?
check "C: stage-in rc=0" test "$rc" = 0
[ "$rc" = 0 ] || cat "$H/run/stagein.out"
check "C: data root unpacked" test -f "$ROOT$PROJ/README"
check "C: toolchain root unpacked" test -x "$STAGE/toolchain-root$P/tc/bin/mytool"
check "C: secret copied in" grep -q v1 "$ROOT$P/home/pi/.pi/agent/auth.json"
check "C: stage copies of the images removed" bash -c "test ! -e '$STAGE/data.sqfs' && test ! -e '$STAGE/toolchain.sqfs'"
touch "$ROOT$PROJ/.workset-seen"

echo "== D: hold + exec from another process =="
ws hold > "$H/run/hold.out" 2>&1 &
wait_for 30 grep -q '^NS_PID=' "$STAGE/ns.env" || bad "D: holder never published NS_PID"
NSPID="$(sed -n 's/^NS_PID=//p' "$STAGE/ns.env")"
check "D: holder alive" kill -0 "$NSPID"
check "D: holder is in another mount namespace" test "$(readlink /proc/$NSPID/ns/mnt)" != "$(readlink /proc/self/ns/mnt)"
echo "joined via $( [ $OVERLAP = 1 ] && echo 'srun --overlap step' || echo 'setsid process')"
out="$(in_ns 'readlink /proc/self/ns/mnt' 2>"$H/run/exec.err")"
check "D: exec joins the holder's namespace (nsenter, no fallback)" test "$out" = "$(readlink /proc/$NSPID/ns/mnt)"
[ "$out" = "$(readlink /proc/$NSPID/ns/mnt)" ] || cat "$H/run/exec.err"
check "D: tmp tree visible at the /work path" bash -c "[ \"\$(cd /tmp && bash '$WS' --project '$PROJ' exec -- test -e '$PROJ/.workset-seen' && echo y)\" = y ]"
check "D: outside the namespace the Lustre tree is unchanged" test ! -e "$PROJ/.workset-seen"
in_ns "echo made-inside > '$PROJ/new-in-ns.txt'"
check "D: write to the project lands on node tmp" test -f "$ROOT$PROJ/new-in-ns.txt"
check "D: ... and not on Lustre" test ! -e "$PROJ/new-in-ns.txt"
in_ns "echo live > '$PROJ/runs/live.txt'"
check "D: write to the live runs/ lands on Lustre" test -f "$PROJ/runs/live.txt"
check "D: ... and not on node tmp" test ! -e "$ROOT$PROJ/runs/live.txt"
check "D: Lustre results visible inside" bash -c "[ \"\$(bash '$WS' --project '$PROJ' exec -- cat '$PROJ/runs/result.dat')\" = 'big result' ]"
check "D: toolchain comes from the image" bash -c "[ \"\$(bash '$WS' --project '$PROJ' exec -- '$P/tc/bin/mytool')\" = tool-v1 ]"
check "D: caches on node tmp, WORKSET_ACTIVE=1" bash -c "bash '$WS' --project '$PROJ' exec -- env | grep -qx 'UV_CACHE_DIR=$H/node/workset-cache-$UIDN/uv' && bash '$WS' --project '$PROJ' exec -- env | grep -qx WORKSET_ACTIVE=1"
check "D: cwd follows into the namespace" bash -c "[ \"\$(cd '$PROJ/docs' && bash '$WS' --project '$PROJ' exec -- pwd)\" = '$PROJ/docs' ]"
# state made inside: a commit in the lane worktree, a WAL-mode SQLite DB with
# uncheckpointed rows held open, live logs, a refreshed secret
in_ns "cd '$PROJ/.lane-worktrees/harden' && echo c > lanefile && git add lanefile && git -c user.name=l -c user.email=l@x commit -q -m 'lane: inside ns' && git rev-parse HEAD" > "$H/run/commit.out"
LANEC="$(tail -1 "$H/run/commit.out")"
check "D: commit made inside" test -n "$LANEC"
check "D: commit not in the Lustre repo" bash -c "! git -C '$PROJ' cat-file -e '$LANEC' 2>/dev/null"
cat > "$H/run/walwriter.py" <<'PY'
import sqlite3, sys, time
c = sqlite3.connect(sys.argv[1]); c.execute("pragma journal_mode=wal"); c.execute("pragma wal_autocheckpoint=0")
c.execute("create table t(x)"); c.executemany("insert into t values (?)", [(i,) for i in range(1000)]); c.commit()
open(sys.argv[2], "w").write("ready"); time.sleep(120)
PY
bash "$WS" --project "$PROJ" exec -- python3 "$H/run/walwriter.py" "$PROJ/.git/langywrap/lane-turns.sqlite3" "$H/run/wal.ready" &
WALPID=$!
wait_for 30 test -f "$H/run/wal.ready" || bad "D: WAL writer never ready"
check "D: WAL holds the rows (main db not checkpointed)" test -s "$ROOT$PROJ/.git/langywrap/lane-turns.sqlite3-wal"
in_ns "mkdir -p '$PROJ/.log' '$PROJ/.langywrap/activity-spool/v2'; echo n > '$PROJ/.log/new.log'; echo n > '$PROJ/.langywrap/activity-spool/v2/n'; echo '{\"auth\":\"refreshed\"}' > '$P/home/pi/.pi/agent/auth.json'"

echo "== E: spill =="
OLDSHA="$(sha "$IMG")"
ws spill --mode manual --reason test --pre "touch '$H/run/pre-ran'" > "$H/run/spill1.out" 2>&1; rc=$?
check "E: spill rc=0" test "$rc" = 0
[ "$rc" = 0 ] || cat "$H/run/spill1.out"
check "E: image replaced" test "$(sha "$IMG")" != "$OLDSHA"
check "E: previous image kept as .prev" test "$(sha "$IMG.prev")" = "$OLDSHA"
check "E: manifest kind=spill mode=manual" test "$(mjson "$IMG.manifest.json" 'd["kind"]+"/"+d["mode"]')" = spill/manual
check "E: --pre hook ran" test -f "$H/run/pre-ran"
check "E: file made inside is in the image" in_img "$PROJ/new-in-ns.txt"
check "E: live logs excluded" bash -c "! unsquashfs -lls -d R '$IMG' | grep -qE '/\\.log/|activity-spool'"
check "E: WAL/SHM excluded" bash -c "! unsquashfs -lls -d R '$IMG' | grep -qE 'sqlite3-(wal|shm)'"
check "E: live runs file NOT in the image (it is on Lustre)" not_in_img "$PROJ/runs/live.txt"
check "E: secret NOT in the image" not_in_img "$P/home/pi/.pi/agent/auth.json"
check "E: refreshed secret written back to Lustre" grep -q refreshed "$P/home/pi/.pi/agent/auth.json"
rm -rf "$H/run/unp"; unsquashfs -q -d "$H/run/unp" "$IMG" >/dev/null 2>&1
check "E: the commit is in the image's repo" git -C "$H/run/unp$PROJ" cat-file -e "$LANEC"
check "E: SQLite WAL rows captured (backup API)" test "$(python3 -c 'import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute("select count(*) from t").fetchone()[0])' "$H/run/unp$PROJ/.git/langywrap/lane-turns.sqlite3" 2>/dev/null)" = 1000
kill "$WALPID" 2>/dev/null; wait "$WALPID" 2>/dev/null

echo "== F: failed spills never replace the image =="
GOOD="$(sha "$IMG")"
mkdir -p "$H/badbin"
printf '#!/bin/sh\nfor a; do last=$a; done\necho garbage > "$2"\nexit 1\n' > "$H/badbin/mksquashfs"; chmod +x "$H/badbin/mksquashfs"
PATH="$H/badbin:$PATH" ws spill --mode manual > "$H/run/spill-bad.out" 2>&1; rc=$?
check "F: failing mksquashfs -> rc=2" test "$rc" = 2
check "F: image unchanged" test "$(sha "$IMG")" = "$GOOD"
check "F: no .new left" test ! -e "$IMG.new"
printf '#!/bin/sh\necho partial > "$2"\nexec sleep 60\n' > "$H/badbin/mksquashfs"
PATH="$H/badbin:$PATH" ws spill --mode final --deadline "$(( $(date +%s) + 8 ))" > "$H/run/spill-slow.out" 2>&1; rc=$?
check "F: mksquashfs past the deadline -> rc=2" test "$rc" = 2
check "F: image unchanged after the timeout" test "$(sha "$IMG")" = "$GOOD"
check "F: no partial image anywhere" bash -c "test ! -e '$IMG.new' && ! grep -q partial '$IMG'"
ws spill --mode final --deadline "$(( $(date +%s) - 1 ))" > "$H/run/spill-late.out" 2>&1; rc=$?
check "F: deadline already passed -> rc=2, unchanged" bash -c "[ $rc = 2 ] && [ \"\$(sha256sum '$IMG' | cut -d' ' -f1)\" = '$GOOD' ]"
rm -rf "$H/badbin"
( flock 9; ws spill --mode periodic > "$H/run/spill-busy.out" 2>&1; echo $? > "$H/run/busy.rc" ) 9>"$STAGE/spill.lock"
check "F: periodic spill while another holds the lock -> rc=3 (busy)" test "$(cat "$H/run/busy.rc")" = 3

echo "== G: build refuses to overwrite a spilled image =="
ws build > "$H/run/build2.out" 2>&1; rc=$?
check "G: build without --from-lustre refused (rc=3)" test "$rc" = 3
check "G: image unchanged" test "$(sha "$IMG")" = "$GOOD"

kill "$NSPID" 2>/dev/null

echo "== H: supervisor.sbatch with fleet.workset =="
run_sbatch() {
  local jid="$1" gen="$2"; shift 2
  ( cd "$H/run" && exec env -i PATH="$H/stubs:/usr/bin:/bin" USER="$USER" LOGNAME="$USER" "${SHENV[@]}" \
      SLURM_JOB_ID="$jid" SLURM_RESTART_COUNT="$gen" SLURM_JOB_NAME=agent-supervisor \
      SLURM_JOB_NODELIST=fake-node SLURM_JOB_PARTITION=fake-long "$@" \
      setsid bash "$H/run/supervisor.sbatch" > "$H/run/sbatch-$jid-$gen.out" 2>&1 ) &
  echo $! > "$H/run/batch-$jid.pid"
  echo $!
}
sbatch_done() { ! kill -0 "$1" 2>/dev/null; }
alloc_has() { grep -q "^$1=$2\$" "$P/agents/state/alloc.env" 2>/dev/null; }
daemon_up() { [ -s "$H/run/daemon.pid" ] && kill -0 "$(cat "$H/run/daemon.pid")" 2>/dev/null; }
lane_up() { [ -s "$H/run/lane.pid" ] && kill -0 "$(cat "$H/run/lane.pid")" 2>/dev/null; }
reset_run() { rm -f "$H/run"/{daemon-*.env,daemon.pid,daemon.ctl,lane.pid,events.log,requeue.log,squeue.out}; }
ALOG="$P/agents/logs/alloc.log"
ly() { ( cd "$H/run" && env PATH="$H/stubs:$PATH" "${SHENV[@]}" bash "$SHHOME/fleet/ly" "$@" ); }
cp "$SHHOME/fleet/supervisor.sbatch" "$H/run/supervisor.sbatch"
rm -rf "$STAGE"   # a new job: /tmp is empty
reset_run
PID="$(run_sbatch 700001 0)"
wait_for 90 alloc_has WORKSET 1 || { bad "H: alloc.env never published WORKSET=1"; tail -30 "$ALOG"; }
wait_for 30 daemon_up || bad "H: fake daemon never started"
check "H: daemon runs inside the workset namespace" grep -Fqx "mntns=$(readlink "/proc/$(sed -n 's/^WORKSET_NS_PID=//p' "$P/agents/state/alloc.env")/ns/mnt")" "$H/run/daemon-0.env"
check "H: daemon sees the image tree at the project path" grep -qx "seen=tmp-tree" "$H/run/daemon-0.env"
check "H: daemon serves the ORIGINAL project path" grep -q "argv=-m langywrap.sessions.supervisor --serve $PROJ\$" "$H/run/daemon-0.env"
check "H: daemon caches on node tmp" grep -qx "UV_CACHE_DIR=$H/node/workset-cache-$UIDN/uv" "$H/run/daemon-0.env"
check "H: daemon got the config's MPI env (site.mpi_env)" grep -qx "OMPI_MCA_pml=ob1" "$H/run/daemon-0.env"
check "H: alloc.env publishes the namespace pid" bash -c "kill -0 \$(sed -n 's/^WORKSET_NS_PID=//p' '$P/agents/state/alloc.env')"
check "H: new-in-ns.txt (from the spill) restored by stage-in" test -f "$STAGE/root$PROJ/new-in-ns.txt"
wait_for 20 lane_up || bad "H: fake lane never started"
echo 700001 > "$H/run/squeue.out"
out="$(ly exec "cat '$PROJ/new-in-ns.txt'; readlink /proc/self/ns/mnt" 2>&1)"
check "H: ly exec joins the namespace" bash -c "grep -qx made-inside <<<'$out' && grep -Fqx '$(readlink "/proc/$(sed -n 's/^WORKSET_NS_PID=//p' "$P/agents/state/alloc.env")/ns/mnt")' <<<'$out'"
ly exec "echo late > '$PROJ/lane-late.txt'" >/dev/null 2>&1
ly spill > "$H/run/ly-spill.out" 2>&1
check "H: ly spill -> manual spill published" test "$(mjson "$IMG.manifest.json" 'd["mode"]')" = manual
SST="$(ly spill status 2>&1)"
check "H: ly spill status" grep -q 'namespace holder' <<<"$SST"
rm -f "$H/run/squeue.out"
echo "after manual" > "$STAGE/root$PROJ/after-manual.txt"
LANEPID="$(cat "$H/run/lane.pid")"
T_SIG="$(date +%s.%N)"
kill -TERM "$PID"
wait_for 60 sbatch_done "$PID" || bad "H: batch script did not exit within 60 s of TERM"
T_EXIT="$(date +%s.%N)"
check "H: TERM -> final spill (mode=final, reason=TERM)" test "$(mjson "$IMG.manifest.json" 'd["mode"]+"/"+d["reason"]')" = final/TERM
check "H: final image written before the batch script exited" python3 -c "import json,sys,os; t=os.path.getmtime('$IMG'); sys.exit(0 if $T_SIG <= t <= $T_EXIT else 1)"
check "H: final spill holds the late edit" in_img "$PROJ/after-manual.txt"
check "H: lane quiesced before the spill" bash -c "! kill -0 $LANEPID 2>/dev/null"
check "H: lane heartbeat log NOT spilled" bash -c "! unsquashfs -lls -d R '$IMG' | grep -q hb.log"
check "H: alloc.log: FINAL spill done rc=0" grep -q "workset: FINAL spill done rc=0" "$ALOG"
check "H: no requeue on TERM" test ! -s "$H/run/requeue.log"
check "H: namespace holder stopped" bash -c "! pgrep -f 'workset-ns' -u $UIDN >/dev/null || ! kill -0 \$(sed -n 's/^NS_PID=//p' '$STAGE/ns.env') 2>/dev/null"

echo "== H2: requeue (gen 1): stage-in restores the final spill; USR1 -> final + requeue =="
rm -rf "$STAGE"
reset_run
PID="$(run_sbatch 700001 1)"
wait_for 90 daemon_up || bad "H2: fake daemon never started"
check "H2: daemon sees the state restored from the final spill" grep -qx "late=present" "$H/run/daemon-0.env"
check "H2: late edit restored" test -f "$STAGE/root$PROJ/after-manual.txt"
check "H2: lane commit restored" git -C "$STAGE/root$PROJ" cat-file -e "$LANEC"
kill -USR1 "$PID"
wait_for 90 sbatch_done "$PID" || bad "H2: batch script did not exit within 90 s of USR1"
check "H2: USR1 -> final spill" test "$(mjson "$IMG.manifest.json" 'd["mode"]+"/"+d["reason"]')" = final/USR1
check "H2: requeue issued after the image was published" python3 -c "import os,sys; t=float(open('$H/run/requeue.log').read().split()[-1]); sys.exit(0 if t > os.path.getmtime('$IMG') else 1)"

echo "== H3: both enabled -> workset wins =="
lanes_json true true > "$STAGE/root$PROJ/.langywrap/lanes.json" 2>/dev/null
lanes_json true true > "$PROJ/.langywrap/lanes.json"
rm -rf "$STAGE"; reset_run
PID="$(run_sbatch 700002 0)"
wait_for 90 daemon_up || bad "H3: fake daemon never started"
check "H3: WARN logged" grep -q "both enabled; the workset wins" "$ALOG"
check "H3: alloc.env WORKSET=1, no NODE_LOCAL" bash -c "grep -qx WORKSET=1 '$P/agents/state/alloc.env' && ! grep -q '^NODE_LOCAL=' '$P/agents/state/alloc.env'"
check "H3: no node-local clone" test ! -e "$H/node/proj-$UIDN"
echo exit0 > "$H/run/daemon.ctl"
wait_for 90 sbatch_done "$PID" || bad "H3: batch script did not exit after daemon exit0"
check "H3: normal end -> final spill (reason end)" test "$(mjson "$IMG.manifest.json" 'd["reason"]')" = end
lanes_json true false > "$PROJ/.langywrap/lanes.json"
[ -s "$H/run/lane.pid" ] && kill -KILL "$(cat "$H/run/lane.pid")" 2>/dev/null

echo "== I: view + writeback =="
ws view > "$H/run/view.out" 2>&1; rc=$?
if [ "$rc" = 0 ]; then
  check "I: view mounts the image read-only" test -f "$P/images/view$PROJ/after-manual.txt"
  check "I: view is read-only" bash -c "! touch '$P/images/view$PROJ/x' 2>/dev/null"
  ws view --umount > /dev/null 2>&1
  check "I: view --umount" bash -c "! grep -q ' $P/images/view ' /proc/self/mounts"
else
  bad "I: view failed: $(tail -2 "$H/run/view.out")"
fi
ws writeback > "$H/run/wb-dry.out" 2>&1
check "I: writeback dry run changes nothing" test ! -e "$PROJ/after-manual.txt"
check "I: writeback dry run reports changes" grep -q "would apply" "$H/run/wb-dry.out"
ws writeback --apply > "$H/run/wb.out" 2>&1; rc=$?
check "I: writeback --apply rc=0" test "$rc" = 0
check "I: spilled work now on Lustre" bash -c "test -f '$PROJ/after-manual.txt' && test -f '$PROJ/new-in-ns.txt'"
check "I: lane commit now in the Lustre repo" git -C "$PROJ" cat-file -e "$LANEC"
check "I: Lustre results untouched" bash -c "test -f '$PROJ/runs/result.dat' && test -f '$PROJ/runs/live.txt'"
check "I: excluded Lustre paths untouched (.log, spool, big-scratch)" bash -c "test -f '$PROJ/.log/old.log' && test -f '$PROJ/.langywrap/activity-spool/v1/a' && test -f '$PROJ/big-scratch/blob'"
check "I: secret untouched by writeback" grep -q refreshed "$P/home/pi/.pi/agent/auth.json"
check "I: no embedded manifest written to Lustre" test ! -e "$P/.workset"

for f in "$STAGE/ns.env"; do [ -f "$f" ] && kill "$(sed -n 's/^NS_PID=//p' "$f")" 2>/dev/null; done
echo
echo "== summary: $PASS passed, $FAIL failed =="
printf '%s\n' "${RESULTS[@]}" | grep '^FAIL' || true
[ "$FAIL" -eq 0 ]
