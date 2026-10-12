#!/usr/bin/env bash
# tests/shell/test-remote.sh SCRATCH_DIR — workstation-side helpers: bin/mirror_mount.sh (the
# project mirror mount, unchanged, and named mounts) and bin/code_sync.sh.
# A real workstation sshfs mount cannot be made from the cluster, so this runs against a fake
# project config with stubs on PATH: sshfs (adds a line to a fake mount
# table), fusermount3 (removes it), ssh (runs the command locally) and rsync (real rsync with the
# fake host prefix stripped). The cluster-node refusal is checked for real first; stubbed cases
# set SLURMHELPS_REMOTE_ALLOW_CLUSTER=1. Wrapped by tests/test_remote_shell.py for pytest.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH="$(cd "$HERE/../.." && pwd)"
MM="$SH/bin/mirror_mount.sh"; CS="$SH/bin/code_sync.sh"
T="${1:?usage: test-remote.sh SCRATCH_DIR}"
case "$T" in /home/*) echo "refusing a scratch dir under /home" >&2; exit 2 ;; esac
rm -rf "$T"; mkdir -p "$T"/{stubs,home,proj,remote}; T="$(cd "$T" && pwd -P)"
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); echo "PASS $*"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL $*"; }
check() { local w="$1"; shift; if "$@"; then ok "$w"; else bad "$w"; fi; }

if command -v shellcheck >/dev/null 2>&1; then
  check "shellcheck" shellcheck -x "$MM" "$CS" "$SH/lib/named_mounts.sh"
else
  echo "NOTE shellcheck not installed; bash -n only"
fi
for f in "$MM" "$CS" "$SH/lib/named_mounts.sh"; do check "bash -n $(basename "$f")" bash -n "$f"; done

# ---- fake project config, stubs
MT="$T/mounts"; : > "$MT"
cat > "$T/proj/slurmhelps.json" <<'EOF'
{"cluster": {"ssh": "fakehost", "mirror": "/srv/mirror"},
 "mount": {"mountpoint": "~/mnt/mirror"},
 "mounts": {"test": {"remote": "me@fakehost:/srv/x", "mountpoint": "~/mnt/t", "mode": "rw"},
            "ro-view": {"remote": "me@fakehost:/srv/v", "mountpoint": "~/mnt/v", "mode": "ro"},
            "badmode": {"remote": "me@fakehost:/srv/x", "mountpoint": "~/mnt/b", "mode": "rx"},
            "relative": {"remote": "me@fakehost:srv/x", "mountpoint": "~/mnt/r"}}}
EOF
cat > "$T/stubs/sshfs" <<EOF
#!/usr/bin/env bash
echo "\$*" >> "$T/sshfs.argv"
printf 'sshfs#%s %s fuse.sshfs rw 0 0\n' "\$1" "\$2" >> "$MT"
EOF
cat > "$T/stubs/fusermount3" <<EOF
#!/usr/bin/env bash
echo "\$*" >> "$T/fusermount.argv"
mp="\${@: -1}"
awk -v m="\$mp" '\$2 != m' "$MT" > "$MT.tmp" && mv "$MT.tmp" "$MT"
EOF
cat > "$T/stubs/ssh" <<EOF
#!/usr/bin/env bash
echo "\$1" >> "$T/ssh.hosts"; shift
exec bash -c "\$*"
EOF
cat > "$T/stubs/rsync" <<EOF
#!/usr/bin/env bash
a=()
for x in "\$@"; do a+=("\${x#fakehost:}"); done
/usr/bin/rsync "\${a[@]}"; rc=\$?
# CORRUPT=<relpath>: damage one file on the "cluster" after a real transfer
if [ -n "\${CORRUPT:-}" ] && [ -f "$T/remote/dst/\$CORRUPT" ]; then echo corrupted >> "$T/remote/dst/\$CORRUPT"; fi
exit \$rc
EOF
chmod +x "$T"/stubs/*
SPATH="$T/stubs:/usr/bin:/bin"   # no sbatch on this PATH
B=(env -i HOME="$T/home" PATH="$SPATH" SLURMHELPS_PROJECT="$T/proj" SLURMHELPS_REMOTE_MOUNTS_FILE="$MT")
E=("${B[@]}" SLURMHELPS_REMOTE_ALLOW_CLUSTER=1)
M()  { "${E[@]}" bash "$MM" "$@"; }
C()  { "${E[@]}" bash "$CS" "$@"; }

# ---- refusal on the cluster (real environment, no override)
out="$(env SLURMHELPS_PROJECT="$T/proj" bash "$MM" --list 2>&1)"; rc=$?
check "named: refuses on this cluster node by default" bash -c "[ $rc = 1 ] && grep -q refusing <<<'$out'"
mkdir -p "$T/globproj"; echo '{"site": {"cluster_host_glob": "*"}}' > "$T/globproj/slurmhelps.json"
out="$("${B[@]}" SLURMHELPS_PROJECT="$T/globproj" bash "$MM" --list 2>&1)"; rc=$?
check "named: refuses on a host matching site.cluster_host_glob (no Slurm env)" bash -c "[ $rc = 1 ] && grep -q refusing <<<'$out'"
"${B[@]}" SLURM_JOB_ID=1 bash "$MM" --list >/dev/null 2>&1
check "named: refuses inside a Slurm job (SLURM_JOB_ID)" test $? = 1
out="$(env SLURMHELPS_PROJECT="$T/proj" bash "$CS" "$T/proj" /tmp/x 2>&1)"; rc=$?
check "code_sync: refuses on this cluster node by default" bash -c "[ $rc = 1 ] && grep -q refusing <<<'$out'"

# ---- the project mirror mount (no NAME): behaviour unchanged
"${B[@]}" bash "$MM" > "$T/compat.out" 2>&1; rc=$?
check "compat: no-name mount calls sshfs with the original arguments" grep -qx "fakehost:/srv/mirror $T/home/mnt/mirror -o BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o idmap=user,follow_symlinks" "$T/sshfs.argv"
out="$("${B[@]}" bash "$MM" status 2>&1)"; rc=$?
check "compat: status reads /proc/mounts as before (not mounted -> 1)" bash -c "[ $rc = 1 ] && [ '$out' = 'not mounted: $T/home/mnt/mirror' ]"
out="$("${B[@]}" bash "$MM" umount 2>&1)"; rc=$?
check "compat: umount when not mounted -> 0, same message" bash -c "[ $rc = 0 ] && [ '$out' = 'not mounted: $T/home/mnt/mirror' ]"
: > "$T/sshfs.argv"; : > "$MT"

# ---- argument / config errors
out="$(M --list | awk '{print $1}' | sort | tr '\n' ' ')"
check "--list prints the project's mounts" test "$out" = "badmode relative ro-view test "
M nosuch >/dev/null 2>&1; check "unknown mount name -> exit 2" test $? = 2
M badmode >/dev/null 2>&1; check "bad mode rejected" test $? = 2
M relative >/dev/null 2>&1; check "relative remote dir rejected" test $? = 2
M test frobnicate >/dev/null 2>&1; check "unknown action -> exit 2" test $? = 2
out="$("${E[@]}" SSHFS=/nonexistent/sshfs bash "$MM" test 2>&1)"; rc=$?
check "missing sshfs reported" bash -c "[ $rc = 1 ] && grep -q 'sshfs not found' <<<'$out'"

# ---- named mount / status / umount
M test status >/dev/null; check "status before mount -> 1" test $? = 1
M test mount >/dev/null 2>&1; check "mount -> 0" test $? = 0
check "mountpoint ~ expanded and created" test -d "$T/home/mnt/t"
check "sshfs got remote + mountpoint" grep -q "^me@fakehost:/srv/x $T/home/mnt/t -o " "$T/sshfs.argv"
check "options: reconnect + keepalive + BatchMode" grep -q 'BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3' "$T/sshfs.argv"
check "options: no allow_other; rw has a short dcache, no ro" bash -c "! grep -q allow_other '$T/sshfs.argv' && grep -q 'dcache_timeout=5' '$T/sshfs.argv' && ! grep -q ',ro,' '$T/sshfs.argv'"
M test >/dev/null; check "second mount is a no-op (sshfs called once)" test "$(wc -l < "$T/sshfs.argv")" = 1
M test status >/dev/null; check "status mounted -> 0" test $? = 0
chmod 000 "$T/home/mnt/t"; M test status >/dev/null; rc=$?; chmod 755 "$T/home/mnt/t"
check "status on an unresponsive mount -> 2 (stale)" test "$rc" = 2
M ro-view >/dev/null; check "ro mount passes -o ro" bash -c "tail -1 '$T/sshfs.argv' | grep -q ',ro,'"
M ro_view status >/dev/null; check "'_' finds the '-' named mount" test $? = 0
M test umount >/dev/null; check "umount -> fusermount3 -u" grep -qx -- "-u $T/home/mnt/t" "$T/fusermount.argv"
M test status >/dev/null; check "status after umount -> 1" test $? = 1
M test umount >/dev/null; check "umount when not mounted -> 0" test $? = 0
M ro-view umount --lazy >/dev/null; check "umount --lazy -> fusermount3 -u -z" grep -qx -- "-u -z $T/home/mnt/v" "$T/fusermount.argv"
"${E[@]}" SH_SSHFS_OPTS=allow_other bash "$MM" test >/dev/null 2>&1
check "allow_other refused" test $? = 1
echo x > "$T/home/mnt/t/stray"; M test >/dev/null 2>&1; rc=$?; rm -f "$T/home/mnt/t/stray"
check "non-empty mountpoint refused" test "$rc" = 1

# ---- code_sync.sh
SRC="$T/src"; DST="$T/remote/dst"
mkdir -p "$SRC/sub" "$SRC/.git" "$SRC/build"
echo a > "$SRC/a.py"; echo "b c" > "$SRC/sub/b c.txt"; echo g > "$SRC/.git/HEAD"; echo o > "$SRC/build/x.o"
C >/dev/null 2>&1; check "code_sync: missing args -> 1" test $? = 1
C "$SRC" rel/dir >/dev/null 2>&1; check "code_sync: relative DEST refused" test $? = 1
C "$SRC" "$DST" --dry-run >/dev/null 2>&1
check "code_sync --dry-run copies nothing" test ! -e "$DST/a.py"
: > "$T/ssh.hosts"
out="$(C "$SRC" "$DST" --exclude build/ 2>&1)"; rc=$?
check "code_sync -> 0 and verified" bash -c "[ $rc = 0 ] && grep -q '2 file(s) synced .* verified' <<<'$out'"
check "code_sync: default host = cluster.ssh from the project config" bash -c "[ \"\$(sort -u '$T/ssh.hosts')\" = fakehost ]"
check "code_sync: files arrived (incl. a name with a space)" bash -c "[ -f '$DST/a.py' ] && [ -f '$DST/sub/b c.txt' ]"
check "code_sync: .git excluded by default, --exclude honoured" bash -c "[ ! -e '$DST/.git' ] && [ ! -e '$DST/build' ]"
echo stale > "$DST/stale.txt"
C "$SRC" "$DST" --delete >/dev/null 2>&1
check "code_sync --delete removes a stale remote file" test ! -e "$DST/stale.txt"
echo a2 > "$SRC/a.py"
out="$("${E[@]}" CORRUPT=a.py bash "$CS" "$SRC" "$DST" 2>&1)"; rc=$?
check "code_sync: corrupted remote file -> exit 3 (checksum mismatch)" bash -c "[ $rc = 3 ] && grep -q 'CHECKSUM MISMATCH' <<<'$out'"
C "$SRC" "$DST" --no-default-excludes >/dev/null 2>&1
check "code_sync --no-default-excludes copies .git" test -f "$DST/.git/HEAD"
mkdir -p "$T/noproj"; echo '{}' > "$T/noproj/slurmhelps.json"
out="$("${E[@]}" SLURMHELPS_PROJECT="$T/noproj" bash "$CS" "$SRC" "$DST" 2>&1)"; rc=$?
check "code_sync: no --host and no cluster.ssh -> error" bash -c "[ $rc = 1 ] && grep -q 'no host' <<<'$out'"
: > "$T/ssh.hosts"
"${E[@]}" SLURMHELPS_PROJECT="$T/noproj" bash "$CS" "$SRC" "$DST" --host fakehost >/dev/null 2>&1
check "code_sync: --host used" bash -c "[ \"\$(sort -u '$T/ssh.hosts')\" = fakehost ]"

echo
echo "== summary: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
