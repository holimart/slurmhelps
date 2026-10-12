#!/bin/bash
# =============================================================================
#  fuse-bench.sh REPO [LUSTRE_SCRATCH_DIR] [BRANCH] — fuse2fs vs tmpfs vs the
#  shared filesystem: metadata, git, concurrency, image copy throughput.
# =============================================================================
#  Source of truth: slurmhelps maint/fuse-bench.sh. Run in a Slurm job (node
#  /tmp). REPO is cloned (read-only for it); LUSTRE_SCRATCH_DIR (default
#  <work_root>/tmp/fuse-bench) receives one temporary 20 GB sparse image.
# =============================================================================
export GIT_OPTIONAL_LOCKS=0
REPO="${1:?usage: fuse-bench.sh REPO [LUSTRE_SCRATCH_DIR] [BRANCH]}"
BRANCH="${3:-$(git -C "$REPO" symbolic-ref -q --short HEAD)}"
T=/tmp/fb-$USER; rm -rf $T; mkdir -p $T/mnt $T/plain
L="${2:-${SH_WORK_ROOT:-${WORK_ROOT:?set WORK_ROOT or pass LUSTRE_SCRATCH_DIR}}/tmp/fuse-bench}"; mkdir -p $L
t() { local s=$(date +%s.%N); "$@" >/dev/null 2>&1; printf '%8.2fs  %s\n' "$(echo "$(date +%s.%N) - $s" | bc)" "$*"; }
smallfiles() { mkdir -p "$1/sf"; for i in $(seq 5000); do echo x > "$1/sf/f$i"; done; }
statall()    { find "$1/sf" -type f | xargs stat >/dev/null; }
readall()    { cat "$1"/sf/* >/dev/null; }
rmall()      { rm -rf "$1/sf"; }
gitwork() { cd "$1/repo" && git status --porcelain && git log --oneline -2000 >/dev/null && git grep -q zzzznotfound || true; }
par4()    { for k in 1 2 3 4; do ( cd "$1/repo" && git status --porcelain >/dev/null ) & done; wait; }

echo "== setup: 20G sparse ext4 image (no journal) on /tmp, fuse2fs mount"
truncate -s 20G $T/a.img; mkfs.ext4 -q -F -O ^has_journal $T/a.img
fuse2fs $T/a.img $T/mnt -o rw,fakeroot || { echo FUSE FAILED; exit 1; }
for tgt in $T/plain $T/mnt; do
  echo "==== target: $tgt"
  t smallfiles $tgt; t statall $tgt; t readall $tgt; t rmall $tgt
  t git clone -q --no-local --branch "$BRANCH" "$REPO" $tgt/repo
  echo "   files: $(cd $tgt/repo && git ls-files | wc -l)"
  t gitwork $tgt; t gitwork $tgt   # second = warm
  t par4 $tgt
  t dd if=/dev/zero of=$tgt/big bs=4M count=512 conv=fsync
  t dd if=$tgt/big of=/dev/null bs=4M
  rm -f $tgt/big
done
echo "==== shared-filesystem baseline (read-only, the REPO checkout itself)"
cd "$REPO" && { t git status --porcelain; t git status --porcelain; }
echo "==== image copy: unmount, copy /tmp -> Lustre (striped) -> /tmp"
fusermount -u $T/mnt; e2fsck -fn $T/a.img | tail -1
du -h --apparent-size $T/a.img | cut -f1; du -h $T/a.img | cut -f1
lfs setstripe -c 8 $L/a.img 2>/dev/null
t cp --sparse=always $T/a.img $L/a.img
t cp --sparse=always $L/a.img $T/b.img
ls -ls $L/a.img $T/b.img
rm -f $L/a.img; rmdir $L; rm -rf $T
