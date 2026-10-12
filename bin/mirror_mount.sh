#!/usr/bin/env bash
# Mount the project's cluster mirror read-write via user-space sshfs (FUSE via the system fusermount3,
# no sudo). Remote and mountpoint come from the calling project's slurmhelps.json:
#   mount.remote      (default cluster.ssh:cluster.mirror)
#   mount.mountpoint  (e.g. ~/mnt/troja-whitebox; $MIRROR_MNT overrides)
# sshfs: `command -v sshfs`, else ~/.local/bin/sshfs (extracted from the distro .deb); $SSHFS overrides.
#
#   ../slurmhelps/bin/mirror_mount.sh            # mount (no-op if already mounted)
#   ../slurmhelps/bin/mirror_mount.sh status     # print whether it is mounted
#   ../slurmhelps/bin/mirror_mount.sh umount     # unmount
#
# Code for jobs is still synced with rsync (bin/hpc_bridge_gpu_queue.py, steward rsync, or
# bin/code_sync.sh): editing job code through the mount while a job runs would change it mid-run.
#
# NAMED MOUNTS (workstation side only; refused on a cluster node, where /work is local):
#   ../slurmhelps/bin/mirror_mount.sh --list                    # every named mount
#   ../slurmhelps/bin/mirror_mount.sh NAME [mount|status|umount [--lazy]]
# Named mounts come from slurmhelps.json "mounts": {"NAME": {"remote": "user@host:/dir",
#   "mountpoint": "~/mnt/x", "mode": "rw"|"ro"}}   ('-' and '_' in NAME are interchangeable).
# Named mounts never pass allow_other, refuse a non-empty mountpoint, and mount `ro` read-only.
# status exits 0 mounted, 1 not mounted, 2 stale (not answering).
# No-sudo sshfs on Debian/Ubuntu (system fuse3 provides fusermount3 + libfuse3):
#   cd "$(mktemp -d)" && apt-get download sshfs && dpkg -x sshfs_*.deb ~/.local/pkg-sshfs
#   mkdir -p ~/.local/bin && ln -sf ~/.local/pkg-sshfs/usr/bin/sshfs ~/.local/bin/sshfs
# The workset view (images/view on the cluster) is a squashfuse mount on ONE login node, so its
# "mounts" entry must name that pinned host (site.login_host_pinned, e.g. hpc-head), not the
# round-robin login alias. Cluster-node refusal: SLURM_JOB_ID set, sbatch on PATH, or the hostname
# matching site.cluster_host_glob.
# Test hooks: SLURMHELPS_REMOTE_MOUNTS_FILE (mount table, default /proc/mounts),
# SLURMHELPS_REMOTE_ALLOW_CLUSTER=1 (skip the cluster-node refusal).
set -euo pipefail
SLURMHELPS_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "${1:-mount}" in
  mount|status|umount|unmount) ;;  # the project mirror mount, unchanged below
  *)
    # shellcheck source=../lib/named_mounts.sh
    . "$SLURMHELPS_HOME/lib/named_mounts.sh"
    named_main "$@"
    exit $? ;;
esac
eval "$(python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" shell)"
REMOTE=$SH_MOUNT_REMOTE
MNT="${MIRROR_MNT:-${TROJA_MNT:-$SH_MOUNT_POINT}}"
[ -n "$REMOTE" ] && [ -n "$MNT" ] || { echo "mirror_mount.sh: set mount.mountpoint and cluster.ssh/mirror in slurmhelps.json" >&2; exit 2; }
SSHFS="${SSHFS:-$(command -v sshfs || echo "$HOME/.local/bin/sshfs")}"

mounted() { grep -qs " $MNT fuse.sshfs " /proc/mounts; }

case "${1:-mount}" in
  status)
    if mounted; then echo "mounted: $MNT"; else echo "not mounted: $MNT"; exit 1; fi ;;
  umount|unmount)
    if mounted; then fusermount3 -u "$MNT" && echo "unmounted: $MNT"; else echo "not mounted: $MNT"; fi ;;
  mount)
    if mounted; then echo "already mounted: $MNT"; exit 0; fi
    mkdir -p "$MNT"
    "$SSHFS" "$REMOTE" "$MNT" \
      -o BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
      -o idmap=user,follow_symlinks
    mounted && echo "mounted: $MNT -> $REMOTE" ;;
  *)
    echo "usage: $0 [mount|status|umount]" >&2; exit 2 ;;
esac
