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
# Code for jobs is still synced with rsync (bin/hpc_bridge_gpu_queue.py, steward rsync): editing job
# code through the mount while a job runs would change it mid-run.
set -euo pipefail
SLURMHELPS_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
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
