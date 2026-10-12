# shellcheck shell=bash
# Named sshfs mounts for bin/mirror_mount.sh (SOURCED; workstation side only).
# Source: the project config "mounts" object (lib/slurmhelps/config.py mounts).
# Also used by bin/code_sync.sh for the cluster-node refusal.

nm_die() { echo "mirror_mount.sh: $*" >&2; exit "${2:-1}"; }

# nm_cfg KEY.PATH — a project config value ("" if unset).
nm_cfg() { python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" get "$1" 2>/dev/null || true; }

# nm_on_cluster — true on a cluster login/compute node (/work is local there).
nm_on_cluster() {
  [ "${SLURMHELPS_REMOTE_ALLOW_CLUSTER:-0}" = 1 ] && return 1
  [ -n "${SLURM_JOB_ID:-}" ] && return 0
  command -v sbatch >/dev/null 2>&1 && return 0
  local h glob
  glob="$(nm_cfg site.cluster_host_glob)"
  h="$(hostname -f 2>/dev/null || hostname)"
  # shellcheck disable=SC2053
  [ -n "$glob" ] && [[ "$h" == $glob ]] && return 0
  return 1
}

nm_refuse_on_cluster() {
  if nm_on_cluster; then
    nm_die "refusing: this looks like a cluster node ($(hostname -f 2>/dev/null || hostname)); /work is local here. Run this on your workstation."
  fi
}

# nm_project_mounts — name<TAB>remote<TAB>mountpoint<TAB>mode from the project config.
nm_project_mounts() { python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" mounts; }

nm_mounted() {
  local tab="${SLURMHELPS_REMOTE_MOUNTS_FILE:-/proc/mounts}"
  [ -r "$tab" ] || return 1
  awk -v m="${1// /\\040}" '$2 == m { f = 1 } END { exit !f }' "$tab"
}

named_main() {
  local rows name key row remote mp mode src action
  nm_refuse_on_cluster
  rows="$(nm_project_mounts | sed 's/^/project\t/')" || nm_die "cannot read the project config" 2

  case "${1:-}" in
    --list)
      [ -n "$rows" ] && printf '%s\n' "$rows" | awk -F'\t' '{ printf "%-20s %s  %s  %s\n", $2, $3, $4, $5 }'
      return 0 ;;
    -h|--help)
      sed -n '2,36p' "$SLURMHELPS_HOME/bin/mirror_mount.sh"; return 0 ;;
  esac

  name="$1"; action="${2:-mount}"
  key="${name//-/_}"
  # Names compare with '-' read as '_'.
  row="$(printf '%s\n' "$rows" | awk -F'\t' -v k="$key" '{ n = $2; gsub(/-/, "_", n) } n == k { print; exit }')"
  [ -n "$row" ] || nm_die "no mount '$name' in the project config 'mounts'; see: mirror_mount.sh --list" 2
  IFS=$'\t' read -r src _ remote mp mode <<<"$row"
  [ -n "$remote" ] && [ -n "$mp" ] || nm_die "$src mount '$name' needs a remote and a mountpoint" 2
  case "$mode" in rw|ro) ;; *) nm_die "$src mount '$name': mode must be rw or ro (got '$mode')" 2 ;; esac
  case "$remote" in *:/*|*:~*) ;; *) nm_die "$src mount '$name': remote must be host:/absolute/dir (got $remote)" 2 ;; esac
  case "$mp" in "~") mp="$HOME" ;; "~/"*) mp="$HOME/${mp#\~/}" ;; esac
  case "$mp" in /*) ;; *) nm_die "$src mount '$name': mountpoint must be absolute or ~/... (got $mp)" 2 ;; esac
  mp="${mp%/}"

  case "$action" in
    mount)
      if nm_mounted "$mp"; then echo "already mounted: $mp"; return 0; fi
      local sshfs fm opts
      sshfs="${SSHFS:-$(command -v sshfs || echo "$HOME/.local/bin/sshfs")}"
      [ -x "$sshfs" ] || nm_die "sshfs not found (PATH or ~/.local/bin); see the no-sudo install in: mirror_mount.sh --help"
      fm="${FUSERMOUNT:-$(command -v fusermount3 || true)}"
      [ -n "$fm" ] || nm_die "fusermount3 not found; install the system fuse3 package"
      mkdir -p "$mp"
      [ -z "$(ls -A "$mp" 2>/dev/null)" ] || nm_die "$mp is not empty; refusing to mount over it"
      opts="BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,idmap=user,follow_symlinks,dir_cache=yes"
      if [ "$mode" = ro ]; then opts="$opts,ro,dcache_timeout=60"; else opts="$opts,dcache_timeout=5"; fi
      [ -n "${SH_SSHFS_OPTS:-}" ] && opts="$opts,$SH_SSHFS_OPTS"
      case ",$opts," in *,allow_other,*) nm_die "allow_other is not allowed" ;; esac
      "$sshfs" "$remote" "$mp" -o "$opts" || nm_die "$name: sshfs $remote $mp failed"
      echo "mounted: $mp -> $remote ($mode)" ;;
    status)
      if ! nm_mounted "$mp"; then echo "not mounted: $mp"; return 1; fi
      if ! timeout 10 ls "$mp" >/dev/null 2>&1; then
        echo "STALE: $mp (not answering); run: mirror_mount.sh $name umount --lazy"; return 2
      fi
      echo "mounted: $mp -> $remote ($mode)" ;;
    umount|unmount)
      if ! nm_mounted "$mp"; then echo "not mounted: $mp"; return 0; fi
      local fm
      fm="${FUSERMOUNT:-$(command -v fusermount3 || true)}"
      [ -n "$fm" ] || nm_die "fusermount3 not found"
      if [ "${3:-}" = --lazy ]; then
        "$fm" -u -z "$mp" || nm_die "fusermount3 -u -z $mp failed"
      else
        "$fm" -u "$mp" || nm_die "fusermount3 -u $mp failed (busy? or use --lazy)"
      fi
      echo "unmounted: $mp" ;;
    *)
      nm_die "unknown action '$action' (mount|status|umount)" 2 ;;
  esac
}
