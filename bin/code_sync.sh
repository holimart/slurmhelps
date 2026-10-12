#!/usr/bin/env bash
# Push job code to the cluster and VERIFY it there (workstation side; refused on a cluster node).
# rsync -a SRC/ to HOST:DEST/, with a local sha256 manifest of exactly the files rsync sends,
# then `sha256sum -c` of that manifest on the cluster. For code a job will run; read/write
# outputs in place through bin/mirror_mount.sh instead.
#
#   ../slurmhelps/bin/code_sync.sh SRC DEST [--host USER@HOST] [--delete] [--dry-run]
#                                  [--exclude PATTERN]... [--no-default-excludes]
#
# DEST: absolute cluster dir. Host: --host, else cluster.ssh from the project's slurmhelps.json.
# Default exclude: .git/. Exit 0 synced+verified, 1 error, 3 checksum mismatch on the cluster.
set -uo pipefail
SLURMHELPS_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/named_mounts.sh
. "$SLURMHELPS_HOME/lib/named_mounts.sh"
die() { echo "code_sync.sh: $*" >&2; exit 1; }

src="" dest="" host="" delete=0 dry=0 defaults=1
excludes=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --host) host="${2:?--host needs a value}"; shift 2 ;;
    --delete) delete=1; shift ;;
    --dry-run|-n) dry=1; shift ;;
    --exclude) excludes+=("${2:?--exclude needs a pattern}"); shift 2 ;;
    --no-default-excludes) defaults=0; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) if [ -z "$src" ]; then src="$1"; elif [ -z "$dest" ]; then dest="$1"; else die "too many arguments"; fi; shift ;;
  esac
done
[ -n "$src" ] && [ -n "$dest" ] || die "usage: code_sync.sh SRC DEST [--host USER@HOST] [--delete] [--dry-run] [--exclude P]..."
nm_on_cluster && die "refusing: this looks like a cluster node; code_sync.sh runs on the workstation"
[ -d "$src" ] || die "source $src is not a directory"
case "$dest" in /*) ;; *) die "DEST must be an absolute cluster path (got $dest)" ;; esac
if [ -z "$host" ]; then
  host="$(python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" get cluster.ssh 2>/dev/null || true)"
fi
[ -n "$host" ] || die "no host (--host, or cluster.ssh in slurmhelps.json)"
[ "$defaults" = 1 ] && excludes=(".git/" ${excludes[@]+"${excludes[@]}"})

rargs=(-a)
for e in ${excludes[@]+"${excludes[@]}"}; do rargs+=(--exclude "$e"); done
[ "$delete" = 1 ] && rargs+=(--delete)
src="${src%/}/"

if [ "$dry" = 1 ]; then
  rsync "${rargs[@]}" -n -i "$src" "$host:$dest/"
  exit $?
fi

tmp="$(mktemp -d)" || die "mktemp failed"
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/empty"
# The files rsync sends = what a dry run into an empty dir would create (dirs end in '/').
rsync "${rargs[@]}" -n --out-format='%n' "$src" "$tmp/empty/" > "$tmp/all" || die "rsync dry run failed"
grep -v '/$' "$tmp/all" > "$tmp/files" || true
( cd "$src" && tr '\n' '\0' < "$tmp/files" | xargs -0 -r sha256sum -- ) > "$tmp/manifest" \
  || die "local sha256sum failed"
n="$(wc -l < "$tmp/manifest")"

ssh "$host" "mkdir -p '$dest'" || die "ssh $host mkdir $dest failed"
rsync "${rargs[@]}" "$src" "$host:$dest/" || die "rsync to $host:$dest failed"
if ! ssh "$host" "cd '$dest' && sha256sum -c --quiet -" < "$tmp/manifest"; then
  echo "code_sync.sh: CHECKSUM MISMATCH on $host:$dest (see lines above)" >&2
  exit 3
fi
echo "code_sync.sh: $n file(s) synced to $host:$dest and verified (sha256)"
