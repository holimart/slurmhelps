#!/usr/bin/env bash
# install-engine.sh — install the Strata Terminal + the CUDA 13 and CUDA 12
# engine components into /work/ffa/holecem/strata-engine (idempotent).
#
# What this does, following the Strata docs (docs/INSTALL.md, docs/TERMINAL_LAUNCH.md,
# packaging/UPDATES.md, packaging/ENGINE-COMPONENT.md):
#
#  1. Runs the official one-line Terminal installer
#       curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
#     with STRATA_DATA_DIR/STRATA_BIN_DIR pointed at $STRATA_ROOT, so we get
#     $STRATA_ROOT/terminal/<version>/bin/strata  (+ .../terminal/current) and
#     $STRATA_ROOT/bin/strata. The installer verifies the Terminal archive's
#     SHA-256 against the release's install-manifest.txt and runs
#     `strata update --check` to validate the signed feed before keeping it.
#
#  2. Installs BOTH engine packages (variants cuda13 and cuda12) from the same
#     signed release feed into $STRATA_ROOT/components/engine/<version>-
#     linux-x86_64-cuda-<variant>/ — the exact layout Control scans
#     (strata_control/engine_component.py: scan()/resolve()). The variant
#     name in the folder plus the manifest's archs let Control pick per node:
#     every card >= sm_75 -> the CUDA 13 engine; a Pascal/Volta card (L40 is
#     sm_89, V100 is sm_70) -> only the CUDA 12 package covers it, so Volta
#     nodes automatically run the CUDA 12 engine (docs/OLDER_GPUS.md,
#     docs/NVIDIA_V100.md: "CUDA 13 cannot run Volta"). No --cuda flag needed.
#
#     Why not plain `strata update` here: the updater selects one package for
#     "this PC" via a live nvidia-smi probe and refuses to install engines on
#     a box with no supported GPU — the login node has none. We instead read
#     the release feed (strata-feed.json), download exactly the entries the
#     feed names, and verify each archive's SHA-256 against the feed's value
#     (the same check `strata update` performs before installing). Packages
#     stay signed; an unmodified official package loads as "unverified"
#     (packaging/ENGINE-COMPONENT.md signature policy).
#
#  3. Logs everything to /work/ffa/holecem/logs/strata-install.log.
#
# Run on the LOGIN NODE only (network install; no model loading, no GPU).
#
# Usage:  install-engine.sh [--force] [--only cuda13|cuda12|both]
set -euo pipefail

STRATA_ROOT=${STRATA_ROOT:-/work/ffa/holecem/strata-engine}
FEED=${STRATA_FEED:-https://github.com/Niko1221/Strata/releases/latest/download}
LOG=${LOG:-/work/ffa/holecem/logs/strata-install.log}
PLATFORM=linux-x86_64

ONLY=both
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --only) ONLY=${2:?--only needs cuda13|cuda12|both}; shift 2 ;;
    --only=*) ONLY=${1#*=}; shift ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install-engine: unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ONLY" in cuda13|cuda12|both) ;; *) echo "install-engine: --only must be cuda13|cuda12|both" >&2; exit 2 ;; esac

mkdir -p "$STRATA_ROOT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
log() { printf '[strata-install] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
[ -n "${SLURM_JOB_ID:-}" ] && log "WARNING: running inside Slurm job $SLURM_JOB_ID — this installer belongs on the login node"

FEED="${FEED%/}"
export STRATA_COMPONENTS_DIR="$STRATA_ROOT/components"
mkdir -p "$STRATA_COMPONENTS_DIR/engine" "$STRATA_ROOT/bin" "$STRATA_ROOT/dl" "$STRATA_ROOT/feed"

# ---- 1. Terminal (the `strata` CLI) ------------------------------------------
if [ -x "$STRATA_ROOT/bin/strata" ] && [ "$FORCE" != 1 ]; then
  log "Terminal already installed: $STRATA_ROOT/bin/strata ($( "$STRATA_ROOT/bin/strata" --version 2>/dev/null || echo version-unknown )); skip (use --force to redo)"
else
  log "fetching the one-line Terminal installer from $FEED/install.sh"
  tmpsh=$(mktemp "$STRATA_ROOT/dl/install.sh.XXXXXX")
  curl -fsSL --retry 3 -o "$tmpsh" "$FEED/install.sh"
  log "running install.sh with STRATA_DATA_DIR=$STRATA_ROOT STRATA_BIN_DIR=$STRATA_ROOT/bin"
  STRATA_DATA_DIR="$STRATA_ROOT" STRATA_BIN_DIR="$STRATA_ROOT/bin" sh "$tmpsh"
  rm -f "$tmpsh"
  [ -x "$STRATA_ROOT/bin/strata" ] || { log "FATAL: installer finished but $STRATA_ROOT/bin/strata is missing"; exit 4; }
fi
log "strata version: $("$STRATA_ROOT/bin/strata" --version 2>/dev/null || true)"

# ---- 2. Engine components from the signed feed --------------------------------
FEED_JSON=$STRATA_ROOT/feed/strata-feed.json
log "fetching $FEED/strata-feed.json"
curl -fsSL --retry 3 -o "$FEED_JSON.tmp" "$FEED/strata-feed.json"
mv "$FEED_JSON.tmp" "$FEED_JSON"

# emit one line "variant version url size sha256" per wanted cuda variant
pick=$(python3 - "$FEED_JSON" "$PLATFORM" "$ONLY" <<'PY'
import json, sys
feed = json.load(open(sys.argv[1]))
platform, want = sys.argv[2], sys.argv[3]
for p in feed.get("engine", {}).get("packages", []):
    if p.get("platform") != platform or p.get("backend") != "cuda":
        continue
    if want != "both" and p.get("variant") != want:
        continue
    print("\t".join([str(p.get(k, "")) for k in ("variant", "version", "url", "size", "sha256")]))
PY
)
[ -n "$pick" ] || { log "FATAL: feed names no linux-x86_64 CUDA engine packages"; exit 4; }

printf '%s\n' "$pick" | while IFS=$'\t' read -r variant version url size sha256; do
  [ -n "$variant" ] || continue
  dest="$STRATA_ROOT/components/engine/$version-$PLATFORM-cuda-$variant"
  if [ -f "$dest/engine-component.json" ] && [ "$FORCE" != 1 ]; then
    log "engine cuda-$variant $version already installed at $dest; skip"
    continue
  fi
  name=$(basename "$url")
  archive="$STRATA_ROOT/dl/$name"
  if [ ! -f "$archive" ] || [ "$(sha256sum "$archive" | cut -d' ' -f1)" != "$sha256" ]; then
    log "downloading engine cuda-$variant $version ($name, ${size:-?} bytes)"
    # GitHub release assets are flat; feed urls may carry a folder — try both.
    if ! curl -fsSL --retry 3 -C - -o "$archive.tmp" "$FEED/$url"; then
      curl -fsSL --retry 3 -C - -o "$archive.tmp" "$FEED/$name"
    fi
    mv "$archive.tmp" "$archive"
    got=$(sha256sum "$archive" | cut -d' ' -f1)
    if [ "$got" != "$sha256" ]; then
      log "FATAL: $name sha256 mismatch (got $got, feed says $sha256)"
      rm -f "$archive"
      exit 5
    fi
    log "  sha256 ok ($got)"
  else
    log "engine cuda-$variant archive already in dl/, sha256 pre-verified"
  fi
  rm -rf "$dest.incoming"; mkdir -p "$dest.incoming"
  tar -xzf "$archive" -C "$dest.incoming"
  # accept an archive with one top folder
  if [ ! -f "$dest.incoming/engine-component.json" ]; then
    inner=$(find "$dest.incoming" -mindepth 2 -maxdepth 2 -name engine-component.json -print -quit 2>/dev/null || true)
    if [ -n "$inner" ]; then
      top=$(dirname "$inner")
      mv "$top" "$dest.flat" && rm -rf "$dest.incoming" && mv "$dest.flat" "$dest.incoming"
    fi
  fi
  [ -f "$dest.incoming/engine-component.json" ] || { log "FATAL: $name unpacked without engine-component.json"; exit 5; }
  ep=$(python3 -c "import json;print(json.load(open('$dest.incoming/engine-component.json')).get('entrypoint','strata'))")
  [ -f "$dest.incoming/$ep" ] || { log "FATAL: package entrypoint $ep missing"; exit 5; }
  chmod +x "$dest.incoming/$ep" 2>/dev/null || true
  rm -rf "$dest"; mv "$dest.incoming" "$dest"
  log "installed $dest (entrypoint $ep)"
done

# ---- 3. Report ----------------------------------------------------------------
log "component inventory (quick check, no engine start):"
STRATA_COMPONENTS_DIR="$STRATA_COMPONENTS_DIR" "$STRATA_ROOT/bin/strata" components engine || true
log "done. Components live in $STRATA_COMPONENTS_DIR/engine; compute nodes pick the"
log "variant automatically from nvidia-smi archs (sm_70/sm_60 -> cuda12, else cuda13)."
log "next steps: llm/strata/fetch-models.sh (models), then sbatch --nice=100 llm/strata/strata-bench.sbatch"
