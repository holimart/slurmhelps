#!/usr/bin/env bash
# fetch-models.sh — pull the Strata benchmark GGUFs into the model library.
#
# Thin helper: runs `hf download` (resumable) for the two Qwen3.8-Flash-Next
# GGUF packs the Strata benchmark matrix measures, and writes a manifest
# (repo, file, size, sha256) per fetched file. Sizes and SHA-256 are the
# Hub LFS pointers pinned in the Strata source checkout
# (/work/ffa/holecem/src/Strata/setup.py, ISTA_FILES / CODER_FILES, repos at
# revisions 2026-09-29). Strata reads shard 1 (experts, RAM) plus shard 2
# (the ~28.8 GB n-gram/PLE table, kept on SSD) at every start, so BOTH
# shards are required per pack. The picture reader (mmproj) is only fetched
# with --with-mmproj; the benchmark runs with --vision off.
#
# Usage:
#   fetch-models.sh [--only iq3s|coder|all] [--verify] [--with-mmproj] [--submit]
#
#   --only X        fetch just this pack (default: all)
#   --verify        audit only: check presence + exact size + sha256 of every
#                   expected file. Read-only — never downloads, never writes
#                   the manifest, safe to run while the download job owns the
#                   library (missing files report "missing" and exit 0; only
#                   size/sha mismatch on a present file exits 1)
#   --with-mmproj   also fetch mmproj-Qwen3.8-Flash-Next-BF16.gguf (~0.9 GB)
#   --submit        wrap the run in `sbatch --nice=100` on a CPU partition
#                   instead of downloading here (runs on the login node too,
#                   it is pure network + IO, no GPU)
#
# Layout produced (bench-matrix.sh expects exactly this):
#   /work/ffa/holecem/models/strata/iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000[12]-of-00002.gguf
#   /work/ffa/holecem/models/strata/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000[12]-of-00002.gguf
#   /work/ffa/holecem/models/strata/manifest.txt   (append-only, one key=value line per file)
#
# Model download is a network operation; nothing in this script loads a model.
set -euo pipefail

MODELS_ROOT=${MODELS_ROOT:-/work/ffa/holecem/models/strata}
LOG_DIR=${LOG_DIR:-/work/ffa/holecem/logs}
MANIFEST=$MODELS_ROOT/manifest.txt
SUBMIT_PARTITION=${PARTITION:-ffa}

ONLY=all
VERIFY=0
WITH_MMPROJ=0
SUBMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=${2:?--only needs iq3s|coder|all}; shift 2 ;;
    --only=*) ONLY=${1#*=}; shift ;;
    --verify) VERIFY=1; shift ;;
    --with-mmproj) WITH_MMPROJ=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "fetch-models: unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ONLY" in iq3s|coder|all) ;; *) echo "fetch-models: --only must be iq3s|coder|all" >&2; exit 2 ;; esac

if [ "$SUBMIT" = 1 ] && [ "$VERIFY" != 1 ]; then
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  exec sbatch --nice=100 --job-name=strata-fetch --partition="$SUBMIT_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=11:00:00 \
    --output="$LOG_DIR/strata-fetch-%j.log" --wrap="bash '$here/fetch-models.sh' --only '$ONLY' $( [ "$VERIFY" = 1 ] && echo --verify ) $( [ "$WITH_MMPROJ" = 1 ] && echo --with-mmproj )"
fi

if [ "$VERIFY" != 1 ]; then
  [ -d "$LOG_DIR" ] || mkdir -p "$LOG_DIR"
  mkdir -p "$MODELS_ROOT"
fi

log() { printf '[strata-fetch] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
[ "$SUBMIT" = 1 ] && [ "$VERIFY" = 1 ] && log "audit mode ignores --submit; running locally"

hf_bin() {
  if command -v hf >/dev/null 2>&1; then echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then echo huggingface-cli
  else echo "fetch-models: neither hf nor huggingface-cli on PATH" >&2; exit 3; fi
}

# repo file expected_bytes expected_sha256 — values from
# /work/ffa/holecem/src/Strata/setup.py (HF_REVISIONS pinned 2026-09-29).
files_iq3s() {
  echo "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf 54817524224 4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"
  echo "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf 28800138432 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
}
files_coder() {
  echo "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf 29608446496 e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"
  echo "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf 28800138432 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
}
mmproj() {
  echo "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF mmproj-Qwen3.8-Flash-Next-BF16.gguf 907543008 b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
}

manifest_line() { # repo file size sha status
  printf 'ts=%s repo=%s file=%s size=%s sha256=%s status=%s path=%s\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "$5" "$6" >> "$MANIFEST"
}

fetch_pack() { # variant_name subdir file_fn
  local name=$1 sub=$2 fn=$3 HF rc repo file want_size want_sha path got
  HF=$(hf_bin)
  mkdir -p "$MODELS_ROOT/$sub"
  while read -r repo file want_size want_sha; do
    [ -n "$repo" ] || continue
    path=$MODELS_ROOT/$sub/$file
    got=""
    if [ -f "$path" ]; then got=$(stat -c %s "$path" 2>/dev/null || echo 0); fi
    if [ "$got" = "$want_size" ]; then
      log "$name: $file already complete ($want_size bytes), skip download"
      manifest_line "$repo" "$file" "$want_size" "$want_sha" pinned "$path"
      continue
    fi
    log "$name: hf download $repo $file -> $path"
    # hf download is resumable; --local-dir keeps the exact Hub filename.
    "$HF" download "$repo" "$file" --local-dir "$MODELS_ROOT/$sub" || rc=$?
    rc=${rc:-1}
    got=$(stat -c %s "$path" 2>/dev/null || echo 0)
    if [ "$got" != "$want_size" ]; then
      log "$name: FAILED $file size $got != $want_size (rc=$rc)"
      manifest_line "$repo" "$file" "$got" "$want_sha" size-mismatch "$path"
      return 1
    fi
    manifest_line "$repo" "$file" "$want_size" "$want_sha" pinned "$path"
  done < <("$fn")
  log "$name: complete in $MODELS_ROOT/$sub"
}

# audit mode (--verify): read-only over the library. Checks presence, exact
# size, sha256. Never downloads, never writes the manifest — safe to run
# while the download job owns the library. Looks in the canonical subdir and
# the fetch job's alternate dir; a file present in either counts.
audit_pack() { # variant_name file_fn dir [dir...]
  local name=$1 fn=$2; shift 2
  local dirs=("$@") repo file want_size want_sha path d got real
  while read -r repo file want_size want_sha; do
    [ -n "$repo" ] || continue
    path=""
    for d in "${dirs[@]}"; do
      if [ -f "$d/$file" ]; then path=$d/$file; break; fi
    done
    if [ -z "$path" ]; then
      log "AUDIT $name: $file -> MISSING (looked in ${dirs[*]})"
      AUDIT_MISSING=$((AUDIT_MISSING + 1))
      continue
    fi
    got=$(stat -c %s "$path" 2>/dev/null || echo 0)
    if [ "$got" != "$want_size" ]; then
      log "AUDIT $name: $file -> SIZE-MISMATCH in $path: got $got want $want_size"
      AUDIT_BAD=$((AUDIT_BAD + 1))
      continue
    fi
    log "AUDIT $name: $file -> hashing $want_size bytes ($path)"
    real=$(sha256sum "$path" | cut -d' ' -f1)
    if [ "$real" = "$want_sha" ]; then
      log "AUDIT $name: $file -> VERIFIED sha256 $real"
    else
      log "AUDIT $name: $file -> SHA-MISMATCH got $real want $want_sha"
      AUDIT_BAD=$((AUDIT_BAD + 1))
    fi
  done < <("$fn")
}

if [ "$VERIFY" = 1 ]; then
  AUDIT_MISSING=0; AUDIT_BAD=0
  if [ "$ONLY" = all ] || [ "$ONLY" = iq3s ]; then
    audit_pack iq3s files_iq3s "$MODELS_ROOT/iq3s" "$MODELS_ROOT/models/IQ3_S"
    [ "$WITH_MMPROJ" = 1 ] && audit_pack iq3s-mmproj mmproj "$MODELS_ROOT/iq3s" "$MODELS_ROOT/models/IQ3_S"
  fi
  if [ "$ONLY" = all ] || [ "$ONLY" = coder ]; then
    audit_pack coder files_coder "$MODELS_ROOT/coder" "$MODELS_ROOT/models/coder-IQ1_M"
    [ "$WITH_MMPROJ" = 1 ] && audit_pack coder-mmproj mmproj "$MODELS_ROOT/coder" "$MODELS_ROOT/models/coder-IQ1_M"
  fi
  log "audit complete: missing=$AUDIT_MISSING corrupt=$AUDIT_BAD (library $MODELS_ROOT)"
  if [ "$AUDIT_BAD" != 0 ]; then
    log "one or more present files are corrupt; re-download them (fetch-models.sh)"
    exit 1
  fi
  exit 0
fi

fail=0
if [ "$ONLY" = all ] || [ "$ONLY" = iq3s ]; then
  fetch_pack iq3s iq3s files_iq3s || fail=1
  if [ "$fail" = 0 ] && [ "$WITH_MMPROJ" = 1 ]; then fetch_pack iq3s-mmproj iq3s mmproj || fail=1; fi
fi
if [ "$ONLY" = all ] || [ "$ONLY" = coder ]; then
  fetch_pack coder coder files_coder || fail=1
  if [ "$fail" = 0 ] && [ "$WITH_MMPROJ" = 1 ]; then fetch_pack coder-mmproj coder mmproj || fail=1; fi
fi

[ -f "$MANIFEST" ] && tail -n 8 "$MANIFEST"
if [ "$fail" = 1 ]; then
  log "one or more files incomplete or mismatched; see $MANIFEST"
  exit 1
fi
log "done; library under $MODELS_ROOT"
