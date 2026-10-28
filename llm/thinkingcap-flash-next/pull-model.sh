#!/usr/bin/env bash
# Fetch the pinned ThinkingCap Qwen3.8-Flash-Next NVFP4 checkpoint.
#
# Usage:
#   llm/thinkingcap-flash-next/pull-model.sh
#   THINKINGCAP_FORCE_PULL=1 llm/thinkingcap-flash-next/pull-model.sh
#
# The model is ~132 GB.  Set THINKINGCAP_MODEL to put it somewhere other than
# $WORK_ROOT/models/thinkingcap-flash-next-nvfp4.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

REPO=${THINKINGCAP_REPO:-bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4}
REV=${THINKINGCAP_REV:-df3922b21fc28a8604cc9b31451495690d155194}
DEST=${THINKINGCAP_MODEL:-$WORK_ROOT/models/thinkingcap-flash-next-nvfp4}
WORKERS=${THINKINGCAP_WORKERS:-4}

if [ -s "$DEST/config.json" ] && [ -s "$DEST/model.safetensors.index.json" ] \
   && [ "${THINKINGCAP_FORCE_PULL:-0}" != 1 ]; then
  printf '[thinkingcap-pull] existing checkpoint retained: %s\n' "$DEST"
  du -sh "$DEST"
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli)
else
  echo "thinkingcap-pull: need 'hf' or 'huggingface-cli' on PATH" >&2
  exit 2
fi

mkdir -p "$DEST"
df -hT "$DEST" || true
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-}" = "" ] \
   && python3 -c 'import hf_transfer' >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
export HF_HUB_DISABLE_TELEMETRY=1

printf '[thinkingcap-pull] repo=%s revision=%s dest=%s\n' "$REPO" "$REV" "$DEST"
"${HF[@]}" download "$REPO" --revision "$REV" --local-dir "$DEST" \
  --max-workers "$WORKERS"

[ -s "$DEST/config.json" ] || { echo "thinkingcap-pull: missing config.json" >&2; exit 1; }
[ -s "$DEST/model.safetensors.index.json" ] || { echo "thinkingcap-pull: missing index" >&2; exit 1; }
du -sh "$DEST"
printf '[thinkingcap-pull] done: %s\n' "$DEST"
