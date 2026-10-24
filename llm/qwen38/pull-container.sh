#!/usr/bin/env bash
# Build the Qwen3.8 Blackwell vLLM container on node-local scratch, then
# copy only the finished SIF to /work. Writing the SIF and extraction cache
# directly on Lustre was the slow path.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
SIF=${QWEN38_SIF:-$WORK_ROOT/images/qwen38-vllm.sif}
IMAGE=${QWEN38_IMAGE:-docker://ghcr.io/local-inference-lab/vllm:release-10152026}
SCRATCH=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
CACHE="$SCRATCH/qwen38-apptainer-cache"
TMP="$SCRATCH/qwen38-apptainer-tmp"
TARGET="$SCRATCH/qwen38-vllm.sif"
mkdir -p "$CACHE" "$TMP" "$(dirname "$SIF")"
if [ -s "$SIF" ] && [ "${QWEN38_FORCE_PULL:-0}" != 1 ]; then
  printf '[pull] existing SIF retained: %s\n' "$SIF"
  ls -lh "$SIF"
  sha256sum "$SIF"
  exit 0
fi
cleanup() { rm -rf "$CACHE" "$TMP" "$TARGET"; }
trap cleanup EXIT
export APPTAINER_CACHEDIR="$CACHE"
export APPTAINER_TMPDIR="$TMP"
export TMPDIR="$TMP"
printf '[pull] image=%s\n' "$IMAGE"
printf '[pull] scratch=%s\n' "$SCRATCH"
df -hT "$SCRATCH" "$(dirname "$SIF")" || true
apptainer pull --force --name "$TARGET" "$IMAGE"
ls -lh "$TARGET"
# Copy the finished SIF; this is a sequential write, not an extraction tree.
cp "$TARGET" "$SIF"
sync
ls -lh "$SIF"
sha256sum "$SIF"
