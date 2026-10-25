#!/usr/bin/env bash
# tests/test_fetch_models.sh — regression guard for fetch-models.sh.
#
# Uses a fake `hf` that creates *sparse* files of the pinned sizes, so this
# takes milliseconds and never touches the network.  It checks the two bugs
# fixed on 2026-10-25:
#   1. hf is called with IQ3_S/ or IQ1_M/ prefixed paths (bare names silently
#      fetch nothing and exit 0);
#   2. the identical 28,800,138,432-byte shard 2 is fetched once and hardlinked
#      into both pack dirs.
#
# Usage: tests/test_fetch_models.sh
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fetch=$here/../fetch-models.sh
root=$(mktemp -d "${TMPDIR:-/tmp}/strata-fetch-test.XXXXXX")
trap 'rm -rf "$root"' EXIT

bin=$root/bin
models=$root/models
logdir=$root/logs
calls=$root/hf-calls.txt
mkdir -p "$bin"

cat > "$bin/hf" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
[ "$1" = download ] || { echo "fake hf: expected download" >&2; exit 2; }
repo=$2; hub=$3; shift 3
local_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --local-dir) local_dir=$2; shift 2 ;;
    *) shift ;;
  esac
done
echo "$repo $hub" >> "${HF_CALLS:?}"
[ "${HF_MODE:-create}" = "noop" ] && exit 0
case "$hub" in
  */Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf) size=54817524224 ;;
  */Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf) size=28800138432 ;;
  */Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf) size=29608446496 ;;
  */Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf) size=28800138432 ;;
  *) echo "fake hf: unknown hub path $hub" >&2; exit 3 ;;
esac
mkdir -p "$local_dir/$(dirname "$hub")"
truncate -s "$size" "$local_dir/$hub"
STUB
chmod +x "$bin/hf"

size() { stat -c %s "$1" 2>/dev/null || echo 0; }
inode() { stat -c %i "$1"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

PATH="$bin:$PATH" HF_CALLS="$calls" MODELS_ROOT="$models" LOG_DIR="$logdir" \
  "$fetch" --only all >/dev/null

# Expected flat layout + exact sizes.
for f in \
  iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf; do
  [ -f "$models/$f" ] || fail "missing $models/$f"
done
[ "$(size "$models/iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")" = 54817524224 ] \
  || fail "wrong IQ3_S shard1 size"
[ "$(size "$models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")" = 29608446496 ] \
  || fail "wrong IQ1_M shard1 size"

# Every hf call must carry a repo subdirectory prefix.
if awk '{print $2}' "$calls" | grep -vqE '^(IQ3_S|IQ1_M)/'; then
  fail "hf was called with a bare filename: $(cat "$calls")"
fi

# Shard 2 fetched exactly once, and both pack copies share its inode.
shared=$models/_shared/Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf
[ -f "$shared" ] || fail "missing canonical shared shard $shared"
[ "$(grep -c '00002-of-00002' "$calls")" = 1 ] \
  || fail "shard 2 was downloaded more than once: $(grep '00002-of-00002' "$calls")"
[ "$(inode "$shared")" = "$(inode "$models/iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")" ] \
  || fail "IQ3_S shard 2 is not hardlinked to _shared"
[ "$(inode "$shared")" = "$(inode "$models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf")" ] \
  || fail "Coder shard 2 is not hardlinked to _shared"

# A fake hf that exits 0 without creating the file must fail loudly.
root2=$root/noop
mkdir -p "$root2"
if PATH="$bin:$PATH" HF_CALLS="$root/noop-calls.txt" HF_MODE=noop \
   MODELS_ROOT="$root2/models" LOG_DIR="$root2/logs" \
   "$fetch" --only coder >"$root2/out.txt" 2>&1; then
  fail "fetch-models accepted an hf download that created nothing"
fi
grep -q 'does not exist' "$root2/out.txt" \
  || fail "missing-file failure did not explain the missing prefixed path"

echo "test_fetch_models: PASS (prefixed paths, one shard-2 download, hardlinks, loud failure)"
