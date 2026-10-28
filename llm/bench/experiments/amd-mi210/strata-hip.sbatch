#!/usr/bin/env bash
#SBATCH --job-name=amd-mi210-strata-hip
#SBATCH --partition=ffa-preempt
#SBATCH --nodelist=r50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/amd-mi210-strata-hip-%j.out
#
# Optional step 6: attempt a Strata HIP build for gfx90a.  Strata's HIP
# backend is documented for wave32 RDNA (gfx11/gfx12); MI210 is wave64 CDNA2,
# so either CMake, the build, or the runtime will refuse it.  A clean failure
# is the expected finding and is recorded verbatim.
set -uo pipefail

JOB=${SLURM_JOB_ID:?}
SRC_SHARED=${SRC_SHARED:-/work/ffa/holecem/src/Strata}
GGML_SHARED=${GGML_SHARED:-/work/ffa/holecem/strata-engine/install/third_party/llama.cpp}
OUT=/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210/strata-hip
LOCAL=/dev/shm/amd-mi210-strata-$JOB
mkdir -p "$LOCAL" "$OUT"
exec > >(tee "$LOCAL/strata-hip.log") 2>&1
log() { printf '[strata-hip] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "job=$JOB node=$(hostname)"
hipcc --version | head -2

t0=$(date +%s)
log "copy Strata + pinned llama.cpp"
cp -a "$SRC_SHARED" "$LOCAL/Strata" || exit 2
cp -a "$GGML_SHARED" "$LOCAL/llama.cpp" || exit 2
log "copy_s=$(( $(date +%s) - t0 ))"
ls "$LOCAL/Strata/third_party" 2>/dev/null | head
ls "$LOCAL/llama.cpp/src/llama-arch.cpp" >/dev/null && echo "pinned llama.cpp present"

CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=/usr/bin/gcc
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DSTRATA_ENABLE_HIP=ON
  -DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_GGML_DIR="$LOCAL/llama.cpp"
)
printf '%s\n' "cmake -S $LOCAL/Strata -B $LOCAL/build ${CMAKE_ARGS[*]}" > "$LOCAL/cmake-command.txt"
log "configure"
timeout 900 cmake -S "$LOCAL/Strata" -B "$LOCAL/build" "${CMAKE_ARGS[@]}" 2>&1 | tee "$LOCAL/configure.log"
cfg_rc=${PIPESTATUS[0]}
log "configure_rc=$cfg_rc"
if [ "$cfg_rc" -eq 0 ]; then
  log "build target strata"
  t1=$(date +%s)
  timeout 3000 ninja -C "$LOCAL/build" strata -j 32 2>&1 | tee "$LOCAL/build.log"
  build_rc=${PIPESTATUS[0]}
  log "build_rc=$build_rc build_s=$(( $(date +%s) - t1 ))"
  BIN=$(find "$LOCAL/build" -maxdepth 3 -type f -name 'strata' -perm -u+x 2>/dev/null | head -1)
  log "binary=$BIN"
  if [ -n "$BIN" ]; then
    "$BIN" --version > "$LOCAL/run-version.txt" 2>&1 || true
    cat "$LOCAL/run-version.txt"
    # startup is where the wave32 requirement is checked
    timeout 120 "$BIN" --gpus 0,1 --help > "$LOCAL/run-help.txt" 2>&1 || true
    head -40 "$LOCAL/run-help.txt"
  fi
fi

{
  echo "job=$JOB configure_rc=$cfg_rc build_rc=${build_rc:-none}"
  echo "-- configure command --"; cat "$LOCAL/cmake-command.txt"
  echo "-- configure tail --"; tail -60 "$LOCAL/configure.log" 2>/dev/null
  echo "-- build tail --"; tail -80 "$LOCAL/build.log" 2>/dev/null
  echo "-- run version --"; cat "$LOCAL/run-version.txt" 2>/dev/null
  echo "-- run help head --"; head -60 "$LOCAL/run-help.txt" 2>/dev/null
} > "$OUT/strata-hip-facts.txt"
cp "$LOCAL"/configure.log "$LOCAL"/build.log "$LOCAL"/cmake-command.txt "$OUT/" 2>/dev/null
cp "$LOCAL/strata-hip.log" "$OUT/" 2>/dev/null
log "done"
