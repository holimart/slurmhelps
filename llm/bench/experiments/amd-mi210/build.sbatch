#!/usr/bin/env bash
#SBATCH --job-name=amd-mi210-build
#SBATCH --partition=ffa-preempt
#SBATCH --nodelist=r50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/tmp/amd-mi210-build-%j.out
#
# Step 2 of the AMD MI210 campaign: build llama.cpp b11785 with HIP/gfx90a
# on the node (never on the login node), plus the HBM bandwidth probe.
#
# The build tree is copied to node-local /tmp, built there, then the bin/ and
# lib/ trees are copied to Lustre so the measurement job can reuse them.
# Logs stay in node-local /tmp until the job ends; only small summaries are
# copied afterwards.
set -uo pipefail

JOB=${SLURM_JOB_ID:?inside allocation only}
ROOT=${SLURMHELPS_ROOT:-/work/ffa/holecem/slurmhelps-fleet}
EXTRAS=${AMD_MI210_DIR:-$ROOT/llm/bench/experiments/amd-mi210}
SRC_SHARED=${SRC_SHARED:-/work/ffa/holecem/src/llama-hip-b11785}
PREFIX_SHARED=${PREFIX_SHARED:-/work/ffa/holecem/build/llama-hip-b11785-gfx90a}
OUT=/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210/build
LOCAL=/tmp/amd-mi210-build-$JOB
SRC=$LOCAL/src
BUILD=$LOCAL/build
mkdir -p "$LOCAL" "$OUT" "$PREFIX_SHARED"

exec > >(tee "$LOCAL/build.log") 2>&1

echo "== job $JOB on $(hostname) $(date -u +%FT%TZ) =="
echo "== compiler =="
hipcc --version
/opt/rocm/llvm/bin/clang++ --version | head -2
cmake --version | head -1
ninja --version
gcc --version | head -1
nproc
free -g

echo "== copy source off Lustre ($SRC_SHARED) =="
t0=$(date +%s)
cp -a "$SRC_SHARED" "$SRC" || { echo "COPY FAILED"; exit 1; }
echo "copy_s=$(( $(date +%s) - t0 ))"
cd "$SRC"
git log --oneline -1
grep -n "class llm_graph_input_qwen4exp_ple" src/models/qwen4exp.cpp
grep -n "option(GGML_HIP_MMQ_MFMA\|option(GGML_HIP_GRAPHS\|option(GGML_CUDA_FA\b" ggml/CMakeLists.txt

# ROCWMMA flash-attention option: absent in b11785 (checked in ggml/CMakeLists.txt
# and ggml/src/ggml-hip/CMakeLists.txt); do not pass a flag CMake would ignore
# silently.  Recorded here on purpose.
if grep -q "GGML_HIP_ROCWMMA_FATTN" ggml/CMakeLists.txt ggml/src/ggml-hip/CMakeLists.txt 2>/dev/null; then
  echo "rocwmma_flag=present"
  ROCWMMA=ON
else
  echo "rocwmma_flag=absent"
  ROCWMMA=OFF
fi

CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=/usr/bin/gcc
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DAMDGPU_TARGETS=gfx90a
  -DGGML_HIP=ON
  -DGGML_HIP_MMQ_MFMA=ON
  -DGGML_HIP_GRAPHS=ON
  -DGGML_HIP_NO_VMM=ON
  -DGGML_CUDA_FA=ON
  -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DGGML_NATIVE=ON
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
)
printf '%s\n' "cmake -S . -B $BUILD ${CMAKE_ARGS[*]}" | tee "$LOCAL/cmake-command.txt"

echo "== configure =="
if ! cmake -S "$SRC" -B "$BUILD" "${CMAKE_ARGS[@]}"; then
  echo "== configure with gcc/g++ failed; retrying with hipcc as CXX (legacy path) =="
  rm -rf "$BUILD"
  CMAKE_ARGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
    -DAMDGPU_TARGETS=gfx90a
    -DGGML_HIP=ON
    -DGGML_HIP_MMQ_MFMA=ON
    -DGGML_HIP_GRAPHS=ON
    -DGGML_HIP_NO_VMM=ON
    -DGGML_CUDA_FA=ON
    -DGGML_CUDA_FA_ALL_QUANTS=ON
    -DGGML_NATIVE=ON
    -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=ON
  )
  printf '%s\n' "cmake -S . -B $BUILD ${CMAKE_ARGS[*]}" > "$LOCAL/cmake-command.txt"
  cmake -S "$SRC" -B "$BUILD" "${CMAKE_ARGS[@]}" || { echo "CONFIGURE FAILED"; exit 1; }
fi

echo "== build =="
t0=$(date +%s)
cmake --build "$BUILD" --config Release -j 64 2>&1 | tee "$LOCAL/build-make.log"
rc=${PIPESTATUS[0]}
echo "build_rc=$rc build_s=$(( $(date +%s) - t0 ))"
[ "$rc" -eq 0 ] || { echo "BUILD FAILED"; exit 1; }

echo "== artifacts =="
ls -l "$BUILD/bin" | head -40
"$BUILD/bin/llama-server" --version 2>&1 | head -10 || true
"$BUILD/bin/llama-bench" --help 2>&1 | head -5 || true

echo "== persist build =="
rm -rf "$PREFIX_SHARED"
mkdir -p "$PREFIX_SHARED"
cp -a "$BUILD/bin" "$PREFIX_SHARED/bin"
[ -d "$BUILD/lib" ] && cp -a "$BUILD/lib" "$PREFIX_SHARED/lib"
cp -a "$LOCAL/cmake-command.txt" "$OUT/cmake-command.txt"
cp -a "$SRC/ggml/CMakeLists.txt" "$OUT/ggml-CMakeLists.txt"
cp -a "$BUILD/CMakeCache.txt" "$OUT/CMakeCache.txt"
{
  echo "job=$JOB"
  echo "source=$SRC_SHARED commit=$(git -C "$SRC" rev-parse HEAD) describe=$(git -C "$SRC" describe --tags --always)"
  echo "hipcc=$(hipcc --version | head -1)"
  echo "clang=$(/opt/rocm/llvm/bin/clang++ --version | head -1)"
  echo "cmake=$(cmake --version | head -1)"
  echo "rocwmma_flag=$ROCWMMA"
  echo "built=$(date -u +%FT%TZ)"
  echo "cmd=$(cat "$LOCAL/cmake-command.txt")"
} > "$OUT/build-facts.txt"
cat "$OUT/build-facts.txt"

echo "== HBM bandwidth baseline =="
hipcc -O3 --offload-arch=gfx90a "$EXTRAS/hbm-bw.cpp" -o "$LOCAL/hbm-bw" 2>&1 | tee "$LOCAL/hbm-build.log"
if [ -x "$LOCAL/hbm-bw" ]; then
  for d in 0 1; do
    echo "-- device $d --"
    HIP_VISIBLE_DEVICES=$d "$LOCAL/hbm-bw" $((1<<28)) 10 2>&1 | tee -a "$LOCAL/hbm-bw.txt"
  done
  cp "$LOCAL/hbm-bw.txt" "$OUT/hbm-bw.txt"
else
  echo "hbm-bw build failed; see $LOCAL/hbm-build.log" | tee "$OUT/hbm-bw.txt"
fi

echo "== small bundle to Lustre =="
cp "$LOCAL/build.log" "$LOCAL/cmake-command.txt" "$OUT"/ 2>/dev/null
cp "$LOCAL/build-make.log" "$OUT"/ 2>/dev/null
du -sh "$PREFIX_SHARED"
echo "== done $(date -u +%FT%TZ) =="
