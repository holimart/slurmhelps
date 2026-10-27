#!/usr/bin/env bash
#SBATCH --job-name=amd-mi210-probe
#SBATCH --partition=ffa-preempt
#SBATCH --nodelist=r50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/tmp/amd-mi210-probe-%j.out
#
# Step 1 of the AMD MI210 campaign: facts only, no model, no build.
# Everything is written under /tmp on the node; only the final small text
# bundle is copied to Lustre.  Never tail/cat/grep live logs on Lustre.
set -uo pipefail

JOB=${SLURM_JOB_ID:?inside allocation only}
OUT=/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210/probe
LOCAL=/tmp/amd-mi210-probe-$JOB
mkdir -p "$LOCAL" "$OUT"

exec > >(tee "$LOCAL/probe.log") 2>&1

echo "== job $JOB on $(hostname) $(date -u +%FT%TZ) =="
echo "== slurm env =="
env | grep -E '^SLURM_|^ROCM|^HIP|^CUDA' | sort

echo "== os / kernel =="
cat /etc/redhat-release 2>/dev/null || cat /etc/os-release
uname -a

echo "== cpu / memory =="
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|CPU\(s\)'
free -g
numactl --hardware 2>/dev/null || true

echo "== local scratch =="
df -h /tmp /dev/shm 2>&1
ls -ld /tmp /dev/shm

echo "== rocm =="
hipcc --version 2>&1
cat /opt/rocm/.info/version 2>/dev/null || true
ls -l /opt/rocm 2>/dev/null | head
rocm-smi --version 2>&1 | head -5

echo "== rocminfo =="
rocminfo > "$LOCAL/rocminfo.txt" 2>&1
rc=$?
echo "rocminfo exit=$rc lines=$(wc -l < "$LOCAL/rocminfo.txt")"
grep -n "Name:\|gfx\|Marketing Name\|Uuid\|Compute Unit\|Wavefront\|Memory Size\|L2 Cache" "$LOCAL/rocminfo.txt" | head -80

echo "== rocm-smi product/hw =="
rocm-smi --showproductname --showhw > "$LOCAL/rocm-smi-hw.txt" 2>&1
cat "$LOCAL/rocm-smi-hw.txt"

echo "== rocm-smi topo =="
rocm-smi --showtopo > "$LOCAL/rocm-smi-topo.txt" 2>&1
cat "$LOCAL/rocm-smi-topo.txt"

echo "== rocm-smi topo weights =="
rocm-smi --showtopoweights > "$LOCAL/rocm-smi-topo-weights.txt" 2>&1 || true
cat "$LOCAL/rocm-smi-topo-weights.txt" 2>/dev/null | head -40

echo "== rocm-smi vram / idle clocks / power / temp =="
rocm-smi --showmeminfo vram > "$LOCAL/rocm-smi-vram.txt" 2>&1
rocm-smi --showclocks > "$LOCAL/rocm-smi-clocks-idle.txt" 2>&1
rocm-smi --showpower > "$LOCAL/rocm-smi-power-idle.txt" 2>&1
rocm-smi --showtemp > "$LOCAL/rocm-smi-temp-idle.txt" 2>&1
cat "$LOCAL/rocm-smi-vram.txt" "$LOCAL/rocm-smi-clocks-idle.txt" "$LOCAL/rocm-smi-power-idle.txt" "$LOCAL/rocm-smi-temp-idle.txt"

echo "== amd-smi (if any) =="
command -v amd-smi && amd-smi version 2>&1 | head -5
amd-smi static 2>&1 | head -60 || true

echo "== tools =="
for t in cmake ninja make gcc g++ clang clang++ python3 git numactl rocm-bandwidth-test babelstream; do
  printf '%-22s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo MISSING)"
done
cmake --version 2>/dev/null | head -1
ninja --version 2>/dev/null
gcc --version 2>/dev/null | head -1
clang --version 2>/dev/null | head -1

echo "== infiniband / xgmi fabric =="
ibstat 2>/dev/null | grep -E 'CA |State|Rate|LinkUp' | head -20 || true

echo "== source tree present =="
ls -ld /work/ffa/holecem/src/llama-hip-b11785
git -C /work/ffa/holecem/src/llama-hip-b11785 log --oneline -1 2>/dev/null || true
grep -n "class llm_graph_input_qwen4exp_ple" /work/ffa/holecem/src/llama-hip-b11785/src/models/qwen4exp.cpp 2>/dev/null || true

echo "== model shard sizes =="
ls -l /work/ffa/holecem/models/strata/iq3s /work/ffa/holecem/models/strata/coder /work/ffa/holecem/models/strata/q2_0 2>&1

echo "== copy small bundle to Lustre =="
cp "$LOCAL"/*.txt "$LOCAL/probe.log" "$OUT"/ 2>/dev/null
ls -l "$OUT"
echo "== done $(date -u +%FT%TZ) =="
