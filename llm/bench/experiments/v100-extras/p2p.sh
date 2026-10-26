#!/usr/bin/env bash
# p2p.sh — GPU topology / NVLink / P2P evidence (M10).
#
# usage: p2p.sh OUTDIR
# Writes OUTDIR/topology.txt, OUTDIR/nvlink.txt, OUTDIR/p2p.txt.  If a CUDA
# p2pBandwidthLatencyTest binary is present it is run; otherwise the report
# says so and the engine's own host->device PCIe probe (archived in
# engine-facts.txt) is the bandwidth evidence.
set -uo pipefail
out=${1:?outdir}
mkdir -p "$out"

{
  echo "== nvidia-smi -L"
  nvidia-smi -L 2>&1
  echo
  echo "== nvidia-smi topo -m"
  nvidia-smi topo -m 2>&1
  echo
  echo "== nvidia-smi topo -p2p r"
  nvidia-smi topo -p2p r 2>&1 || true
  echo
  echo "== nvidia-smi topo -p2p w"
  nvidia-smi topo -p2p w 2>&1 || true
} > "$out/topology.txt" 2>&1

{
  for i in 0 1 2 3; do
    echo "== nvidia-smi nvlink -s -i $i"
    nvidia-smi nvlink -s -i "$i" 2>&1 || true
    echo
  done
  echo "== nvidia-smi nvlink -c"
  nvidia-smi nvlink -c 2>&1 || true
} > "$out/nvlink.txt" 2>&1

bin=""
for cand in p2pBandwidthLatencyTest /usr/local/cuda/samples/bin/x86_64/linux/release/p2pBandwidthLatencyTest \
            /usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest; do
  if command -v "$cand" >/dev/null 2>&1; then bin=$(command -v "$cand"); break; fi
  [ -x "$cand" ] && { bin=$cand; break; }
done
{
  if [ -n "$bin" ]; then
    echo "== $bin"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$bin" 2>&1 | head -80
  else
    echo "no p2pBandwidthLatencyTest binary found on the node"
    echo "topology and NVLink status above are the hardware evidence;"
    echo "the engine's layer-split probe reports host->device PCIe bandwidth"
    echo "(12.4 GB/s on this V100 node, from engine-facts.txt)."
  fi
} > "$out/p2p.txt" 2>&1

echo "[p2p] wrote $out/{topology,nvlink,p2p}.txt"
