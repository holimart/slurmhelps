# RUNBOOK — RTX 3090 C16 batch campaign (`vast-3090-c16`)

**Status: prepared, not rented.** The Vast account is at −$0.78 with
`can_pay=false` after the 2026-10-25 bandwidth incident; nothing here spends
money until credits are recharged and the commands are run with `--yes`.

## Why
The economics in `opportunityscan/modelsquantdb/analyses/20-iq3s-economics.md`
§4b assumes a C16 multiplier somewhere between 1.30× and 2.88×. The 2.88× is a
projection (`cheap_compare.csv`), not a 3090 measurement. On a 3090 the C16
aggregate decides whether the node is €0.370 (1.3×), €0.240 (2.0×) or €0.167
(2.88×) per Mtok versus the Mac's €0.234. This campaign measures it directly.

## Offer selection (live 2026-10-28; re-check before renting)
Filter: `gpu_name=RTX 3090`, 1 GPU, `min_ram_gb >= 60`, disk >= 150 GB,
**`driver >= 580.65`**, **`internet_down_cost_per_tb <= $5/TB`**,
reliability >= 0.92, dph <= $0.25/h.

| offer | $/h | host RAM | down | driver | rel | note |
|---|---:|---:|---:|---|---:|---|
| **57519016** (TH) | 0.176 | 125 GB | $2.67/TB | 595.99 | 0.93 | **primary — headroom for IQ3_S** |
| **57836122** (RS) | 0.176 | 62 GB | $2.67/TB | 595.99 | 0.95 | fallback (86 GB total, tight) |
| 38849861 (CA) | 0.122 | 125 GB | $13.33/TB | **570.181** | 0.99 | excluded: driver too old |
| 30239905 (CZ) | 0.141 | 41 GB | $5.33/TB | 610.57 | 0.99 | excluded: 65 GB total < 83.6 GB |
| any host at $33–53/TB | — | — | — | — | — | excluded: the 2026-10-25 trap |

**Cost estimate:** 113.2 GB × $2.67/TB ≈ **$0.30** download + 3 h × $0.176 ≈
**$0.53** GPU + disk ≈ **$0.85–1.00** total. Cap the run at 3 h (`DEADLINE`).

## Fit and fallback
- 24 GB VRAM + 62–125 GB host; IQ3_S is 83.6 GB → 86 GB on the 62 GB host
  (2.4 GB headroom) and comfortable on 125 GB.
- KV at C16: **51 MB at 4K, 410 MB at 32K**; 128K is out of scope on this host.
- If IQ3_S OOMs on the 62 GB host, run **Q2_0 (66 GB)** — the campaign already
  includes it in `PACKS`.

## Matrix (exact)
- contexts **4096, 16384, 32768** × concurrency **1, 2, 4, 8, 16**
- 512 generated tokens, 3 requests/slot, `pad_safety 2048`, temperature 0
- 3 repeats at C1, 1 at C2/C4/C8/C16
- capture per-stream decode, aggregate tok/s, TTFT p50/p95, expert-cache hit,
  VRAM/RAM, and power (`nvidia-smi dmon` or the engine's power bracket)

## Commands (once credit is in)
```bash
cd /work/ffa/holecem/slurmhelps-fleet
export VAST_API_KEY=$(grep '^VAST_API_KEY=' /home/holecem/Projects/adgfem.stale-20261024/.env | cut -d= -f2)

# 0) must show can_pay=true
python3 llm/cloud/vast/vast.py balance

# 1) plan (no billing) — shows offer, download + GPU cost
llm/bench/benchctl vast plan --recipe vast-3090-c16

# 2) rent the primary offer (or 57836122 if 57519016 is gone)
python3 llm/cloud/vast/vast.py rent --recipe-file llm/bench/recipes/vast-3090-c16.json \
  --offer-id 57519016 --outdir /work/ffa/holecem/strata-runs/vast-3090-c16-$(date +%Y%m%d) --yes

# 3) stage the pinned packs + engine + runner (uses the recipe's setup3090.sh)
python3 llm/cloud/vast/vast.py stage --instance <ID> --recipe-file llm/bench/recipes/vast-3090-c16.json --yes

# 4) run the C16 matrix (remote logs stay on the host; pull at the end)
python3 llm/cloud/vast/vast.py run --instance <ID> --script /root/run-matrix2.sh \
  --set 'CTXS=4096 16384 32768' --set 'BATCHES=1 2 4 8 16' --set 'MAXTOK=512' \
  --set 'MAXCTX=32768' --set 'PACKS=iq3s q2_0' --set 'RPS=3' --set 'SAFETY=2048' \
  --set 'PARALLEL=16' --set 'DEADLINE=<epoch>' \
  --background --yes

# 5) pull summaries, then ALWAYS destroy
python3 llm/cloud/vast/vast.py pull --instance <ID> \
  --outdir /work/ffa/holecem/strata-runs/vast-3090-c16-<stamp> --yes
python3 llm/cloud/vast/vast.py destroy --instance <ID> --yes
```

`run-matrix2.sh` now sets `--parallel` from `PARALLEL` (auto = max(BATCHES)); the
old hardcoded `--parallel 4` would have made every C8/C16 cell a C4 cell. Pass
`PARALLEL=16` explicitly or rely on the auto value.

Rules: never tail logs on the remote host or on Lustre; poll the instance
status; pull only small summaries; destroy even on failure.

## Acceptance
- C16 aggregate measured at 4K and 16K (32K if memory holds), with the
  per-stream number.
- Gain = C16 aggregate / C1 aggregate; report it next to 1.30×, 2.0×, 2.88×.
- €/Mtok at C16 with €1,900 capex amortised and €0.20/kWh (state the method).
- If per-stream < 10 tok/s, say so — C16 is a throughput mode, not latency.
- No fabrication: an OOM or a missing cell is recorded with the real error.
