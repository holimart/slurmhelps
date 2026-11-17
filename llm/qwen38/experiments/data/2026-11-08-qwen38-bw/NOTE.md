# 2026-11-08 Blackwell (bw01, job 38374335) re-measure

Measured 02:40-03:00 right after the five ADGFEM lanes moved back to this endpoint, so the server was
**shared with live lane traffic** for the whole run. Treat it as "under load", not as a clean baseline;
compare against `2026-10-24-qwen38-bw` (idle) only with that in mind.

- greedy5: 5/5 correct.
- batch-500 C=16: 44.2 tok/s out (tpot p50 115 ms); peak 164 tok/s at C=32.
- batch-800 C=16: 72.9 tok/s out; peak 132 tok/s at C=32.
- ctx 4096 C=8: peak 278 tok/s; ctx 16384 C=4: 91.7; ctx 65536 C=2: 25.6; ctx 262144 C=1: 6.8 (ttft 18.2 s).
- Earlier idle numbers for reference: bw01 208 tok/s and hopper 431 tok/s at C=16 (2026-11-07 note).

Log: /work/ffa/holecem/scratch-claude/fleet20/t9/measure-bw-38374335.log
