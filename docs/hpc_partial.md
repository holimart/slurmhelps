# HPC partial results and failure write-ups

**Job authors.** Import `PartialResults` from `lib/hpc_partial.py` (stdlib only; on PYTHONPATH via `slurmhelps.pth`, see README) and wrap `main`:
`with PartialResults(args.out, meta={...}) as pr:`. Call `pr.add(key, record)` after each finished
unit (net, layer, or property); each call atomically rewrites `--out` with `status: "partial"`.
End with `pr.finish(summary)`, which sets `status: "complete"` and merges `summary` at top level
(`units` is kept). An exception inside the `with` block is written as `status: "failed"` with
`error` and `traceback`, then re-raised. Keep the CLI and computations unchanged; the `--out` path
stays the same, so `{output}` in `hpc_run.sh` needs no change. Cards: `outputs/hpc/k3anatomy/run.py`,
`outputs/hpc/layerlab/exp_lowrank_diag.py`.

**Failed or killed jobs.** `bin/hpc_run.sh` tees the job's output to `$JOB_TMP/job.log`. On a
non-zero exit, or on SIGTERM at the Slurm time limit, it annotates `eval.json` in place with
`exit_code`, `terminated` (true only for SIGTERM) and `log_tail` (last 60 lines). If the job wrote no
JSON, the result is minimal: `{"status": "failed", "exit_code": N, "terminated": false, "log_tail": "..."}`.
A timed-out job therefore shows `"status": "partial", "terminated": true, "exit_code": 143` and keeps
every unit it finished.

**Steward reading.** Check `status` first. `complete` means use the top-level summary as before.
`partial` or `failed` means use `units` (keys are the per-unit names, for example `net3-seed...` or
`prop:<name>`) and `units_done`, and read `log_tail` for the cause. Partial units are not a decision.
