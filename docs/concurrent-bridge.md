# Opt-in concurrent queue bridge

Default invocation still holds `bridge.lock` through one complete job. No scheduler or
running process is changed by this source update. A reviewer may enable independent
one-shot workers at the next clean legacy-helper boundary:

```
python3 ../slurmhelps/bin/hpc_bridge_gpu_queue.py --concurrent --active-job-limit 5
```

Each invocation runs at most one admitted job (after up to three pre-submit refusals).
The configured cluster maximum remains authoritative. Root must first check the
combined fleet census and other submission routes; this lock cannot serialize unrelated
custom submitters. Existing chain behavior is unchanged, and no live lock should be
removed or helper restarted to activate this option.

All new workers serialize census, reservation and atomic pending-to-running claim under
`bridge-admission.lock`, then release that short lock before transfer/monitoring. Cluster
names visible in squeue count once per allocation; reserved names not visible yet count
as additional slots. Duplicate actual names remain separate allocations. Reservation multiplicity is preserved
with one-to-one matching against visible names; a pending job colliding with a reserved
cluster output name is skipped. Serial workers
conservatively count actual jobs plus reservations, possibly underfilling but never
ignoring a reservation. Existing atomic LAN claims remain the authority for winning a
pending job.

A claim records UUID, host, boot, PID and process start identity. Its durable lease lock is
inherited by blocking transfer and hpc_run children, so parent death does not make a
still-running submission child stealable. Live owners are skipped. Missing, unreadable,
foreign-host or ambiguous identity fails closed. A definitely dead pre-submit owner can
release its claim; a submitted/pulling claim without a terminal receipt remains reserved
for explicit reconciliation, even after its owner dies. Missing terminal output is not
proof that a cluster allocation was never accepted. Review the submit log, actual Slurm
handle/state and output before any manual recovery; this implementation never retries
such work. Worker identity files are retained as evidence.

A concurrent worker also respects an active legacy serial lock and its ownerless claim.
Prefer rollout after the old helper finishes: already-loaded old Python code cannot be
made aware of new ownership semantics by editing files underneath it.

Local tests use synthetic queues, mocked cluster commands, real file locks, and two
bounded test processes racing for the last slot. They submit no cluster/LAN work:

```
/path/to/python -m pytest tests/test_bridge_concurrent_dispatch.py tests/test_hpc_bridge_gpu_queue.py tests/test_gpu_queue_hardening.py -q
```
