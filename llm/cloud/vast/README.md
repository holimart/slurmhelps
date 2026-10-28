# `llm/cloud/vast` — canonical Vast.ai campaign tooling

Copied out of the read-only 2026-10-25 run dirs so a rented host can be
reproduced from the repo. The originals under
`/work/ffa/holecem/strata-runs/2026-10-25-vast-*` stay untouched as historical
evidence.

| file | origin | notes |
|---|---|---|
| `setup.sh` | `2026-10-25-vast-titanx2/remote/setup.sh` | Titan/Q2_0 bring-up; paths now env-configurable |
| `setup3090.sh` | `2026-10-25-vast-3090x1/remote/setup3090.sh` | 3090 bring-up; `PACKS_TO_FETCH="iq3s coder q2_0"`, pinned revisions |
| `install-engine.sh` | `2026-10-25-vast-titanx2/remote/install-engine.sh` | sha256 identical to `llm/strata/install-engine.sh` at copy time |
| `run-matrix.sh` | `2026-10-25-vast-titanx2/remote/run-matrix.sh` | no-Slurm matrix: one launch per pack, deadline, skip list |
| `run-matrix2.sh` | `2026-10-25-vast-titanx2/remote/run-matrix2.sh` | same runner plus the `q2_0` pack path and `CELLS` |
| `strata_probe.py` | `2026-10-25-vast-titanx2/remote/strata_probe.py` | **frozen evidence** (one-point calibration build). The canonical probe is `llm/bench/probe.py`. |
| `vast.py` | none — the campaign's `tools/vast.py` was a 14-byte `404: Not Found` | new stdlib client; see below |

SHA-256 at the time of copy:

```text
292912855b4f360d170cb4340fdc56c391f55f6703d2ba894b80dffe5fbb59a6  install-engine.sh
058d3317c9ad19111672aaeaa5369931bc0c827db252f9eb26d522b0aa8ebeb9  run-matrix.sh
7f470205d6b90927e18b1986cfcc3deeb4881088f1cd8f577518d8eadbd531c3  run-matrix2.sh
97f2fec8991c78ed289e5571b753c65354048a6b3e7b769af50422a0cf8d035d  strata_probe.py
```

`setup.sh` and `setup3090.sh` were edited **only** to make paths configurable:

```text
RESULTS_DIR   default /root/results
MODELS_DIR    default /root/models
SETUP_DIR     directory containing install-engine.sh, default /root
STRATA_ROOT   default /root/strata-engine
```

The embedded Python verification uses `os.environ["MODELS_DIR"]` for `BASE`.
Everything else (pinned revisions, sizes, sha256, hardlinks) is unchanged.

## vast.py

```bash
python3 llm/cloud/vast/vast.py balance
python3 llm/cloud/vast/vast.py plan --recipe-file llm/bench/recipes/vast-titan.json
python3 llm/cloud/vast/vast.py rent --offer-id ID --label NAME --yes
python3 llm/cloud/vast/vast.py stage --instance ID --recipe-file llm/bench/recipes/vast-titan.json --yes
python3 llm/cloud/vast/vast.py run   --instance ID --script /root/run-matrix.sh --yes
python3 llm/cloud/vast/vast.py pull  --instance ID --dest DIR
python3 llm/cloud/vast/vast.py destroy --instance ID --yes
```

Encoded lessons:

- **Bandwidth is a filter, not a detail.** Offers are rejected when
  `internet_down_cost_per_tb > 5`. The Oman 3090 host billed $0.052/GB
  ($53.33/TB): 73 GB cost $3.80 while the hourly price was $0.26.
- **Driver floor.** `driver_version >= 580.65` (cuda13) and
  `cuda_max_good >= 13.0`; the cheapest 3090s on driver 570.181 cannot even run
  the cuda12 engine (floor 575.51).
- **Always destroy.** `destroy` verifies `GET /api/v1/instances/` afterwards
  and reports `still_listed`.
- **API endpoints** as observed: create `PUT /api/v0/asks/{offer}/`, list
  `GET /api/v1/instances/`, destroy `DELETE /api/v0/instances/{id}/`.
  `/api/v0/instances/` is deprecated (301).
- **`--dry-run`** prints the PUT body / rsync / ssh line and never bills.
  `rent`, `stage`, `run`, `destroy` require `--yes`.
- `--offers-file` accepts a saved `/bundles/` JSON fixture, which is how the
  tests run the filter without network.

The client stores nothing between commands. `rent --outdir DIR` writes
`DIR/instance.json` (offer, contract response, SSH endpoint); pass
`--instance`/`--ssh` explicitly afterwards.
