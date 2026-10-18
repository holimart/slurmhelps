#!/usr/bin/env python3
"""Generic Slurm/GPU batch helpers — project-agnostic, stdlib only.

Two subcommands, usable from the CLI or by importing the functions:

``probe``
    Read-only scan of a Slurm cluster over ssh: keep only *idle* nodes with a
    ``gpu:`` GRES, apply node/GPU exclusions and partition allow-lists, rank by
    GPU cost then partition, de-duplicate per node and print the cheapest first.

``run``
    Select a GPU (same rules), stage a local batch directory to the remote,
    generate an sbatch script, submit it, poll until the job leaves the queue,
    read its final ``sacct`` state, rsync the results back and clean up.

Nothing here knows anything about a particular cluster: the ssh target,
account, directories, partitions and GPU ranks are all parameters. The remote
work directories are created only under the ``--remote-tmp`` / ``--workdir``
you pass — never ``$HOME``.

Environment: pure standard library (no third-party imports). Remote access is
exclusively via the ``ssh``/``rsync`` CLIs (``BatchMode=yes``, no prompts).

Reusable API::

    from hpchelpers.gpu_batch import probe_gpus, select_gpu, submit_run

    nodes = probe_gpus("user@login", account="acct", exclude_gpu=("V100",))
    path = submit_run(
        "user@login", account="acct", workdir="/scratch/env",
        remote_tmp="/scratch/tmp", local_batch="./batch", remote_name="job1",
        run_cmd="python run.py --out {output}", yes=True,
    )

CLI::

    gpu_batch.py probe --ssh SSH [--account ACCT] [--json]
        [--exclude-node-prefix P]... [--exclude-gpu NAME]...
        [--gpu-rank NAME=INT]... [--partition-rank NAME=INT]...
        [--allowed-partition NAME]...

    gpu_batch.py run --ssh SSH --account ACCT --workdir DIR --remote-tmp DIR
        --local-batch DIR --remote-name NAME --run-cmd 'CMD' [--setup-cmd 'CMD']
        [--gres gpu:1] [--cpus 4] [--mem 16G] [--minutes 10]
        [--partition P] [--nodelist N] [--results-name results.jsonl]
        [--results-local PATH] [--poll-timeout SECONDS] [--yes] [--dry-run]
        [plus all probe selection flags]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

# Rank used for a GPU model / partition the caller did not rank explicitly:
# deliberately large so any explicitly ranked option sorts ahead of it.
DEFAULT_RANK = 1_000_000

# A job in one of these ``squeue %T`` states has left the queue.
_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}

_SINFO_CMD = 'sinfo -h -N -o "%P|%N|%G|%t|%a"'
_SCONTROL_CMD = "scontrol show partition -o"
# GPUs in use per node. A "mix" node (CPUs partly allocated, often by CPU-only
# jobs) can still have every GPU free, so node state alone is not enough.
_SINFO_USED_CMD = 'sinfo -h -N -O "NodeList:60,GresUsed:120"'


class HpcError(RuntimeError):
    """A recoverable failure: bad ssh call, unusable cluster reply, no node."""


@dataclass(frozen=True)
class Node:
    """One candidate GPU node (already filtered and ranked)."""

    partition: str
    node: str
    gpu: str
    state: str
    gpu_rank: int
    partition_rank: int


# --------------------------------------------------------------------------
# External process plumbing (monkeypatchable in tests)
# --------------------------------------------------------------------------


def run_local(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a local command (rsync/ssh) and capture its output.

    This is the single choke point for local process execution, so tests can
    replace it to assert that no transfer/submission happens in dry-run.
    """
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)


def run_ssh(ssh: str, command: str, *, timeout: float = 120) -> str:
    """Run ``command`` on ``ssh`` and return stdout; raise ``HpcError`` on failure.

    Single choke point for remote execution: tests monkeypatch this to serve
    canned ``sinfo``/``scontrol``/``squeue`` output with no network.
    """
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh, command]
    try:
        proc = run_local(argv, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise HpcError(f"ssh {ssh!r} timed out after {timeout}s") from e
    except OSError as e:
        raise HpcError(f"cannot run ssh for {ssh!r}: {e}") from e
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise HpcError(f"ssh {ssh!r} command failed (rc={proc.returncode}): {err}")
    return proc.stdout


def _rsync(ssh: str, src: str, dst: str, *, timeout: float = 3600) -> None:
    argv = ["rsync", "-az", "--timeout=600", "-e", "ssh", src, dst]
    try:
        proc = run_local(argv, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise HpcError(f"rsync timed out: {src} -> {dst}") from e
    except OSError as e:
        raise HpcError(f"cannot run rsync: {e}") from e
    if proc.returncode != 0:
        raise HpcError(f"rsync failed ({proc.returncode}): {(proc.stderr or '').strip()}")


# --------------------------------------------------------------------------
# Probing / selection
# --------------------------------------------------------------------------


def _split_gpu(gres: str) -> tuple[str, str] | None:
    """Parse a Slurm GRES string into ``(model, canonical_model)``.

    Returns ``None`` when the GRES carries no GPU. Examples::

        gpu:tesla:V100:2   -> ("tesla:V100", "V100")
        gpu:V100:2         -> ("V100", "V100")
        gpu:t4:2(S:0-1)    -> ("t4", "t4")
        gpu:2              -> ("", "")
    """
    for entry in gres.split(","):
        entry = entry.strip()
        if not entry.startswith("gpu:"):
            continue
        base = entry.split("(", 1)[0]
        segs = base.split(":")[1:]
        if segs and segs[-1].isdigit():
            segs = segs[:-1]
        return ":".join(segs), (segs[-1] if segs else "")
    return None


def _gpu_count(gres: str) -> int | None:
    """Number of GPUs in a GRES string (``gpu:L40:2(S:0-1)`` -> 2), else ``None``."""
    for entry in gres.split(","):
        entry = entry.strip()
        if not entry.startswith("gpu:"):
            continue
        last = entry.split("(", 1)[0].split(":")[-1]
        return int(last) if last.isdigit() else None
    return None


def _load_gpus_used(ssh: str) -> dict[str, int]:
    """Map node -> GPUs in use; nodes that fail to parse are left out."""
    used: dict[str, int] = {}
    try:
        raw = run_ssh(ssh, _SINFO_USED_CMD)
    except Exception:  # noqa: BLE001 - fall back to idle-only selection
        return used
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        n = _gpu_count(parts[1])
        if n is not None:
            used[parts[0]] = n
    return used


def _gpu_rank(model: str, canonical: str, ranks: Mapping[str, int]) -> int:
    """Lowest rank whose key appears in the model (else ``DEFAULT_RANK``)."""
    best: int | None = None
    low_model, low_canonical = model.lower(), canonical.lower()
    for name, rank in ranks.items():
        key = name.lower()
        if key and (key in low_model or key == low_canonical) and (best is None or rank < best):
            best = rank
    return DEFAULT_RANK if best is None else best


def _partition_rank(partition: str, ranks: Mapping[str, int]) -> int:
    return ranks.get(partition, DEFAULT_RANK)


def _load_allow_accounts(ssh: str) -> dict[str, str]:
    """Map partition name -> ``AllowAccounts`` value from ``scontrol``.

    Missing partitions map to ``""`` (nothing permitted) — fail closed.
    """
    raw = run_ssh(ssh, _SCONTROL_CMD)
    out: dict[str, str] = {}
    for line in raw.splitlines():
        pname = re.search(r"PartitionName=(\S+)", line)
        if not pname:
            continue
        allow = re.search(r"AllowAccounts=(\S+)", line)
        out[pname.group(1)] = allow.group(1) if allow else ""
    return out


def _partition_permitted(
    partition: str,
    allowed_partition: Sequence[str],
    account: str | None,
    allow_accounts: Mapping[str, str],
) -> bool:
    """Allow-list wins; otherwise, when an account is given, check AllowAccounts."""
    if allowed_partition:
        return partition in allowed_partition
    if account:
        value = allow_accounts.get(partition)
        if value is None:
            return False
        accounts = [a for a in value.split(",") if a]
        return "ALL" in accounts or account in accounts
    return True


def probe_gpus(
    ssh: str,
    *,
    account: str | None = None,
    exclude_node_prefix: Sequence[str] = (),
    exclude_gpu: Sequence[str] = (),
    gpu_rank: Mapping[str, int] | None = None,
    partition_rank: Mapping[str, int] | None = None,
    allowed_partition: Sequence[str] = (),
) -> list[Node]:
    """Return eligible idle GPU nodes, cheapest-first.

    Filters (all applied before ranking):

    * state must be ``idle``;
    * GRES must start with ``gpu:``;
    * node must not start with any ``exclude_node_prefix``;
    * GPU model must not contain any ``exclude_gpu`` string (case-insensitive);
    * partition must be permitted: in ``allowed_partition`` when given,
      otherwise (when ``account`` is given) listed in the partition's
      ``AllowAccounts`` (``ALL`` or the account).

    Each node is de-duplicated to its cheapest ``(gpu_rank, partition_rank)``
    occurrence, then sorted by ``(gpu_rank, partition_rank, node)``.
    """
    ranks = dict(gpu_rank or {})
    pranks = dict(partition_rank or {})
    excluded_gpu = [g.lower() for g in exclude_gpu]
    allow_accounts = _load_allow_accounts(ssh) if (account and not allowed_partition) else {}

    raw = run_ssh(ssh, _SINFO_CMD)
    gpus_used = _load_gpus_used(ssh)
    candidates: dict[str, Node] = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        partition = parts[0].strip().rstrip("*")
        node = parts[1].strip()
        gres = parts[2].strip()
        state = parts[3].strip()
        if not node:
            continue
        if state != "idle":
            # "mix" qualifies only when sinfo shows a GPU still free on it.
            total = _gpu_count(gres)
            if state != "mix" or total is None or node not in gpus_used:
                continue
            if gpus_used[node] >= total:
                continue
        if any(node.startswith(p) for p in exclude_node_prefix):
            continue
        parsed = _split_gpu(gres)
        if parsed is None:
            continue
        model, canonical = parsed
        if any(g in model.lower() for g in excluded_gpu):
            continue
        if not _partition_permitted(partition, allowed_partition, account, allow_accounts):
            continue
        cand = Node(
            partition=partition,
            node=node,
            gpu=model,
            state=state,
            gpu_rank=_gpu_rank(model, canonical, ranks),
            partition_rank=_partition_rank(partition, pranks),
        )
        best = candidates.get(node)
        better = best is None or (cand.gpu_rank, cand.partition_rank) < (
            best.gpu_rank,
            best.partition_rank,
        )
        if better:
            candidates[node] = cand
    return sorted(candidates.values(), key=lambda n: (n.gpu_rank, n.partition_rank, n.node))


def select_gpu(
    ssh: str,
    *,
    nodelist: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    exclude_node_prefix: Sequence[str] = (),
    exclude_gpu: Sequence[str] = (),
    gpu_rank: Mapping[str, int] | None = None,
    partition_rank: Mapping[str, int] | None = None,
    allowed_partition: Sequence[str] = (),
) -> Node:
    """Pick the single best GPU node for a run.

    ``nodelist`` is honored verbatim (no probing needed). Otherwise ``probe_gpus``
    is consulted; when ``partition`` is given the result is restricted to it.
    Raises ``HpcError`` when nothing suitable exists.
    """
    if nodelist:
        return Node(
            partition=partition or "",
            node=nodelist,
            gpu="",
            state="",
            gpu_rank=0,
            partition_rank=0,
        )
    nodes = probe_gpus(
        ssh,
        account=account,
        exclude_node_prefix=exclude_node_prefix,
        exclude_gpu=exclude_gpu,
        gpu_rank=gpu_rank,
        partition_rank=partition_rank,
        allowed_partition=allowed_partition,
    )
    if partition:
        nodes = [n for n in nodes if n.partition == partition]
    if not nodes:
        where = f" in partition {partition!r}" if partition else ""
        raise HpcError(f"no suitable idle GPU node found{where}")
    return nodes[0]


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


def _subst(cmd: str, batch_dir: str, output: str, workdir: str) -> str:
    return (
        cmd.replace("{batch_dir}", batch_dir)
        .replace("{output}", output)
        .replace("{workdir}", workdir)
    )


def build_sbatch_script(
    *,
    remote_name: str,
    remote_dir: str,
    account: str,
    gres: str = "gpu:1",
    cpus: int = 4,
    mem: str = "16G",
    minutes: int = 10,
    partition: str | None = None,
    nodelist: str | None = None,
    exclude: Sequence[str] = (),
    setup_cmd: str | None = None,
    run_cmd: str = "",
    workdir: str = "",
    results_name: str = "results.jsonl",
) -> str:
    """Render the sbatch script (``--output`` -> ``<remote_dir>/slurm-%j.out``).

    ``{batch_dir}`` / ``{output}`` / ``{workdir}`` in ``setup_cmd`` and
    ``run_cmd`` are substituted before embedding. ``exclude`` becomes
    ``#SBATCH --exclude=`` (node names/hostlists), so a job can never be placed
    on forbidden nodes even though selection only ranked eligible ones.
    """
    output = f"{remote_dir}/{results_name}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={remote_name}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={minutes}",
        f"#SBATCH --account={account}",
        f"#SBATCH --output={remote_dir}/slurm-%j.out",
    ]
    if partition:
        lines.insert(2, f"#SBATCH --partition={partition}")
    if nodelist:
        lines.insert(3, f"#SBATCH --nodelist={nodelist}")
    if exclude:
        lines.insert(4, f"#SBATCH --exclude={','.join(exclude)}")
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append(f"cd {shlex.quote(remote_dir)}")
    if setup_cmd:
        lines.append(_subst(setup_cmd, remote_dir, output, workdir))
    lines.append(_subst(run_cmd, remote_dir, output, workdir))
    return "\n".join(lines) + "\n"


def _write_remote_file(ssh: str, path: str, content: str) -> None:
    quoted = shlex.quote(path)
    heredoc = f"cat > {quoted} <<'LANGYWRAP_HPC_EOF'\n{content}LANGYWRAP_HPC_EOF"
    run_ssh(ssh, heredoc)


def submit_run(
    ssh: str,
    *,
    account: str,
    workdir: str,
    remote_tmp: str,
    local_batch: str,
    remote_name: str,
    run_cmd: str,
    setup_cmd: str | None = None,
    gres: str = "gpu:1",
    cpus: int = 4,
    mem: str = "16G",
    minutes: int = 10,
    partition: str | None = None,
    nodelist: str | None = None,
    exclude: Sequence[str] = (),
    results_name: str = "results.jsonl",
    results_local: str | None = None,
    poll_timeout: int = 3600,
    yes: bool = False,
    dry_run: bool = False,
    exclude_node_prefix: Sequence[str] = (),
    exclude_gpu: Sequence[str] = (),
    gpu_rank: Mapping[str, int] | None = None,
    partition_rank: Mapping[str, int] | None = None,
    allowed_partition: Sequence[str] = (),
) -> str:
    """Submit one GPU batch job and return the local results path.

    Dry-run (``not yes`` or ``dry_run=True``) prints the plan and returns the
    intended local results path without touching the remote.

    Steps when real: select GPU -> ``mkdir -p`` batch dir + workdir -> rsync the
    batch contents up -> write sbatch script -> ``sbatch`` -> poll ``squeue``
    until it leaves -> read ``sacct`` state -> rsync ``<results_name>`` back ->
    ``rm -rf`` the remote dir.
    """
    node = select_gpu(
        ssh,
        nodelist=nodelist,
        partition=partition,
        account=account,
        exclude_node_prefix=exclude_node_prefix,
        exclude_gpu=exclude_gpu,
        gpu_rank=gpu_rank,
        partition_rank=partition_rank,
        allowed_partition=allowed_partition,
    )
    sel_partition = partition or node.partition or None
    batch_dir = f"{remote_tmp.rstrip('/')}/{remote_name}"
    output = f"{batch_dir}/{results_name}"
    if results_local is None:
        results_local = f"./{remote_name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    script = build_sbatch_script(
        remote_name=remote_name,
        remote_dir=batch_dir,
        account=account,
        gres=gres,
        cpus=cpus,
        mem=mem,
        minutes=minutes,
        partition=sel_partition,
        nodelist=nodelist,
        exclude=exclude,
        setup_cmd=setup_cmd,
        run_cmd=run_cmd,
        workdir=workdir,
        results_name=results_name,
    )
    sbatch_path = f"{batch_dir}/job.sbatch"

    if dry_run or not yes:
        print(f"[dry-run] ssh target      : {ssh}")
        print(f"[dry-run] selected node   : {node.node or '(unspecified)'}")
        print(f"[dry-run] partition       : {sel_partition or '(unspecified)'}")
        print(f"[dry-run] remote batch dir: {batch_dir}")
        print(f"[dry-run] workdir         : {workdir}")
        print(f"[dry-run] results local   : {results_local}")
        print("[dry-run] sbatch script:")
        for line in script.rstrip("\n").splitlines():
            print(f"    {line}")
        print("[dry-run] commands:")
        print(f"    ssh {ssh} mkdir -p {batch_dir} {workdir}")
        print(f"    rsync -az {local_batch.rstrip('/')}/ {ssh}:{batch_dir}/")
        print(f"    ssh {ssh} (write {sbatch_path})")
        print(f"    ssh {ssh} sbatch --parsable {sbatch_path}")
        print(f"    ssh {ssh} squeue -h -j <id> -o %T   (poll, timeout {poll_timeout}s)")
        print(f"    ssh {ssh} sacct -n -j <id> -o State")
        print(f"    rsync -az {ssh}:{output} {results_local}")
        print(f"    ssh {ssh} rm -rf {batch_dir}")
        return results_local

    run_ssh(ssh, f"mkdir -p {shlex.quote(batch_dir)} {shlex.quote(workdir)}")
    _rsync(ssh, f"{local_batch.rstrip('/')}/", f"{ssh}:{batch_dir}/")
    _write_remote_file(ssh, sbatch_path, script)

    jid = run_ssh(ssh, f"sbatch --parsable {shlex.quote(sbatch_path)}").strip()
    jid = jid.split(";", 1)[0].split(".", 1)[0]
    if not jid:
        raise HpcError(f"sbatch returned no job id for {remote_name!r}")

    deadline = time.monotonic() + poll_timeout
    while True:
        alive = run_ssh(ssh, f"squeue -h -j {jid} -o %T").strip()
        if not alive or alive.splitlines()[0].strip() in _TERMINAL_STATES:
            break
        if time.monotonic() > deadline:
            raise HpcError(f"job {jid} not finished after {poll_timeout}s (queued or still running; results not synced -- raise --poll-timeout or fetch them manually)")
        time.sleep(10)

    state = run_ssh(ssh, f"sacct -n -j {jid} -o State").strip()
    print(f"[hpc] job {jid} finished: {state or 'unknown'}", file=sys.stderr)

    _rsync(ssh, f"{ssh}:{output}", results_local)
    run_ssh(ssh, f"rm -rf {shlex.quote(batch_dir)}")
    return results_local


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _rank_pair(text: str) -> tuple[str, int]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected NAME=INT, got {text!r}")
    name, _, value = text.partition("=")
    try:
        return name, int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected NAME=INT, got {text!r}") from e


def _add_selection_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--exclude-node-prefix", action="append", default=[], metavar="P")
    p.add_argument("--exclude-gpu", action="append", default=[], metavar="NAME")
    p.add_argument("--gpu-rank", action="append", default=[], type=_rank_pair, metavar="NAME=INT")
    p.add_argument(
        "--partition-rank", action="append", default=[], type=_rank_pair, metavar="NAME=INT"
    )
    p.add_argument("--allowed-partition", action="append", default=[], metavar="NAME")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gpu_batch.py",
        description="Generic Slurm/GPU probe and batch-run helpers (see module docstring).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="list eligible idle GPU nodes")
    probe.add_argument("--ssh", required=True, help="ssh target (user@host)")
    probe.add_argument("--account", default=None, help="slurm account (AllowAccounts filter)")
    probe.add_argument("--json", action="store_true", help="emit JSON instead of text")
    _add_selection_flags(probe)

    run = sub.add_parser("run", help="stage, submit and collect one GPU batch job")
    run.add_argument("--ssh", required=True)
    run.add_argument("--account", required=True)
    run.add_argument("--workdir", required=True, help="remote working dir (never $HOME)")
    run.add_argument("--remote-tmp", required=True, dest="remote_tmp")
    run.add_argument("--local-batch", required=True, help="local dir whose contents are staged")
    run.add_argument("--remote-name", required=True)
    run.add_argument("--run-cmd", required=True)
    run.add_argument("--setup-cmd", default=None)
    run.add_argument("--gres", default="gpu:1")
    run.add_argument("--cpus", type=int, default=4)
    run.add_argument("--mem", default="16G")
    run.add_argument("--minutes", type=int, default=10)
    run.add_argument("--partition", default=None)
    run.add_argument("--nodelist", default=None)
    run.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NODE",
        help="node name/hostlist to forbid via #SBATCH --exclude (repeatable)",
    )
    run.add_argument("--results-name", default="results.jsonl")
    run.add_argument("--results-local", default=None)
    run.add_argument("--poll-timeout", type=int, default=3600)
    run.add_argument("--yes", action="store_true", help="actually submit (else dry-run)")
    run.add_argument("--dry-run", action="store_true", help="force dry-run")
    _add_selection_flags(run)
    return ap


def _cmd_probe(args: argparse.Namespace) -> int:
    nodes = probe_gpus(
        args.ssh,
        account=args.account,
        exclude_node_prefix=args.exclude_node_prefix,
        exclude_gpu=args.exclude_gpu,
        gpu_rank=dict(args.gpu_rank),
        partition_rank=dict(args.partition_rank),
        allowed_partition=args.allowed_partition,
    )
    if args.json:
        print(json.dumps([asdict(n) for n in nodes], indent=2))
    else:
        for n in nodes:
            print(f"{n.partition}:{n.node} [{n.gpu}, {n.state}]")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    results = submit_run(
        args.ssh,
        account=args.account,
        workdir=args.workdir,
        remote_tmp=args.remote_tmp,
        local_batch=args.local_batch,
        remote_name=args.remote_name,
        run_cmd=args.run_cmd,
        setup_cmd=args.setup_cmd,
        gres=args.gres,
        cpus=args.cpus,
        mem=args.mem,
        minutes=args.minutes,
        partition=args.partition,
        nodelist=args.nodelist,
        exclude=args.exclude,
        results_name=args.results_name,
        results_local=args.results_local,
        poll_timeout=args.poll_timeout,
        yes=args.yes,
        dry_run=args.dry_run,
        exclude_node_prefix=args.exclude_node_prefix,
        exclude_gpu=args.exclude_gpu,
        gpu_rank=dict(args.gpu_rank),
        partition_rank=dict(args.partition_rank),
        allowed_partition=args.allowed_partition,
    )
    print(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.cmd == "probe":
            return _cmd_probe(args)
        return _cmd_run(args)
    except HpcError as e:
        print(f"gpu_batch: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
