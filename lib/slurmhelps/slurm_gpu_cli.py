"""On-demand GPU Slurm jobs CLI (was ``langywrap jobs gpu``). See :mod:`slurmhelps.slurm_gpu`.

    PYTHONPATH=../slurmhelps/lib python3 -m slurmhelps.slurm_gpu_cli gpu plan --project .
"""

from __future__ import annotations

import sys

import click

from slurmhelps import slurm_gpu as sg


def _fail(exc: Exception) -> None:
    click.echo(f"error: {exc}", err=True)
    sys.exit(2)


@click.group("jobs")
def jobs() -> None:
    """Project-scoped Slurm job helpers."""


@jobs.group("gpu")
def gpu() -> None:
    """On-demand GPU job: plan, submit, status, wait, cancel (config: fleet.slurm.gpu)."""


_project_opt = click.option(
    "--project", default=".", show_default=True, help="Project root holding .langywrap/lanes.json."
)


@gpu.command("plan")
@_project_opt
def gpu_plan(project: str) -> None:
    """Show which GPU partition/node a submit would use, and why."""
    try:
        policy = sg.load_policy(project)
        click.echo(sg.render_plan(policy, sg.plan(policy)))
    except sg.SlurmGpuError as exc:
        _fail(exc)


@gpu.command("submit", context_settings={"ignore_unknown_options": True})
@_project_opt
@click.option("--dry-run", is_flag=True, help="Print the sbatch call; submit nothing.")
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
def gpu_submit(project: str, dry_run: bool, command: tuple[str, ...]) -> None:
    """Submit COMMAND as one on-demand GPU job: submit [--project P] -- CMD ARGS..."""
    try:
        policy = sg.load_policy(project)
        rec = sg.submit(policy, list(command), dry_run=dry_run)
    except sg.SlurmGpuError as exc:
        _fail(exc)
    if dry_run:
        click.echo("dry-run: " + " ".join(rec["sbatch"]))
        return
    click.echo(f"job_id:    {rec['job_id']}")
    click.echo(f"job_name:  {rec['job_name']}")
    click.echo(f"partition: {rec['partition']}" + (" (queued)" if rec["queued"] else ""))
    click.echo(f"log:       {rec['log']}")
    click.echo(f"ledger:    {policy.ledger}")


@gpu.command("status")
@_project_opt
@click.argument("job_id")
def gpu_status(project: str, job_id: str) -> None:
    """Show the state of JOB_ID (squeue, falling back to sacct)."""
    try:
        st = sg.status(job_id)
    except sg.SlurmGpuError as exc:
        _fail(exc)
    click.echo(" ".join(f"{k}={v}" for k, v in st.items() if v != ""))


@gpu.command("wait")
@_project_opt
@click.option(
    "--timeout",
    default=3600.0,
    show_default=True,
    type=float,
    help="Seconds to wait before giving up (the job keeps running).",
)
@click.argument("job_id")
def gpu_wait(project: str, timeout: float, job_id: str) -> None:
    """Poll JOB_ID until it finishes; exit 0 only if it COMPLETED with exit code 0."""
    try:
        policy = sg.load_policy(project)
        st = sg.wait(job_id, timeout_s=timeout)
    except sg.SlurmGpuError as exc:
        _fail(exc)
    click.echo(" ".join(f"{k}={v}" for k, v in st.items() if v != ""))
    done_ok = st.get("state") == "COMPLETED" and st.get("exit_code", "").startswith("0:")
    try:
        sg.record_event(
            policy,
            "wait-finished",
            job_id=job_id,
            state=st.get("state"),
            exit_code=st.get("exit_code"),
            node=st.get("node"),
        )
    except OSError as exc:
        click.echo(f"warning: ledger not updated: {exc}", err=True)
    if not done_ok:
        sys.exit(1)


@gpu.command("cancel")
@_project_opt
@click.argument("job_id")
def gpu_cancel(project: str, job_id: str) -> None:
    """Cancel JOB_ID (scancel) and note it in the ledger."""
    try:
        policy = sg.load_policy(project)
        sg.cancel(job_id)
        sg.record_event(policy, "cancel", job_id=job_id)
    except sg.SlurmGpuError as exc:
        _fail(exc)
    click.echo(f"cancelled {job_id}")


if __name__ == "__main__":
    jobs()
