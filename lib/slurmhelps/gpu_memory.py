"""Per-job GPU memory admission shared by the LAN queue and cluster bridge."""

import math
import subprocess

FIELDS = ("min_vram_gib", "min_free_vram_gib", "cupy_pool_limit_gib")


def resources(job):
    r = job.get("gpu_memory", {})
    if not isinstance(r, dict) or set(r) - set(FIELDS):
        raise ValueError("unknown GPU memory schema")
    for key, value in r.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("GPU memory fields require finite positive GiB")
    if "cupy_pool_limit_gib" in r and "min_free_vram_gib" not in r:
        raise ValueError("pool requires explicit free-memory guard")
    if r.get("cupy_pool_limit_gib", 0) > r.get("min_free_vram_gib", float("inf")):
        raise ValueError("pool exceeds free guard")
    return r


def admit(job, device, *, phase):
    """Routing requires known total; allocated execution requires fresh total+free.

    Static route capacity is NEVER represented as a measurement of current free memory.
    Legacy jobs have no new restrictions. No mutation, claim, retry or ordering here.
    """
    assert phase in ["route", "execute"]
    try:
        r = resources(job)
    except ValueError as e:
        return False, str(e)
    if not r:
        return True, "legacy"
    total = device.get("total_gib")
    free = device.get("free_gib")
    minimum = max(
        r.get("min_vram_gib", 0),
        r.get("min_free_vram_gib", 0),
        r.get("cupy_pool_limit_gib", 0),
    )
    if not valid_number(total) or total < minimum:
        return False, "total VRAM insufficient or unknown"
    if free is not None and (not valid_number(free) or free > total):
        return False, "invalid free VRAM measurement"
    if phase == "execute" and "min_free_vram_gib" in r:
        if not valid_number(free) or free < r["min_free_vram_gib"]:
            return False, "free VRAM insufficient or unknown"
    return True, "eligible"


def eligible_in_order(ordered_jobs, device, *, phase="route"):
    """Filter AFTER consumer's existing priority/fairness sort, BEFORE atomic claim."""
    return [j for j in ordered_jobs if admit(j, device, phase=phase)[0]]


def environment(job):
    r = resources(job)
    return (
        {"EVAL_CUPY_POOL_LIMIT_GB": str(r["cupy_pool_limit_gib"])}
        if "cupy_pool_limit_gib" in r
        else {}
    )


def validate_command_pool(job, command):
    """Reject known immediate env/CLI pool overrides; never rewrite user argv.

    Shell/script interiors are opaque and must honor the documented pool contract.
    """
    import os

    r = resources(job)
    if "cupy_pool_limit_gib" not in r:
        return
    if not command or not all(isinstance(x, str) for x in command):
        raise ValueError("invalid constrained command argv")
    expected = r["cupy_pool_limit_gib"]

    def check(value):
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError("invalid explicit CuPy pool value") from e
        if not valid_number(number) or number != expected:
            raise ValueError("explicit CuPy pool contradicts GPU memory metadata")

    # Follow immediate env launchers, including nested env, but no shell parsing.
    cursor = 0
    while cursor < len(command) and os.path.basename(command[cursor]) == "env":
        cursor += 1
        while cursor < len(command):
            token = command[cursor]
            if token == "--":
                cursor += 1
                continue
            if token in ("-u", "--unset"):
                if cursor + 1 >= len(command):
                    raise ValueError("missing env unset name")
                if command[cursor + 1] == "EVAL_CUPY_POOL_LIMIT_GB":
                    raise ValueError("env unsets declared CuPy pool")
                cursor += 2
                continue
            if token.startswith("--unset="):
                if token.split("=", 1)[1] == "EVAL_CUPY_POOL_LIMIT_GB":
                    raise ValueError("env unsets declared CuPy pool")
                cursor += 1
                continue
            if token.startswith("-"):
                # -i/--ignore-environment/-S and unrecognized options can erase
                # or reinterpret the inherited pool; require an explicit wrapper.
                raise ValueError("unsupported env option for declared CuPy pool")
            if "=" not in token:
                break
            key, value = token.split("=", 1)
            if key == "EVAL_CUPY_POOL_LIMIT_GB":
                check(value)
            cursor += 1
    for i, token in enumerate(command):
        if token == "--cupy-pool-limit-gb":
            if i + 1 >= len(command):
                raise ValueError("missing explicit CuPy pool value")
            check(command[i + 1])
        elif token.startswith("--cupy-pool-limit-gb="):
            check(token.split("=", 1)[1])


def valid_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def parse_probe(text):
    """nvidia-smi CSV MiB, one selected physical device only."""
    lines = text.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("expected exactly one selected GPU")
    fields = lines[0].split(",")
    if len(fields) != 2:
        raise ValueError("expected total,free MiB")
    total, free = (float(x.strip()) / 1024 for x in fields)
    if not valid_number(total) or not valid_number(free) or free > total:
        raise ValueError("invalid device memory")
    return {"total_gib": total, "free_gib": free}


def probe():
    import os
    import subprocess

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if not visible or "," in visible:
        raise ValueError("one visible GPU required")
    r = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + visible,
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return parse_probe(r.stdout)


def guarded_exec(requirements, command):
    import json
    import os
    import sys
    import time

    job = {"gpu_memory": requirements}
    try:
        resources(job)
        validate_command_pool(job, command)
        device = probe()
        ok, reason = admit(job, device, phase="execute")
    except (ValueError, OSError, RuntimeError) as e:
        device = {}
        ok = False
        reason = str(e)
    except subprocess.SubprocessError as e:
        device = {}
        ok = False
        reason = str(e)
    print(
        json.dumps(
            {
                "gpu_memory_admission": {
                    "requirements": requirements,
                    "observed": device,
                    "accepted": ok,
                    "reason": reason,
                    "timestamp": time.time(),
                }
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    if not ok:
        return 78
    env = {**os.environ, **environment(job)}
    os.execvpe(command[0], command, env)


def main():
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--requirements", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("command required")
    return guarded_exec(json.loads(a.requirements), command)


if __name__ == "__main__":
    raise SystemExit(main())
