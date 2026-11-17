"""llm/qwen38/serve-strata.sh against a fake ``strata`` CLI: no GPU, no Slurm, loopback only.

The fake engine reports an empty /v1/models while "loading" (as Strata does)
and writes an ``engine started:`` line; the script must stage the pack,
seed a private data dir, gate the node gateway on a loaded model, and stop
the engine on SIGTERM.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "llm" / "qwen38" / "serve-strata.sh"
SHARDS = ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
          "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
STRATA_ID = "qwen3.8-flash-next-iq3_s"

FAKE_STRATA = r'''#!/usr/bin/env python3
"""Fake `strata launch` / `strata launch --stop` (records calls in $FAKE_LOG)."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
argv = sys.argv[1:]
log = Path(os.environ["FAKE_LOG"])
with log.open("a") as f:
    f.write(json.dumps(argv) + "\n")
runtime = Path(os.environ["STRATA_RUNTIME_DIR"])
if "--stop" in argv:
    pid = runtime / "server.pid"
    if pid.exists():
        os.kill(int(pid.read_text()), signal.SIGTERM)
        pid.unlink()
    sys.exit(0)
opt = lambda k: argv[argv.index(k) + 1]
sets = dict(v.split("=", 1) for i, v in enumerate(argv) if i and argv[i - 1] == "--set")
assert Path(sets["mtp"], "experts.bin").is_file(), "private mtp/rt not seeded"
model = Path(opt("--model-path"))
assert all((model / s).is_file() for s in %(shards)r), "pack not staged"
state = Path(os.environ["XDG_STATE_HOME"], "strata", "launch")
state.mkdir(parents=True, exist_ok=True)
(state / "x.engine.log").write_text(
    "engine started: strata --serve --native %%s --mtp %%s --kv %%s --max-context %%s\n"
    %% (model, sets["mtp"], opt("--kv"), opt("--context")))
server = r"""
import http.server, json, sys, time
t0 = time.time()
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        loaded = time.time() - t0 > 2.0
        doc = {"object": "list", "data": [{"id": "%(model)s", "object": "model"}] if loaded else []}
        body = json.dumps(doc).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""
proc = subprocess.Popen([sys.executable, "-c", server, opt("--port"), str(model)],
                        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
runtime.mkdir(parents=True, exist_ok=True)
(runtime / "server.pid").write_text(str(proc.pid))
print("serving on 127.0.0.1:%%s" %% opt("--port"))
''' % {"shards": SHARDS, "model": STRATA_ID}

FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
case "$*" in
  *compute_cap*) echo 7.0 ;;
  *-L*) for i in 0 1 2 3; do echo "GPU $i: Tesla V100-SXM2-32GB (UUID: GPU-$i)"; done ;;
esac
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def _exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def test_bash_syntax():
    for script in ("serve-strata.sh", "serve.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "llm" / "qwen38" / script)], check=True)


def test_plan_prints_the_pinned_launch_command():
    env = dict(os.environ, PORT="8030", MAX_MODEL_LEN="262144", MAX_NUM_SEQS="4", KV_DTYPE="int8")
    out = subprocess.run(["bash", str(SCRIPT), "--plan"], env=env, text=True, capture_output=True,
                         timeout=30, check=True).stdout
    command = next(line for line in out.splitlines() if line.startswith("command:"))
    for want in ("launch", "--context 262144", "--parallel 4", "--kv int8", "--vision off", "--mtp on",
                 r"--gpu 0\,1\,2\,3", "--access local", "--port 8031", "--no-api-key", "--set mtp="):
        assert want in command, (want, command)
    assert "jobs/strata-volta-" in command and "bad_words=none" in out


def test_refuses_without_slurm_job():
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=30)
    assert proc.returncode == 2 and "never on the login node" in proc.stdout


def test_stages_gates_gateway_on_loaded_model_and_stops_on_sigterm(tmp_path):
    strata_root, models, bindir = tmp_path / "engine", tmp_path / "models", tmp_path / "bin"
    for d in (strata_root / "bin", strata_root / "install", models / "iq3s", bindir):
        d.mkdir(parents=True)
    _exe(strata_root / "bin" / "strata", FAKE_STRATA)
    _exe(bindir / "nvidia-smi", FAKE_NVIDIA_SMI)
    for shard in SHARDS:
        (models / "iq3s" / shard).write_bytes(b"x" * 4096)
    rt = models / "_strata-data" / "mtp" / "rt"
    rt.mkdir(parents=True)
    for name in ("dense.bin", "experts.bin", "draft_vocab.bin"):
        (rt / name).write_bytes(b"shared")
    port = _free_port()
    while True:
        uport = _free_port()
        if uport != port:
            break
    job = f"t{os.getpid()}"
    scratch, state = tmp_path / "scratch", tmp_path / "state"
    env = dict(os.environ, SLURM_JOB_ID=job, PORT=str(port), STRATA_PORT=str(uport),
               MAX_MODEL_LEN="262144", MAX_NUM_SEQS="4", KV_DTYPE="int8",
               STRATA_ROOT=str(strata_root), MODELS_ROOT=str(models), SCRATCH_ROOT=str(scratch),
               SLURMHELPS_SERVE_DIR=str(state), STRATA_TEST_SHARD_SIZE="4096",
               FAKE_LOG=str(tmp_path / "calls.jsonl"), PATH=f"{bindir}:{os.environ['PATH']}")
    out = (tmp_path / "out.log").open("w")
    proc = subprocess.Popen(["bash", str(SCRIPT)], env=env, stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 60
        saw_loading = False
        while time.time() < deadline:
            assert proc.poll() is None, (tmp_path / "out.log").read_text()
            status, doc = _get(f"http://127.0.0.1:{uport}/v1/models")
            gw_status, _ = _get(f"http://127.0.0.1:{port}/health")
            if status == 200 and not doc["data"]:
                saw_loading = True
                assert gw_status is None, "gateway must not answer /health while Strata is loading"
            if gw_status == 200:
                break
            time.sleep(0.2)
        else:
            raise AssertionError((tmp_path / "out.log").read_text())
        assert saw_loading
        status, doc = _get(f"http://127.0.0.1:{port}/v1/models")  # through the node gateway
        assert status == 200 and [m["id"] for m in doc["data"]] == [STRATA_ID]
        assert (scratch / f"strata-volta-{job}" / "model" / SHARDS[0]).stat().st_size == 4096
        private = models / "_strata-data" / "jobs" / f"strata-volta-{job}"
        assert (private / "mtp" / "rt" / "experts.bin").is_file()

        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=30)
        log = (tmp_path / "out.log").read_text()
        assert rc == 143, log
        calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        assert calls[0][0] == "launch" and "--stop" in calls[-1]
        assert _get(f"http://127.0.0.1:{uport}/v1/models")[0] is None  # engine stopped
        assert _get(f"http://127.0.0.1:{port}/health")[0] is None      # gateway stopped
        assert not (scratch / f"strata-volta-{job}").exists() and not private.exists()
        assert (rt / "experts.bin").read_bytes() == b"shared"           # shared tree untouched
        assert (state / "strata-logs" / job / "launch" / "x.engine.log").is_file()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        out.close()


def test_early_exit_pipelines_are_sigpipe_safe():
    """Under `set -euo pipefail` a `x=$(producer | reader-that-exits-early)` assignment dies with
    the producer's SIGPIPE (exit 141) and kills the backend: job 38585702 on volta05 died at
    `lib=$(ldconfig -p | awk '...{print $NF; exit}')`. Every such substitution must end in `|| true`."""
    import re
    early = re.compile(r"\|\s*(head\b|awk\b[^|]*\bexit\b|grep\s+-[a-zA-Z]*[qm])")
    bad = []
    for no, line in enumerate(SCRIPT.read_text().splitlines(), 1):
        code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        if "=$(" in code and early.search(code) and "|| true" not in code:
            bad.append(f"{no}: {line.strip()}")
    assert not bad, "unguarded early-exit pipelines:\n" + "\n".join(bad)
