"""Local synthetic ownership/admission tests; no network, model or real queue."""
import fcntl
import json
import multiprocessing as mp
import os
import subprocess
import signal
import time
import sys

import pytest

import bridge_dispatch as dispatch
import gpu_queue
import hpc_bridge_gpu_queue as bridge
from test_hpc_bridge_gpu_queue import FakeCluster, claimed, make_job, repo  # noqa: F401


def dirs_at(root):
    d = {name: root / name for name in ('pending', 'running', 'done')}
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    return d


def test_live_unknown_pid_reuse_and_read_gap(tmp_path, monkeypatch):
    d = dirs_at(tmp_path)
    with dispatch.WorkerLease(d) as worker:
        owner = worker.owner
        assert dispatch.owner_status(d, owner) == 'alive'
    assert dispatch.owner_status(d, owner) == 'unknown'  # live PID, unlocked lease
    monkeypatch.setattr(dispatch, 'process_start', lambda _: 'different-start')
    assert dispatch.owner_status(d, owner) == 'dead'
    worker.path.unlink()
    assert dispatch.owner_status(d, owner) == 'unknown'
    assert dispatch.owner_status(d, {**owner, 'host': 'other-host'}) == 'unknown'


def test_orphan_helper_keeps_lease(tmp_path, monkeypatch):
    d = dirs_at(tmp_path)
    worker = dispatch.WorkerLease(d)
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                             stdin=subprocess.PIPE, pass_fds=(worker.fd,))
    try:
        worker.__exit__()
        # Even a missing owner PID cannot override the inherited child's live lock.
        with monkeypatch.context() as patch:
            def dead_pid(_):
                raise FileNotFoundError('synthetic dead bridge owner')
            patch.setattr(dispatch, 'process_start', dead_pid)
            assert dispatch.owner_status(d, worker.owner) == 'alive'
    finally:
        child.communicate(timeout=5)
    assert dispatch.owner_status(d, worker.owner) == 'unknown'


def test_recovery_never_steals_live_or_ambiguous_submitted(repo, monkeypatch):
    d, job = claimed(repo)
    with dispatch.WorkerLease(d) as worker:
        bridge.set_stage(d, job, 'syncing', bridge_owner=worker.owner)
        bridge.recover_stale(d)
        assert (d['running'] / (job['id'] + '.json')).exists()
    monkeypatch.setattr(dispatch, 'process_start', lambda _: 'reused-pid')
    bridge.set_stage(d, job, 'submitted')
    bridge.recover_stale(d)
    assert (d['running'] / (job['id'] + '.json')).exists()
    bridge.set_stage(d, job, 'syncing')
    bridge.recover_stale(d)
    assert (d['pending'] / (job['id'] + '.json')).exists()
    assert 'bridge_owner' not in json.loads((d['pending'] / (job['id'] + '.json')).read_text())


def test_unknown_owner_and_malformed_running_fail_closed(repo):
    d, job = claimed(repo)
    bridge.set_stage(d, job, 'claimed', bridge_owner={'schema': 1})
    with pytest.raises(RuntimeError):
        bridge.recover_stale(d)
    with pytest.raises(RuntimeError):
        dispatch.claims(d)
    (d['running'] / 'bad.json').write_text('{')
    with pytest.raises((RuntimeError, ValueError)):
        dispatch.claims(d)


def test_capacity_counts_invisible_reservations_and_duplicate_allocations():
    rows = [{'name': 'a'}, {'name': 'b'}]
    assert dispatch.occupied_slots(['wb-a', 'wb-a', 'foreign'], rows, 'wb-',
                                   lambda row: row['name']) == 3


def race_worker(root, start, release, results):
    """Real independent flock owners competing for the fifth slot."""
    d = dirs_at(root)
    with dispatch.WorkerLease(d) as worker:
        start.wait(5)
        with dispatch.admission_lock(d):
            rows = dispatch.claims(d)
            occupied = dispatch.occupied_slots(['wb-existing' + str(i) for i in range(4)],
                                               rows, 'wb-', lambda row: row['id'])
            won = occupied < 5
            if won:
                candidate = sorted(d['pending'].glob('*.json'))[0]
                job = gpu_queue.claim_job(d, candidate.stem, 'hpc-bridge')
                bridge.set_stage(d, job, 'claimed', bridge_owner=worker.owner)
        results.put(won)
        release.wait(5)


def test_two_process_last_slot_race(tmp_path):
    d = dirs_at(tmp_path)
    for jid in ('one', 'two'):
        (d['pending'] / (jid + '.json')).write_text(json.dumps({'id': jid}))
    ctx = mp.get_context('fork')
    start, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=race_worker, args=(tmp_path, start, release, results))
                for _ in range(2)]
    try:
        for child in children:
            child.start()
        start.set()
        assert sorted([results.get(timeout=5), results.get(timeout=5)]) == [False, True]
        assert len(list(d['running'].glob('*.json'))) == 1
        assert len(list(d['pending'].glob('*.json'))) == 1
    finally:
        release.set()
        for child in children:
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join()
        assert all(child.exitcode == 0 for child in children)


@pytest.mark.parametrize('failure', ['no-receipt', 'interrupted'])
def test_ambiguous_owned_submit_stays_reserved(repo, monkeypatch, failure):
    d, job = claimed(repo)
    fake = FakeCluster(repo, write_result=False, hpc_rc=1)
    def run(argv, **kwargs):
        if failure == 'interrupted' and argv[:2] == ['bash', str(bridge.HPC_RUN)]:
            raise bridge.Interrupted('synthetic stop')
        return fake(argv, **kwargs)
    monkeypatch.setattr(bridge.subprocess, 'run', run)
    with dispatch.WorkerLease(d) as worker:
        bridge.set_stage(d, job, 'claimed', bridge_owner=worker.owner)
        if failure == 'interrupted':
            with pytest.raises(bridge.Interrupted):
                bridge.run_job(d, job)
        else:
            assert bridge.run_job(d, job) == 1
    saved = json.loads((d['running'] / (job['id'] + '.json')).read_text())
    assert saved['bridge_reconcile_required'] is True
    assert not list(d['done'].glob('*.json'))


def test_concurrent_respects_legacy_owner_without_stealing(repo, monkeypatch):
    d, job = claimed(repo)
    with (gpu_queue.Q / 'bridge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(bridge, 'our_cluster_job_names', lambda: [])
        monkeypatch.setattr(bridge, 'pick_for_bridge', lambda _: [])
        assert bridge.dispatch_one(d, 5, concurrent=True) == bridge.EXIT_NOTHING
    assert (d['running'] / (job['id'] + '.json')).exists()


def test_serial_respects_concurrent_reservation(repo, monkeypatch):
    d, job = claimed(repo)
    with dispatch.WorkerLease(d) as worker:
        bridge.set_stage(d, job, 'claimed', bridge_owner=worker.owner)
        monkeypatch.setattr(bridge, 'our_cluster_jobs', lambda: 0)
        monkeypatch.setattr(bridge, 'pick_for_bridge', lambda _: pytest.fail('capacity already reserved'))
        assert bridge.dispatch_one(d, 1) == bridge.EXIT_NOTHING


def test_submit_children_inherit_owner_lease(repo, monkeypatch):
    d, job = claimed(repo)
    fake = FakeCluster(repo)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return fake(argv, **kwargs)
    monkeypatch.setattr(bridge.subprocess, 'run', run)
    with dispatch.WorkerLease(d) as worker:
        monkeypatch.setattr(bridge, 'WORKER_LEASE_FD', worker.fd)
        bridge.set_stage(d, job, 'claimed', bridge_owner=worker.owner)
        assert bridge.run_job(d, job) == 0
        assert calls and all(kw['pass_fds'] == (worker.fd,) for _, kw in calls)


def test_dead_submit_owner_with_live_inherited_helper_retains_reservation(tmp_path):
    d = dirs_at(tmp_path)
    script = """
import json, os, subprocess, sys
from pathlib import Path
import bridge_dispatch as dispatch
root = Path(sys.argv[1])
d = {name: root / name for name in ('pending', 'running', 'done')}
worker = dispatch.WorkerLease(d)
helper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                          pass_fds=(worker.fd,), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
job = {'id': 'owned-submit', 'claimed_by': 'hpc-bridge', 'bridge_stage': 'submitted',
       'bridge_owner': worker.owner}
dispatch.atomic_json(d['running'] / 'owned-submit.json', job)
print(json.dumps({'owner': worker.owner, 'helper': helper.pid}), flush=True)
os._exit(0)
"""
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            env={**os.environ, 'PYTHONPATH': str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'bin')},
                            capture_output=True, text=True, timeout=5, check=True)
    evidence = json.loads(result.stdout)
    try:
        assert dispatch.owner_status(d, evidence['owner']) == 'alive'
        bridge.recover_stale(d)
        assert len(dispatch.claims(d)) == 1
    finally:
        os.kill(evidence['helper'], signal.SIGTERM)
    deadline = time.monotonic() + 3
    while dispatch.owner_status(d, evidence['owner']) == 'alive' and time.monotonic() < deadline:
        time.sleep(.01)
    assert dispatch.owner_status(d, evidence['owner']) == 'dead'
    bridge.recover_stale(d)
    assert (d['running'] / 'owned-submit.json').exists()
    assert not list(d['pending'].glob('*.json'))


def test_reservation_name_collision_preserves_multiplicity():
    rows = [{'name': 'collision'}, {'name': 'collision'}]
    name = lambda row: row['name']
    assert dispatch.occupied_slots([], rows, 'wb-', name) == 2
    assert dispatch.occupied_slots(['wb-collision'], rows, 'wb-', name) == 2
    assert dispatch.occupied_slots(['wb-collision'] * 3, rows, 'wb-', name) == 3


def test_dispatch_does_not_claim_colliding_output_name(repo, monkeypatch):
    d = gpu_queue.dirs()
    # Distinct IDs collide under the real eighty-character cluster-name truncation.
    prefix = 'x' * 90
    for jid in (prefix + 'one', prefix + 'two'):
        make_job(d, jid, '2026-10-10T00:00:00+00:00')
    job = gpu_queue.claim_job(d, prefix + 'one', 'hpc-bridge')
    with dispatch.WorkerLease(d) as worker:
        bridge.set_stage(d, job, 'claimed', bridge_owner=worker.owner)
        monkeypatch.setattr(bridge, 'our_cluster_job_names', lambda: [])
        monkeypatch.setattr(bridge, 'pick_for_bridge', lambda _: bridge.load_pending(d))
        monkeypatch.setattr(bridge, 'run_job', lambda *_: pytest.fail('colliding output claimed'))
        assert bridge.dispatch_one(d, 5, concurrent=True) == bridge.EXIT_NOTHING
    assert len(list(d['pending'].glob('*.json'))) == 1
