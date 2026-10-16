"""Local ownership and slot reservations for opt-in concurrent GPU queue bridges."""
from __future__ import annotations

import contextlib
from collections import Counter
import fcntl
import json
import os
import socket
import uuid
from pathlib import Path


def queue_root(d):
    return d['running'].parent


def atomic_json(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def process_start(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


class WorkerLease:
    """Locked durable identity, inherited by blocking sync/submit/monitor children."""
    def __init__(self, d):
        directory = queue_root(d) / 'bridge-workers'
        directory.mkdir(exist_ok=True)
        self.owner = {'schema': 1, 'id': uuid.uuid4().hex, 'host': socket.gethostname(),
                      'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      'pid': os.getpid(), 'starttime': process_start(os.getpid())}
        self.path = directory / (self.owner['id'] + '.json')
        self.stream = self.path.open('x+')
        fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        json.dump(self.owner, self.stream)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    @property
    def fd(self):
        return self.stream.fileno()


def owner_status(d, owner):
    """alive/dead/unknown. A held lease also protects orphaned blocking children."""
    try:
        if (owner['schema'] != 1 or owner['host'] != socket.gethostname()
                or len(owner['id']) != 32 or any(c not in '0123456789abcdef' for c in owner['id'])):
            return 'unknown'
        path = queue_root(d) / 'bridge-workers' / (owner['id'] + '.json')
        with path.open('r+') as stream:
            if json.load(stream) != owner:
                return 'unknown'
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 'alive'
            if owner['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
                return 'dead'
            try:
                current = process_start(owner['pid'])
            except (FileNotFoundError, ProcessLookupError):
                return 'dead'
            # A matching PID without its promised lease is ambiguous, never steal it.
            return 'unknown' if current == owner['starttime'] else 'dead'
    except (OSError, ValueError, KeyError, TypeError):
        return 'unknown'


@contextlib.contextmanager
def admission_lock(d):
    with (queue_root(d) / 'bridge-admission.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def legacy_probe(d):
    """Concurrent dispatch never releases somebody else's serial bridge lock."""
    with (queue_root(d) / 'bridge.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        except BlockingIOError:
            held = False
        yield not held


def claims(d):
    rows = []
    for path in d['running'].glob('*.json'):
        # Malformed/unreadable running metadata is a census gap: fail closed.
        row = json.loads(path.read_text())
        if row.get('claimed_by') == 'hpc-bridge':
            owner = row.get('bridge_owner')
            if owner is not None and owner_status(d, owner) == 'unknown':
                raise RuntimeError('bridge owner liveness unknown: ' + row['id'])
            rows.append(row)
    return rows


def occupied_slots(active_names, reserved, prefix, job_name):
    """Count actual allocations plus not-yet-visible reservations, without double count."""
    active = [name for name in active_names if name.startswith(prefix)]
    actual = Counter(active)
    reservations = Counter(prefix + job_name(row) for row in reserved)
    # One visible allocation can satisfy at most one reservation, even on name collisions.
    absent = sum(max(count - actual[name], 0) for name, count in reservations.items())
    return len(active) + absent


def inherited_lease(fd):
    return {} if fd is None else {'pass_fds': (fd,)}
