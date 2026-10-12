"""Tests for cluster/agents/index-hygiene.py. Run: python3 -m unittest cluster/agents/test_index_hygiene.py

Each test builds its own temporary repository; nothing touches the ADGFEM checkout.
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("index_hygiene", os.path.join(HERE, "index-hygiene.py"))
ih = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ih)


def _git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=repo, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()


def _write(repo, rel, text):
    with open(os.path.join(repo, rel), "w") as fh:
        fh.write(text)


def _read(repo, rel):
    with open(os.path.join(repo, rel)) as fh:
        return fh.read()


def _run_main(repo, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ih.main(["--repo", repo, *extra])
    return rc, json.loads(buf.getvalue())


class IndexHygieneTest(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="idxhyg-")
        r = self.repo
        _git(r, "init", "-q", "-b", "main")
        _write(r, "stale.txt", "version one, long content line\n")
        _write(r, "d.txt", "present in worktree and HEAD\n")
        _write(r, "keep.txt", "base\n")
        _git(r, "add", ".")
        _git(r, "commit", "-qm", "c1")
        _write(r, "stale.txt", "v1\n")
        _git(r, "commit", "-qam", "c2")          # older blob "v1\n"
        _write(r, "stale.txt", "version two, a longer stale line\n")
        _git(r, "commit", "-qam", "c3")
        older = _git(r, "rev-parse", "HEAD~1:stale.txt")
        # A "lane" commits newer content through a private index: HEAD moves, shared index does not.
        _write(r, "stale.txt", "version three, newest content\n")
        priv = os.path.join(r, ".lane-private-index")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_INDEX_FILE"] = priv
        subprocess.run(["git", "read-tree", "HEAD"], cwd=r, env=env, check=True)
        subprocess.run(["git", "update-index", "--add", "stale.txt"], cwd=r, env=env, check=True)
        tree = subprocess.run(["git", "write-tree"], cwd=r, env=env, check=True,
                              stdout=subprocess.PIPE).stdout.decode().strip()
        commit = _git(r, "commit-tree", tree, "-p", "HEAD", "-m", "lane")
        _git(r, "update-ref", "HEAD", commit)
        os.remove(priv)
        # Shared index is now stale for stale.txt (older blob), as the lanes leave it.
        _git(r, "update-index", "--cacheinfo", f"100644,{older},stale.txt")
        _git(r, "rm", "-q", "--cached", "d.txt")                     # staged D, file still exists
        _write(r, "keep.txt", "genuinely edited\n")
        _git(r, "add", "keep.txt")                                    # genuine staged edit
        _write(r, "new.txt", "brand new\n")
        _git(r, "add", "new.txt")                                     # genuine new file

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_dry_run_classifies_and_does_not_touch_index(self):
        before = _git(self.repo, "diff", "--cached", "--name-status")
        rc, out = _run_main(self.repo)
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"stale": ["d.txt", "stale.txt"], "kept": ["keep.txt", "new.txt"], "applied": False})
        self.assertEqual(_git(self.repo, "diff", "--cached", "--name-status"), before)

    def test_apply_unstages_only_stale_and_keeps_worktree(self):
        rc, out = _run_main(self.repo, "--apply")
        self.assertEqual(rc, 0)
        self.assertTrue(out["applied"])
        self.assertEqual(out["stale"], ["d.txt", "stale.txt"])
        self.assertEqual(_git(self.repo, "diff", "--cached", "--name-status").split(),
                         ["M", "keep.txt", "A", "new.txt"])
        self.assertEqual(_read(self.repo, "stale.txt"), "version three, newest content\n")
        self.assertTrue(os.path.exists(os.path.join(self.repo, "d.txt")))
        self.assertEqual(_read(self.repo, "keep.txt"), "genuinely edited\n")
        rc, again = _run_main(self.repo)
        self.assertEqual(again["stale"], [])

    def test_apply_waits_on_index_lock_and_never_deletes_it(self):
        lock = os.path.join(self.repo, ".git", "index.lock")
        open(lock, "w").close()
        old_wait = ih.LOCK_WAIT_S, ih.LOCK_POLL_S
        ih.LOCK_WAIT_S, ih.LOCK_POLL_S = 1.0, 0.2
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                rc, out = _run_main(self.repo, "--apply")
        finally:
            ih.LOCK_WAIT_S, ih.LOCK_POLL_S = old_wait
        self.assertEqual(rc, 0)
        self.assertFalse(out["applied"])
        self.assertTrue(os.path.exists(lock))
        os.remove(lock)

    def test_non_repository_fails(self):
        plain = tempfile.mkdtemp(prefix="idxhyg-plain-")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                rc = ih.main(["--repo", plain])
            self.assertEqual(rc, 1)
        finally:
            shutil.rmtree(plain, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
