#!/usr/bin/env python3
"""Small local OpenCode agent benchmark for Qwen3.5 (tool use/coding/repo search).

This is intentionally a *local* task suite: it runs the real OpenCode CLI
against a local OpenAI-compatible endpoint and a disposable repository. It
does not need Docker or a public dataset. Public benchmark options are listed
in llm/qwen35/README.md.

Usage:
  python llm/qwen35/opencode_tasks.py \
    --model slurmhelps-qwen/qwen35-122b-a10b \
    --output llm/qwen35/experiments/data/2026-10-23-opencode-qwen35-122b.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

BASE = {
    "README.md": "# Local fixture\n\nSmall Python package for an agent smoke test.\n",
    "calc.py": (
        "def add(a, b):\n"
        "    return a - b\n\n\n"
        "def area_triangle(base, height):\n"
        "    return base * height / 2\n"
    ),
    "ranges.py": (
        "def parse_ranges(text):\n"
        "    \"\"\"Parse '1-3,5,8-9' into [1,2,3,5,8,9].\"\"\"\n"
        "    raise NotImplementedError\n"
    ),
    "tests/test_calc.py": (
        "import unittest\n"
        "from calc import add, area_triangle\n\n\n"
        "class CalcTest(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
        "    def test_area(self):\n"
        "        self.assertEqual(area_triangle(4, 3), 6)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
    "tests/test_ranges.py": (
        "import unittest\n"
        "from ranges import parse_ranges\n\n\n"
        "class RangesTest(unittest.TestCase):\n"
        "    def test_single(self):\n"
        "        self.assertEqual(parse_ranges('5'), [5])\n"
        "    def test_range(self):\n"
        "        self.assertEqual(parse_ranges('1-3,5,8-9'), [1, 2, 3, 5, 8, 9])\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
}

TASKS = {
    "repo_search": (
        "Use repository search (not memory) to find where `area_triangle` is "
        "defined. Write exactly `calc.py:<line-number>` into ANSWER.txt and do "
        "not change any other file. Then stop."
    ),
    "tool_use_fix": (
        "Run the test suite with `python3 -m unittest discover -s tests`. One "
        "test fails because of a bug in calc.py. Find it, fix only calc.py, "
        "rerun the tests, and leave the tests unchanged."
    ),
    "coding": (
        "Implement parse_ranges in ranges.py so `1-3,5,8-9` becomes "
        "[1,2,3,5,8,9]. Run `python3 -m unittest discover -s tests`, fix your "
        "implementation until all tests pass, and do not edit the tests."
    ),
    "multi_step": (
        "Search the repository for the implementation of area_triangle and "
        "the failing add function. Fix the bug in calc.py, run "
        "`python3 -m unittest discover -s tests`, and report the exact test "
        "command and result. Do not edit the tests."
    ),
}


GOOD_RANGES = '''def parse_ranges(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out
'''


def make_repo(root: Path, task: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=False)
    for rel, text in BASE.items():
        if rel == "ranges.py" and task in {"tool_use_fix", "multi_step", "repo_search"}:
            text = GOOD_RANGES
        if rel == "calc.py" and task in {"coding", "repo_search"}:
            text = text.replace("return a - b", "return a + b")
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def run_tests(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", "tests"],
        cwd=root, text=True, capture_output=True, timeout=120,
    )
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def run_task(opencode: str, model: str, task: str, prompt: str, root: Path,
             timeout: int) -> dict:
    env = dict(os.environ, PWD=str(root), OLDPWD=str(root))
    cmd = [
        opencode, "run", "--standalone", "--format", "json", "--auto",
        "--model", model, prompt,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=root, text=True, capture_output=True, timeout=timeout, env=env,
        )
        elapsed = time.perf_counter() - started
        return {
            "task": task, "ok": proc.returncode == 0, "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 3), "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-4000:], "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "task": task, "ok": False, "returncode": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timeout": True,
        }


def evaluate(task: str, root: Path, result: dict) -> dict:
    checks: dict = {}
    if task == "repo_search":
        answer = (root / "ANSWER.txt").read_text().strip() if (root / "ANSWER.txt").is_file() else ""
        checks["answer"] = answer
        checks["answer_ok"] = answer.startswith("calc.py:") and answer.split(":", 1)[1].isdigit()
    elif task in {"tool_use_fix", "coding", "multi_step"}:
        rc, output = run_tests(root)
        checks["tests_returncode"] = rc
        checks["tests_output"] = output
        checks["tests_ok"] = rc == 0
    result["checks"] = checks
    result["passed"] = bool(result["ok"] and all(
        v for k, v in checks.items() if k.endswith("_ok")
    ))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencode", default=shutil.which("opencode") or "opencode")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--work-root", default="/work/ffa/holecem/opencode-bench")
    parser.add_argument("--output", default="")
    parser.add_argument("--only", default="", help="comma-separated task names")
    args = parser.parse_args(argv)
    only = {x for x in args.only.replace(",", " ").split() if x}
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    results = []
    for task, prompt in TASKS.items():
        if only and task not in only:
            continue
        root = Path(tempfile.mkdtemp(prefix=f"task-{task}-", dir=work_root))
        try:
            make_repo(root, task)
            print(f"[opencode-bench] {task}: starting in {root}", flush=True)
            result = run_task(args.opencode, args.model, task, prompt, root, args.timeout)
            result = evaluate(task, root, result)
            results.append(result)
            print(f"[opencode-bench] {task}: {'PASS' if result['passed'] else 'FAIL'} "
                  f"{result['elapsed_s']:.1f}s", flush=True)
        finally:
            shutil.rmtree(root, ignore_errors=True)
    report = {"model": args.model, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "results": results,
              "passed": sum(1 for r in results if r["passed"]), "total": len(results)}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
