"""`benchctl import`: hand new result dirs to the opportunityscan dataset importer.

The dataset importer (modelsquantdb/scripts/import_strata_runs.py) stays the
single writer of the dataset.  benchctl only decides *which* run dirs are new
(a small state file under the work root), invokes it, and records success.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config as config_mod
from . import util


def _scan(root):
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    try:
        for entry in root.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                out.append({"path": str(entry), "mtime": entry.stat().st_mtime})
            except OSError:
                continue
    except OSError:
        return []
    out.sort(key=lambda x: x["mtime"])
    return out


def _load_state(cfg):
    return util.read_json(cfg.state_path("import.json"), {}) or {}


def collect(cfg, dataset=None, force=False):
    importer = Path(dataset).expanduser() / "modelsquantdb/scripts/import_strata_runs.py" \
        if dataset else cfg.dataset_importer()
    if not importer.is_file():
        raise FileNotFoundError(
            "dataset importer not found: %s (pass --dataset PATH)" % importer)
    root = Path(cfg.path("strata_runs"))
    state = _load_state(cfg)
    imported = state.get("imported", {})
    current = _scan(root)
    if force:
        new = current
    else:
        new = [d for d in current
               if d["path"] not in imported
               or float(imported[d["path"]].get("mtime", 0)) < d["mtime"]]
    return {
        "importer": str(importer),
        "runs_root": str(root),
        "state_file": str(cfg.state_path("import.json")),
        "new_dirs": new,
        "known_dirs": len(imported),
        "total_dirs": len(current),
    }


def invoke(cfg, plan, dry_run=False, replace=False, date=None, configs=None,
           python=None, timeout=900):
    cmd = [python or sys.executable, plan["importer"],
           "--runs-root", plan["runs_root"]]
    if dry_run:
        cmd.append("--dry-run")
    if replace:
        cmd.append("--replace")
    if date:
        cmd.extend(["--date", date])
    if configs:
        cmd.extend(["--configs", configs])
    result = util.run(cmd, timeout=timeout)
    result["argv"] = cmd
    if result["rc"] == 0 and not dry_run:
        state = _load_state(cfg)
        imported = state.setdefault("imported", {})
        now = util.utc_iso()
        for entry in plan["new_dirs"]:
            imported[entry["path"]] = {"mtime": entry["mtime"], "imported_at": now}
        state["last_run"] = {"at": now, "runs_root": plan["runs_root"],
                             "new_dirs": [e["path"] for e in plan["new_dirs"]]}
        util.write_json(plan["state_file"], state)
        result["state_updated"] = True
    else:
        result["state_updated"] = False
    return result
