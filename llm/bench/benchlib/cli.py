"""benchctl command line: one interface over the existing benchmark backends."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import adapter as adapter_mod
from . import config as config_mod
from . import doctor as doctor_mod
from . import importer as importer_mod
from . import plan as plan_mod
from . import prepare as prepare_mod
from . import recipes as recipes_mod
from . import report as report_mod
from . import status as status_mod
from . import util


def _add_common(sp):
    sp.add_argument("--config", help="benchctl config JSON (default $BENCH_CONFIG)")
    sp.add_argument("--work-root", help="override config work_root (default $BENCH_WORK_ROOT)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")


def _cfg(args):
    return config_mod.load(path=getattr(args, "config", None),
                           work_root=getattr(args, "work_root", None))


def cmd_doctor(args):
    cfg = _cfg(args)
    result = doctor_mod.Doctor(cfg, backends=args.backend,
                               offline=args.offline,
                               deep_sha=args.deep_sha).run()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(doctor_mod.render_text(result))
    if args.strict and not result["ok"]:
        return 1
    return 0


def cmd_plan(args):
    cfg = _cfg(args)
    try:
        recipe = recipes_mod.load_recipe(args.recipe)
    except recipes_mod.RecipeError as exc:
        print("benchctl plan: %s" % exc, file=sys.stderr)
        return 2
    plan = plan_mod.build(recipe, cfg, stamp=args.stamp, dry_run=True)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(plan_mod.render_text(plan))
    return 0


def cmd_prepare(args):
    cfg = _cfg(args)
    try:
        steps = prepare_mod.build(cfg, args.backend, args.model, force=args.force,
                                  submit=args.submit, engine=args.engine,
                                  variant=args.variant, allow_heavy=args.allow_heavy,
                                  recipe=args.recipe, instance=args.instance,
                                  ssh=args.ssh)
    except ValueError as exc:
        print("benchctl prepare: %s" % exc, file=sys.stderr)
        return 2
    results = []
    for step in steps:
        record = {"id": step["id"], "status": "skipped", "command": None,
                  "note": step["note"]}
        if step["argv"] is None:
            results.append(record)
            continue
        record["command"] = util.shell_join(step["argv"])
        if step.get("blocked"):
            record["status"] = "blocked"
            results.append(record)
            continue
        if args.dry_run:
            record["status"] = "dry-run"
            results.append(record)
            continue
        env = dict(os.environ)
        env.update(step.get("env") or {})
        env.setdefault("WORK_ROOT", cfg.work_root)
        proc = subprocess.run(step["argv"], env=env, cwd=step.get("cwd"))
        record["status"] = "ok" if proc.returncode == 0 else "failed"
        record["rc"] = proc.returncode
        results.append(record)
        if proc.returncode != 0:
            break
    if args.json:
        print(json.dumps({"backend": args.backend, "model": args.model,
                          "dry_run": args.dry_run, "steps": results},
                         indent=2, sort_keys=True))
    else:
        print(prepare_mod.render_text(steps, args.dry_run))
        if not args.dry_run:
            print("")
            for record in results:
                print("[%s] %s" % (record["status"], record["id"]))
    if args.dry_run:
        return 0 if all(r["status"] != "failed" for r in results) else 1
    return 0 if all(r["status"] in ("ok", "skipped") for r in results) else 1


def cmd_run(args):
    cfg = _cfg(args)
    try:
        recipe = recipes_mod.load_recipe(args.recipe)
    except recipes_mod.RecipeError as exc:
        print("benchctl run: %s" % exc, file=sys.stderr)
        return 2
    plan = plan_mod.build(recipe, cfg, stamp=args.stamp, dry_run=args.dry_run)
    if plan["estimate"]["outdir_exists"] and not args.force:
        print("benchctl run: output directory already exists: %s\n"
              "refusing to mix runs; pass --force or a different --stamp"
              % plan["outdir"], file=sys.stderr)
        return 2
    if recipe.backend == "vast" and not args.dry_run and not args.yes:
        print(plan_mod.render_text(plan))
        print("\nnot renting: pass --yes to create the instance (billing starts immediately)")
        return 0
    if (recipe.backend in ("vllm", "glm53")
            and recipe.runner in ("measure", "tasks", "bench", "context", "cache")
            and config_mod.is_login_node(cfg) and not args.allow_heavy and not args.dry_run):
        print("benchctl run: %s/%s runs a benchmark client (and sometimes a server); "
              "do not run it on the login node.\n"
              "  - submit it with the backend's own sbatch (see `benchctl plan`), or\n"
              "  - pass --allow-heavy if it only talks to an already-running server."
              % (recipe.backend, recipe.runner), file=sys.stderr)
        return 2
    if not args.dry_run and plan["estimate"]["outdir_exists"] is False:
        Path(plan["outdir"]).mkdir(parents=True, exist_ok=True)
    argv = adapter_mod.adapter_argv(recipe, cfg, Path(plan["outdir"]),
                                    dry_run=args.dry_run, yes=args.yes)
    env = dict(os.environ)
    env.update(adapter_mod.adapter_env(cfg))
    if args.json:
        proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        out = {"plan": plan, "argv": argv, "rc": proc.returncode,
               "stdout": proc.stdout, "stderr": proc.stderr}
        print(json.dumps(out, indent=2, sort_keys=True))
        return proc.returncode
    if args.dry_run:
        print(plan_mod.render_text(plan))
        return 0
    print("submitting: %s" % util.shell_join(argv))
    sys.stdout.flush()
    return subprocess.call(argv, env=env)


def cmd_status(args):
    cfg = _cfg(args)
    data = status_mod.collect(cfg, include_all_jobs=args.all, limit=args.limit)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(status_mod.render_text(data))
    return 0


def cmd_import(args):
    cfg = _cfg(args)
    try:
        plan = importer_mod.collect(cfg, dataset=args.dataset, force=args.force)
    except FileNotFoundError as exc:
        print("benchctl import: %s" % exc, file=sys.stderr)
        return 2
    result = importer_mod.invoke(cfg, plan, dry_run=args.dry_run, replace=args.replace,
                                 date=args.date, configs=args.configs,
                                 timeout=args.timeout)
    out = {"plan": plan, "argv": result["argv"], "rc": result["rc"],
           "state_updated": result["state_updated"],
           "stdout_tail": util.tail(result["out"], 40),
           "stderr_tail": util.tail(result["err"], 20)}
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("importer   %s" % plan["importer"])
        print("runs root  %s" % plan["runs_root"])
        print("new dirs   %d (of %d; %d already imported)"
              % (len(plan["new_dirs"]), plan["total_dirs"], plan["known_dirs"]))
        for entry in plan["new_dirs"]:
            print("  %s" % entry["path"])
        print("command    %s" % util.shell_join(result["argv"]))
        if out["stdout_tail"]:
            print(out["stdout_tail"])
        if out["stderr_tail"]:
            print(out["stderr_tail"], file=sys.stderr)
        print("state      %s" % ("updated" if result["state_updated"] else "unchanged"))
    return 0 if result["rc"] == 0 else result["rc"]


def cmd_report(args):
    cfg = _cfg(args)
    result = report_mod.build(cfg, args.experiment, max_rows=args.max_rows,
                              extra_dirs=args.dir)
    if args.append:
        target = report_mod.append(cfg, result, force=args.force)
        result["appended_to"] = str(target)
        if not args.json:
            print(result["markdown"])
            print("appended to %s" % target, file=sys.stderr)
    elif args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["markdown"])
    return 0


def cmd_recipes(args):
    rows = []
    bad = 0
    for path in sorted(recipes_mod.RECIPE_DIR.glob("*.json")):
        try:
            recipe = recipes_mod.load_recipe(path)
            rows.append({"name": recipe.name, "backend": recipe.backend,
                         "runner": recipe.runner, "model": recipe.model,
                         "experiment": recipe.get("experiment"),
                         "cells": len(recipe.cells), "path": str(path),
                         "ok": True})
        except recipes_mod.RecipeError as exc:
            bad += 1
            rows.append({"name": path.stem, "ok": False, "error": str(exc),
                         "path": str(path)})
    if args.json:
        print(json.dumps({"recipes": rows, "invalid": bad}, indent=2, sort_keys=True))
    else:
        print("%-28s %-7s %-8s %-30s %-6s %s"
              % ("recipe", "backend", "runner", "model", "cells", "experiment"))
        for row in rows:
            if not row["ok"]:
                print("%-28s INVALID %s" % (row["name"], row["error"]))
                continue
            print("%-28s %-7s %-8s %-30s %-6d %s"
                  % (row["name"], row["backend"], row["runner"], row["model"],
                     row["cells"], row.get("experiment") or "-"))
    return 1 if bad else 0


def cmd_config(args):
    cfg = _cfg(args)
    print(json.dumps(cfg.as_dict(), indent=2, sort_keys=True))
    return 0


def cmd_vast(args):
    cfg = _cfg(args)
    vast_py = config_mod.REPO_ROOT / "llm" / "cloud" / "vast" / "vast.py"
    argv = [sys.executable, str(vast_py)] + list(args.args)
    # Accept `--recipe NAME` anywhere and turn it into a recipe path.
    if "--recipe" in argv:
        idx = argv.index("--recipe")
        if idx + 1 >= len(argv):
            print("benchctl vast: --recipe needs a name", file=sys.stderr)
            return 2
        try:
            recipe = recipes_mod.load_recipe(argv[idx + 1])
        except recipes_mod.RecipeError as exc:
            print("benchctl vast: %s" % exc, file=sys.stderr)
            return 2
        argv[idx:idx + 2] = ["--recipe-file", str(recipe.path)]
    env = dict(os.environ)
    env.setdefault("BENCH_CONFIG", str(cfg.config_path))
    return subprocess.call(argv, env=env)


def build_parser():
    p = argparse.ArgumentParser(
        prog="benchctl",
        description="One interface over the Strata, vLLM and GLM benchmark backends.",
        epilog="The backend scripts remain the source of truth; see llm/bench/README.md.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("doctor", help="preconditions and known footguns")
    _add_common(sp)
    sp.add_argument("--backend", action="append", default=None,
                    choices=["all", "strata", "vllm", "glm53", "cloud"],
                    help="limit checks (repeatable; default all)")
    sp.add_argument("--offline", action="store_true", help="skip Vast API calls")
    sp.add_argument("--deep-sha", action="store_true", help="sha256 model + engine files")
    sp.add_argument("--strict", action="store_true", help="exit 1 on any failed check")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("plan", help="print what a recipe would run")
    _add_common(sp)
    sp.add_argument("--recipe", required=True)
    sp.add_argument("--stamp", help="fixed output timestamp (default now)")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("prepare", help="install an engine / fetch a model pack")
    _add_common(sp)
    sp.add_argument("--backend", required=True, choices=["strata", "vllm", "glm53", "cloud"])
    sp.add_argument("--model", required=True, help="pack or model name (all for strata)")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--submit", action="store_true", help="strata: wrap the fetch in sbatch")
    sp.add_argument("--engine", choices=["sglang", "vllm", "llamacpp"])
    sp.add_argument("--variant", choices=["nvfp4", "fp8", "q4_k_s", "q4_k_m", "iq4_xs", "iq3_m"])
    sp.add_argument("--allow-heavy", action="store_true",
                    help="allow image builds/compiled installs on the login node")
    sp.add_argument("--recipe", help="cloud: recipe whose setup script/packs to stage")
    sp.add_argument("--instance", help="cloud: Vast instance id")
    sp.add_argument("--ssh", help="cloud: root@host:port when no instance lookup is wanted")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("run", help="submit a recipe (or print its command)")
    _add_common(sp)
    sp.add_argument("--recipe", required=True)
    sp.add_argument("--stamp")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--yes", action="store_true", help="required for a Vast rent flow")
    sp.add_argument("--force", action="store_true", help="allow an existing output dir")
    sp.add_argument("--allow-heavy", action="store_true",
                    help="allow a client benchmark on the login node")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="our jobs, result dirs, last experiment doc")
    _add_common(sp)
    sp.add_argument("--all", action="store_true", help="all of the user's jobs")
    sp.add_argument("--limit", type=int, default=15)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("import", help="import new result dirs into opportunityscan")
    _add_common(sp)
    sp.add_argument("--dataset", help="dataset checkout (default from config)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--force", action="store_true", help="ignore the import state file")
    sp.add_argument("--replace", action="store_true", help="pass --replace to the importer")
    sp.add_argument("--date")
    sp.add_argument("--configs")
    sp.add_argument("--timeout", type=int, default=900)
    sp.set_defaults(func=cmd_import)

    sp = sub.add_parser("report", help="assemble a markdown section from result dirs")
    _add_common(sp)
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--dir", action="append", default=[], help="extra result dir")
    sp.add_argument("--max-rows", type=int, default=report_mod.MAX_ROWS_DEFAULT)
    sp.add_argument("--append", action="store_true",
                    help="append to llm/bench/experiments/<experiment>.md")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("recipes", help="list/validate recipes")
    _add_common(sp)
    sp.set_defaults(func=cmd_recipes)

    sp = sub.add_parser("config", help="print the resolved configuration")
    _add_common(sp)
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("vast", help="Vast.ai adapter (plan|rent|stage|run|pull|destroy)")
    _add_common(sp)
    sp.add_argument("args", nargs=argparse.REMAINDER,
                    help="subcommand and flags, passed to llm/cloud/vast/vast.py")
    sp.set_defaults(func=cmd_vast)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
