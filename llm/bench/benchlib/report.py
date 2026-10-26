"""`benchctl report`: project result artifacts into a markdown section.

No prose is invented.  Each table names the file it came from; caveats come
verbatim from the recipe.  The command writes to stdout unless --append is
given, and --append is restricted to llm/bench/experiments/ so a generated
section never rewrites a hand-written campaign document.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config as config_mod
from . import recipes as recipes_mod
from . import util

PREFERRED_SUMMARY = ["ctx", "context", "concurrency", "variant", "pack", "mode",
                     "repeat", "decode_tok_s_p50", "decode_p50", "aggregate_tok_s",
                     "aggregate", "ttft_s_p50", "ttft_p50", "ttft_s_p95",
                     "ttft_p95", "prompt_tokens", "status", "error"]
SKIP_DIRS = {"attic", "data", "deps", "engine-logs", "procs", "tokenizer",
             "packs", "mtp", "__pycache__", ".cache", "_attempts"}
MAX_ROWS_DEFAULT = 200


def find_doc(slug):
    hits = []
    for pattern in ("*/experiments/%s.md" % slug, "bench/experiments/%s.md" % slug):
        hits.extend(config_mod.REPO_ROOT.glob("llm/" + pattern))
    return hits[0] if hits else None


def evidence_for(cfg, slug):
    out = []
    seen = set()
    for name in recipes_mod.recipe_names():
        try:
            rec = recipes_mod.load_recipe(name)
        except recipes_mod.RecipeError:
            continue
        if rec.get("experiment") != slug:
            continue
        for item in rec.evidence_paths():
            path = config_mod.resolve_evidence(cfg, item)
            key = str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _read_csv(path):
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _md_cell(value):
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")[:120]


def _md_table(columns, rows, max_rows):
    lines = ["| " + " | ".join(_md_cell(c) for c in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    shown = rows[:max_rows]
    for row in shown:
        lines.append("| " + " | ".join(_md_cell(row.get(c, "")) for c in columns) + " |")
    if len(rows) > len(shown):
        lines.append("| ... | %d more row(s) not shown |%s"
                     % (len(rows) - len(shown), " |" * (len(columns) - 2)))
    return lines


def _summary_table(path, max_rows):
    rows = _read_csv(path)
    if not rows:
        return None
    header = list(rows[0].keys())
    columns = [c for c in PREFERRED_SUMMARY if c in header]
    if len(columns) < 3:
        columns = header
    return {"source": str(path), "columns": columns, "rows": rows[:max_rows],
            "total_rows": len(rows)}


def _json_table(path, max_rows):
    data = util.read_json(path)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("rows"), list) and data["rows"]:
        first = data["rows"][0]
        if "prefix_tokens" in first:
            wanted = ["prefix_tokens", "status", "fill_ttft_s", "hit_ttft_s",
                      "hit_reuse_ratio", "hit_decode_tok_s", "repeat_ttft_s",
                      "cache_reuse_verified"]
        else:
            wanted = ["context", "concurrency", "status", "output_tokens_per_s",
                      "total_tokens_per_s", "ttft_s", "tpot_ms"]
        cols = [c for c in wanted if c in first]
        if len(cols) < 3:
            cols = list(first.keys())
        return {"source": str(path), "columns": cols,
                "rows": data["rows"][:max_rows], "total_rows": len(data["rows"])}
    if isinstance(data.get("points"), list) and data["points"]:
        rows = []
        for point in data["points"]:
            runs = point.get("runs") or [{}]
            run = runs[-1]
            ttft = (run.get("ttft_ms") or {}).get("p50")
            tpot = (run.get("tpot_ms") or {}).get("p50")
            rows.append({
                "concurrency": point.get("concurrency"),
                "output_tokens_per_s": run.get("output_tokens_per_s"),
                "total_tokens_per_s": run.get("total_tokens_per_s"),
                "ttft_ms_p50": ttft,
                "tpot_ms_p50": tpot,
                "completed": run.get("completed"),
                "failed": run.get("failed"),
            })
        return {"source": str(path),
                "columns": ["concurrency", "output_tokens_per_s", "total_tokens_per_s",
                            "ttft_ms_p50", "tpot_ms_p50", "completed", "failed"],
                "rows": rows[:max_rows], "total_rows": len(rows)}
    if isinstance(data.get("results"), list) and data["results"]:
        first = data["results"][0]
        if "name" in first and "status" in first:
            rows = [{"check": r.get("name"), "status": r.get("status"),
                     "detail": r.get("detail")} for r in data["results"]]
            return {"source": str(path), "columns": ["check", "status", "detail"],
                    "rows": rows[:max_rows], "total_rows": len(rows)}
    return None


def _cells_table(path, max_rows):
    rows = []
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                rows.append({
                    "mode": d.get("mode", "cold"),
                    "context": d.get("context"),
                    "concurrency": d.get("concurrency"),
                    "decode_tok_s_p50": d.get("decode_tok_s_p50"),
                    "aggregate_tok_s": d.get("aggregate_tok_s"),
                    "ttft_s_p50": d.get("ttft_s_p50"),
                    "prompt_tokens": d.get("prompt_tokens"),
                    "error": (str(d.get("error"))[:60] if d.get("error") else ""),
                })
    except OSError:
        return None
    if not rows:
        return None
    return {"source": str(path),
            "columns": ["mode", "context", "concurrency", "decode_tok_s_p50",
                        "aggregate_tok_s", "ttft_s_p50", "prompt_tokens", "error"],
            "rows": rows[:max_rows], "total_rows": len(rows)}


def collect_tables(directory, max_rows, seen=None):
    directory = Path(directory)
    seen = seen if seen is not None else set()
    tables = []
    if not directory.exists():
        return tables, ["evidence path does not exist: %s" % directory]
    notes = []
    if directory.is_file():
        name = directory.name
        if name.endswith(".csv"):
            table = _summary_table(directory, max_rows)
        elif name.endswith(".jsonl"):
            table = _cells_table(directory, max_rows)
        elif name.endswith(".json"):
            table = _json_table(directory, max_rows)
        else:
            table = None
        if table and table["source"] not in seen:
            seen.add(table["source"])
            tables.append(table)
        return tables, notes
    try:
        files = sorted(directory.rglob("*"),
                       key=lambda p: (len(p.relative_to(directory).parts), str(p)))
    except OSError as exc:
        return tables, ["cannot read %s: %s" % (directory, exc)]

    def add(table):
        if table and table["source"] not in seen:
            seen.add(table["source"])
            tables.append(table)

    # Summaries first (compact, one row per cell), then sweeps and raw cells.
    for path in files:
        if not path.is_file() or path.name.startswith("."):
            continue
        rel_parts = path.relative_to(directory).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if ".bak" in path.name or ".bak-" in str(path):
            continue
        name = path.name
        if name in ("summary.csv", "summary-warm.csv", "knobs.csv", "warm-vs-cold.csv"):
            add(_summary_table(path, max_rows))
        elif name.endswith(".json") and name not in ("launch-config.json",):
            # Skip multi-GB artifacts and per-request dumps.
            if util.file_size(path) > 8 * 1024 * 1024:
                notes.append("skipped large json: %s" % path)
                continue
            add(_json_table(path, max_rows))
    if not tables:
        for path in files:
            if path.is_file() and path.name == "cells.jsonl":
                rel_parts = path.relative_to(directory).parts
                if any(part in SKIP_DIRS for part in rel_parts[:-1]):
                    continue
                add(_cells_table(path, max_rows))
    return tables, notes


def build(cfg, slug, max_rows=MAX_ROWS_DEFAULT, extra_dirs=None):
    doc = find_doc(slug)
    dirs = evidence_for(cfg, slug)
    for item in extra_dirs or []:
        dirs.append(Path(item).expanduser().resolve())
    sections = []
    all_notes = []
    seen = set()
    for directory in dirs:
        tables, notes = collect_tables(directory, max_rows, seen)
        all_notes.extend(notes)
        if tables:
            sections.append({"dir": str(directory), "tables": tables})
    caveats = []
    grouped_caveats = []
    for name in recipes_mod.recipe_names():
        try:
            rec = recipes_mod.load_recipe(name)
        except recipes_mod.RecipeError:
            continue
        if rec.get("experiment") == slug:
            for caveat in rec.caveats:
                if caveat not in caveats:
                    caveats.append(caveat)
                grouped_caveats.append({"recipe": rec.name, "caveat": caveat})

    lines = ["## benchctl evidence — %s" % slug, ""]
    lines.append("_Generated %s from %d result dir(s). Each table is a projection of the "
                 "file named above it; no prose is inferred._" % (util.utc_iso(), len(sections)))
    lines.append("")
    if doc:
        lines.append("Campaign document: `%s`" % doc.relative_to(config_mod.REPO_ROOT))
        lines.append("")
    if not sections:
        lines.append("_No result artifacts matched this experiment. Record evidence paths in "
                     "the recipe or pass --dir._")
        lines.append("")
    for section in sections:
        try:
            rel = str(Path(section["dir"]).relative_to(config_mod.REPO_ROOT))
        except ValueError:
            rel = section["dir"]
        lines.append("### %s" % rel)
        lines.append("")
        for table in section["tables"]:
            src = table["source"]
            try:
                src = str(Path(src).relative_to(config_mod.REPO_ROOT))
            except ValueError:
                pass
            lines.append("`%s` (%d row(s))" % (src, table["total_rows"]))
            lines.append("")
            lines.extend(_md_table(table["columns"], table["rows"], max_rows))
            lines.append("")
    if grouped_caveats:
        lines.append("### Caveats (verbatim from the recipes)")
        lines.append("")
        seen_caveat = set()
        for item in grouped_caveats:
            key = (item["recipe"], item["caveat"])
            if key in seen_caveat:
                continue
            seen_caveat.add(key)
            lines.append("- `%s`: %s" % (item["recipe"], item["caveat"]))
        lines.append("")
    if all_notes:
        lines.append("### Artifact notes")
        lines.append("")
        for note in all_notes:
            lines.append("- %s" % note)
        lines.append("")
    return {
        "experiment": slug,
        "doc": str(doc) if doc else None,
        "directories": [str(d) for d in dirs],
        "sections": sections,
        "caveats": caveats,
        "caveats_by_recipe": grouped_caveats,
        "notes": all_notes,
        "markdown": "\n".join(lines).rstrip() + "\n",
    }


def append(cfg, result, force=False):
    slug = result["experiment"]
    target = config_mod.BENCH_DIR / "experiments" / ("%s.md" % slug)
    allowed = target.resolve().is_relative_to((config_mod.BENCH_DIR / "experiments").resolve()) \
        if hasattr(Path, "is_relative_to") else str(target).startswith(
            str(config_mod.BENCH_DIR / "experiments"))
    if not allowed and not force:
        raise ValueError("refusing to append outside llm/bench/experiments: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with open(str(target), "w", encoding="utf-8") as fh:
            fh.write("# %s — benchctl generated report\n\n" % slug)
            fh.write("This file is machine-assembled. Hand-written campaign notes live next to "
                     "the backend that ran the measurement.\n\n")
    with open(str(target), "a", encoding="utf-8") as fh:
        fh.write("\n" + result["markdown"])
    return target
