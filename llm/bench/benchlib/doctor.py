"""`benchctl doctor`: preconditions and the known footguns.

Nothing here starts a server, loads a model or downloads a pack.  Checks are
read-only (a few shell probes and at most one Vast API call) and each one names
the fix when it is not ok.  Missing optional backends are warnings, not
failures; `--strict` turns real failures into a non-zero exit.
"""
from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path

from . import config as config_mod
from . import recipes as recipes_mod
from . import util

# Pinned from llm/strata/README.md and the 2026-10-25 experiments doc.
MODEL_PACKS = {
    "iq3s": [
        ("iq3s", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", 54817524224,
         "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
        ("iq3s", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf", 28800138432,
         "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    ],
    "coder": [
        ("coder", "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", 29608446496,
         "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
        ("coder", "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf", 28800138432,
         "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    ],
    "q2_0": [
        ("q2_0", "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", 37623740192,
         "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
        ("q2_0", "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", 28800138432,
         "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    ],
}

GLIBCXX_FLOOR = (3, 4, 30)  # cuda13 engine; cuda12 floor is lower (575.51 driver)
DRIVER_CUDA13 = "580.65"
DRIVER_CUDA12 = "575.51"

BACKEND_GROUPS = {
    "strata": {"host", "repo", "slurm", "nodes", "engine.strata", "libs",
               "models.strata", "storage", "footguns"},
    "vllm": {"host", "repo", "slurm", "nodes", "engine.vllm", "models.qwen",
             "storage"},
    "glm53": {"host", "repo", "slurm", "nodes", "engine.glm53", "models.glm53",
              "storage"},
    "cloud": {"host", "repo", "cloud", "storage"},
}


def _status_rank(status):
    return {"fail": 3, "warn": 2, "ok": 1, "info": 0, "skip": 0}.get(status, 0)


class Doctor:
    def __init__(self, cfg, backends=("all",), offline=False, deep_sha=False):
        self.cfg = cfg
        self.backends = set(backends or ["all"])
        self.offline = offline
        self.deep_sha = deep_sha
        self.checks = []
        self._groups = self._wanted_groups()

    def _wanted_groups(self):
        if "all" in self.backends:
            return None
        groups = set()
        for b in self.backends:
            groups |= BACKEND_GROUPS.get(b, set())
        return groups

    def wanted(self, group):
        return self._groups is None or group in self._groups

    def add(self, cid, status, detail, fix=None, group="host", data=None):
        self.checks.append({
            "id": cid,
            "status": status,
            "detail": detail,
            "fix": fix,
            "group": group,
            "data": data or {},
        })

    # ---- host / repo -------------------------------------------------
    def check_host(self):
        host = socket.gethostname()
        login = config_mod.is_login_node(self.cfg)
        self.add("host.login_node", "ok" if login else "warn",
                 "%s (%s)" % (host, "login node" if login else "not the configured login node"),
                 None if login else "heavy runs belong in Slurm; only prepare/plan/doctor here",
                 data={"hostname": host, "login": login})
        self.add("host.python", "ok", "%s (%s)" % (
            __import__("sys").version.split()[0], __import__("sys").executable))
        rc = util.run(["nvidia-smi", "-L"], timeout=5)
        if rc["rc"] != 0:
            self.add("host.driver", "info",
                     "no driver on this host (expected on a login node); compute-node floor is "
                     "driver >= %s for cuda13, >= %s for cuda12" % (DRIVER_CUDA13, DRIVER_CUDA12),
                     "`srun --partition=gpu-ffa nvidia-smi` to read a node's driver")
        else:
            self.add("host.driver", "ok", rc["out"].strip().splitlines()[0] if rc["out"] else "driver visible")
        try:
            self.add("host.ram", "info", "%s total, %s available" % (
                util.human_bytes(_meminfo("MemTotal") * 1024),
                util.human_bytes(_meminfo("MemAvailable") * 1024)))
        except OSError:
            pass

    def check_repo(self):
        r = util.run(["git", "-C", str(config_mod.REPO_ROOT), "rev-parse", "--short", "HEAD"], timeout=10)
        branch = util.run(["git", "-C", str(config_mod.REPO_ROOT), "branch", "--show-current"], timeout=10)
        dirty = util.run(["git", "-C", str(config_mod.REPO_ROOT), "status", "--porcelain"], timeout=15)
        if r["rc"] == 0:
            n = len([x for x in dirty["out"].splitlines() if x.strip()])
            self.add("repo.git", "ok" if n == 0 else "warn",
                     "%s on %s, %d changed path(s)" % (r["out"].strip(), branch["out"].strip() or "?", n),
                     "commit or stash other sessions' work before a campaign")
        else:
            self.add("repo.git", "warn", "not a git checkout: %s" % config_mod.REPO_ROOT)
        self.add("repo.config", "ok" if self.cfg.loaded else "warn",
                 "%s (%s)" % (self.cfg.config_path, "loaded" if self.cfg.loaded else "missing, using defaults"),
                 "add llm/bench/config.json or set BENCH_CONFIG")
        if not Path(self.cfg.work_root).is_dir():
            self.add("repo.work_root", "warn", "work root does not exist: %s" % self.cfg.work_root,
                     "set BENCH_WORK_ROOT or fix config.json")

    # ---- Slurm -------------------------------------------------------
    def check_slurm(self):
        missing = [t for t in ("sbatch", "squeue", "sinfo") if not util.which(t)]
        if missing:
            self.add("slurm.tools", "fail", "missing: %s" % ", ".join(missing),
                     "run on a submit host or add the Slurm bin directory to PATH")
            return
        self.add("slurm.tools", "ok", "sbatch/squeue/sinfo on PATH")
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        jobs = util.run(["squeue", "-h", "-u", user, "-o", "%i|%j|%T"], timeout=20)
        if jobs["rc"] == 0:
            rows = [x for x in jobs["out"].splitlines() if x.strip()]
            ours = [x for x in rows if _is_our_job(x.split("|")[1] if "|" in x else x)]
            self.add("slurm.queue", "info",
                     "%d job(s) for %s, %d benchmark job(s)" % (len(rows), user, len(ours)),
                     data={"jobs": ours[:20]})

    def check_nodes(self):
        hw_parts = set()
        for name in recipes_mod.recipe_names():
            try:
                rec = recipes_mod.load_recipe(name)
            except recipes_mod.RecipeError:
                continue
            if rec.hardware.get("kind") == "cluster" and rec.hardware.get("partition"):
                hw_parts.add(rec.hardware["partition"])
        out = util.run(["sinfo", "-h", "-N", "-o", "%N|%P|%G|%m|%t"], timeout=25)
        if out["rc"] != 0:
            self.add("nodes.sinfo", "warn", "sinfo failed: %s" % util.tail(out["err"], 3))
            return
        nodes = []
        for line in out["out"].splitlines():
            bits = line.split("|")
            if len(bits) < 5:
                continue
            node, part, gres, mem, state = bits[0], bits[1].rstrip("*"), bits[2], bits[3], bits[4]
            m = re.search(r"gpu:([A-Za-z0-9_.-]+):([0-9]+)", gres)
            if not m:
                continue
            nodes.append({
                "node": node,
                "partition": part,
                "gpu": m.group(1),
                "gpu_count": int(m.group(2)),
                "ram_mb": int(mem) if mem.isdigit() else 0,
                "state": state,
            })
        for part in sorted(hw_parts):
            rows = [n for n in nodes if n["partition"] == part]
            if not rows:
                continue
            idle = [n for n in rows if n["state"] in ("idle", "mixed")]
            by_type = {}
            for n in idle:
                by_type.setdefault(n["gpu"], [0, 0])
                by_type[n["gpu"]][0] += 1
                by_type[n["gpu"]][1] += n["gpu_count"]
            detail = ", ".join("%s: %d node(s)/%d card(s)" % (k, v[0], v[1])
                               for k, v in sorted(by_type.items())) or "no idle GPU node"
            self.add("nodes.%s" % part, "ok" if by_type else "warn",
                     "%s — %s (of %d GPU node(s))" % (part, detail, len(rows)),
                     None if by_type else "choose another partition or wait; never queue on fuuk*" ,
                     group="nodes", data={"partition": part, "idle": by_type})
        self.add("nodes.driver_floor", "info",
                 "driver/CUDA per node is not visible from the login node; the runners precheck "
                 "GLIBCXX and cuda13 needs driver >= %s" % DRIVER_CUDA13,
                 group="nodes")

    # ---- engines -----------------------------------------------------
    def check_strata(self):
        root = Path(self.cfg.path("strata_engine"))
        binary = root / "bin" / "strata"
        if not binary.exists():
            self.add("engine.strata", "warn", "missing %s" % binary,
                     "llm/strata/install-engine.sh (login node, network only)",
                     group="engine.strata")
            return
        comps = sorted((root / "components" / "engine").glob("*cuda-*"))
        variants = [p.name for p in comps]
        detail = "%s -> %s; components: %s" % (
            binary, os.path.realpath(str(binary)), ", ".join(variants) or "none")
        status = "ok" if len(comps) >= 2 else "warn"
        self.add("engine.strata", status, detail,
                 None if status == "ok" else "install-engine.sh installs both cuda13 and cuda12",
                 group="engine.strata", data={"components": variants})
        for comp in comps:
            exe = comp / "strata"
            sha = util.sha256_file(exe, limit=None) if self.deep_sha and exe.is_file() else None
            data = {"path": str(exe), "size": util.file_size(exe)}
            if sha:
                data["sha256"] = sha
            self.add("engine.strata.%s" % ("cuda13" if "cuda13" in comp.name else "cuda12"),
                     "ok" if exe.is_file() else "warn",
                     "%s %s" % (exe.name, util.human_bytes(util.file_size(exe))),
                     group="engine.strata", data=data)
        self.add("engine.strata.cli", "info",
                 "`strata --version` can block on the login node (no GPU); doctor reads the "
                 "component directory instead",
                 group="engine.strata")

    def check_libs(self):
        floor = "%d.%d.%d" % GLIBCXX_FLOOR
        sys_lib = "/lib64/libstdc++.so.6"
        best = _glibcxx_max(sys_lib)
        cxx = Path(self.cfg.path("cxx_lib")) / "libstdc++.so.6"
        cxx_best = _glibcxx_max(str(cxx)) if cxx.exists() else None
        if best and _ver_tuple(best) >= GLIBCXX_FLOOR:
            self.add("libs.glibcxx.system", "ok", "%s has GLIBCXX_%s" % (sys_lib, best), group="libs")
        else:
            fix = ("the cuda13 engine needs GLIBCXX_%s; point CXX_LIB_DIR at a newer libstdc++"
                   % floor)
            if cxx_best and _ver_tuple(cxx_best) >= GLIBCXX_FLOOR:
                fix = "use CXX_LIB_DIR=%s (has GLIBCXX_%s)" % (cxx.parent, cxx_best)
            self.add("libs.glibcxx.system", "warn",
                     "%s has GLIBCXX_%s (< %s)" % (sys_lib, best or "?", floor),
                     fix, group="libs")
        self.add("libs.glibcxx.cxx_lib", "ok" if cxx_best else "warn",
                 "%s: GLIBCXX_%s" % (cxx, cxx_best or "missing"),
                 None if cxx_best else "the l40x1 deps/cxx/lib tree was not found",
                 group="libs")

    # ---- models ------------------------------------------------------
    def check_strata_models(self):
        root = Path(self.cfg.path("strata_models"))
        for pack, files in sorted(MODEL_PACKS.items()):
            missing, wrong, ok = [], [], 0
            total = 0
            for sub, name, size, sha in files:
                p = root / sub / name
                if not p.exists():
                    missing.append(name)
                    continue
                actual = util.file_size(p)
                total += max(0, actual)
                if actual != size:
                    wrong.append("%s=%d (want %d)" % (name, actual, size))
                    continue
                if self.deep_sha:
                    got = util.sha256_file(p)
                    if got != sha:
                        wrong.append("%s sha256=%s (want %s)" % (name, got[:12], sha[:12]))
                        continue
                ok += 1
            if missing or wrong:
                status = "warn"
                detail = "pack %s: %d/%d file(s) ok" % (pack, ok, len(files))
                if missing:
                    detail += "; missing: " + ", ".join(missing)
                if wrong:
                    detail += "; " + "; ".join(wrong)
                self.add("models.strata.%s" % pack, status, detail,
                         "llm/strata/fetch-models.sh --only %s" % pack, group="models.strata")
            else:
                self.add("models.strata.%s" % pack, "ok",
                         "pack %s complete (%s)%s" % (
                             pack, util.human_bytes(total),
                             " sha256 verified" if self.deep_sha else "; sizes match"),
                         group="models.strata")
        table = root / "iq3s" / MODEL_PACKS["iq3s"][1][1]
        try:
            nlink = table.stat().st_nlink
            self.add("models.strata.shared_table", "ok" if nlink >= 2 else "warn",
                     "shared PLE shard has %d hardlink(s)" % nlink,
                     None if nlink >= 2 else "packs should share one 28.8 GB table (fetch-models.sh does this)",
                     group="models.strata")
        except OSError:
            pass

    def check_vllm(self):
        sif = Path(self.cfg.path("qwen38_sif"))
        model = Path(self.cfg.path("qwen38_model"))
        if sif.is_file():
            self.add("engine.vllm.qwen38_sif", "ok",
                     "%s (%s)" % (sif, util.human_bytes(util.file_size(sif))),
                     group="engine.vllm")
        else:
            self.add("engine.vllm.qwen38_sif", "warn", "missing %s" % sif,
                     "llm/qwen38/pull-container.sh (inside a Slurm job; heavy)", group="engine.vllm")
        if (model / "config.json").is_file():
            self.add("models.qwen38", "ok", "%s" % model, group="models.qwen")
        else:
            self.add("models.qwen38", "warn", "missing checkpoint at %s" % model,
                     "the FP4 local checkpoint is a site artifact; see llm/qwen38/README.md",
                     group="models.qwen")
        vllm = Path(self.cfg.path("venvs")) / "vllm032py" / "bin" / "vllm"
        self.add("engine.vllm.venv", "ok" if vllm.is_file() else "warn",
                 "%s" % vllm, None if vllm.is_file() else "stock vLLM venv missing",
                 group="engine.vllm")
        for label, key in (("qwen35-122b", "qwen35_122b_model"),
                           ("qwen35-35b", "qwen35_35b_model")):
            p = Path(self.cfg.path(key))
            ok = (p / "config.json").is_file()
            self.add("models.%s" % label, "ok" if ok else "warn",
                     "%s%s" % (p, "" if ok else " (missing)"),
                     None if ok else "hf download cyankiwi/Qwen3.5-%s-AWQ-4bit --local-dir %s"
                     % ("122B-A10B" if "122" in label else "35B-A3B", p),
                     group="models.qwen")

    def check_glm53(self):
        model = Path(self.cfg.path("glm_model"))
        has_model = any(model.glob("*")) if model.is_dir() else False
        self.add("models.glm53", "ok" if has_model else "warn",
                 "%s%s" % (model, "" if has_model else " (empty/missing)"),
                 None if has_model else
                 "llm/glm53-reap50/install.sh --engine llamacpp --variant iq4_xs (heavy: run in Slurm)",
                 group="models.glm53")
        vllm = Path(self.cfg.path("venvs")) / "vllm032py" / "bin" / "vllm"
        self.add("engine.glm53.vllm", "ok" if vllm.is_file() else "warn",
                 "%s" % vllm, None if vllm.is_file() else "vLLM 0.32 venv missing",
                 group="engine.glm53")

    # ---- storage -----------------------------------------------------
    def check_storage(self):
        checks = [("work", Path(self.cfg.work_root)),
                  ("tmp", Path(self.cfg.path("node_tmp")))]
        for label, path in checks:
            try:
                usage = __import__("shutil").disk_usage(str(path))
            except OSError as exc:
                self.add("storage.%s" % label, "warn", "cannot stat %s: %s" % (path, exc),
                         group="storage")
                continue
            free = usage.free
            status = "ok"
            fix = None
            if label == "tmp" and free < 90 * (1 << 30):
                status = "warn"
                fix = ("IQ3_S staging needs ~86 GiB; use SCRATCH_ROOT on node NVMe or "
                       "COPY_MODELS=0")
            self.add("storage.%s" % label, status,
                     "%s: %s free of %s" % (path, util.human_bytes(free),
                                            util.human_bytes(usage.total)),
                     fix, group="storage")

    # ---- cloud -------------------------------------------------------
    def check_cloud(self):
        vast_py = config_mod.BENCH_DIR.parent / "cloud" / "vast" / "vast.py"
        self.add("cloud.vast_client", "ok" if vast_py.is_file() else "fail",
                 "%s" % vast_py, None if vast_py.is_file() else "llm/cloud/vast/vast.py is missing",
                 group="cloud")
        key = os.environ.get("VAST_API_KEY") or ""
        self.add("cloud.vast_key", "ok" if key else "warn",
                 "VAST_API_KEY %s" % ("set" if key else "not set"),
                 None if key else "export VAST_API_KEY=... (never commit it)", group="cloud")
        if self.offline:
            self.add("cloud.vast_credit", "skip", "--offline: balance not queried", group="cloud")
        elif key and vast_py.is_file():
            bal = util.run([__import__("sys").executable, str(vast_py), "balance", "--json"],
                           timeout=15)
            if bal["rc"] == 0:
                try:
                    data = json.loads(bal["out"])
                    credit = float(data.get("credit", 0))
                    can_pay = bool(data.get("can_pay"))
                    self.add("cloud.vast_credit", "ok" if (credit > 0 and can_pay) else "warn",
                             "credit $%.2f, can_pay=%s" % (credit, can_pay),
                             None if (credit > 0 and can_pay) else
                             "top up before renting; a $53/TB host once burned $3.80 on 73 GB",
                             group="cloud", data=data)
                except ValueError:
                    self.add("cloud.vast_credit", "warn", "unparseable balance output", group="cloud")
            else:
                self.add("cloud.vast_credit", "warn",
                         "balance query failed: %s" % util.tail(bal["err"] or bal["out"], 2),
                         group="cloud")
        else:
            self.add("cloud.vast_credit", "skip", "no VAST_API_KEY; balance not queried", group="cloud")
        max_down = float(self.cfg.vast.get("max_down_cost_per_tb", 5.0))
        min_driver = str(self.cfg.vast.get("min_driver", DRIVER_CUDA13))
        self.add("cloud.offer_filter", "info",
                 "offers must satisfy driver >= %s, internet_down_cost_per_tb <= $%.1f/TB, "
                 "reliability >= %.2f" % (min_driver, max_down,
                                          float(self.cfg.vast.get("min_reliability", 0.97))),
                 group="cloud")

    # ---- footguns ----------------------------------------------------
    def check_footguns(self):
        root = Path(self.cfg.path("strata_engine"))
        bad = []
        for path in sorted((root / "install").glob("strata-*.json")):
            data = util.read_json(path, {}) or {}
            if data.get("mtp") is False:
                bad.append(path.name)
        if bad:
            self.add("footgun.mtp_persisted", "warn",
                     "installed config has mtp=false in: %s" % ", ".join(bad),
                     "every runner passes --mtp on explicitly; a concurrent `strata launch` of "
                     "the same pack can rewrite this shared file",
                     group="footguns")
        else:
            self.add("footgun.mtp_persisted", "ok",
                     "no install/strata-*.json has mtp=false", group="footguns")
        shared = Path(self.cfg.path("strata_data"))
        rt = shared / "mtp" / "rt"
        self.add("footgun.mtp_tree", "ok" if rt.is_dir() else "warn",
                 "%s %s" % (rt, "present" if rt.is_dir() else "missing"),
                 None if rt.is_dir() else "first launch downloads the ~5.2 GB drafter; "
                 "DRAFTER-READY is NOT a readiness signal, check the engine log",
                 group="footguns")
        probe = config_mod.BENCH_DIR / "probe.py"
        upstream = config_mod.REPO_ROOT / "llm" / "strata" / "strata_probe.py"
        if probe.is_file() and upstream.is_file():
            same = util.sha256_file(probe) == util.sha256_file(upstream)
            text = probe.read_text(encoding="utf-8", errors="replace")
            calibrated = "calibrate_chars_per_token" in text and "--max-context" in text
            self.add("footgun.probe", "ok" if (same and calibrated) else "warn",
                     "probe.py %s upstream, calibration=%s" % (
                         "matches" if same else "DIFFERS from", calibrated),
                     None if (same and calibrated) else
                     "copy llm/strata/strata_probe.py again (do not edit either file in place)",
                     group="footguns")
        self.add("footgun.slurm_export", "info",
                 "never put comma lists in sbatch --export (Slurm splits at commas); "
                 "the adapters pass matrix knobs as trailing argv",
                 group="footguns")
        self.add("footgun.engine_config", "info",
                 "strata launch rewrites install/strata-<pack>.json; the runners pin the recipe "
                 "with --set and verify the engine start line",
                 group="footguns")
        fetch = util.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n",
                          "strata-fetch", "-o", "%i"], timeout=15)
        if fetch["rc"] == 0 and fetch["out"].strip():
            self.add("footgun.verify_concurrency", "warn",
                     "a strata-fetch job is running (%s); concurrent --verify audits of the "
                     "114 GB library contend badly" % ", ".join(fetch["out"].split()),
                     "run one audit at a time", group="footguns")

    # ---- driver ------------------------------------------------------
    def run(self):
        self.check_host()
        self.check_repo()
        if self.wanted("slurm"):
            self.check_slurm()
        if self.wanted("nodes"):
            self.check_nodes()
        if self.wanted("engine.strata"):
            self.check_strata()
        if self.wanted("libs"):
            self.check_libs()
        if self.wanted("models.strata"):
            self.check_strata_models()
        if self.wanted("engine.vllm") or self.wanted("models.qwen"):
            self.check_vllm()
        if self.wanted("engine.glm53") or self.wanted("models.glm53"):
            self.check_glm53()
        if self.wanted("storage"):
            self.check_storage()
        if self.wanted("cloud"):
            self.check_cloud()
        if self.wanted("footguns"):
            self.check_footguns()
        summary = {"ok": 0, "warn": 0, "fail": 0, "info": 0, "skip": 0}
        for c in self.checks:
            summary[c["status"]] = summary.get(c["status"], 0) + 1
        return {
            "generated_at": util.utc_iso(),
            "backends": sorted(self.backends),
            "offline": self.offline,
            "config": self.cfg.as_dict(),
            "summary": summary,
            "checks": self.checks,
            "ok": summary["fail"] == 0,
        }


def _meminfo(key):
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return 0


def _glibcxx_max(path):
    if not os.path.isfile(path):
        return None
    out = util.run(["strings", "-a", path], timeout=30)
    if out["rc"] != 0:
        return None
    versions = re.findall(r"GLIBCXX_([0-9]+\.[0-9]+\.[0-9]+)", out["out"])
    if not versions:
        return None
    return max(versions, key=_ver_tuple)


def _ver_tuple(text):
    try:
        return tuple(int(x) for x in str(text).split("."))
    except ValueError:
        return (0,)


def _is_our_job(name):
    name = str(name)
    for prefix in ("strata-", "glm53-", "qwen38-", "qwen35-", "bench-"):
        if name.startswith(prefix):
            return True
    return False


def render_text(result):
    symbol = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "info": "[info]", "skip": "[skip]"}
    lines = []
    order = []
    groups = {}
    for check in result["checks"]:
        if check["group"] not in groups:
            groups[check["group"]] = []
            order.append(check["group"])
        groups[check["group"]].append(check)
    for group in order:
        lines.append("")
        lines.append("== %s ==" % group)
        for check in groups[group]:
            lines.append("%s %-32s %s" % (symbol.get(check["status"], "[????]"),
                                          check["id"], check["detail"]))
            if check.get("fix") and check["status"] in ("warn", "fail"):
                lines.append("       fix: %s" % check["fix"])
    s = result["summary"]
    lines.append("")
    lines.append("summary: %d ok, %d warn, %d fail, %d info, %d skipped"
                 % (s.get("ok", 0), s.get("warn", 0), s.get("fail", 0),
                    s.get("info", 0), s.get("skip", 0)))
    return "\n".join(lines)
