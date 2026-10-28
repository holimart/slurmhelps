#!/usr/bin/env python3
"""vast.py — canonical, stdlib-only Vast.ai adapter for the Strata campaigns.

Why this file exists: the 2026-10-25 campaigns drove Vast with ad-hoc curl
calls; tools/vast.py in that run dir was a 14-byte "404: Not Found" download.
The lessons are encoded here instead of in shell history:

  * filter offers on `internet_down_cost_per_tb` (a $53/TB host burned $3.80 on
    a 73 GB download while its $0.26/h headline looked cheap);
  * require driver >= 580.65 for the cuda13 engine (and never rent the
    driver-570 3090s that look cheapest);
  * always destroy the instance; `destroy` verifies the list afterwards.

Subcommands: balance, offers, plan, rent, stage, run, upload, pull, list, destroy.
Every mutating command needs --yes; --dry-run prints the request instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_API = "https://console.vast.ai/api/v0"
# Vast deprecated GET /api/v0/instances/ (HTTP 410, 2026-10-28).  Instance
# listing lives on v1; asks/delete still answer on v0.
INSTANCES_API = "https://console.vast.ai/api/v1"
DEFAULT_IMAGE = "vastai/base-image:cuda-13.0.1-auto"
DEFAULT_MAX_DOWN_TB = 5.0
DEFAULT_MIN_DRIVER = "580.65"
DEFAULT_MIN_CUDA = "13.0"


class VastError(RuntimeError):
    pass


def _api_base(args):
    return (args.api or os.environ.get("VAST_API_URL") or DEFAULT_API).rstrip("/")


def api(method, path, body=None, args=None, timeout=30):
    url = _api_base(args) + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    key = (getattr(args, "api_key", None) if args else None) or os.environ.get("VAST_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise VastError("%s %s -> HTTP %s: %s" % (method, url, exc.code, detail))
    except urllib.error.URLError as exc:
        raise VastError("%s %s -> %s" % (method, url, exc.reason))
    try:
        return json.loads(raw)
    except ValueError:
        raise VastError("%s %s -> non-JSON response: %r" % (method, url, raw[:200]))


def api_v1(method, path, body=None, args=None, timeout=30):
    """Call the v1 API (instance listing moved there on 2026-10-28)."""
    if args is None:
        proxy = argparse.Namespace(api=INSTANCES_API, api_key=None)
    else:
        proxy = argparse.Namespace(**vars(args))
        proxy.api = os.environ.get("VAST_API_V1_URL") or INSTANCES_API
    return api(method, path, body=body, args=proxy, timeout=timeout)


def _ver(text):
    out = []
    for part in str(text or "0").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


def _offer_down_tb(offer):
    for key in ("internet_down_cost_per_tb", "inet_down_cost_per_tb",
                "internet_down_cost"):
        value = offer.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _offer_reliability(offer):
    for key in ("reliability2", "reliability"):
        value = offer.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def offer_verdict(offer, opts):
    """Return (eligible, reasons). Filters are explicit so plan can explain."""
    reasons = []
    if offer.get("rentable") is False:
        reasons.append("not rentable")
    want_gpu = (opts.get("gpu_name") or "").strip().lower()
    if want_gpu and want_gpu not in str(offer.get("gpu_name", "")).lower():
        reasons.append("gpu %r != %r" % (offer.get("gpu_name"), opts.get("gpu_name")))
    want_n = int(opts.get("num_gpus") or 1)
    if int(offer.get("num_gpus") or 0) != want_n:
        reasons.append("num_gpus %s != %s" % (offer.get("num_gpus"), want_n))
    # A "1x RTX 3090" offer can still be a fractional slice of the card
    # (gpu_frac 0.25 here); benchmarking on a shared slice is meaningless.
    min_frac = float(opts.get("min_gpu_frac") or 1.0)
    frac = float(offer.get("gpu_frac") or 1.0)
    if frac < min_frac:
        reasons.append("gpu_frac %.3f < %.3f (fractional GPU)" % (frac, min_frac))
    ram_gb = float(offer.get("cpu_ram") or 0) / 1024.0
    if opts.get("min_ram_gb") and ram_gb < float(opts["min_ram_gb"]):
        reasons.append("host RAM %.0f GB < %s GB" % (ram_gb, opts["min_ram_gb"]))
    disk = float(offer.get("disk_space") or 0)
    if opts.get("min_disk_gb") and disk < float(opts["min_disk_gb"]):
        reasons.append("disk %.0f GB < %s GB" % (disk, opts["min_disk_gb"]))
    min_driver = str(opts.get("min_driver") or DEFAULT_MIN_DRIVER)
    driver = str(offer.get("driver_version") or "0")
    if _ver(driver) < _ver(min_driver):
        reasons.append("driver %s < %s" % (driver, min_driver))
    min_cuda = str(opts.get("min_cuda") or DEFAULT_MIN_CUDA)
    cuda = str(offer.get("cuda_max_good") or offer.get("cuda_vers") or "0")
    if _ver(cuda) < _ver(min_cuda):
        reasons.append("cuda %s < %s" % (cuda, min_cuda))
    max_down = float(opts.get("max_down_cost_per_tb") or DEFAULT_MAX_DOWN_TB)
    down = _offer_down_tb(offer)
    if down is None:
        reasons.append("internet_down_cost_per_tb unknown")
    elif down > max_down:
        reasons.append("download $%.2f/TB > $%.2f/TB" % (down, max_down))
    min_rel = float(opts.get("min_reliability") or 0.0)
    rel = _offer_reliability(offer)
    if min_rel and rel < min_rel:
        reasons.append("reliability %.3f < %.3f" % (rel, min_rel))
    if opts.get("max_dph") is not None:
        dph = float(offer.get("dph_total") or 0)
        if dph > float(opts["max_dph"]):
            reasons.append("$%.4f/h > $%.4f/h" % (dph, float(opts["max_dph"])))
    return (not reasons), reasons


def annotate_offer(offer, model_gb, hours):
    down = _offer_down_tb(offer) or 0.0
    dph = float(offer.get("dph_total") or 0.0)
    dl = float(model_gb or 0) / 1000.0 * down
    gpu = dph * float(hours or 0)
    return {
        "id": offer.get("id"),
        "gpu_name": offer.get("gpu_name"),
        "num_gpus": offer.get("num_gpus"),
        "gpu_frac": offer.get("gpu_frac"),
        "dph_total": dph,
        "driver_version": offer.get("driver_version"),
        "cuda_max_good": offer.get("cuda_max_good"),
        "cpu_ram_gb": round(float(offer.get("cpu_ram") or 0) / 1024.0, 1),
        "disk_space_gb": offer.get("disk_space"),
        "internet_down_cost_per_tb": down,
        "down_mbps": offer.get("inet_down") or offer.get("internet_down"),
        "reliability": _offer_reliability(offer),
        "geolocation": offer.get("geolocation"),
        "machine_id": offer.get("machine_id"),
        "download_cost_usd": round(dl, 4),
        "gpu_cost_usd": round(gpu, 4),
        "total_usd": round(dl + gpu, 4),
        "verified": offer.get("verified"),
    }


def filter_offers(offers, opts):
    keep, reject = [], []
    for offer in offers:
        ok, reasons = offer_verdict(offer, opts)
        if ok:
            keep.append(offer)
        else:
            reject.append({"id": offer.get("id"), "gpu_name": offer.get("gpu_name"),
                           "dph_total": offer.get("dph_total"),
                           "reasons": reasons})
    keep.sort(key=lambda o: float(o.get("dph_total") or 1e9))
    return keep, reject


def read_offers(args, opts):
    if args.offers_file:
        data = json.loads(Path(args.offers_file).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("offers") or data.get("bundles") or []
        return data
    query = {
        "rentable": {"eq": True},
        "num_gpus": {"eq": int(opts.get("num_gpus") or 1)},
    }
    if opts.get("gpu_name"):
        query["gpu_name"] = {"eq": opts["gpu_name"]}
    path = "/bundles/?q=" + urllib.parse.quote(json.dumps(query))
    data = api("GET", path, args=args, timeout=args.timeout)
    return data.get("offers") or data.get("bundles") or []


def recipe_options(path):
    """Read offer/filter defaults from a benchctl recipe file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    offer = (raw.get("hardware") or {}).get("offer") or {}
    out = dict(offer)
    out["recipe"] = raw.get("name")
    out["options"] = raw.get("options") or {}
    out["caveats"] = raw.get("caveats") or []
    out["output"] = raw.get("output") or {}
    return out


# ---------------------------------------------------------------- commands
def cmd_balance(args):
    data = api("GET", "/users/current/", args=args, timeout=args.timeout)
    out = {
        "credit": data.get("credit"),
        "balance": data.get("balance"),
        "can_pay": data.get("can_pay"),
        "email": data.get("email"),
    }
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("credit    $%.4f" % float(out["credit"] or 0))
        print("balance   $%.4f" % float(out["balance"] or 0))
        print("can_pay   %s" % out["can_pay"])
        if not out["can_pay"] or float(out["credit"] or 0) <= 0:
            print("fix: top up before renting anything", file=sys.stderr)
    return 0


def _opts_from(args):
    opts = {
        "gpu_name": args.gpu_name,
        "num_gpus": args.num_gpus,
        "min_ram_gb": args.min_ram_gb,
        "min_disk_gb": args.min_disk_gb,
        "min_driver": args.min_driver,
        "min_cuda": args.min_cuda,
        "max_down_cost_per_tb": args.max_down_cost_per_tb,
        "min_reliability": args.min_reliability,
        "max_dph": args.max_dph,
        "min_gpu_frac": getattr(args, "min_gpu_frac", None),
    }
    if args.recipe_file:
        rec = recipe_options(args.recipe_file)
        for key in ("gpu_name", "num_gpus", "min_ram_gb", "min_disk_gb",
                    "min_driver", "max_dph", "max_down_cost_per_tb",
                    "min_reliability", "min_gpu_frac"):
            if opts.get(key) in (None, False) and rec.get(key) is not None:
                opts[key] = rec[key]
        opts["_recipe"] = rec
    opts["num_gpus"] = int(opts.get("num_gpus") or 1)
    return opts


def cmd_offers(args):
    opts = _opts_from(args)
    offers = read_offers(args, opts)
    keep, reject = filter_offers(offers, opts)
    keep = keep[:args.limit]
    model_gb = args.model_gb
    hours = args.hours
    if opts.get("_recipe"):
        model_gb = model_gb or opts["_recipe"].get("model_gb")
        hours = hours or opts["_recipe"].get("hours")
    rows = [annotate_offer(o, model_gb, hours) for o in keep]
    out = {"filters": {k: v for k, v in opts.items() if not k.startswith("_")},
           "eligible": rows, "rejected": len(reject),
           "rejected_examples": reject[:8]}
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("filters: driver >= %s, cuda >= %s, download <= $%.2f/TB, reliability >= %.2f"
              % (opts["min_driver"], opts["min_cuda"], opts["max_down_cost_per_tb"],
                 opts["min_reliability"]))
        if not rows:
            print("no eligible offers (%d rejected)" % len(reject))
            for item in reject[:5]:
                print("  reject %s: %s" % (item["id"], "; ".join(item["reasons"])))
            return 1
        print("%-10s %-24s %-9s %-12s %-7s %-11s %-6s %s"
              % ("offer", "gpu", "$/h", "driver", "down/TB", "down Mbps", "reliab", "geo"))
        for r in rows:
            print("%-10s %-24s %-9.4f %-12s $%-6.2f %-11s %-6.2f %s"
                  % (r["id"], "%dx %s" % (r["num_gpus"], r["gpu_name"]),
                     r["dph_total"], r["driver_version"], r["internet_down_cost_per_tb"],
                     r["down_mbps"], r["reliability"], r["geolocation"]))
    return 0 if rows else 1


def cmd_plan(args):
    opts = _opts_from(args)
    recipe = opts.get("_recipe") or {}
    model_gb = args.model_gb or recipe.get("model_gb") or 0
    hours = args.hours or recipe.get("hours") or 2.0
    offers = read_offers(args, opts)
    keep, reject = filter_offers(offers, opts)
    if args.offer_id:
        keep = [o for o in keep if str(o.get("id")) == str(args.offer_id)]
        if not keep:
            raise VastError("offer %s is not eligible under the filters" % args.offer_id)
    if not keep:
        out = {"recipe": recipe.get("recipe"), "eligible": [], "rejected": len(reject),
               "rejected_examples": reject[:8],
               "error": "no offer passes the filters"}
        print(json.dumps(out, indent=2) if args.json else
              "no eligible offer; %d rejected. Run `vast.py offers --all-rejects` for reasons."
              % len(reject))
        return 1
    picks = [annotate_offer(o, model_gb, hours) for o in keep[:args.limit]]
    best = picks[0]
    out = {"recipe": recipe.get("recipe"), "hours": hours, "model_gb": model_gb,
           "filters": {k: v for k, v in opts.items() if not k.startswith("_")},
           "best": best, "offers": picks, "rejected": len(reject),
           "caveats": recipe.get("caveats") or []}
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("recipe    %s" % (recipe.get("recipe") or "(ad hoc)"))
        print("download  %.1f GB, GPU time %.2f h" % (model_gb, hours))
        print("")
        print("%-10s %-24s %-9s %-10s %-12s %-11s %s"
              % ("offer", "gpu", "$/h", "gpu $", "download $", "total $", "geo"))
        for pick in picks:
            print("%-10s %-24s %-9.4f %-10.2f %-12.4f %-11.2f %s"
                  % (pick["id"], "%dx %s" % (pick["num_gpus"], pick["gpu_name"]),
                     pick["dph_total"], pick["gpu_cost_usd"],
                     pick["download_cost_usd"], pick["total_usd"], pick["geolocation"]))
        print("")
        print("chosen    %s (%s), driver %s, download $%.2f/TB"
              % (best["id"], best["gpu_name"], best["driver_version"],
                 best["internet_down_cost_per_tb"]))
        print("teardown  ALWAYS run `vast.py destroy --instance ID --yes`")
    return 0


def cmd_rent(args):
    opts = _opts_from(args)
    if not args.offer_id:
        raise VastError("rent needs --offer-id (from `vast.py plan`)")
    offers = read_offers(args, opts)
    match = next((o for o in offers if str(o.get("id")) == str(args.offer_id)), None)
    if match is None and not args.offers_file:
        raise VastError("offer %s not in the current list; re-plan" % args.offer_id)
    price = args.price
    if price is None and match is not None:
        price = float(match.get("dph_total") or 0)
    if price is None:
        raise VastError("rent needs --price when the offer is not in the fetched list")
    body = {
        "client_id": "me",
        "image": args.image,
        "env": {},
        "price": price,
        "disk": args.disk,
        "label": args.label,
        "runtype": args.runtype,
        "onstart": args.onstart or "mkdir -p /root/results && echo onstart-ok > /root/results/onstart-ok",
        "cancel_unavail": True,
    }
    if args.dry_run:
        print(json.dumps({"method": "PUT", "path": "/asks/%s/" % args.offer_id,
                          "body": body}, indent=2, sort_keys=True))
        return 0
    if not args.yes:
        raise VastError("refusing to create a billable instance without --yes")
    resp = api("PUT", "/asks/%s/" % args.offer_id, body=body, args=args, timeout=args.timeout)
    contract = resp.get("new_contract") or resp.get("contract_id") or resp.get("id")
    out = {"offer_id": args.offer_id, "instance_id": contract, "response": resp,
           "label": args.label, "price": price, "created_at": _now()}
    if os.environ.get("VAST_SSH_KEY"):
        out["ssh_key"] = os.environ["VAST_SSH_KEY"]
    if contract and not args.no_wait:
        out["instance"] = wait_running(args, contract, args.wait)
    if args.outdir:
        path = Path(args.outdir) / "instance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out["written"] = str(path)
    print(json.dumps(out, indent=2, sort_keys=True) if args.json else
          "instance %s created from offer %s (%s); label %s"
          % (contract, args.offer_id, "$%.4f/h" % float(price or 0), args.label))
    print("remember: pull results, then `vast.py destroy --instance %s --yes`" % contract)
    return 0


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cmd_list(args):
    data = api_v1("GET", "/instances/", args=args, timeout=args.timeout)
    rows = data.get("instances") if isinstance(data, dict) else data
    rows = rows or []
    out = []
    for inst in rows:
        out.append({
            "id": inst.get("id"),
            "label": inst.get("label"),
            "status": inst.get("actual_status") or inst.get("status"),
            "gpu_name": inst.get("gpu_name"),
            "num_gpus": inst.get("num_gpus"),
            "dph_total": inst.get("dph_total"),
            "ssh_host": inst.get("ssh_host") or inst.get("public_ipaddr"),
            "ssh_port": inst.get("ssh_port"),
        })
    if args.json:
        print(json.dumps({"instances": out}, indent=2, sort_keys=True))
    else:
        if not out:
            print("no instances")
        for r in out:
            print("%-10s %-28s %-10s %-22s %s:%s"
                  % (r["id"], r["label"], r["status"],
                     "%sx %s" % (r["num_gpus"], r["gpu_name"]),
                     r["ssh_host"], r["ssh_port"]))
    return 0


def wait_running(args, instance_id, timeout=900):
    deadline = time.time() + float(timeout)
    last = None
    while time.time() < deadline:
        data = api_v1("GET", "/instances/", args=args, timeout=args.timeout)
        rows = data.get("instances") if isinstance(data, dict) else (data or [])
        inst = next((x for x in rows if str(x.get("id")) == str(instance_id)), None)
        if inst:
            last = {
                "id": inst.get("id"),
                "status": inst.get("actual_status") or inst.get("status"),
                "ssh_host": inst.get("ssh_host") or inst.get("public_ipaddr"),
                "ssh_port": inst.get("ssh_port"),
                "gpu_name": inst.get("gpu_name"),
                "dph_total": inst.get("dph_total"),
            }
            if last["status"] in ("running", "exited"):
                return last
        time.sleep(max(5, args.poll))
    return last or {"id": instance_id, "status": "timeout"}


def ssh_argv(args, host, port):
    key = os.path.expanduser(args.ssh_key or os.environ.get("VAST_SSH_KEY")
                             or "~/.ssh/id_ed25519")
    opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15", "-p", str(port)]
    if os.path.exists(key):
        opts += ["-i", key, "-o", "IdentitiesOnly=yes"]
    return ["ssh"] + opts + ["root@%s" % host]


def _instance_ssh(args):
    if args.ssh:
        host, _, port = args.ssh.partition(":")
        return host, int(port or 22)
    if not args.instance:
        raise VastError("need --ssh host:port or --instance ID")
    data = api_v1("GET", "/instances/", args=args, timeout=args.timeout)
    rows = data.get("instances") if isinstance(data, dict) else (data or [])
    inst = next((x for x in rows if str(x.get("id")) == str(args.instance)), None)
    if not inst:
        raise VastError("instance %s not found" % args.instance)
    host = inst.get("ssh_host") or inst.get("public_ipaddr")
    port = inst.get("ssh_port") or 22
    if not host:
        raise VastError("instance %s has no ssh host yet" % args.instance)
    return host, int(port)


def cmd_upload(args):
    host, port = _instance_ssh(args)
    if args.instance and not args.ssh:
        data = api_v1("GET", "/instances/", args=args, timeout=args.timeout)
        rows = data.get("instances") if isinstance(data, dict) else (data or [])
        inst = next((x for x in rows if str(x.get("id")) == str(args.instance)), {})
        host = inst.get("ssh_host") or inst.get("public_ipaddr") or host
        port = inst.get("ssh_port") or port
    cmd = ["rsync", "-az", "--no-owner", "--no-group",
           "-e", " ".join(shlex.quote(x) for x in ssh_argv(args, host, port)[:-1]),
           args.source, "root@%s:%s" % (host, args.dest)]
    if args.dry_run:
        print(" ".join(shlex.quote(x) for x in cmd))
        return 0
    rc = subprocess.call(cmd)
    return rc


def cmd_stage(args):
    """Upload the campaign scripts and run setup.sh on the instance."""
    host, port = (args.ssh.partition(":")[0], int(args.ssh.partition(":")[2] or 22)) \
        if args.ssh else _instance_ssh(args)
    cloud_dir = Path(__file__).resolve().parent
    # The canonical probe (llm/strata/strata_probe.py) carries the two-point
    # chars/token + chat-template intercept calibration.  The cloud copy went
    # stale, so stage the canonical file under the remote name the runner uses.
    canonical_probe = cloud_dir.parents[1] / "strata" / "strata_probe.py"
    files = ["setup.sh", "install-engine.sh",
             str(canonical_probe) if canonical_probe.exists() else "strata_probe.py",
             "run-matrix.sh", "run-matrix2.sh"]
    packs = args.packs
    recipe_setup = None
    if args.recipe_file:
        rec = recipe_options(args.recipe_file)
        recipe_packs = rec.get("options", {}).get("packs")
        recipe_setup = rec.get("options", {}).get("setup_script")
        if recipe_packs:
            if not packs:
                packs = recipe_packs
        if recipe_packs or recipe_setup:
            files.append("setup3090.sh")
    script = args.script or recipe_setup or "setup.sh"
    remote = args.remote_dir or "/root"
    ssh = ssh_argv(args, host, port)
    rsync = ["rsync", "-az", "-e", " ".join(shlex.quote(x) for x in ssh[:-1])]
    rsync += [str(cloud_dir / f) for f in files] + ["root@%s:%s/" % (host, remote)]
    run = ssh + ["cd %s && PACKS_TO_FETCH=%s STRATA_ROOT=%s bash %s"
                 % (shlex.quote(remote), shlex.quote(str(packs or "iq3s coder")),
                    shlex.quote(args.strata_root), shlex.quote(script))]
    if args.dry_run:
        print(" ".join(shlex.quote(x) for x in rsync))
        print(" ".join(shlex.quote(x) for x in run))
        return 0
    if args.instance and not args.yes:
        raise VastError("refusing to run setup on a billable instance without --yes")
    rc = subprocess.call(rsync)
    if rc != 0:
        return rc
    if args.no_run:
        return 0
    if not args.yes:
        raise VastError("refusing to run setup on a billable instance without --yes")
    return subprocess.call(run)


def cmd_run(args):
    host, port = _instance_ssh(args)
    if args.script and os.path.exists(args.script):
        up = argparse.Namespace(**vars(args))
        up.source = args.script
        up.dest = args.remote_dir or "/root"
        rc = cmd_upload(up)
        if rc != 0:
            return rc
        remote_script = (args.remote_dir or "/root").rstrip("/") + "/" + os.path.basename(args.script)
    else:
        remote_script = args.script or "/root/run-matrix.sh"
    assigns = []
    for item in (args.set_env or []):
        key, _, value = item.partition("=")
        assigns.append("%s=%s" % (key, shlex.quote(value)))
    inner = "cd /root && %s bash %s" % (
        " ".join(assigns), shlex.quote(remote_script))
    if args.background:
        inner = "setsid nohup bash -c %s >/root/results/run.log 2>&1 < /dev/null & echo pid=$!" \
                % shlex.quote(inner)
    cmd = ssh_argv(args, host, port) + [inner]
    if args.dry_run:
        print(" ".join(shlex.quote(x) for x in cmd))
        return 0
    if not args.yes:
        raise VastError("refusing to run on a billable instance without --yes")
    return subprocess.call(cmd)


def cmd_pull(args):
    host, port = _instance_ssh(args)
    ssh = ssh_argv(args, host, port)
    dest = Path(args.dest or "vast-results").expanduser()
    cmd = ["rsync", "-az", "--no-owner", "--no-group",
           "-e", " ".join(shlex.quote(x) for x in ssh[:-1]),
           "root@%s:%s" % (host, args.remote or "/root/results/"), str(dest) + "/"]
    if args.dry_run:
        print(" ".join(shlex.quote(x) for x in cmd))
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    return subprocess.call(cmd)


def cmd_destroy(args):
    if not args.instance:
        raise VastError("destroy needs --instance ID")
    if args.dry_run:
        print("DELETE %s/instances/%s/ (then verify the list)" % (_api_base(args), args.instance))
        return 0
    if not args.yes:
        raise VastError("refusing to destroy without --yes")
    resp = api("DELETE", "/instances/%s/" % args.instance, args=args, timeout=args.timeout)
    time.sleep(2)
    data = api_v1("GET", "/instances/", args=args, timeout=args.timeout)
    rows = data.get("instances") if isinstance(data, dict) else (data or [])
    remaining = [x.get("id") for x in rows if str(x.get("id")) == str(args.instance)]
    out = {"instance": args.instance, "response": resp, "still_listed": remaining,
           "total_instances": len(rows)}
    print(json.dumps(out, indent=2, sort_keys=True) if args.json else
          "destroyed %s: success=%s, still_listed=%s, total_instances=%d"
          % (args.instance, resp.get("success"), remaining or "none", len(rows)))
    return 0 if not remaining else 1


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--api", default=None, help="API base (default $VAST_API_URL)")
    p.add_argument("--api-key", default=None, help="API key (default $VAST_API_KEY)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print, do not create/destroy")
    p.add_argument("--timeout", type=int, default=30)
    sub = p.add_subparsers(dest="command", required=True)

    def add_offer_flags(sp):
        sp.add_argument("--gpu-name")
        sp.add_argument("--num-gpus", type=int, default=None)
        sp.add_argument("--min-ram-gb", type=int)
        sp.add_argument("--min-disk-gb", type=int)
        sp.add_argument("--min-driver", default=None)
        sp.add_argument("--min-cuda", default=None)
        sp.add_argument("--max-down-cost-tb", dest="max_down_cost_per_tb", type=float)
        sp.add_argument("--min-reliability", type=float)
        sp.add_argument("--min-gpu-frac", type=float, default=None)
        sp.add_argument("--max-dph", type=float)
        sp.add_argument("--model-gb", type=float)
        sp.add_argument("--hours", type=float)
        sp.add_argument("--offers-file", help="offline JSON fixture of /bundles/ offers")
        sp.add_argument("--recipe-file", help="benchctl recipe JSON")

    sub.add_parser("balance")

    sp = sub.add_parser("offers")
    add_offer_flags(sp)
    sp.add_argument("--limit", type=int, default=10)

    sp = sub.add_parser("plan")
    add_offer_flags(sp)
    sp.add_argument("--limit", type=int, default=5)
    sp.add_argument("--offer-id")

    sp = sub.add_parser("rent")
    add_offer_flags(sp)
    sp.add_argument("--offer-id", required=True)
    sp.add_argument("--label", default="strata-bench")
    sp.add_argument("--disk", type=int, default=160)
    sp.add_argument("--image", default=DEFAULT_IMAGE)
    sp.add_argument("--price", type=float)
    sp.add_argument("--runtype", default="ssh_direc ssh_proxy")
    sp.add_argument("--onstart")
    sp.add_argument("--outdir")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--no-wait", action="store_true")
    sp.add_argument("--wait", type=int, default=900)
    sp.add_argument("--poll", type=int, default=10)

    sub.add_parser("list")

    sp = sub.add_parser("destroy")
    sp.add_argument("--instance", required=True)
    sp.add_argument("--yes", action="store_true")

    for name in ("upload", "pull"):
        sp = sub.add_parser(name)
        sp.add_argument("--instance")
        sp.add_argument("--ssh", help="host:port")
        sp.add_argument("--ssh-key")
        sp.add_argument("--source")
        sp.add_argument("--dest")
        sp.add_argument("--remote")
        sp.add_argument("--yes", action="store_true")

    sp = sub.add_parser("stage")
    sp.add_argument("--instance")
    sp.add_argument("--ssh", help="host:port")
    sp.add_argument("--ssh-key")
    sp.add_argument("--recipe-file")
    sp.add_argument("--script")
    sp.add_argument("--remote-dir")
    sp.add_argument("--strata-root", default="/root/strata-engine")
    sp.add_argument("--packs")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--no-run", action="store_true")

    sp = sub.add_parser("run")
    sp.add_argument("--instance")
    sp.add_argument("--ssh", help="host:port")
    sp.add_argument("--ssh-key")
    sp.add_argument("--script")
    sp.add_argument("--remote-dir")
    sp.add_argument("--set", dest="set_env", action="append", default=[])
    sp.add_argument("--background", action="store_true")
    sp.add_argument("--yes", action="store_true")

    return p


GLOBAL_FLAGS = ("--json", "--dry-run", "--api", "--api-key", "--timeout")


def _normalize_globals(argv):
    """Move global flags (and their values) before the subcommand.

    argparse only accepts parent-parser options before the subcommand, but
    `vast.py plan --json` is the natural thing to type.
    """
    argv = list(argv)
    head, tail, i = [], [], 0
    while i < len(argv):
        token = argv[i]
        if token in ("--json", "--dry-run"):
            head.append(token)
            i += 1
        elif token in ("--api", "--api-key", "--timeout") and i + 1 < len(argv):
            head.extend([token, argv[i + 1]])
            i += 2
        elif any(token.startswith(flag + "=") for flag in ("--api", "--api-key", "--timeout")):
            head.append(token)
            i += 1
        else:
            tail.append(token)
            i += 1
    return head + tail


def main(argv=None):
    argv = _normalize_globals(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        return {
            "balance": cmd_balance,
            "offers": cmd_offers,
            "plan": cmd_plan,
            "rent": cmd_rent,
            "list": cmd_list,
            "destroy": cmd_destroy,
            "upload": cmd_upload,
            "pull": cmd_pull,
            "stage": cmd_stage,
            "run": cmd_run,
        }[args.command](args)
    except VastError as exc:
        print("vast.py: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
