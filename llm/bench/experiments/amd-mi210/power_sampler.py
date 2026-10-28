#!/usr/bin/env python3
"""power_sampler.py — 1 Hz ROCm power/clock/temperature/energy sampler.

Reads the amdgpu hwmon sysfs directly (cheap, no rocm-smi fork per sample) and
falls back to rocm-smi text parsing if a file is missing.

CSV columns:
  epoch,gpu,power_w,sclk_mhz,mclk_mhz,temp_c,use_pct,energy_j

`energy_j` is the cumulative hwmon energy counter when available; the reporter
prefers its delta over integrating power.  `power_w` is power1_average.

usage: power_sampler.py --out power.csv [--duration -]
"""
import argparse
import glob
import os
import re
import subprocess
import time


def find_hwmon(card):
    # /sys/class/drm/card0/device/hwmon/hwmon3
    for h in glob.glob("/sys/class/drm/card%d/device/hwmon/hwmon*" % card):
        return h
    return None


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def read_sclk(card):
    for p in ("/sys/class/drm/card%d/device/pp_dpm_sclk" % card,
              "/sys/class/drm/card%d/device/hwmon/hwmon*/pp_dpm_sclk" % card):
        for f in glob.glob(p):
            try:
                cur = None
                with open(f) as fh:
                    for line in fh:
                        m = re.search(r"(\d+)\s*Mhz\s*\*", line)
                        if m:
                            cur = int(m.group(1))
                if cur is not None:
                    return cur
            except Exception:
                pass
    return None


def rocm_smi_sample():
    """Fallback: one rocm-smi call, parse the text output."""
    out = {}
    try:
        txt = subprocess.run(
            ["rocm-smi", "--showpower", "--showclocks", "--showtemp", "--showuse"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return out
    gpu = None
    for line in txt.splitlines():
        m = re.match(r"GPU\[(\d+)\]", line)
        if m:
            gpu = int(m.group(1))
            out.setdefault(gpu, {})
        if gpu is None:
            continue
        if "Graphics Package Power" in line:
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                out[gpu]["power_w"] = float(m.group(1))
        if "sclk clock level" in line:
            m = re.search(r"\((\d+)Mhz\)", line)
            if m:
                out[gpu]["sclk_mhz"] = int(m.group(1))
        if "mclk clock level" in line:
            m = re.search(r"\((\d+)Mhz\)", line)
            if m:
                out[gpu]["mclk_mhz"] = int(m.group(1))
        if "Sensor edge" in line:
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                out[gpu]["temp_c"] = float(m.group(1))
        if "GPU use" in line:
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                out[gpu]["use_pct"] = float(m.group(1))
        if "Energy counter" in line:
            m = re.search(r":\s*(\d+)", line)
            if m:
                out[gpu]["energy_j"] = int(m.group(1)) / 1e6  # rocm-smi reports uJ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", default="-", help="seconds, or - until killed")
    ap.add_argument("--cards", type=int, default=2)
    args = ap.parse_args()

    hw = [find_hwmon(c) for c in range(args.cards)]
    start = time.time()
    with open(args.out, "w") as f:
        f.write("epoch,gpu,power_w,sclk_mhz,mclk_mhz,temp_c,use_pct,energy_j\n")
        while True:
            ts = time.time()
            fallback = None
            for c in range(args.cards):
                p = None
                if hw[c]:
                    p = read_int(os.path.join(hw[c], "power1_average"))
                    if p is None:
                        p = read_int(os.path.join(hw[c], "power1_input"))
                    e = read_int(os.path.join(hw[c], "energy1_input"))
                    t = read_int(os.path.join(hw[c], "temp1_input"))
                    u = read_int(os.path.join(hw[c], "gpu_busy_percent"))
                    row = {
                        "power_w": None if p is None else p / 1e6,
                        "sclk_mhz": read_sclk(c),
                        "mclk_mhz": None,
                        "temp_c": None if t is None else t / 1000.0,
                        "use_pct": u,
                        "energy_j": None if e is None else e / 1e6,
                    }
                else:
                    if fallback is None:
                        fallback = rocm_smi_sample()
                    row = fallback.get(c, {})
                f.write("%.3f,%d,%s,%s,%s,%s,%s,%s\n" % (
                    ts, c,
                    "" if row.get("power_w") is None else "%.2f" % row["power_w"],
                    "" if row.get("sclk_mhz") is None else row["sclk_mhz"],
                    "" if row.get("mclk_mhz") is None else row["mclk_mhz"],
                    "" if row.get("temp_c") is None else "%.1f" % row["temp_c"],
                    "" if row.get("use_pct") is None else "%.1f" % row["use_pct"],
                    "" if row.get("energy_j") is None else "%.3f" % row["energy_j"]))
            f.flush()
            if args.duration != "-" and time.time() - start >= float(args.duration):
                break
            time.sleep(max(0.0, 1.0 - (time.time() - ts)))


if __name__ == "__main__":
    main()
