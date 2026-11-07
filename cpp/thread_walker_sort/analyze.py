#!/usr/bin/env python3
"""Join a Thread Walker Sort run with its powermetrics recording.

For every (label, threads) step it prints the median sort time and speedup, then for each CPU
cluster: how many cores were busy on average (the sum of per-CPU active residency, so 2.0 = two
cores fully busy) and the cluster's average active clock. With 1 or 2 heavy threads it also lists
which CPU ids were busy, which shows the fastest tier. (On the M6, Super and Performance share
the "PS" cluster, so the busy CPU ids are what tells them apart.)
Usage: python3 analyze.py results/<host>_<timestamp>
"""
import collections, csv, re, statistics, sys
from datetime import datetime
from pathlib import Path

out = Path(sys.argv[1])
runs = collections.defaultdict(list)  # (label, threads) -> [(sec, start, end)]
order = []
for label, threads, rep, sec, start, end in csv.reader(open(out / "walk.csv")):
    key = (label, int(threads))
    if key not in runs: order.append(key)
    runs[key].append((float(sec), float(start), float(end)))

hdr = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
cl_freq = re.compile(r"^(\S+?)-Cluster HW active frequency:\s+([\d.]+)")
cpu_res = re.compile(r"^CPU (\d+) active residency:\s+([\d.]+)%")
samples, cur, cluster = [], None, None
for line in open(out / "powermetrics.txt", errors="replace"):
    line = line.strip()
    m = hdr.search(line)
    if m:
        t_end = datetime.strptime(" ".join(m.group(1).split()), "%a %b %d %H:%M:%S %Y %z").timestamp()
        cur = {"mid": t_end - float(m.group(2)) / 2000.0, "freq": {}, "busy": collections.Counter(), "cpu": {}}
        samples.append(cur)
        continue
    if cur is None: continue
    m = cl_freq.match(line)
    if m: cluster = m.group(1); cur["freq"][cluster] = float(m.group(2)); continue
    m = cpu_res.match(line)
    if m and cluster:
        cur["busy"][cluster] += float(m.group(2)) / 100.0
        cur["cpu"][int(m.group(1))] = float(m.group(2))

clusters = sorted({c for s in samples for c in s["freq"]})
print("Per cluster: average busy cores (sum of per-CPU active residency) @ average active MHz")
print(f"{'label':16} {'thr':>4} {'sec':>8} {'speedup':>8}  " + "  ".join(f"{c + ' cluster':>18}" for c in clusters) + "  busy CPU ids (>50%)")
base = {}
for label, t in order:
    rs = runs[(label, t)]
    med = statistics.median(r[0] for r in rs)
    base.setdefault(label, med)
    inside = [s for s in samples if any(a <= s["mid"] <= b for _, a, b in rs)]
    cols = []
    for c in clusters:
        if inside:
            cols.append(f"{statistics.mean(s['busy'][c] for s in inside):5.1f} @ {statistics.mean(s['freq'].get(c, 0) for s in inside):5.0f}")
        else:
            cols.append("-")
    hot = ""
    if inside:
        avg = collections.defaultdict(float)
        for s in inside:
            for cpu, r in s["cpu"].items(): avg[cpu] += r / len(inside)
        hot = ",".join(str(c) for c in sorted(avg) if avg[c] > 50)
    print(f"{label:16} {t:>4} {med:8.3f} {base[label] / med:7.2f}x  " + "  ".join(f"{c:>18}" for c in cols) + f"  {hot}")
