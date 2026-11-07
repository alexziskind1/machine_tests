# Thread Walker Sort

Sorts the **same data** with 1, 2, 3 … N threads and times each step, to show how
performance scales as each extra core joins, and which **core types** do the work. It's built
for chips with mixed core types, such as the M6 (2 Super + 4 Performance + 6 Efficiency) vs the
M5 (4 Super + 6 Efficiency).

It uses the `../sorting_v2` multi-threaded sort: `std::mt19937` data (seed 42, full 32-bit
range), chunks quicksorted in parallel, then every merge pass split across all threads. With 1
thread it's a plain single-threaded quicksort. Only the sort is timed; every run is verified
(sorted + same values), and a failure stops the walk.

## Why it works without core pinning

macOS has no API to pin threads to cores. Instead:

- **All-cores walk:** macOS puts busy threads on the fastest free cores first. On the M6, threads
  1–2 land on the Super cores, 3–6 add the Performance cores, and 7–12 add the Efficiency cores.
  On the M5, threads 1–4 are Super and 5–10 are Efficiency.
- **Lowest-tier walk:** `taskpolicy -b` (background priority) keeps the process on the lowest core
  tier: the Efficiency cores on M5/M6, and the Performance cores on M5 Max, which has no E-cores.
  So 1..E threads measures that tier alone. Background priority also holds those cores at a **low
  clock** (about 1 GHz on the M6's E-cores), so treat this walk as "that tier alone, throttled".
  The cores' real contribution at full clock is the tail of the all-cores walk.
- **Proof:** `POWERMETRICS=1` records per-cluster activity with `powermetrics`, and `analyze.py`
  shows how busy each cluster was, and at what clock, during each step.

## Run (macOS)

```bash
./run_walk.sh                     # n=200M ints, 3 reps per step
POWERMETRICS=1 ./run_walk.sh      # + per-cluster activity (asks for your password)
./run_walk.sh 100000000 5         # custom n and reps
```

The output goes to `results/<host>_<timestamp>/`:
- `layout.txt`: core tiers from `hw.perflevel*`.
- `walk_all.txt` and `walk_bg.txt`: the per-run lines plus a summary table.
- `walk.csv`: every run, as `label,threads,rep,sort_seconds,start_epoch,end_epoch`.
- `analysis.txt` (with powermetrics): for each step, each cluster's average busy cores and clock
  (MHz), plus which CPU ids were busy. On the M6, Super and Performance share one "PS" cluster, so
  the CPU ids tell them apart: the ones that fill first at 1–2 threads are the Super cores.

The binary on its own:
`./thread_walker_sort [n=200000000] [reps=3] [first=1] [last=all] [seed=42]`.
Memory is about 2.4 GB at n=200M (original + working copy + merge buffer).

## Reading the summary

| Column | Meaning |
|---|---|
| speedup | vs 1 thread |
| step-gain | this step vs the previous one. A big jump means a fast core joined; a small one means a slow core (or memory) is limiting |
| efficiency | speedup ÷ threads |

The step where step-gain drops shows where the next core tier starts.
