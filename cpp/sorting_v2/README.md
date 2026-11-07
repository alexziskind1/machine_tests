# C++ sorting v2

A cleaner version of `../sorting`. The original tests are left unchanged, so older results
stay comparable.

| | v1 (`../sorting`) | v2 (this folder) |
|---|---|---|
| Data | `rand() % 100`: 0–99, millions of duplicates | `std::mt19937` raw output: full 32-bit range |
| Same data everywhere | Only on the same C library (seed 42 since fe1c76c) | Yes. mt19937 output is fixed by the C++ standard, so macOS, Windows and Linux get identical data |
| What's timed | Whole process: `rand()` fill + sort (+ printing 10M numbers in single) | Only the sort (`std::chrono`); fill time is printed separately |
| Single-threaded | Lomuto quicksort; with only 100 distinct values it degrades toward O(n²) | Quicksort, median-of-three pivot, Hoare partition, insertion sort for small ranges: O(n log n) |
| Multi-threaded | Chunks sorted in parallel, then **every merge pass on one thread** | Chunks quicksorted in parallel, then every merge pass split across **all** threads (merge path), so all cores stay busy to the end |
| Correctness | Not checked | `std::is_sorted` + a value fingerprint; exit code 1 if either fails |

## Build and run (macOS)

```bash
clang++ -O3 -std=c++17 -o sort_single sort_single.cpp
clang++ -O3 -std=c++17 -pthread -o sort_multi sort_multi.cpp

./sort_single              # n=100,000,000, seed 42, 1 thread
./sort_multi               # n=500,000,000, seed 42, all logical cores
```

Optional arguments: `./sort_single [n] [seed]`, `./sort_multi [n] [seed] [threads]`.

On Windows use `cl /O2 /std:c++17 /EHsc sort_single.cpp` (and the same for `sort_multi.cpp`), or clang/g++ with the same flags as above.

## Output

```
sort_multi v2: n=500000000 seed=42 threads=20
fill_seconds=0.411
verify: sorted=yes same_values=yes
RESULT sort_seconds=1.941
```

Report `RESULT sort_seconds`. Memory use: about 0.4 GB for single and about 4 GB for multi
(the data plus one merge buffer), which is safe on 16 GB machines.
