#!/bin/bash
# Thread Walker Sort on macOS: walks all cores (1..N threads), then the Efficiency cores
# alone (1..E threads at background priority, which macOS keeps on E-cores).
#
# Usage:  ./run_walk.sh [n=200000000] [reps=3]
#   POWERMETRICS=1  also record per-cluster activity with powermetrics (asks for sudo),
#                   then analyze.py shows which core groups were busy at each step.
# Output: results/<host>_<timestamp>/
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-200000000}; REPS=${2:-3}
HOST=$(hostname -s); OUT="results/${HOST}_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"

clang++ -O3 -std=c++17 -pthread -o thread_walker_sort thread_walker_sort.cpp

# Core layout: hw.perflevel0 is the fastest tier. The highest perflevel is the Efficiency tier.
{
  echo "host=$HOST chip=$(sysctl -n machdep.cpu.brand_string) logical=$(sysctl -n hw.logicalcpu)"
  levels=$(sysctl -n hw.nperflevels)
  for ((i=0; i<levels; i++)); do
    echo "perflevel$i name=$(sysctl -n hw.perflevel$i.name) cores=$(sysctl -n hw.perflevel$i.logicalcpu)"
  done
  sw_vers -productVersion; clang++ --version | head -1
  pmset -g | grep -Ei "powermode|lowpowermode" || true; pmset -g batt | head -1
} | tee "$OUT/layout.txt"
LEVELS=$(sysctl -n hw.nperflevels)
LOWNAME=$(sysctl -n hw.perflevel$((LEVELS-1)).name)       # e.g. Efficiency (M5/M6), Performance (M5 Max)
LOWCORES=$(sysctl -n hw.perflevel$((LEVELS-1)).logicalcpu)
LOWLABEL=$(echo "$LOWNAME" | tr 'A-Z' 'a-z')-bg
ALL=$(sysctl -n hw.logicalcpu)

PM_PID=""
if [ "${POWERMETRICS:-0}" = 1 ]; then
  if [ -n "${SUDO_STDIN:-}" ]; then sudo -S -p '' -v; else sudo -v; fi
  sudo -n powermetrics --samplers cpu_power -i 500 -o "$OUT/powermetrics.txt" >/dev/null 2>&1 &
  PM_PID=$!
  sleep 2
fi
cleanup() { [ -n "$PM_PID" ] && sudo -n pkill -INT -f "powermetrics --samplers cpu_power" 2>/dev/null || true; }
trap cleanup EXIT

echo "== all cores: 1..$ALL threads"
WALK_LABEL=all-cores ./thread_walker_sort "$N" "$REPS" 1 "$ALL" | tee "$OUT/walk_all.txt"
sleep 20
# Background priority keeps work on the lowest core tier (and at a low clock), so this walk
# shows that tier alone. With powermetrics on, analysis.txt confirms which cluster ran it.
echo "== $LOWNAME cores at background priority: 1..$LOWCORES threads"
WALK_LABEL=$LOWLABEL taskpolicy -b ./thread_walker_sort "$N" "$REPS" 1 "$LOWCORES" | tee "$OUT/walk_bg.txt"

cleanup; PM_PID=""; sleep 1
grep -h '^CSV,' "$OUT"/walk_*.txt | grep -v '^CSV,label' | sed 's/^CSV,//' > "$OUT/walk.csv"
if [ -f "$OUT/powermetrics.txt" ]; then
  python3 analyze.py "$OUT" | tee "$OUT/analysis.txt"
fi
echo "== done: $OUT"
