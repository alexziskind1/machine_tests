// Thread Walker Sort: sorts the SAME data with 1, 2, 3, ... N threads and reports the time
// at each step, to show how performance scales as each extra core (and core type) joins.
//
// Build: clang++ -O3 -std=c++17 -pthread -o thread_walker_sort thread_walker_sort.cpp
// Run:   ./thread_walker_sort [n=200000000] [reps=3] [first=1] [last=all logical cores] [seed=42]
//        WALK_LABEL=<name> sets the label column (run_walk.sh uses "all-cores" / "ecores-only").
//
// The algorithm is the sorting_v2 multi-threaded sort: chunks quicksorted in parallel, then
// every merge pass split across all threads (merge path). With 1 thread it's a plain
// single-threaded quicksort. Only the sort is timed; every run is verified.
#include "../sorting_v2/common.h"
#include <cstring>
#include <thread>

template <class F>
static void parallel_for(unsigned T, F f) {
    std::vector<std::thread> ts;
    ts.reserve(T);
    for (unsigned k = 0; k < T; k++) ts.emplace_back(f, k);
    for (auto& t : ts) t.join();
}

static int64_t co_rank(int64_t d, const int32_t* A, int64_t na, const int32_t* B, int64_t nb) {
    int64_t lo = std::max<int64_t>(0, d - nb), hi = std::min<int64_t>(d, na);
    while (lo < hi) {
        int64_t i = lo + (hi - lo) / 2;
        if (A[i] < B[d - i - 1]) lo = i + 1; else hi = i;
    }
    return lo;
}

static void parallel_merge(const int32_t* A, int64_t na, const int32_t* B, int64_t nb,
                           int32_t* out, unsigned T) {
    const int64_t total = na + nb;
    parallel_for(T, [&](unsigned k) {
        int64_t d0 = total * k / T, d1 = total * (k + 1) / T;
        int64_t i0 = co_rank(d0, A, na, B, nb), i1 = co_rank(d1, A, na, B, nb);
        std::merge(A + i0, A + i1, B + (d0 - i0), B + (d1 - i1), out + d0);
    });
}

// Sorts a[0..n) using T threads; buf is scratch of the same size. Result ends up in a.
static void parallel_sort(std::vector<int32_t>& a, std::vector<int32_t>& buf, unsigned T) {
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<int64_t> bounds(T + 1);
    for (unsigned k = 0; k <= T; k++) bounds[k] = n * k / T;
    if (T == 1) { quicksort(a.data(), 0, n - 1); return; }
    parallel_for(T, [&](unsigned k) { quicksort(a.data(), bounds[k], bounds[k + 1] - 1); });
    int32_t* src = a.data();
    int32_t* dst = buf.data();
    while (bounds.size() > 2) {
        std::vector<int64_t> next{0};
        for (size_t r = 0; r + 1 < bounds.size(); r += 2) {
            int64_t l = bounds[r], m = bounds[r + 1];
            int64_t h = (r + 2 < bounds.size()) ? bounds[r + 2] : m;
            if (h > m) parallel_merge(src + l, m - l, src + m, h - m, dst + l, T);
            else       std::copy(src + l, src + m, dst + l);
            next.push_back(h);
        }
        bounds.swap(next);
        std::swap(src, dst);
    }
    if (src != a.data()) a.swap(buf);
}

static double epoch_now() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int64_t n = arg_or(argc, argv, 1, 200000000);
    const int reps = static_cast<int>(arg_or(argc, argv, 2, 3));
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned first = static_cast<unsigned>(arg_or(argc, argv, 3, 1));
    const unsigned last = static_cast<unsigned>(arg_or(argc, argv, 4, hw));
    const uint32_t seed = static_cast<uint32_t>(arg_or(argc, argv, 5, 42));
    const char* label = std::getenv("WALK_LABEL") ? std::getenv("WALK_LABEL") : "all-cores";
    if (first < 1 || last < first || reps < 1 || n < 1) { std::fprintf(stderr, "bad arguments\n"); return 2; }

    std::vector<int32_t> orig(static_cast<size_t>(n)), a(static_cast<size_t>(n)), buf(static_cast<size_t>(n));
    fill_data(orig, seed);
    const Fingerprint before = fingerprint(orig.data(), orig.size());

    std::printf("thread_walker_sort: n=%lld seed=%u reps=%d threads=%u..%u label=%s logical_cores=%u\n",
                (long long)n, seed, reps, first, last, label, hw);
    std::fflush(stdout);

    // Untimed warm-up at the highest thread count (page in memory, spin up clocks).
    std::copy(orig.begin(), orig.end(), a.begin());
    parallel_sort(a, buf, last);

    std::printf("CSV,label,threads,rep,sort_seconds,start_epoch,end_epoch\n");
    std::vector<double> medians;
    for (unsigned t = first; t <= last; t++) {
        std::vector<double> times;
        for (int r = 1; r <= reps; r++) {
            std::copy(orig.begin(), orig.end(), a.begin());
            double e0 = epoch_now();
            auto t0 = Clock::now();
            parallel_sort(a, buf, t);
            double s = seconds_since(t0);
            double e1 = epoch_now();
            if (!std::is_sorted(a.begin(), a.end()) || !(fingerprint(a.data(), a.size()) == before)) {
                std::printf("VERIFY FAILED at threads=%u rep=%d\n", t, r);
                return 1;
            }
            times.push_back(s);
            std::printf("CSV,%s,%u,%d,%.3f,%.3f,%.3f\n", label, t, r, s, e0, e1);
            std::fflush(stdout);
        }
        std::sort(times.begin(), times.end());
        medians.push_back(times[times.size() / 2]);
    }

    std::printf("\nSUMMARY label=%s (median of %d; speedup vs %u thread%s)\n", label, reps, first, first == 1 ? "" : "s");
    // speedup: vs the first step. step-gain: how much faster this step is than the one before it.
    std::printf("%8s %12s %9s %10s %11s\n", "threads", "seconds", "speedup", "step-gain", "efficiency");
    for (unsigned t = first; t <= last; t++) {
        double m = medians[t - first], sp = medians[0] / m;
        double step = (t > first) ? medians[t - first - 1] / m : 1.0;
        std::printf("%8u %12.3f %8.2fx %9.2fx %10.0f%%\n", t, m, sp, step, 100.0 * sp * first / t);
    }
    return 0;
}
