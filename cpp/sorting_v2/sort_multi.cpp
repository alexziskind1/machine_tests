// sorting v2: multi-threaded sort that keeps every core busy for the whole run.
// Build: clang++ -O3 -std=c++17 -pthread -o sort_multi sort_multi.cpp
// Run:   ./sort_multi [n=500000000] [seed=42] [threads=all logical cores]
//
// Phase 1: split into T chunks; each thread quicksorts its chunk (the same quicksort as
//          sort_single).
// Phase 2: merge passes. Every merge (including the final one, which joins two halves
//          of the whole array) is split across ALL threads with the "merge path"
//          partitioning, so no pass falls back to a single thread.
// Times ONLY the sort (data generation and verification are reported separately).
#include "common.h"
#include <thread>

template <class F>
static void parallel_for(unsigned T, F f) {
    std::vector<std::thread> ts;
    ts.reserve(T);
    for (unsigned k = 0; k < T; k++) ts.emplace_back(f, k);
    for (auto& t : ts) t.join();
}

// Merge path: how many of the first d output elements come from A (the rest from B).
static int64_t co_rank(int64_t d, const int32_t* A, int64_t na, const int32_t* B, int64_t nb) {
    int64_t lo = std::max<int64_t>(0, d - nb), hi = std::min<int64_t>(d, na);
    while (lo < hi) {
        int64_t i = lo + (hi - lo) / 2;  // take i from A, d-i from B
        if (A[i] < B[d - i - 1]) lo = i + 1; else hi = i;
    }
    return lo;
}

// Merge sorted A and B into out, split evenly across T threads.
static void parallel_merge(const int32_t* A, int64_t na, const int32_t* B, int64_t nb,
                           int32_t* out, unsigned T) {
    const int64_t total = na + nb;
    parallel_for(T, [&](unsigned k) {
        int64_t d0 = total * k / T, d1 = total * (k + 1) / T;
        int64_t i0 = co_rank(d0, A, na, B, nb), i1 = co_rank(d1, A, na, B, nb);
        std::merge(A + i0, A + i1, B + (d0 - i0), B + (d1 - i1), out + d0);
    });
}

int main(int argc, char** argv) {
    const int64_t n = arg_or(argc, argv, 1, 500000000);
    const uint32_t seed = static_cast<uint32_t>(arg_or(argc, argv, 2, 42));
    unsigned T = static_cast<unsigned>(arg_or(argc, argv, 3, std::thread::hardware_concurrency()));
    if (T == 0) T = 1;

    std::vector<int32_t> a(static_cast<size_t>(n)), buf(static_cast<size_t>(n));
    auto t0 = Clock::now();
    fill_data(a, seed);
    double fill_s = seconds_since(t0);
    Fingerprint before = fingerprint(a.data(), a.size());

    std::printf("sort_multi v2: n=%lld seed=%u threads=%u\n", (long long)n, seed, T);
    t0 = Clock::now();

    // Phase 1: sort T chunks in parallel.
    std::vector<int64_t> bounds(T + 1);
    for (unsigned k = 0; k <= T; k++) bounds[k] = n * k / T;
    parallel_for(T, [&](unsigned k) { quicksort(a.data(), bounds[k], bounds[k + 1] - 1); });

    // Phase 2: pairwise merge passes, each one using all T threads.
    int32_t* src = a.data();
    int32_t* dst = buf.data();
    while (bounds.size() > 2) {
        std::vector<int64_t> next{0};
        for (size_t r = 0; r + 1 < bounds.size(); r += 2) {
            int64_t l = bounds[r], m = bounds[r + 1];
            int64_t h = (r + 2 < bounds.size()) ? bounds[r + 2] : m;
            if (h > m) parallel_merge(src + l, m - l, src + m, h - m, dst + l, T);
            else       std::copy(src + l, src + m, dst + l);  // odd run out: carry over
            next.push_back(h);
        }
        bounds.swap(next);
        std::swap(src, dst);
    }
    if (src != a.data()) a.swap(buf);
    double sort_s = seconds_since(t0);

    std::printf("fill_seconds=%.3f\n", fill_s);
    int rc = verify(a, before);
    std::printf("RESULT sort_seconds=%.3f\n", sort_s);
    return rc;
}
