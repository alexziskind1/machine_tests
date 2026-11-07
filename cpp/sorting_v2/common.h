// Shared pieces for sort_single.cpp and sort_multi.cpp (sorting v2).
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Full-range 32-bit ints from std::mt19937 with a fixed seed. mt19937's raw output is
// fixed by the C++ standard, so every platform (macOS, Windows, Linux) gets identical
// data. (std::uniform_int_distribution is implementation-defined, so it isn't used.)
static void fill_data(std::vector<int32_t>& a, uint32_t seed) {
    std::mt19937 gen(seed);
    for (auto& x : a) x = static_cast<int32_t>(gen());
}

// Order-independent fingerprint, so the sorted output can be checked against the input.
struct Fingerprint {
    uint64_t sum = 0, xr = 0;
    bool operator==(const Fingerprint& o) const { return sum == o.sum && xr == o.xr; }
};
static Fingerprint fingerprint(const int32_t* a, size_t n) {
    Fingerprint f;
    for (size_t i = 0; i < n; i++) {
        uint64_t v = static_cast<uint32_t>(a[i]);
        f.sum += v;
        f.xr ^= v * 0x9E3779B97F4A7C15ull;
    }
    return f;
}

static void insertion_sort(int32_t* a, int64_t lo, int64_t hi) {
    for (int64_t i = lo + 1; i <= hi; i++) {
        int32_t v = a[i];
        int64_t j = i - 1;
        while (j >= lo && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

// Quicksort: median-of-three pivot, Hoare partition (handles duplicates well),
// insertion sort for small ranges, recurse into the smaller side only, so the stack
// depth stays O(log n). Single-threaded.
static void quicksort(int32_t* a, int64_t lo, int64_t hi) {
    while (hi - lo > 24) {
        int64_t mid = lo + (hi - lo) / 2;
        if (a[mid] < a[lo]) std::swap(a[mid], a[lo]);
        if (a[hi] < a[lo]) std::swap(a[hi], a[lo]);
        if (a[hi] < a[mid]) std::swap(a[hi], a[mid]);
        int32_t pivot = a[mid];
        int64_t i = lo - 1, j = hi + 1;
        while (true) {
            do { i++; } while (a[i] < pivot);
            do { j--; } while (a[j] > pivot);
            if (i >= j) break;
            std::swap(a[i], a[j]);
        }
        if (j - lo < hi - j) { quicksort(a, lo, j); lo = j + 1; }
        else                 { quicksort(a, j + 1, hi); hi = j; }
    }
    insertion_sort(a, lo, hi);
}

static int64_t arg_or(int argc, char** argv, int idx, int64_t dflt) {
    return argc > idx ? std::atoll(argv[idx]) : dflt;
}

// Fails the run (non-zero exit) if the output isn't sorted or isn't a permutation of the input.
static int verify(const std::vector<int32_t>& a, const Fingerprint& before) {
    bool sorted = std::is_sorted(a.begin(), a.end());
    bool same = fingerprint(a.data(), a.size()) == before;
    std::printf("verify: sorted=%s same_values=%s\n", sorted ? "yes" : "NO", same ? "yes" : "NO");
    return (sorted && same) ? 0 : 1;
}
