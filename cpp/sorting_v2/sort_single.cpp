// sorting v2: single-threaded quicksort.
// Build: clang++ -O3 -std=c++17 -o sort_single sort_single.cpp
// Run:   ./sort_single [n=100000000] [seed=42]
// Times ONLY the sort (data generation and verification are reported separately).
#include "common.h"

int main(int argc, char** argv) {
    const int64_t n = arg_or(argc, argv, 1, 100000000);
    const uint32_t seed = static_cast<uint32_t>(arg_or(argc, argv, 2, 42));

    std::vector<int32_t> a(static_cast<size_t>(n));
    auto t0 = Clock::now();
    fill_data(a, seed);
    double fill_s = seconds_since(t0);
    Fingerprint before = fingerprint(a.data(), a.size());

    std::printf("sort_single v2: n=%lld seed=%u threads=1\n", (long long)n, seed);
    t0 = Clock::now();
    quicksort(a.data(), 0, n - 1);
    double sort_s = seconds_since(t0);

    std::printf("fill_seconds=%.3f\n", fill_s);
    int rc = verify(a, before);
    std::printf("RESULT sort_seconds=%.3f\n", sort_s);
    return rc;
}
