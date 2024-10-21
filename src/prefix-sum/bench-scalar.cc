#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdlib>

void psum_naive_kernel(int *arr, size_t n) {
  for (int i = 1; i < n; i++)
    arr[i] += arr[i - 1];
}

static void BM_scalar(benchmark::State &state) {
  int *arr = new int[state.range()];
  for (int i = 0; i < state.range(); i++)
    arr[i] = std::rand();

  for (auto _ : state) {
    psum_naive_kernel(arr, state.range());
    benchmark::DoNotOptimize(arr);
    benchmark::ClobberMemory();
  }
  delete[] arr;
}
BENCHMARK(BM_scalar)->RangeMultiplier(2)->Range(16, 8192);

BENCHMARK_MAIN();
