#include <benchmark/benchmark.h>

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#include <cstdint>
#include <cstdlib>

void matmul_naive(const std::uint32_t *__restrict__ a,
                  const std::uint32_t *__restrict__ b,
                  std::uint32_t *__restrict__ c, const int M, const int N,
                  const int K) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++) {
      std::uint32_t res = 0.0f;
      for (int k = 0; k < N; k++)
        res += a[N * i + k] * b[K * k + j];
      c[K * i + j] = res;
    }
}

static void BM_matmul_naive(benchmark::State &state) {
  const int M = state.range(0), N = state.range(1), K = state.range(2);
  std::uint32_t *a = new std::uint32_t[M * N], *b = new std::uint32_t[N * K],
                *c = new std::uint32_t[M * K];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      a[N * i + j] = std::rand();
    for (int j = 0; j < K; j++)
      c[K * i + j] = std::rand();
  }
  for (int i = 0; i < N; i++)
    for (int j = 0; j < K; j++)
      b[K * i + j] = std::rand();

  for (auto _ : state) {
    matmul_naive(a, b, c, M, N, K);
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
    benchmark::DoNotOptimize(c);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_matmul_naive)
    ->ArgsProduct({
        benchmark::CreateRange(64, 512, 2),
        benchmark::CreateRange(64, 512, 2),
        benchmark::CreateRange(64, 512, 2),
    });

BENCHMARK_MAIN();
