#include <benchmark/benchmark.h>

#include <cstddef>

const std::size_t LEN_2D = 256;

void s1115_kernel_blocked(float aa[LEN_2D][LEN_2D],
                          const float bb[LEN_2D][LEN_2D],
                          const float cc[LEN_2D][LEN_2D],
                          std::size_t blocksize_y, std::size_t blocksize_x) {
  for (int ii = 0; ii < LEN_2D; ii += blocksize_y)
    for (int jj = 0; jj < LEN_2D; jj += blocksize_x)
      for (int i = 0; i < blocksize_y && ii + i < LEN_2D; i++)
        for (int j = 0; j < blocksize_x && jj + j < LEN_2D; j++)
          aa[ii + i][jj + j] =
              aa[ii + i][jj + j] * cc[jj + j][ii + i] + bb[ii + i][jj + j];
}

static void BM_scalar_blocked(benchmark::State &state) {
  float aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D];
  for (int i = 0; i < LEN_2D; i++)
    for (int j = 0; j < LEN_2D; j++) {
      aa[i][j] = 0.000001f;
      bb[i][j] = 0.000001f;
      cc[i][j] = 0.000001f;
    }

  for (auto _ : state) {
    s1115_kernel_blocked(aa, bb, cc, state.range(0), state.range(1));
    benchmark::DoNotOptimize(aa);
    benchmark::DoNotOptimize(bb);
    benchmark::DoNotOptimize(cc);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_scalar_blocked)
    ->ArgsProduct({benchmark::CreateRange(16, 256, 2),
                   benchmark::CreateRange(16, 256, 2)});

void s1115_kernel_naive(float aa[LEN_2D][LEN_2D],
                        const float bb[LEN_2D][LEN_2D],
                        const float cc[LEN_2D][LEN_2D]) {
  for (int i = 0; i < LEN_2D; i++)
    for (int j = 0; j < LEN_2D; j++)
      aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j];
}

static void BM_scalar_naive(benchmark::State &state) {
  float aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D];
  for (int i = 0; i < LEN_2D; i++)
    for (int j = 0; j < LEN_2D; j++) {
      aa[i][j] = 0.000001f;
      bb[i][j] = 0.000001f;
      cc[i][j] = 0.000001f;
    }

  for (auto _ : state) {
    s1115_kernel_naive(aa, bb, cc);
    benchmark::DoNotOptimize(aa);
    benchmark::DoNotOptimize(bb);
    benchmark::DoNotOptimize(cc);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_scalar_naive);

BENCHMARK_MAIN();
