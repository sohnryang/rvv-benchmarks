#include <benchmark/benchmark.h>

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#include <cstddef>

const std::size_t LEN_2D = 256;

#ifdef __riscv_v_intrinsic
void s1115_kernel_intrinsic(float aa[LEN_2D][LEN_2D],
                            const float bb[LEN_2D][LEN_2D],
                            const float cc[LEN_2D][LEN_2D]) {
  for (int i = 0; i < LEN_2D; i++) {
    int leftover = LEN_2D;
    float *ptr_aa = &aa[i][0];
    const float *ptr_bb = &bb[i][0], *ptr_cc = &cc[0][i];
    while (leftover > 0) {
      const size_t vl = __riscv_vsetvl_e32m2(leftover);
      vfloat32m2_t vec_aa = __riscv_vle32_v_f32m2(ptr_aa, vl),
                   vec_bb = __riscv_vle32_v_f32m2(ptr_bb, vl),
                   vec_cc = __riscv_vlse32_v_f32m2(ptr_cc,
                                                   sizeof(float) * LEN_2D, vl);
      __riscv_vse32_v_f32m2(
          ptr_aa, __riscv_vfmacc_vv_f32m2(vec_bb, vec_aa, vec_cc, vl), vl);
      leftover -= vl;
      ptr_aa += vl;
      ptr_bb += vl;
      ptr_cc += vl * LEN_2D;
    }
  }
}

static void BM_vector_intrinsic(benchmark::State &state) {
  float aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D];
  for (int i = 0; i < LEN_2D; i++)
    for (int j = 0; j < LEN_2D; j++) {
      aa[i][j] = 0.000001f;
      bb[i][j] = 0.000001f;
      cc[i][j] = 0.000001f;
    }

  for (auto _ : state) {
    s1115_kernel_intrinsic(aa, bb, cc);
    benchmark::DoNotOptimize(aa);
    benchmark::DoNotOptimize(bb);
    benchmark::DoNotOptimize(cc);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_vector_intrinsic);

void s1115_kernel_intrinsic_blocked(float aa[LEN_2D][LEN_2D],
                                    const float bb[LEN_2D][LEN_2D],
                                    const float cc[LEN_2D][LEN_2D],
                                    std::size_t blocksize_y,
                                    std::size_t blocksize_x) {
  for (int ii = 0; ii < LEN_2D; ii += blocksize_y) {
    for (int jj = 0; jj < LEN_2D; jj += blocksize_x) {
      for (int i = 0; i < blocksize_y; i++) {
        float *ptr_aa = &aa[ii + i][jj];
        const float *ptr_bb = &bb[ii + i][jj], *ptr_cc = &cc[jj][ii + i];
        int leftover = LEN_2D - jj >= blocksize_x ? blocksize_x : LEN_2D - jj;
        while (leftover > 0) {
          const size_t vl = __riscv_vsetvl_e32m2(leftover);
          vfloat32m2_t vec_aa = __riscv_vle32_v_f32m2(ptr_aa, vl),
                       vec_bb = __riscv_vle32_v_f32m2(ptr_bb, vl),
                       vec_cc = __riscv_vlse32_v_f32m2(
                           ptr_cc, sizeof(float) * LEN_2D, vl);
          __riscv_vse32_v_f32m2(
              ptr_aa, __riscv_vfmacc_vv_f32m2(vec_bb, vec_aa, vec_cc, vl), vl);
          leftover -= vl;
          ptr_aa += vl;
          ptr_bb += vl;
          ptr_cc += vl * LEN_2D;
        }
      }
    }
  }
}

static void BM_vector_intrinsic_blocked(benchmark::State &state) {
  float aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D];
  for (int i = 0; i < LEN_2D; i++)
    for (int j = 0; j < LEN_2D; j++) {
      aa[i][j] = 0.000001f;
      bb[i][j] = 0.000001f;
      cc[i][j] = 0.000001f;
    }

  for (auto _ : state) {
    s1115_kernel_intrinsic_blocked(aa, bb, cc, state.range(0), state.range(1));
    benchmark::DoNotOptimize(aa);
    benchmark::DoNotOptimize(bb);
    benchmark::DoNotOptimize(cc);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_vector_intrinsic_blocked)
    ->ArgsProduct({benchmark::CreateRange(16, 256, 2),
                   benchmark::CreateRange(16, 256, 2)});
#endif

void s1115_kernel_naive(float aa[LEN_2D][LEN_2D],
                        const float bb[LEN_2D][LEN_2D],
                        const float cc[LEN_2D][LEN_2D]) {
  for (int i = 0; i < LEN_2D; i++)
    for (int j = 0; j < LEN_2D; j++)
      aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j];
}

static void BM_vector_autovec(benchmark::State &state) {
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
BENCHMARK(BM_vector_autovec);

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

static void BM_vector_autovec_blocked(benchmark::State &state) {
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
BENCHMARK(BM_vector_autovec_blocked)
    ->ArgsProduct({benchmark::CreateRange(16, 256, 2),
                   benchmark::CreateRange(16, 256, 2)});

BENCHMARK_MAIN();
