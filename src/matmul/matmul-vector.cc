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

static void BM_matmul_autovec(benchmark::State &state) {
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
BENCHMARK(BM_matmul_autovec)
    ->ArgsProduct({
        benchmark::CreateRange(64, 512, 2),
        benchmark::CreateRange(64, 512, 2),
        benchmark::CreateRange(64, 512, 2),
    });

#ifdef __riscv_v_intrinsic
void matmul_vector(const std::uint32_t *__restrict__ a,
                   const std::uint32_t *__restrict__ b,
                   std::uint32_t *__restrict__ c, const int M, const int N,
                   const int K) {
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  vuint32m1_t vec_zero = __riscv_vmv_v_x_u32m1(0, vlmax);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++) {
      const std::uint32_t *ptr_a = &a[N * i], *ptr_b = &b[j];
      int leftover = N;
      vuint32m2_t vec_s = __riscv_vmv_v_x_u32m2(0, vlmax);
      while (leftover > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(leftover);
        vuint32m2_t vec_a = __riscv_vle32_v_u32m2(ptr_a, vl),
                    vec_b = K == 1 ? __riscv_vle32_v_u32m2(ptr_b, vl)
                                   : __riscv_vlse32_v_u32m2(
                                         ptr_b, sizeof(std::uint32_t) * K, vl);
        vec_s = __riscv_vmacc_vv_u32m2(vec_s, vec_a, vec_b, vl);
        leftover -= vl;
        ptr_a += vl;
        ptr_b += vl * K;
      }
      vuint32m1_t vec_sum =
          __riscv_vredsum_vs_u32m2_u32m1(vec_s, vec_zero, vlmax);
      std::uint32_t sum = __riscv_vmv_x_s_u32m1_u32(vec_sum);
      /*
      std::uint32_t sum;
      asm("vsetvli t0,zero,e32,m2,ta,ma\n\t"
          "vredsum.vs v9,%1,%2\n\t"
          "vmv.x.s %0,v9"
          : "=r"(sum)
          : "vr"(vec_s), "vr"(vec_zero)
          : "t0", "v9");
      */
      c[K * i + j] = sum;
    }
}

static void BM_matmul_intrinsic(benchmark::State &state) {
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
    matmul_vector(a, b, c, M, N, K);
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
    benchmark::DoNotOptimize(c);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_matmul_intrinsic)
    ->ArgsProduct({
        benchmark::CreateRange(64, 512, 2),
        benchmark::CreateRange(64, 512, 2),
        benchmark::CreateRange(64, 512, 2),
    });
#endif

BENCHMARK_MAIN();
