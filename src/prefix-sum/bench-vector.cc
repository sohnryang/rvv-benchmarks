#include <benchmark/benchmark.h>

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#include <cstddef>
#include <cstdlib>

void psum_naive_kernel(int *arr, size_t n) {
  for (int i = 1; i < n; i++)
    arr[i] += arr[i - 1];
}

static void BM_naive(benchmark::State &state) {
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
BENCHMARK(BM_naive)->RangeMultiplier(2)->Range(16, 8192);

#ifdef __riscv_v_intrinsic
void psum_vector_kernel(int *a, int n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m8();
  int carry = 0;
  vint32m8_t vec_zero, va, va_su;
  vec_zero = __riscv_vmv_v_x_i32m8(0, vlmax);
  for (; n > 0; n -= vl) {
    vl = __riscv_vsetvl_e32m8(n);
    va = __riscv_vle32_v_i32m8(a, vl);
    for (size_t offset = 1; offset < vl; offset = offset << 1) {
      va_su = __riscv_vslideup_vx_i32m8(vec_zero, va, offset, vl);
      va = __riscv_vadd_vv_i32m8(va, va_su, vl);
    }
    va = __riscv_vadd_vx_i32m8(va, carry, vl);
    __riscv_vse32_v_i32m8(a, va, vl);
    carry = a[vl - 1];
    a += vl;
  }
}

static void BM_vector(benchmark::State &state) {
  int *arr = new int[state.range()];
  for (int i = 0; i < state.range(); i++)
    arr[i] = std::rand();

  for (auto _ : state) {
    psum_vector_kernel(arr, state.range());
    benchmark::DoNotOptimize(arr);
    benchmark::ClobberMemory();
  }
  delete[] arr;
}
BENCHMARK(BM_vector)->RangeMultiplier(2)->Range(16, 8192);
#endif

BENCHMARK_MAIN();
