add_executable(prefix-sum-scalar bench-scalar.cc)
target_link_libraries(prefix-sum-scalar benchmark::benchmark)

add_executable(prefix-sum-vector bench-vector.cc)
target_link_libraries(prefix-sum-vector benchmark::benchmark)

if (CMAKE_CROSSCOMPILING)
  target_compile_options(prefix-sum-scalar PRIVATE -march=rv64gc PRIVATE -fno-vectorize PRIVATE -fno-slp-vectorize)
  target_compile_options(prefix-sum-vector PRIVATE -march=rv64gcv)
else()
  target_compile_options(prefix-sum-scalar PRIVATE -march=x86-64-v4 PRIVATE -fno-vectorize PRIVATE -fno-slp-vectorize)
  target_compile_options(prefix-sum-vector PRIVATE -march=x86-64-v4)
endif()
