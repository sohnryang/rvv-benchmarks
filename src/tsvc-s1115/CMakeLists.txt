add_executable(tsvc-s1115-scalar bench-scalar.cc)
target_link_libraries(tsvc-s1115-scalar benchmark::benchmark)

add_executable(tsvc-s1115-vector bench-vector.cc)
target_link_libraries(tsvc-s1115-vector benchmark::benchmark)

if (CMAKE_CROSSCOMPILING)
  target_compile_options(tsvc-s1115-scalar PRIVATE -march=rv64gc PRIVATE -fno-vectorize PRIVATE -fno-slp-vectorize)
  target_compile_options(tsvc-s1115-vector PRIVATE -march=rv64gcv)
else()
  target_compile_options(tsvc-s1115-scalar PRIVATE -march=x86-64-v4 PRIVATE -fno-vectorize PRIVATE -fno-slp-vectorize)
  target_compile_options(tsvc-s1115-vector PRIVATE -march=x86-64-v4)
endif()
