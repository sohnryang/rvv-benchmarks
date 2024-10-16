add_executable(matmul-scalar matmul-scalar.cc)
target_link_libraries(matmul-scalar benchmark::benchmark)

add_executable(matmul-vector matmul-vector.cc)
target_link_libraries(matmul-vector benchmark::benchmark)

if (CMAKE_CROSSCOMPILING)
  target_compile_options(matmul-scalar PRIVATE -march=rv64gc PRIVATE -fno-vectorize PRIVATE -fno-slp-vectorize)
  target_compile_options(matmul-vector PRIVATE -march=rv64gcv)
else()
  target_compile_options(matmul-scalar PRIVATE -march=x86-64-v4 PRIVATE -fno-vectorize PRIVATE -fno-slp-vectorize)
  target_compile_options(matmul-vector PRIVATE -march=x86-64-v4)
endif()
