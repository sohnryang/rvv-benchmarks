if (CMAKE_CROSSCOMPILING)
  find_package(benchmark REQUIRED)
else()
  include(FetchContent)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
  FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.0
  )
  FetchContent_MakeAvailable(google_benchmark)
endif()
add_executable(gbench-test gbench-test.cc)
target_link_libraries(gbench-test benchmark::benchmark)

add_subdirectory(tsvc-s1115)
add_subdirectory(matmul)
