cmake_minimum_required(VERSION 3.19)

project(RVV-Benchmarks)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3>")

add_subdirectory(src)
