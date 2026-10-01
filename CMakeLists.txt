cmake_minimum_required(VERSION 3.16)
project(ezNN C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(EZNN_OPENMP "Parallelize wide layers with OpenMP" OFF)
option(EZNN_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(EZNN_NOVEC "Pass -fno-tree-vectorize" OFF)

add_compile_options(-Wall -Wextra -Werror)

add_library(eznn STATIC ezNN.c)
target_include_directories(eznn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(eznn PUBLIC m)

add_executable(runezNN main.c readwrite_csv.c)
target_link_libraries(runezNN PRIVATE eznn)

add_executable(test_eznn tests/test_eznn.c readwrite_csv.c)
target_link_libraries(test_eznn PRIVATE eznn)

add_executable(bench_eznn bench/bench_eznn.c readwrite_csv.c)
target_link_libraries(bench_eznn PRIVATE eznn)

if(EZNN_OPENMP)
    find_package(OpenMP REQUIRED)
    target_compile_definitions(eznn PUBLIC EZNN_OPENMP)
    target_link_libraries(eznn PUBLIC OpenMP::OpenMP_C)
endif()

if(EZNN_NOVEC)
    add_compile_options(-fno-tree-vectorize)
endif()

if(EZNN_SANITIZE)
    set(_san -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined)
    foreach(tgt eznn runezNN test_eznn bench_eznn)
        target_compile_options(${tgt} PRIVATE ${_san})
        target_link_options(${tgt} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()

enable_testing()
add_test(NAME unit COMMAND test_eznn)
