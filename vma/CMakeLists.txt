cmake_minimum_required(VERSION 3.16..3.24)
project(vma VERSION 0.0.1
    DESCRIPTION "Vector Matrix Arithmetic Toolkit"
   LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(vma STATIC
    ${PROJECT_SOURCE_DIR}/src/vma.cpp)

target_include_directories(vma PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/include
    /usr/local/include/lapack
    )

