cmake_minimum_required(VERSION 3.16..3.24)
project(vmalib VERSION 0.0.1
    DESCRIPTION "Vector Matrix Arithmetic Toolkit"
   LANGUAGES CXX)

add_library(vmalib STATIC
    ${PROJECT_SOURCE_DIR}/src/vma.cpp)

target_include_directories(vmalib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/include/lapack
    )

