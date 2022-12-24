cmake_minimum_required(VERSION 3.16..3.24)
project(BasicLinearAlgebraToolkit VERSION 0.0.1
    DESCRIPTION "Basic Linear Algebra Toolkit"
   LANGUAGES CXX)

add_library(blat STATIC
    ${PROJECT_SOURCE_DIR}/src/blat.cpp)

target_include_directories(blat PUBLIC
    ${PROJECT_SOURCE_DIR}/include)
