cmake_minimum_required(VERSION 3.16..3.24)

set(CMAKE_CXX_STANDARD 17)

project(VectorMatrixArithmeticToolkit VERSION 0.0.1
    DESCRIPTION "Vector Matrix Arithmetic Toolkit"
    LANGUAGES CXX)

# Update the submodules automatically
include(cmake/UpdateSubmodules.cmake)

# Enable testing for this project
include(CTest)

# Add subdirectories with code
add_subdirectory(external)
add_subdirectory(vma)
add_subdirectory(test)
add_subdirectory(benchmark)
