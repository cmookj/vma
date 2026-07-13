cmake_minimum_required(VERSION 3.25)
project(vma VERSION 0.0.1
                   DESCRIPTION "A C++ library"
                   LANGUAGES CXX)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
include(Dependencies.cmake)

# Main library target
# ------------------------------------------------------------------------------
add_library(vma STATIC
    ${PROJECT_SOURCE_DIR}/src/vma.cpp)

target_include_directories(vma PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

# Test target
# ------------------------------------------------------------------------------
add_subdirectory(tests)
