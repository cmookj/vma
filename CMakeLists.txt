cmake_minimum_required(VERSION 3.14)
project(vma)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
include(Dependencies.cmake)

add_subdirectory(vma)

enable_testing()

add_executable(vma_test
  test.cpp
)

option(USE_OPENBLAS "Link to OpenBLAS library" OFF)

if(APPLE)
  if (USE_OPENBLAS)
    message(STATUS "Link to OpenBLAS library")

    add_definitions(-DUSE_OPENBLAS)

    set(LIBOPENBLAS_PATH /opt/homebrew/opt/openblas/lib)
    set(LIBOMP_PATH /opt/homebrew/opt/libomp/lib)
    set(LIBGFORTRAN_PATH /opt/homebrew/opt/gfortran/lib/gcc/current)

    target_link_libraries(vma_test PRIVATE
      vma
      ${LIBOPENBLAS_PATH}/libopenblas.a
      ${LIBOMP_PATH}/libomp.a
      ${LIBGFORTRAN_PATH}/libgfortran.a
      ${LIBGFORTRAN_PATH}/libgcc_s.1.1.dylib
      ${LIBGFORTRAN_PATH}/libquadmath.dylib
      GTest::gtest_main)
  else()
    message(STATUS "Link to Accelerate framework")

    find_library(ACCELERATE_LIBRARY Accelerate)

    target_link_libraries(vma_test PRIVATE
      vma
      ${ACCELERATE_LIBRARY}
      GTest::gtest_main)
  endif()

elseif(UNIX AND NOT APPLE)
  # Specify the path to libgfortran.a
  set(LIBGFORTRAN_PATH /usr/lib/gcc/aarch64-linux-gnu/14)

  target_link_libraries(vma_test PRIVATE
    vma
    libblas.a
    ${LIBGFORTRAN_PATH}/libgfortran.a
    GTest::gtest_main)
endif()

target_link_libraries(vma_test PRIVATE
  vma
  GTest::gtest_main
)

target_include_directories(vma_test PRIVATE
  ${PROJECT_SOURCE_DIR}/vma/include
)

include(GoogleTest)
gtest_discover_tests(vma_test)

add_subdirectory(benchmark)
