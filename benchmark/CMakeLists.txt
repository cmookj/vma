#
# Bare-Bone CMakeLists.txt
#

# To create compile_commands.json
#
# $ mkdir build; cd build
# $ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
# $ cd ..; ln -s ./build/compile_commands.json ./
#
cmake_minimum_required(VERSION 3.14)

option(USE_OPENBLAS "Link to OpenBLAS library" OFF)

project(benchmark)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_executable(benchmark test.cpp)

if(APPLE)
  if (USE_OPENBLAS)
    message(STATUS "Link to OpenBLAS library")

    set(LIBOPENBLAS_PATH /opt/homebrew/opt/openblas/lib)
    set(LIBOMP_PATH /opt/homebrew/opt/libomp/lib)
    set(LIBGFORTRAN_PATH /opt/homebrew/opt/gfortran/lib/gcc/current)

    target_link_libraries(benchmark PRIVATE 
      blat
      ${LIBOPENBLAS_PATH}/libopenblas.a
      ${LIBOMP_PATH}/libomp.a
      ${LIBGFORTRAN_PATH}/libgfortran.a
      ${LIBGFORTRAN_PATH}/libgcc_s.1.1.dylib
      ${LIBGFORTRAN_PATH}/libquadmath.dylib
      )

  else()
    message(STATUS "Link to Accelerate framework")

    find_library(ACCELERATE_LIBRARY Accelerate)

    target_link_libraries(benchmark PRIVATE 
      blat
      ${ACCELERATE_LIBRARY}
      GTest::gtest_main)
  endif()

elseif(UNIX AND NOT APPLE)
  # Specify the path to libgfortran.a 
  set(LIBGFORTRAN_PATH /usr/lib/gcc/aarch64-linux-gnu/14)

  target_link_libraries(benchmark PRIVATE 
    blat
    libopenblas.a
    ${LIBGFORTRAN_PATH}/libgfortran.a
    )
endif()

target_include_directories(benchmark PRIVATE 
  ${PROJECT_SOURCE_DIR}../blat/include
  )

set(CMAKE_CXX_FLAGS_DEBUG_INIT "--debug -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -Wall -Wextra -Wpedantic")
