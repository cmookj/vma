# BUILD_TESTING variable is created by include(CTest)
# It is set to ON by defaultA
option(USE_OPENBLAS "Link to OpenBLAS library" OFF)

if (BUILD_TESTING)
  add_executable(blat_test test.cpp)

  if(APPLE)
    if (USE_OPENBLAS)
      message(STATUS "Link to OpenBLAS library")

      set(LIBOPENBLAS_PATH /opt/homebrew/opt/openblas/lib)
      set(LIBOMP_PATH /opt/homebrew/opt/libomp/lib)
      set(LIBGFORTRAN_PATH /opt/homebrew/opt/gfortran/lib/gcc/current)

      target_link_libraries(blat_test PRIVATE 
        blat
        ${LIBOPENBLAS_PATH}/libopenblas.a
        ${LIBOMP_PATH}/libomp.a
        ${LIBGFORTRAN_PATH}/libgfortran.a
        ${LIBGFORTRAN_PATH}/libgcc_s.1.1.dylib
        ${LIBGFORTRAN_PATH}/libquadmath.dylib
        GTest::gtest_main)
    else()
      message(STATUS "Link to Accelerate framework")

      find_library(ACCELERATE_LIBRARY Accelerate)

      target_link_libraries(blat_test PRIVATE 
        blat
        ${ACCELERATE_LIBRARY}
        GTest::gtest_main)
    endif()



  elseif(UNIX AND NOT APPLE)
    # Specify the path to libgfortran.a 
    set(LIBGFORTRAN_PATH /usr/lib/gcc/aarch64-linux-gnu/14)

    target_link_libraries(blat_test PRIVATE 
      blat
      libopenblas.a
      ${LIBGFORTRAN_PATH}/libgfortran.a
      GTest::gtest_main)
  endif()

  target_include_directories(blat_test PRIVATE 
    ${PROJECT_SOURCE_DIR}/blat/include
    )

  include(GoogleTest)

  # Finds all the Google tests associated with the executable
  gtest_discover_tests(blat_test)
endif()
