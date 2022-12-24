# BUILD_TESTING variable is created by include(CTest)
# It is set to ON by default
if (BUILD_TESTING)
    add_executable(blat_test test.cpp)

    if(APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)
        set(extra_libs ${ACCELERATE_LIBRARY})
    elseif(UNIX AND NOT APPLE)
        set(extra_libs 
            libopenblas.a
            libgfortran.a # libopenblas requires link to _gfortran_concat_string
            )
    endif()

    target_include_directories(blat_test PRIVATE 
        ${PROJECT_SOURCE_DIR}/blat/include
        )

    target_link_libraries(blat_test PRIVATE 
        blat
        ${extra_libs}
        GTest::gtest_main)

    include(GoogleTest)
    # Finds all the Google tests associated with the executable
    gtest_discover_tests(blat_test)
endif()
