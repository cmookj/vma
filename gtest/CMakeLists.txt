cmake_minimum_required(VERSION 3.14)
project(linalg_test)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    googletest 
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  linalg_test
  test.cpp
  ${PROJECT_BINARY_DIR}/../../src/linear_algebra.cpp
)

target_include_directories(linalg_test PUBLIC
    ${PROJECT_BINARY_DIR}/../../include/lalib
    )

target_link_libraries(
  linalg_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(linalg_test)

