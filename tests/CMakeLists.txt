project(${CMAKE_PROJECT_NAME})


find_package(OsqpEigen REQUIRED)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_NO_EXTRACT TRUE
)

FetchContent_GetProperties(googletest)

# Prevent overriding the parent project's compiler/linker settings
if (NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_FILES
    hello_test.cc
    test_ALQP.cpp)

add_executable(tests ${TEST_FILES})

target_link_libraries(tests PRIVATE ALQP GTest::gtest_main OsqpEigen::OsqpEigen)

include(GoogleTest)
gtest_discover_tests(tests)