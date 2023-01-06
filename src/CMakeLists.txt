project(${CMAKE_PROJECT_NAME})               # Name of project

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(ALQP ALQP.cpp)              # All the cpp files part of the alqp library by default, it's a STATIC lib 

target_link_libraries (ALQP Eigen3::Eigen)

target_compile_options (ALQP PRIVATE -Ofast -O3)

# Where to find the header files
target_include_directories(ALQP
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )