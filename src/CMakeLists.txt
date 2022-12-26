project(${CMAKE_PROJECT_NAME})               # Name of project

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(ALQP ALQP.cpp)              # All the cpp files part of the alqp library by default, it's a STATIC lib 

target_link_libraries (ALQP Eigen3::Eigen)

# Where to find the header files
target_include_directories(ALQP
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )