project(${CMAKE_PROJECT_NAME})               # Name of project

add_executable(main main.cpp)           # target

# Link executable to alqp library
target_link_libraries(main
    PRIVATE                             # private dependency of the main executable (exectubales are usually private)
    ALQP)