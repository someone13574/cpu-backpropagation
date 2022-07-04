cmake_minimum_required(VERSION 3.10)

project(Machine_Learning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (UNIX)
    set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif (UNIX)

add_executable(machine_learning src/main.cpp
                             src/network.cpp
                             src/backpropagation.cpp
                             src/matrix.hpp)