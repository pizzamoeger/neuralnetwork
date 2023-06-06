cmake_minimum_required(VERSION 3.24)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "-O3")

add_executable(neuralnetwork main.cpp Network.cpp layer.cpp misc.cpp)
add_definitions()
