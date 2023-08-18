cmake_minimum_required(VERSION 3.16)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -std=c++17)
#add_definitions(-funroll-loops -Ofast)

add_executable(neuralnetwork train/main.cpp train/Network.cpp train/layer.cpp train/misc.cpp)
