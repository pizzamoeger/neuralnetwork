set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)  # Path to nvcc compiler executable
enable_language(CUDA)

cmake_minimum_required(VERSION 3.16)
project(neuralnetwork)

set(CMAKE_CUDA_ARCHITECTURES 61;70) # idk what this does

#set(CMAKE_CXX_STANDARD 14)

#add_compile_options(-Wall -Wextra -std=c++17)
#add_definitions(-funroll-loops -Ofast)

add_executable(neuralnetwork train/main.cu train/Network.cu train/layer.cu train/misc.cu)
#add_executable(neuralnetwork train/debug.cu)