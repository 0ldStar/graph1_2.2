cmake_minimum_required(VERSION 3.21)
project(graph1_2_2 CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(graph1_2_2 main.cu)
#set(CMAKE_CUDA_ARCHITECTURES 52)
set_target_properties(graph1_2_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
