cmake_minimum_required(VERSION 3.19)
project(graph1_2_2 CUDA)

add_executable(graph1_2_2 main.cu)

set_target_properties(
        graph1_2_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} “-arch=compute_53 -code=sm_53 -rdc=true -O3” )
set(CUDA_VERBOSE_BUILD ON)
