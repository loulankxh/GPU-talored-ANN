cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# 依赖：cuvs-bench 环境 (libcuvs / libraft 26.10)；编译器：cuvs-build 环境 (gcc 13, cmake 4)
set(CUVS_ENV /home/lanlu/miniconda3/envs/cuvs-bench)
set(TOOL_ENV /home/lanlu/miniconda3/envs/cuvs-build)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER ${CUVS_ENV}/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${TOOL_ENV}/bin/x86_64-conda-linux-gnu-g++)
endif()
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_PREFIX_PATH ${CUVS_ENV})

project(evalCuvs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp")

find_package(cuvs CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# cagra_eval: (1) CAGRA 原生建图 baseline (NN-Descent, 64 -> 32)
#             (2) 用 cuvs CAGRA search 扫 itopk，测任意图的 recall@10 / QPS
add_executable(cagra_eval src/cagra_eval.cu)
target_link_libraries(cagra_eval PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX)
set_target_properties(cagra_eval PROPERTIES BUILD_RPATH "${CUVS_ENV}/lib")
