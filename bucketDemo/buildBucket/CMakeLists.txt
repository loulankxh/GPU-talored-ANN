cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

# 必须在 project() 之前设置，使用 conda 工具链，避免系统版本冲突
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /home/lanlu/miniconda3/envs/rapids_raft/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /home/lanlu/miniconda3/envs/rapids_raft/bin/x86_64-conda-linux-gnu-g++)
endif()
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(
    buildBucket
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# 当前只为本机 GPU 编译以加快编译速度
# 完整多架构 (V100 sm_70 / A100 sm_80 / A10/RTX30 sm_86): set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --extended-lambda --expt-relaxed-constexpr")

find_package(Boost QUIET)
if(NOT Boost_FOUND)
    # Try to find Boost in conda
    set(Boost_INCLUDE_DIRS $ENV{CONDA_PREFIX}/include)
    set(Boost_LIBRARY_DIRS $ENV{CONDA_PREFIX}/lib)
    message(STATUS "Using Boost from CONDA: ${Boost_INCLUDE_DIRS}")
endif()
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(raft CONFIG REQUIRED
    PATHS /home/lanlu/raft/cpp/build
    NO_DEFAULT_PATH
)

if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_INCLUDE $ENV{CONDA_PREFIX}/include)
    set(CONDA_LIB $ENV{CONDA_PREFIX}/lib)
else()
    message(FATAL_ERROR "CONDA_PREFIX not found. Activate your conda env first.")
endif()

set(RAFT_INCLUDE /home/lanlu/raft/cpp/include)
set(RAFT_LIB /home/lanlu/raft/build/lib)
include_directories(
    /home/lanlu/raft/cpp/include/
    /home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include
    /home/lanlu/miniconda3/envs/rapids_raft/include
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx
    /home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include
    ${Boost_INCLUDE_DIRS}
)

link_directories(
    /home/lanlu/raft/cpp/build/_deps/benchmark-build/src
    ${Boost_LIBRARY_DIRS}
)

set(BUILD_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BUILD_SOURCES
    bucket_build.cu)
list(TRANSFORM BUILD_SOURCES PREPEND ${BUILD_SRC_PATH}/)

add_executable(bucket ${BUILD_SOURCES})

target_link_libraries(bucket
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(bucket PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# bucket2: bucket.cu (with tensor core matmul + per-vector KNN)
add_executable(bucket2 ${BUILD_SRC_PATH}/bucket.cu)

target_link_libraries(bucket2
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(bucket2 PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# TEMP DIAGNOSTIC: isolates whether linking raft::raft (e.g. RMM's custom
# allocator) breaks OpenMP's num_threads() clause, independent of calling
# any RAFT algorithm. Remove once the actual_num_threads=1 root cause is
# found and fixed. See bucketDemo/debug/omp_probe_raft.cu.
add_executable(omp_probe_raft ${CMAKE_CURRENT_SOURCE_DIR}/../debug/omp_probe_raft.cu)

target_link_libraries(omp_probe_raft
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
)

set_target_properties(omp_probe_raft PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# optimize: standalone tool that runs CAGRA sort_knn_graph + optimize on
# the vector_knn.bin produced by bucket.cu, yielding a pruned CAGRA graph.
add_executable(optimize ${BUILD_SRC_PATH}/optimize.cu)

target_link_libraries(optimize
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(optimize PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# optimize_chunked: chunked reverse-graph + merge for huge graphs that don't
# fit on a single GPU. Implements 3 methods (A: kernel-filter, B: pre-shuffle,
# C: bucket-aligned A) selectable via --method.
add_executable(optimize_chunked ${BUILD_SRC_PATH}/optimize_chunked.cu)

target_link_libraries(optimize_chunked
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    fmt
    boost_program_options
)

set_target_properties(optimize_chunked PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# reorder: CPU-only tool that reorders dataset + vector_knn.bin by bucket
# layout (consecutive new IDs per bucket), enabling optimize_chunked -m C.
add_executable(reorder ${BUILD_SRC_PATH}/reorder.cpp)

target_link_libraries(reorder
    PRIVATE
    OpenMP::OpenMP_CXX
    boost_program_options
)

set_target_properties(reorder PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)
